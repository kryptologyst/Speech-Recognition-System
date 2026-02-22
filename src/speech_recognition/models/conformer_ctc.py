"""Conformer CTC model implementation for speech recognition."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class ConformerBlock(nn.Module):
    """Conformer block implementation."""
    
    def __init__(
        self,
        encoder_dim: int,
        num_attention_heads: int,
        feedforward_expansion_factor: int,
        conv_expansion_factor: int,
        dropout: float,
        conv_kernel_size: int,
    ) -> None:
        """
        Initialize Conformer block.
        
        Args:
            encoder_dim: Encoder dimension.
            num_attention_heads: Number of attention heads.
            feedforward_expansion_factor: Feedforward expansion factor.
            conv_expansion_factor: Convolution expansion factor.
            dropout: Dropout rate.
            conv_kernel_size: Convolution kernel size.
        """
        super().__init__()
        
        self.encoder_dim = encoder_dim
        self.feedforward_expansion_factor = feedforward_expansion_factor
        self.conv_expansion_factor = conv_expansion_factor
        
        # Feed forward module 1
        self.feed_forward1 = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, encoder_dim * feedforward_expansion_factor),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim * feedforward_expansion_factor, encoder_dim),
            nn.Dropout(dropout),
        )
        
        # Multi-head self attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=encoder_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_attention_norm = nn.LayerNorm(encoder_dim)
        
        # Convolution module
        self.conv_module = ConvolutionModule(
            encoder_dim=encoder_dim,
            conv_expansion_factor=conv_expansion_factor,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
        )
        
        # Feed forward module 2
        self.feed_forward2 = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, encoder_dim * feedforward_expansion_factor),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim * feedforward_expansion_factor, encoder_dim),
            nn.Dropout(dropout),
        )
        
        self.final_norm = nn.LayerNorm(encoder_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through Conformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, encoder_dim).
            mask: Optional attention mask.
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, encoder_dim).
        """
        # Feed forward 1
        residual = x
        x = self.feed_forward1(x)
        x = residual + 0.5 * x
        
        # Multi-head self attention
        residual = x
        x = self.self_attention_norm(x)
        x, _ = self.self_attention(x, x, x, key_padding_mask=mask)
        x = residual + x
        
        # Convolution module
        residual = x
        x = self.conv_module(x)
        x = residual + x
        
        # Feed forward 2
        residual = x
        x = self.feed_forward2(x)
        x = residual + 0.5 * x
        
        return self.final_norm(x)


class ConvolutionModule(nn.Module):
    """Convolution module for Conformer."""
    
    def __init__(
        self,
        encoder_dim: int,
        conv_expansion_factor: int,
        conv_kernel_size: int,
        dropout: float,
    ) -> None:
        """
        Initialize convolution module.
        
        Args:
            encoder_dim: Encoder dimension.
            conv_expansion_factor: Convolution expansion factor.
            conv_kernel_size: Convolution kernel size.
            dropout: Dropout rate.
        """
        super().__init__()
        
        expanded_dim = encoder_dim * conv_expansion_factor
        
        self.layer_norm = nn.LayerNorm(encoder_dim)
        self.pointwise_conv1 = nn.Conv1d(encoder_dim, expanded_dim, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            expanded_dim // 2,
            expanded_dim // 2,
            kernel_size=conv_kernel_size,
            padding=(conv_kernel_size - 1) // 2,
            groups=expanded_dim // 2,
        )
        self.batch_norm = nn.BatchNorm1d(expanded_dim // 2)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(expanded_dim // 2, encoder_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through convolution module.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, encoder_dim).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, encoder_dim).
        """
        # Layer norm
        x = self.layer_norm(x)
        
        # Transpose for conv1d: (batch_size, encoder_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Pointwise conv1d
        x = self.pointwise_conv1(x)
        
        # GLU activation
        x = self.glu(x)
        
        # Depthwise conv1d
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        
        # Pointwise conv1d
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        # Transpose back: (batch_size, seq_len, encoder_dim)
        x = x.transpose(1, 2)
        
        return x


class ConformerCTC(nn.Module):
    """Conformer model with CTC loss for speech recognition."""
    
    def __init__(
        self,
        input_dim: int,
        encoder_dim: int,
        num_encoder_layers: int,
        num_attention_heads: int,
        feedforward_expansion_factor: int,
        conv_expansion_factor: int,
        dropout: float,
        conv_kernel_size: int,
        vocab_size: int,
        blank_id: int = 0,
    ) -> None:
        """
        Initialize Conformer CTC model.
        
        Args:
            input_dim: Input feature dimension.
            encoder_dim: Encoder dimension.
            num_encoder_layers: Number of encoder layers.
            num_attention_heads: Number of attention heads.
            feedforward_expansion_factor: Feedforward expansion factor.
            conv_expansion_factor: Convolution expansion factor.
            dropout: Dropout rate.
            conv_kernel_size: Convolution kernel size.
            vocab_size: Vocabulary size.
            blank_id: Blank token ID for CTC.
        """
        super().__init__()
        
        self.encoder_dim = encoder_dim
        self.vocab_size = vocab_size
        self.blank_id = blank_id
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, encoder_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(encoder_dim, dropout)
        
        # Conformer blocks
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                encoder_dim=encoder_dim,
                num_attention_heads=num_attention_heads,
                feedforward_expansion_factor=feedforward_expansion_factor,
                conv_expansion_factor=conv_expansion_factor,
                dropout=dropout,
                conv_kernel_size=conv_kernel_size,
            )
            for _ in range(num_encoder_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(encoder_dim, vocab_size)
        
        # CTC loss
        self.ctc_loss = nn.CTCLoss(blank=blank_id, reduction="mean", zero_infinity=True)
    
    def forward(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through Conformer CTC model.
        
        Args:
            features: Input features of shape (batch_size, seq_len, input_dim).
            feature_lengths: Feature sequence lengths.
            targets: Target sequences for training.
            target_lengths: Target sequence lengths.
            
        Returns:
            Tuple of (logits, loss): Logits and optional CTC loss.
        """
        batch_size, seq_len, _ = features.shape
        
        # Input projection
        x = self.input_projection(features)
        
        # Positional encoding
        x = self.pos_encoding(x)
        
        # Create attention mask
        mask = self._create_attention_mask(feature_lengths, seq_len, x.device)
        
        # Conformer blocks
        for conformer_block in self.conformer_blocks:
            x = conformer_block(x, mask)
        
        # Output projection
        logits = self.output_projection(x)
        
        # Compute CTC loss if targets provided
        loss = None
        if targets is not None and target_lengths is not None:
            # Transpose for CTC: (seq_len, batch_size, vocab_size)
            logits_t = logits.transpose(0, 1)
            loss = self.ctc_loss(logits_t, targets, feature_lengths, target_lengths)
        
        return logits, loss
    
    def _create_attention_mask(
        self,
        lengths: torch.Tensor,
        max_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create attention mask for variable length sequences.
        
        Args:
            lengths: Sequence lengths.
            max_len: Maximum sequence length.
            device: Device to create mask on.
            
        Returns:
            torch.Tensor: Attention mask.
        """
        batch_size = lengths.size(0)
        mask = torch.arange(max_len, device=device).expand(
            batch_size, max_len
        ) >= lengths.unsqueeze(1)
        return mask


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension.
            dropout: Dropout rate.
            max_len: Maximum sequence length.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            
        Returns:
            torch.Tensor: Input with positional encoding added.
        """
        x = x + self.pe[: x.size(1), :].transpose(0, 1)
        return self.dropout(x)
