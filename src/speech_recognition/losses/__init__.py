"""Loss functions for speech recognition models."""

from typing import Optional

import torch
import torch.nn as nn


class CTCLoss(nn.Module):
    """CTC loss with additional options."""
    
    def __init__(
        self,
        blank_id: int = 0,
        reduction: str = "mean",
        zero_infinity: bool = True,
    ) -> None:
        """
        Initialize CTC loss.
        
        Args:
            blank_id: ID of the blank token.
            reduction: Reduction method ('mean', 'sum', 'none').
            zero_infinity: Whether to zero out infinite losses.
        """
        super().__init__()
        self.blank_id = blank_id
        self.reduction = reduction
        self.zero_infinity = zero_infinity
        
        self.ctc_loss = nn.CTCLoss(
            blank=blank_id,
            reduction=reduction,
            zero_infinity=zero_infinity,
        )
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CTC loss.
        
        Args:
            logits: Model predictions of shape (seq_len, batch_size, vocab_size).
            targets: Target sequences of shape (batch_size, max_target_len).
            input_lengths: Input sequence lengths.
            target_lengths: Target sequence lengths.
            
        Returns:
            torch.Tensor: CTC loss value.
        """
        return self.ctc_loss(logits, targets, input_lengths, target_lengths)
