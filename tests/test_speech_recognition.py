"""Test script for speech recognition system."""

import pytest
import torch
import numpy as np

from speech_recognition.models import ConformerCTC, ConformerBlock, ConvolutionModule
from speech_recognition.features import MelSpectrogramExtractor, SpecAugment
from speech_recognition.metrics import WERCalculator, CERCalculator, TokenAccuracyCalculator
from speech_recognition.utils import get_device, set_seed


class TestConformerCTC:
    """Test Conformer CTC model."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = ConformerCTC(
            input_dim=80,
            encoder_dim=512,
            num_encoder_layers=2,  # Small for testing
            num_attention_heads=8,
            feedforward_expansion_factor=4,
            conv_expansion_factor=2,
            dropout=0.1,
            conv_kernel_size=31,
            vocab_size=1000,
            blank_id=0,
        )
        
        assert model.encoder_dim == 512
        assert model.vocab_size == 1000
        assert model.blank_id == 0
    
    def test_model_forward(self):
        """Test model forward pass."""
        model = ConformerCTC(
            input_dim=80,
            encoder_dim=256,  # Smaller for testing
            num_encoder_layers=2,
            num_attention_heads=4,
            feedforward_expansion_factor=2,
            conv_expansion_factor=2,
            dropout=0.1,
            conv_kernel_size=15,
            vocab_size=100,
            blank_id=0,
        )
        
        # Create dummy input
        batch_size = 2
        seq_len = 100
        input_dim = 80
        
        features = torch.randn(batch_size, seq_len, input_dim)
        feature_lengths = torch.tensor([seq_len, seq_len])
        
        # Forward pass
        logits, loss = model(features, feature_lengths)
        
        assert logits.shape == (batch_size, seq_len, 100)
        assert loss is None  # No targets provided
    
    def test_model_with_targets(self):
        """Test model with targets."""
        model = ConformerCTC(
            input_dim=80,
            encoder_dim=256,
            num_encoder_layers=2,
            num_attention_heads=4,
            feedforward_expansion_factor=2,
            conv_expansion_factor=2,
            dropout=0.1,
            conv_kernel_size=15,
            vocab_size=100,
            blank_id=0,
        )
        
        # Create dummy input
        batch_size = 2
        seq_len = 100
        input_dim = 80
        
        features = torch.randn(batch_size, seq_len, input_dim)
        feature_lengths = torch.tensor([seq_len, seq_len])
        targets = torch.tensor([[1, 2, 3], [4, 5, 6]])
        target_lengths = torch.tensor([3, 3])
        
        # Forward pass
        logits, loss = model(features, feature_lengths, targets, target_lengths)
        
        assert logits.shape == (batch_size, seq_len, 100)
        assert loss is not None
        assert loss.item() >= 0


class TestConformerBlock:
    """Test Conformer block."""
    
    def test_conformer_block_forward(self):
        """Test Conformer block forward pass."""
        block = ConformerBlock(
            encoder_dim=256,
            num_attention_heads=4,
            feedforward_expansion_factor=2,
            conv_expansion_factor=2,
            dropout=0.1,
            conv_kernel_size=15,
        )
        
        # Create dummy input
        batch_size = 2
        seq_len = 50
        encoder_dim = 256
        
        x = torch.randn(batch_size, seq_len, encoder_dim)
        
        # Forward pass
        output = block(x)
        
        assert output.shape == (batch_size, seq_len, encoder_dim)


class TestConvolutionModule:
    """Test convolution module."""
    
    def test_convolution_module_forward(self):
        """Test convolution module forward pass."""
        conv_module = ConvolutionModule(
            encoder_dim=256,
            conv_expansion_factor=2,
            conv_kernel_size=15,
            dropout=0.1,
        )
        
        # Create dummy input
        batch_size = 2
        seq_len = 50
        encoder_dim = 256
        
        x = torch.randn(batch_size, seq_len, encoder_dim)
        
        # Forward pass
        output = conv_module(x)
        
        assert output.shape == (batch_size, seq_len, encoder_dim)


class TestFeatureExtraction:
    """Test feature extraction."""
    
    def test_mel_spectrogram_extractor(self):
        """Test mel spectrogram extractor."""
        extractor = MelSpectrogramExtractor(
            sample_rate=16000,
            n_fft=512,
            hop_length=160,
            win_length=400,
            n_mels=80,
        )
        
        # Create dummy audio
        duration = 1.0  # 1 second
        sample_rate = 16000
        audio = torch.randn(int(duration * sample_rate))
        
        # Extract features
        features = extractor(audio)
        
        assert features.shape[0] == 80  # n_mels
        assert features.shape[1] > 0  # time frames
    
    def test_spec_augment(self):
        """Test SpecAugment."""
        spec_augment = SpecAugment(
            time_mask_param=10,
            freq_mask_param=2,
            num_time_masks=1,
            num_freq_masks=1,
        )
        
        # Create dummy features
        n_mels = 80
        time_frames = 100
        features = torch.randn(n_mels, time_frames)
        
        # Apply augmentation
        augmented = spec_augment(features)
        
        assert augmented.shape == features.shape


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_wer_calculator(self):
        """Test WER calculator."""
        wer_calc = WERCalculator()
        
        predictions = ["hello world", "good morning"]
        references = ["hello world", "good evening"]
        
        wer = wer_calc(predictions, references)
        
        assert wer >= 0
        assert wer <= 1
    
    def test_cer_calculator(self):
        """Test CER calculator."""
        cer_calc = CERCalculator()
        
        predictions = ["hello", "world"]
        references = ["hello", "word"]
        
        cer = cer_calc(predictions, references)
        
        assert cer >= 0
        assert cer <= 1
    
    def test_token_accuracy_calculator(self):
        """Test token accuracy calculator."""
        token_acc_calc = TokenAccuracyCalculator()
        
        predictions = ["hello world", "good morning"]
        references = ["hello world", "good morning"]
        
        accuracy = token_acc_calc(predictions, references)
        
        assert accuracy >= 0
        assert accuracy <= 1
        assert accuracy == 1.0  # Perfect match


class TestUtils:
    """Test utility functions."""
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device("auto")
        assert isinstance(device, torch.device)
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Generate some random numbers
        rand1 = torch.randn(10)
        set_seed(42)
        rand2 = torch.randn(10)
        
        # Should be the same with same seed
        assert torch.allclose(rand1, rand2)


if __name__ == "__main__":
    pytest.main([__file__])
