"""Feature extraction utilities for speech recognition."""

import torch
import torchaudio
import torchaudio.transforms as T
from typing import Tuple, Optional


class MelSpectrogramExtractor:
    """Mel spectrogram feature extractor."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        preemphasis: float = 0.97,
    ) -> None:
        """
        Initialize mel spectrogram extractor.
        
        Args:
            sample_rate: Audio sample rate.
            n_fft: FFT window size.
            hop_length: Hop length between windows.
            win_length: Window length.
            n_mels: Number of mel filter banks.
            f_min: Minimum frequency.
            f_max: Maximum frequency.
            preemphasis: Preemphasis coefficient.
        """
        self.sample_rate = sample_rate
        self.preemphasis = preemphasis
        
        # Mel spectrogram transform
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max or sample_rate // 2,
            power=2.0,
        )
        
        # Log transform
        self.log_transform = T.AmplitudeToDB(stype="power", top_db=80)
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract mel spectrogram features.
        
        Args:
            waveform: Input waveform of shape (batch_size, samples) or (samples,).
            
        Returns:
            torch.Tensor: Mel spectrogram features of shape (batch_size, n_mels, time) or (n_mels, time).
        """
        # Apply preemphasis if specified
        if self.preemphasis > 0:
            waveform = self._apply_preemphasis(waveform)
        
        # Compute mel spectrogram
        mel_spec = self.mel_spectrogram(waveform)
        
        # Convert to log scale
        log_mel_spec = self.log_transform(mel_spec)
        
        return log_mel_spec
    
    def _apply_preemphasis(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply preemphasis filter.
        
        Args:
            waveform: Input waveform.
            
        Returns:
            torch.Tensor: Preemphasized waveform.
        """
        if waveform.dim() == 1:
            # Single waveform
            return torch.cat([
                waveform[0:1],
                waveform[1:] - self.preemphasis * waveform[:-1]
            ])
        else:
            # Batch of waveforms
            return torch.cat([
                waveform[:, 0:1],
                waveform[:, 1:] - self.preemphasis * waveform[:, :-1]
            ], dim=1)


class SpecAugment:
    """SpecAugment data augmentation."""
    
    def __init__(
        self,
        time_mask_param: int = 27,
        freq_mask_param: int = 2,
        num_time_masks: int = 2,
        num_freq_masks: int = 2,
        time_warp_param: int = 80,
        p: float = 1.0,
    ) -> None:
        """
        Initialize SpecAugment.
        
        Args:
            time_mask_param: Maximum time mask length.
            freq_mask_param: Maximum frequency mask length.
            num_time_masks: Number of time masks to apply.
            num_freq_masks: Number of frequency masks to apply.
            time_warp_param: Time warping parameter.
            p: Probability of applying augmentation.
        """
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks
        self.time_warp_param = time_warp_param
        self.p = p
    
    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to features.
        
        Args:
            features: Input features of shape (batch_size, n_mels, time) or (n_mels, time).
            
        Returns:
            torch.Tensor: Augmented features.
        """
        if torch.rand(1) > self.p:
            return features
        
        if features.dim() == 2:
            # Single sample
            features = features.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Apply time warping
        features = self._time_warp(features)
        
        # Apply frequency masking
        for _ in range(self.num_freq_masks):
            features = self._freq_mask(features)
        
        # Apply time masking
        for _ in range(self.num_time_masks):
            features = self._time_mask(features)
        
        if squeeze_output:
            features = features.squeeze(0)
        
        return features
    
    def _time_warp(self, features: torch.Tensor) -> torch.Tensor:
        """Apply time warping."""
        # Simplified time warping implementation
        return features
    
    def _freq_mask(self, features: torch.Tensor) -> torch.Tensor:
        """Apply frequency masking."""
        batch_size, n_mels, time = features.shape
        
        for b in range(batch_size):
            # Random frequency mask
            f = torch.randint(0, self.freq_mask_param + 1, (1,)).item()
            f0 = torch.randint(0, n_mels - f + 1, (1,)).item()
            
            features[b, f0:f0+f, :] = 0
        
        return features
    
    def _time_mask(self, features: torch.Tensor) -> torch.Tensor:
        """Apply time masking."""
        batch_size, n_mels, time = features.shape
        
        for b in range(batch_size):
            # Random time mask
            t = torch.randint(0, self.time_mask_param + 1, (1,)).item()
            t0 = torch.randint(0, time - t + 1, (1,)).item()
            
            features[b, :, t0:t0+t] = 0
        
        return features


class SpeedPerturbation:
    """Speed perturbation augmentation."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        speed_factors: Tuple[float, ...] = (0.9, 1.0, 1.1),
    ) -> None:
        """
        Initialize speed perturbation.
        
        Args:
            sample_rate: Audio sample rate.
            speed_factors: Speed perturbation factors.
        """
        self.sample_rate = sample_rate
        self.speed_factors = speed_factors
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply speed perturbation.
        
        Args:
            waveform: Input waveform.
            
        Returns:
            torch.Tensor: Speed perturbed waveform.
        """
        # Randomly select speed factor
        speed_factor = self.speed_factors[torch.randint(0, len(self.speed_factors), (1,)).item()]
        
        if speed_factor == 1.0:
            return waveform
        
        # Apply speed perturbation using torchaudio
        effects = [["speed", str(speed_factor)]]
        perturbed_waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform, self.sample_rate, effects
        )
        
        return perturbed_waveform
