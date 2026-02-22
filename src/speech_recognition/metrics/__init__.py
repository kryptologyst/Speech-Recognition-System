"""Evaluation metrics for speech recognition."""

import time
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from omegaconf import DictConfig


class WERCalculator:
    """Word Error Rate calculator."""
    
    def __init__(self) -> None:
        """Initialize WER calculator."""
        pass
    
    def __call__(self, predictions: List[str], references: List[str]) -> float:
        """
        Calculate Word Error Rate.
        
        Args:
            predictions: List of predicted transcriptions.
            references: List of reference transcriptions.
            
        Returns:
            float: Word Error Rate.
        """
        total_errors = 0
        total_words = 0
        
        for pred, ref in zip(predictions, references):
            pred_words = pred.lower().split()
            ref_words = ref.lower().split()
            
            # Calculate edit distance
            errors = self._edit_distance(pred_words, ref_words)
            total_errors += errors
            total_words += len(ref_words)
        
        return total_errors / total_words if total_words > 0 else 0.0
    
    def _edit_distance(self, pred: List[str], ref: List[str]) -> int:
        """Calculate edit distance between two word sequences."""
        m, n = len(pred), len(ref)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill the DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred[i - 1] == ref[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],      # deletion
                        dp[i][j - 1],      # insertion
                        dp[i - 1][j - 1]   # substitution
                    )
        
        return dp[m][n]


class CERCalculator:
    """Character Error Rate calculator."""
    
    def __init__(self) -> None:
        """Initialize CER calculator."""
        pass
    
    def __call__(self, predictions: List[str], references: List[str]) -> float:
        """
        Calculate Character Error Rate.
        
        Args:
            predictions: List of predicted transcriptions.
            references: List of reference transcriptions.
            
        Returns:
            float: Character Error Rate.
        """
        total_errors = 0
        total_chars = 0
        
        for pred, ref in zip(predictions, references):
            pred_chars = list(pred.lower())
            ref_chars = list(ref.lower())
            
            # Calculate edit distance
            errors = self._edit_distance(pred_chars, ref_chars)
            total_errors += errors
            total_chars += len(ref_chars)
        
        return total_errors / total_chars if total_chars > 0 else 0.0
    
    def _edit_distance(self, pred: List[str], ref: List[str]) -> int:
        """Calculate edit distance between two character sequences."""
        m, n = len(pred), len(ref)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill the DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred[i - 1] == ref[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],      # deletion
                        dp[i][j - 1],      # insertion
                        dp[i - 1][j - 1]   # substitution
                    )
        
        return dp[m][n]


class TokenAccuracyCalculator:
    """Token accuracy calculator."""
    
    def __init__(self) -> None:
        """Initialize token accuracy calculator."""
        pass
    
    def __call__(self, predictions: List[str], references: List[str]) -> float:
        """
        Calculate token accuracy.
        
        Args:
            predictions: List of predicted transcriptions.
            references: List of reference transcriptions.
            
        Returns:
            float: Token accuracy.
        """
        total_correct = 0
        total_tokens = 0
        
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()
            
            # Calculate correct tokens
            correct = sum(1 for p, r in zip(pred_tokens, ref_tokens) if p == r)
            total_correct += correct
            total_tokens += len(ref_tokens)
        
        return total_correct / total_tokens if total_tokens > 0 else 0.0


class LatencyCalculator:
    """Latency calculator for inference time."""
    
    def __init__(self) -> None:
        """Initialize latency calculator."""
        self.times = []
    
    def start_timer(self) -> None:
        """Start timing."""
        self.start_time = time.time()
    
    def end_timer(self) -> None:
        """End timing and record."""
        end_time = time.time()
        self.times.append(end_time - self.start_time)
    
    def get_average_latency(self) -> float:
        """Get average latency in seconds."""
        return np.mean(self.times) if self.times else 0.0
    
    def get_rtf(self, audio_duration: float) -> float:
        """
        Calculate Real Time Factor.
        
        Args:
            audio_duration: Duration of audio in seconds.
            
        Returns:
            float: Real Time Factor.
        """
        avg_latency = self.get_average_latency()
        return avg_latency / audio_duration if audio_duration > 0 else 0.0


class Evaluator:
    """Main evaluator for speech recognition models."""
    
    def __init__(self, config: DictConfig) -> None:
        """
        Initialize evaluator.
        
        Args:
            config: Evaluation configuration.
        """
        self.config = config
        self.metrics = config.get("metrics", ["wer", "cer"])
        
        # Initialize metric calculators
        self.wer_calculator = WERCalculator()
        self.cer_calculator = CERCalculator()
        self.token_accuracy_calculator = TokenAccuracyCalculator()
        self.latency_calculator = LatencyCalculator()
    
    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
        audio_durations: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate predictions against references.
        
        Args:
            predictions: List of predicted transcriptions.
            references: List of reference transcriptions.
            audio_durations: Optional list of audio durations for RTF calculation.
            
        Returns:
            Dict containing evaluation metrics.
        """
        results = {}
        
        # Calculate metrics
        if "wer" in self.metrics:
            results["wer"] = self.wer_calculator(predictions, references)
        
        if "cer" in self.metrics:
            results["cer"] = self.cer_calculator(predictions, references)
        
        if "token_accuracy" in self.metrics:
            results["token_accuracy"] = self.token_accuracy_calculator(predictions, references)
        
        if "latency" in self.metrics:
            results["latency"] = self.latency_calculator.get_average_latency()
        
        if "rtf" in self.metrics and audio_durations:
            avg_rtf = np.mean([
                self.latency_calculator.get_rtf(duration)
                for duration in audio_durations
            ])
            results["rtf"] = avg_rtf
        
        return results
    
    def create_leaderboard(self, results: Dict[str, float]) -> str:
        """
        Create a formatted leaderboard string.
        
        Args:
            results: Evaluation results.
            
        Returns:
            str: Formatted leaderboard.
        """
        leaderboard = "Speech Recognition Evaluation Results\n"
        leaderboard += "=" * 40 + "\n"
        
        for metric, value in results.items():
            if metric in ["wer", "cer"]:
                leaderboard += f"{metric.upper()}: {value:.4f} ({value*100:.2f}%)\n"
            elif metric == "token_accuracy":
                leaderboard += f"Token Accuracy: {value:.4f} ({value*100:.2f}%)\n"
            elif metric == "latency":
                leaderboard += f"Latency: {value:.4f} seconds\n"
            elif metric == "rtf":
                leaderboard += f"RTF: {value:.4f}\n"
            else:
                leaderboard += f"{metric}: {value:.4f}\n"
        
        return leaderboard
