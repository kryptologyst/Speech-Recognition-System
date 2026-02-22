"""Main training script for speech recognition system."""

import logging
from pathlib import Path
from typing import Dict, Any

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from speech_recognition.utils import get_device, set_seed, setup_logging, log_experiment_config
from speech_recognition.models import ConformerCTC
from speech_recognition.data import LibriSpeechDataset
from speech_recognition.losses import CTCLoss
from speech_recognition.metrics import Evaluator


class Trainer:
    """Main trainer class for speech recognition models."""
    
    def __init__(self, config: DictConfig) -> None:
        """
        Initialize trainer.
        
        Args:
            config: Training configuration.
        """
        self.config = config
        
        # Set up device and seeding
        self.device = get_device(config.device)
        set_seed(config.seed)
        
        # Set up logging
        self.logger = setup_logging(
            log_dir=config.logging.log_dir,
            level=config.logging.level,
            use_wandb=config.logging.use_wandb,
            wandb_project=config.logging.wandb_project,
            wandb_entity=config.logging.wandb_entity,
        )
        
        # Log configuration
        log_experiment_config(config, self.logger)
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Initialize datasets
        self.train_dataset = self._create_dataset(is_training=True)
        self.val_dataset = self._create_dataset(is_training=False)
        
        # Initialize data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.data.batch_size,
            shuffle=config.data.shuffle,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            collate_fn=self.train_dataset.collate_fn,
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            collate_fn=self.val_dataset.collate_fn,
        )
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Initialize loss function
        self.criterion = CTCLoss(
            blank_id=config.model.blank_id,
            reduction=config.training.loss.reduction,
            zero_infinity=config.training.loss.zero_infinity,
        )
        
        # Initialize evaluator
        self.evaluator = Evaluator(config.evaluation)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_wer = float('inf')
    
    def _create_model(self) -> nn.Module:
        """Create model from configuration."""
        model_config = self.config.model
        
        return ConformerCTC(
            input_dim=model_config.input_dim,
            encoder_dim=model_config.encoder_dim,
            num_encoder_layers=model_config.num_encoder_layers,
            num_attention_heads=model_config.num_attention_heads,
            feedforward_expansion_factor=model_config.feedforward_expansion_factor,
            conv_expansion_factor=model_config.conv_expansion_factor,
            dropout=model_config.dropout,
            conv_kernel_size=model_config.conv_kernel_size,
            vocab_size=model_config.vocab_size,
            blank_id=model_config.blank_id,
        )
    
    def _create_dataset(self, is_training: bool) -> LibriSpeechDataset:
        """Create dataset from configuration."""
        data_config = self.config.data
        
        return LibriSpeechDataset(
            data_root=data_config.data_root,
            split=data_config.split,
            sample_rate=data_config.sample_rate,
            n_fft=data_config.n_fft,
            hop_length=data_config.hop_length,
            win_length=data_config.win_length,
            n_mels=data_config.n_mels,
            f_min=data_config.f_min,
            f_max=data_config.f_max,
            preemphasis=data_config.preemphasis,
            max_audio_length=data_config.max_audio_length,
            max_text_length=data_config.max_text_length,
            tokenizer_type=data_config.tokenizer_type,
            augmentation=data_config.augmentation if is_training else None,
            is_training=is_training,
        )
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer from configuration."""
        optimizer_config = self.config.training.optimizer
        
        return hydra.utils.instantiate(
            optimizer_config,
            params=self.model.parameters(),
        )
    
    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler from configuration."""
        scheduler_config = self.config.training.scheduler
        
        # Set total steps based on dataset size
        total_steps = len(self.train_loader) * self.config.training.max_epochs
        scheduler_config.total_steps = total_steps
        
        return hydra.utils.instantiate(
            scheduler_config,
            optimizer=self.optimizer,
        )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            features = batch["features"].to(self.device)
            targets = batch["targets"].to(self.device)
            feature_lengths = batch["feature_lengths"].to(self.device)
            target_lengths = batch["target_lengths"].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, loss = self.model(features, feature_lengths, targets, target_lengths)
            
            if loss is None:
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.training.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip_val,
                )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % self.config.training.log_every_n_steps == 0:
                self.logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}, "
                    f"Loss: {loss.item():.4f}, LR: {self.scheduler.get_last_lr()[0]:.6f}"
                )
        
        return {"train_loss": total_loss / num_batches if num_batches > 0 else 0.0}
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_references = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                features = batch["features"].to(self.device)
                targets = batch["targets"].to(self.device)
                feature_lengths = batch["feature_lengths"].to(self.device)
                target_lengths = batch["target_lengths"].to(self.device)
                
                # Forward pass
                logits, loss = self.model(features, feature_lengths, targets, target_lengths)
                
                if loss is not None:
                    total_loss += loss.item()
                    num_batches += 1
                
                # Decode predictions (simplified greedy decoding)
                predictions = self._decode_predictions(logits, feature_lengths)
                
                # Convert targets to text
                references = batch["texts"]
                
                all_predictions.extend(predictions)
                all_references.extend(references)
        
        # Calculate metrics
        val_loss = total_loss / num_batches if num_batches > 0 else 0.0
        metrics = self.evaluator.evaluate(all_predictions, all_references)
        
        return {
            "val_loss": val_loss,
            **metrics,
        }
    
    def _decode_predictions(self, logits: torch.Tensor, lengths: torch.Tensor) -> list[str]:
        """Decode model predictions to text."""
        # Simplified greedy decoding
        predictions = []
        
        for i in range(logits.size(0)):
            # Get logits for this sample
            sample_logits = logits[i, :lengths[i]]
            
            # Greedy decoding
            predicted_ids = torch.argmax(sample_logits, dim=-1)
            
            # Remove consecutive duplicates and blank tokens
            decoded_ids = []
            prev_id = -1
            for token_id in predicted_ids:
                if token_id != prev_id and token_id != self.config.model.blank_id:
                    decoded_ids.append(token_id.item())
                prev_id = token_id
            
            # Convert to text
            text = self.train_dataset._tokens_to_text(decoded_ids)
            predictions.append(text)
        
        return predictions
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.checkpointing.save_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }
        
        # Save last checkpoint
        if self.config.checkpointing.save_last:
            torch.save(checkpoint, checkpoint_dir / "last.ckpt")
        
        # Save best checkpoint
        if self.config.checkpointing.save_best:
            monitor_metric = self.config.checkpointing.monitor
            if monitor_metric in metrics:
                current_value = metrics[monitor_metric]
                is_better = (
                    current_value < self.best_val_loss
                    if self.config.checkpointing.mode == "min"
                    else current_value > self.best_val_loss
                )
                
                if is_better:
                    self.best_val_loss = current_value
                    torch.save(checkpoint, checkpoint_dir / "best.ckpt")
                    self.logger.info(f"New best {monitor_metric}: {current_value:.4f}")
    
    def train(self) -> None:
        """Main training loop."""
        self.logger.info("Starting training...")
        
        for epoch in range(self.config.training.max_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}
            
            # Log metrics
            self.logger.info(f"Epoch {epoch} Metrics:")
            for metric, value in all_metrics.items():
                self.logger.info(f"  {metric}: {value:.4f}")
            
            # Save checkpoint
            self.save_checkpoint(epoch, all_metrics)
            
            # Early stopping check
            if hasattr(self.config.training, "early_stopping"):
                early_stopping = self.config.training.early_stopping
                monitor_metric = early_stopping.monitor
                
                if monitor_metric in all_metrics:
                    current_value = all_metrics[monitor_metric]
                    
                    if current_value < self.best_val_wer:
                        self.best_val_wer = current_value
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= early_stopping.patience:
                        self.logger.info(f"Early stopping at epoch {epoch}")
                        break
        
        self.logger.info("Training completed!")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    """Main training function."""
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
