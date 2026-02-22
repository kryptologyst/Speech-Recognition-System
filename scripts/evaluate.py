"""Evaluation script for speech recognition models."""

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import torch
import hydra
from omegaconf import DictConfig

from speech_recognition.utils import get_device, set_seed, setup_logging
from speech_recognition.models import ConformerCTC
from speech_recognition.data import LibriSpeechDataset
from speech_recognition.metrics import Evaluator


def evaluate_model(
    checkpoint_path: str,
    config: DictConfig,
    output_dir: str = "assets",
) -> Dict[str, float]:
    """
    Evaluate a trained model.
    
    Args:
        checkpoint_path: Path to model checkpoint.
        config: Evaluation configuration.
        output_dir: Output directory for results.
        
    Returns:
        Dict containing evaluation metrics.
    """
    # Set up device and seeding
    device = get_device(config.device)
    set_seed(config.seed)
    
    # Set up logging
    logger = setup_logging(
        log_dir=config.logging.log_dir,
        level=config.logging.level,
    )
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint["config"].model
    
    # Create model
    model = ConformerCTC(
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
    
    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # Create test dataset
    test_dataset = LibriSpeechDataset(
        data_root=config.data.data_root,
        split="test-clean",  # Use test set for evaluation
        sample_rate=config.data.sample_rate,
        n_fft=config.data.n_fft,
        hop_length=config.data.hop_length,
        win_length=config.data.win_length,
        n_mels=config.data.n_mels,
        f_min=config.data.f_min,
        f_max=config.data.f_max,
        preemphasis=config.data.preemphasis,
        max_audio_length=config.data.max_audio_length,
        max_text_length=config.data.max_text_length,
        tokenizer_type=config.data.tokenizer_type,
        augmentation=None,  # No augmentation for evaluation
        is_training=False,
    )
    
    # Create data loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=test_dataset.collate_fn,
    )
    
    # Initialize evaluator
    evaluator = Evaluator(config.evaluation)
    
    # Run evaluation
    logger.info("Starting evaluation...")
    
    all_predictions = []
    all_references = []
    all_audio_durations = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Move batch to device
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)
            feature_lengths = batch["feature_lengths"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            
            # Forward pass
            logits, _ = model(features, feature_lengths, targets, target_lengths)
            
            # Decode predictions
            predictions = decode_predictions(logits, feature_lengths, test_dataset)
            
            # Get references
            references = batch["texts"]
            
            # Calculate audio durations (approximate)
            durations = [length.item() * config.data.hop_length / config.data.sample_rate 
                        for length in feature_lengths]
            
            all_predictions.extend(predictions)
            all_references.extend(references)
            all_audio_durations.extend(durations)
            
            if batch_idx % 10 == 0:
                logger.info(f"Processed {batch_idx * config.data.batch_size} samples")
    
    # Calculate metrics
    logger.info("Calculating metrics...")
    metrics = evaluator.evaluate(all_predictions, all_references, all_audio_durations)
    
    # Create leaderboard
    leaderboard = evaluator.create_leaderboard(metrics)
    logger.info(f"\n{leaderboard}")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    with open(output_path / "evaluation_results.txt", "w") as f:
        f.write(leaderboard)
    
    # Save predictions
    with open(output_path / "predictions.txt", "w") as f:
        for pred, ref in zip(all_predictions, all_references):
            f.write(f"Prediction: {pred}\n")
            f.write(f"Reference: {ref}\n")
            f.write("-" * 50 + "\n")
    
    logger.info(f"Results saved to {output_path}")
    
    return metrics


def decode_predictions(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    dataset: LibriSpeechDataset,
) -> List[str]:
    """
    Decode model predictions to text.
    
    Args:
        logits: Model logits.
        lengths: Sequence lengths.
        dataset: Dataset for token conversion.
        
    Returns:
        List of decoded predictions.
    """
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
            if token_id != prev_id and token_id != dataset.token_to_id["<blank>"]:
                decoded_ids.append(token_id.item())
            prev_id = token_id
        
        # Convert to text
        text = dataset._tokens_to_text(decoded_ids)
        predictions.append(text)
    
    return predictions


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate speech recognition model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="assets",
        help="Output directory for results",
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    metrics = evaluate_model(
        checkpoint_path=args.checkpoint,
        config=config,
        output_dir=args.output_dir,
    )
    
    print(f"Evaluation completed. Metrics: {metrics}")


if __name__ == "__main__":
    main()
