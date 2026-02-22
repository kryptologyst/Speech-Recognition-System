"""Logging utilities for structured logging and experiment tracking."""

import logging
import os
from pathlib import Path
from typing import Optional

import wandb
from omegaconf import DictConfig


def setup_logging(
    log_dir: str = "logs",
    level: str = "INFO",
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
) -> logging.Logger:
    """
    Set up structured logging for the application.
    
    Args:
        log_dir: Directory to store log files.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        use_wandb: Whether to use Weights & Biases for experiment tracking.
        wandb_project: W&B project name.
        wandb_entity: W&B entity name.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path / "speech_recognition.log"),
            logging.StreamHandler(),
        ],
    )
    
    logger = logging.getLogger("speech_recognition")
    
    # Initialize W&B if requested
    if use_wandb and wandb_project:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            config={},
        )
        logger.info(f"Initialized W&B tracking for project: {wandb_project}")
    
    return logger


def log_experiment_config(config: DictConfig, logger: logging.Logger) -> None:
    """
    Log experiment configuration.
    
    Args:
        config: Experiment configuration.
        logger: Logger instance.
    """
    logger.info("Experiment Configuration:")
    logger.info(f"  Name: {config.experiment.name}")
    logger.info(f"  Version: {config.experiment.version}")
    logger.info(f"  Tags: {config.experiment.tags}")
    logger.info(f"  Device: {config.device}")
    logger.info(f"  Seed: {config.seed}")
    
    if hasattr(config, "model"):
        logger.info(f"  Model: {config.model._target_}")
    
    if hasattr(config, "data"):
        logger.info(f"  Dataset: {config.data.dataset_name}")
        logger.info(f"  Batch Size: {config.data.batch_size}")
    
    if hasattr(config, "training"):
        logger.info(f"  Max Epochs: {config.training.max_epochs}")
        logger.info(f"  Learning Rate: {config.training.optimizer.lr}")
