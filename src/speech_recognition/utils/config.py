"""Configuration utilities for loading and managing experiment configs."""

from pathlib import Path
from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str, overrides: Dict[str, Any] = None) -> DictConfig:
    """
    Load configuration from file with optional overrides.
    
    Args:
        config_path: Path to configuration file.
        overrides: Optional configuration overrides.
        
    Returns:
        DictConfig: Loaded configuration.
        
    Raises:
        FileNotFoundError: If config file doesn't exist.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load base configuration
    config = OmegaConf.load(config_file)
    
    # Apply overrides if provided
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.create(overrides))
    
    return config


def save_config(config: DictConfig, output_path: str) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration to save.
        output_path: Output file path.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    OmegaConf.save(config, output_file)


def print_config(config: DictConfig) -> None:
    """
    Print configuration in a readable format.
    
    Args:
        config: Configuration to print.
    """
    print(OmegaConf.to_yaml(config))
