"""Speech Recognition System - A modern ASR implementation with multiple architectures."""

__version__ = "1.0.0"
__author__ = "AI Projects"
__email__ = "projects@example.com"

from .utils.device import get_device, set_seed
from .utils.logging import setup_logging
from .utils.config import load_config

__all__ = [
    "get_device",
    "set_seed", 
    "setup_logging",
    "load_config",
]
