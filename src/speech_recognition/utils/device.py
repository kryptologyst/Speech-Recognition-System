"""Device utilities for automatic device selection and seeding."""

import os
import random
from typing import Literal, Union

import numpy as np
import torch


def get_device(device: Union[str, Literal["auto"]] = "auto") -> torch.device:
    """
    Get the best available device for computation.
    
    Args:
        device: Device specification. If "auto", automatically selects the best device.
        
    Returns:
        torch.device: The selected device.
        
    Raises:
        RuntimeError: If CUDA is requested but not available.
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    
    return torch.device(device)


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set environment variable for additional reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)
