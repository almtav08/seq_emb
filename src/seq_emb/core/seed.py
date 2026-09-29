"""Reproducibility and device management utilities."""

import os
import random
from typing import Optional
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Fix random seeds across python, numpy and torch for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_str: Optional[str] = "auto") -> torch.device:
    """Return torch.device instance based on preference and hardware availability."""
    if device_str is None or device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)
