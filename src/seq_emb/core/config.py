"""Configuration loader for experiment hyperparameters from YAML."""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml


def find_default_config_path() -> Optional[Path]:
    """Search for default.yaml in standard locations."""
    candidates = [
        Path("configs/default.yaml"),
        Path(__file__).resolve().parents[3] / "configs" / "default.yaml",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    If config_path is None, attempts to load from configs/default.yaml.
    """
    path = Path(config_path) if config_path else find_default_config_path()
    if path and path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
    return {}
