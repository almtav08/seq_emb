"""Model factory to instantiate CASER, GRURec, and SASRec models."""

from typing import Dict, Any, Union, Optional
import torch

from .base import BaseSequentialModel
from .caser import CaserModel
from .grurec import GRURecModel
from .sasrec import SASRecModel


DEFAULT_HYPERPARAMS = {
    "caser": {
        "num_vert_filters": 8,
        "num_hori_filters": 2,
        "hori_filter_sizes": (2, 3, 4),
        "dropout": 0.2,
        "pooling": "last",
        "optimizer": torch.optim.Adagrad,
        "lr": 1e-5,
    },
    "grurec": {
        "num_layers": 10,
        "hidden_dim": 16,
        "pooling": "mean",
        "dropout": 0.2,
        "optimizer": torch.optim.Adam,
        "lr": 1e-4,
    },
    "sasrec": {
        "num_layers": 10,
        "num_heads": 10,
        "pooling": "last",
        "dropout": 0.15,
        "optimizer": torch.optim.Adam,
        "lr": 1e-4,
    },
}


def create_model(
    model_name: str,
    use_pog: bool,
    embedding_dim: int = 150,
    max_seq_length: int = 225,
    num_items: Optional[int] = None,
    device: Union[str, torch.device] = "cpu",
    custom_params: Optional[Dict[str, Any]] = None,
) -> BaseSequentialModel:
    """
    Factory function to instantiate models with standard or customized hyperparameters.

    Args:
        model_name: "caser", "grurec", or "sasrec".
        use_pog: True for pre-trained RotatE POG embeddings, False for learnable item embeddings.
        embedding_dim: Latent dimension size.
        max_seq_length: Sequence padding length.
        num_items: Total number of vocabulary items (required when use_pog=False).
        device: Device to allocate model on.
        custom_params: Optional overrides for model hyperparameters.

    Returns:
        Compiled BaseSequentialModel instance.
    """
    name = model_name.lower().strip()
    if name not in DEFAULT_HYPERPARAMS:
        raise ValueError(f"Unknown model name '{model_name}'. Choose from: 'caser', 'grurec', 'sasrec'.")

    params = DEFAULT_HYPERPARAMS[name].copy()
    if custom_params:
        params.update(custom_params)

    opt_cls = params.pop("optimizer", torch.optim.Adam)
    lr = params.pop("lr", 1e-4)

    common_kwargs = {
        "embedding_dim": embedding_dim,
        "max_seq_length": max_seq_length,
        "use_pog": use_pog,
        "num_items": num_items,
        "device": device,
    }

    if name == "caser":
        model = CaserModel(
            num_vert_filters=params.get("num_vert_filters", 8),
            num_hori_filters=params.get("num_hori_filters", 2),
            hori_filter_sizes=params.get("hori_filter_sizes", (2, 3, 4)),
            dropout=params.get("dropout", 0.2),
            pooling=params.get("pooling", "last"),
            **common_kwargs,
        )
    elif name == "grurec":
        model = GRURecModel(
            hidden_dim=params.get("hidden_dim", 16),
            num_layers=params.get("num_layers", 10),
            dropout=params.get("dropout", 0.2),
            pooling=params.get("pooling", "mean"),
            **common_kwargs,
        )
    elif name == "sasrec":
        model = SASRecModel(
            num_layers=params.get("num_layers", 10),
            num_heads=params.get("num_heads", 10),
            dropout=params.get("dropout", 0.15),
            pooling=params.get("pooling", "last"),
            **common_kwargs,
        )

    model.compile(optimizer=opt_cls, lr=lr)
    return model
