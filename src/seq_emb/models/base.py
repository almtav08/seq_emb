"""Base class for sequential embedding models with and without POG."""

from typing import Optional, Union, Type
import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tracerec.losses.supcon import SupConLoss


class BaseSequentialModel(nn.Module):
    """
    Abstract base class providing common embedding, pooling, compile and fit logic
    for sequential encoders (CASER, GRU4Rec, SASRec) supporting both POG and non-POG modes.
    """

    def __init__(
        self,
        embedding_dim: int = 150,
        max_seq_length: int = 225,
        dropout: float = 0.2,
        pooling: str = "last",
        use_pog: bool = False,
        num_items: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.dropout = dropout
        self.pooling = pooling
        self.use_pog = use_pog
        self.num_items = num_items
        self.device = torch.device(device) if isinstance(device, str) else device

        # If non-POG mode, item IDs are projected via internal nn.Embedding
        if not self.use_pog:
            if num_items is None:
                raise ValueError("num_items must be specified when use_pog=False")
            self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        else:
            self.item_embedding = None

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion: Optional[nn.Module] = None
        self.history = {"train_loss": [], "epoch_time": []}

    def embed_input(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Embed input sequence:
        - If use_pog is True: x is already a float tensor (batch_size, seq_len, embedding_dim).
        - If use_pog is False: x is an integer tensor (batch_size, seq_len) of item IDs.
        """
        if self.use_pog:
            embs = x.to(dtype=torch.float32, device=self.device)
        else:
            x_ids = x.to(dtype=torch.long, device=self.device)
            embs = self.item_embedding(x_ids)

        if mask is not None:
            # Mask is 0 for valid tokens, 1 for padding
            mask_valid = torch.logical_not(mask).unsqueeze(-1).to(device=self.device)
            embs = embs * mask_valid

        return embs

    def apply_pooling(self, out: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply last or mean pooling over sequence outputs."""
        batch_size = out.size(0)

        if self.pooling == "last":
            if mask is not None:
                mask_valid = torch.logical_not(mask).to(device=self.device)
                lengths = torch.clamp(mask_valid.sum(dim=1) - 1, min=0)
                user_emb = out[torch.arange(batch_size, device=self.device), lengths]
            else:
                user_emb = out[:, -1, :]
        elif self.pooling == "mean":
            if mask is not None:
                mask_valid = torch.logical_not(mask).to(device=self.device)
                denom = torch.clamp(mask_valid.sum(dim=1, keepdim=True).float(), min=1.0)
                user_emb = (out * mask_valid.unsqueeze(-1)).sum(dim=1) / denom
            else:
                user_emb = out.mean(dim=1)
        else:
            raise ValueError(f"Unsupported pooling mode: '{self.pooling}'")

        return user_emb

    def compile(
        self,
        optimizer: Union[Type[torch.optim.Optimizer], str] = torch.optim.Adam,
        criterion: Optional[nn.Module] = None,
        lr: float = 1e-4,
    ):
        """Configure optimizer and loss criterion."""
        if criterion is None:
            self.criterion = SupConLoss()
        else:
            self.criterion = criterion

        if hasattr(self.criterion, "to"):
            self.criterion = self.criterion.to(self.device)

        if isinstance(optimizer, str):
            opt_lower = optimizer.lower()
            if opt_lower == "adam":
                opt_cls = torch.optim.Adam
            elif opt_lower == "sgd":
                opt_cls = torch.optim.SGD
            elif opt_lower == "adamw":
                opt_cls = torch.optim.AdamW
            elif opt_lower == "adagrad":
                opt_cls = torch.optim.Adagrad
            else:
                raise ValueError(f"Unknown optimizer string: {optimizer}")
        else:
            opt_cls = optimizer

        self.optimizer_cls = opt_cls
        self.base_lr = lr
        self.optimizer = opt_cls(self.parameters(), lr=lr)
        return self

    def fit(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        train_masks: Optional[torch.Tensor] = None,
        num_epochs: int = 500,
        batch_size: int = 4,
        lr: Optional[float] = None,
        shuffle: bool = False,
        verbose: bool = False,
    ):
        """Train the model using supervised contrastive learning."""
        if self.optimizer is None:
            self.compile(lr=lr or 1e-4)
        elif lr is not None and lr != self.base_lr:
            self.base_lr = lr
            self.optimizer = self.optimizer_cls(self.parameters(), lr=lr)

        self.to(self.device)
        self.train()

        if train_masks is None:
            train_masks = torch.zeros_like(train_x if not self.use_pog else train_x[:, :, 0], dtype=torch.long)

        dataset = TensorDataset(train_x, train_y, train_masks)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        for epoch in range(num_epochs):
            start_time = time.time()
            total_loss = 0.0

            for bx, by, bmask in loader:
                bx = bx.to(self.device)
                by = by.to(self.device)
                bmask = bmask.to(self.device)

                self.optimizer.zero_grad()
                user_embs = self(bx, mask=bmask)
                loss = self.criterion(user_embs, by)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / max(len(loader), 1)
            self.history["train_loss"].append(avg_loss)
            self.history["epoch_time"].append(time.time() - start_time)

            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")

        return self
