"""GRU4Rec: Recurrent Neural Network Sequential Recommendation model."""

from typing import Union, Optional
import torch
import torch.nn as nn
from .base import BaseSequentialModel


class GRURecModel(BaseSequentialModel):
    """
    GRU4Rec model supporting both POG (RotatE pre-trained input)
    and non-POG (end-to-end learnable item embeddings).
    """

    def __init__(
        self,
        embedding_dim: int = 150,
        max_seq_length: int = 225,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        pooling: str = "mean",
        use_pog: bool = False,
        num_items: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__(
            embedding_dim=embedding_dim,
            max_seq_length=max_seq_length,
            dropout=dropout,
            pooling=pooling,
            use_pog=use_pog,
            num_items=num_items,
            device=device,
        )

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_dim, embedding_dim)
        self.to(self.device)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor. If use_pog=True, shape is (batch_size, seq_len, embedding_dim).
               If use_pog=False, shape is (batch_size, seq_len) with integer item IDs.
            mask: Optional mask tensor (0 for real, 1 for padding).
        Returns:
            user_emb: (batch_size, embedding_dim)
        """
        input_embs = self.embed_input(x, mask=mask)
        output, _ = self.gru(input_embs)
        pooled = self.apply_pooling(output, mask=mask)
        user_emb = self.fc(pooled)
        return user_emb
