"""SASRec: Self-Attentive Sequential Recommendation model."""

from typing import Union, Optional
import torch
import torch.nn as nn
from .base import BaseSequentialModel


class SASRecModel(BaseSequentialModel):
    """
    SASRec model supporting both POG (RotatE pre-trained input)
    and non-POG (end-to-end learnable item embeddings).
    """

    def __init__(
        self,
        embedding_dim: int = 150,
        max_seq_length: int = 225,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
        pooling: str = "last",
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

        self.num_layers = num_layers
        self.num_heads = num_heads

        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads}). "
                f"Valid divisors of {embedding_dim} include: "
                f"{[d for d in range(1, embedding_dim + 1) if embedding_dim % d == 0 and d <= 20]}."
            )

        self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
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
        batch_size, seq_length, _ = input_embs.shape

        # Positional embeddings
        positions = torch.arange(seq_length, dtype=torch.long, device=self.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_length)
        pos_embs = self.position_embedding(positions)

        x_in = input_embs + pos_embs

        # Key padding mask: True for padded positions
        padding_mask = mask.bool().to(self.device) if mask is not None else None

        out = self.encoder(x_in, src_key_padding_mask=padding_mask)
        user_emb = self.apply_pooling(out, mask=mask)

        return user_emb
