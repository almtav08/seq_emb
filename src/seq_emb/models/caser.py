"""CASER: Convolutional Sequence Embedding Recommendation model."""

from typing import Tuple, Union, Optional
import torch
import torch.nn as nn
from .base import BaseSequentialModel


class CaserModel(BaseSequentialModel):
    """
    CASER model supporting both POG (RotatE pre-trained input)
    and non-POG (end-to-end learnable item embeddings).
    """

    def __init__(
        self,
        embedding_dim: int = 150,
        max_seq_length: int = 225,
        num_vert_filters: int = 8,
        num_hori_filters: int = 16,
        hori_filter_sizes: Tuple[int, ...] = (2, 3, 4),
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

        self.num_vert_filters = num_vert_filters
        self.num_hori_filters = num_hori_filters
        self.hori_filter_sizes = hori_filter_sizes

        # Horizontal convolutions (sliding window over items)
        self.horizontal_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=num_hori_filters,
                    kernel_size=(h, embedding_dim),
                )
                for h in hori_filter_sizes
            ]
        )
        hori_out_dim = num_hori_filters * len(hori_filter_sizes)

        # Vertical convolution (aggregates all items across latent dimensions)
        self.vertical_conv = nn.Conv2d(
            in_channels=1,
            out_channels=num_vert_filters,
            kernel_size=(max_seq_length, 1),
        )
        vert_out_dim = num_vert_filters * embedding_dim

        self.fc = nn.Linear(hori_out_dim + vert_out_dim, embedding_dim)
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
        batch_size, seq_length, emb_dim = input_embs.size()

        # If sequence length is smaller than max_seq_length (e.g. during inference), pad to max_seq_length
        if seq_length < self.max_seq_length:
            pad = torch.zeros(
                (batch_size, self.max_seq_length - seq_length, emb_dim),
                device=input_embs.device,
                dtype=input_embs.dtype,
            )
            input_embs = torch.cat([input_embs, pad], dim=1)
        elif seq_length > self.max_seq_length:
            input_embs = input_embs[:, : self.max_seq_length, :]

        # Add channel dimension: (batch_size, 1, max_seq_length, embedding_dim)
        x_in = input_embs.unsqueeze(1)

        # Horizontal convolutions
        conv_h = []
        for conv in self.horizontal_convs:
            c_out = torch.relu(conv(x_in)).squeeze(3)  # (batch_size, num_filters, length - k + 1)
            p_out = torch.max_pool1d(c_out, c_out.size(2)).squeeze(2)  # (batch_size, num_filters)
            conv_h.append(p_out)
        z_h = torch.cat(conv_h, dim=1)

        # Vertical convolution
        z_v = torch.relu(self.vertical_conv(x_in)).view(batch_size, -1)

        # Concatenate and project
        z = torch.cat([z_h, z_v], dim=1)
        user_emb = self.fc(z)

        return user_emb
