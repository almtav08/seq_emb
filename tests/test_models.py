"""Smoke tests for the 6 model variants (instantiation, forward pass, and shape validation)."""

import sys
from pathlib import Path
import torch

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from seq_emb.models.factory import create_model


def test_models():
    batch_size = 4
    seq_len = 20
    embedding_dim = 150
    num_items = 122
    device = torch.device("cpu")

    models_to_test = ["caser", "grurec", "sasrec"]

    for model_name in models_to_test:
        for use_pog in [True, False]:
            label = f"{model_name.upper()} ({'con POG' if use_pog else 'sin POG'})"
            print(f"Testing {label}...")

            model = create_model(
                model_name=model_name,
                use_pog=use_pog,
                embedding_dim=embedding_dim,
                max_seq_length=seq_len,
                num_items=num_items,
                device=device,
            )

            mask = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
            mask[:, -5:] = 1  # 5 padding positions

            if use_pog:
                x = torch.randn((batch_size, seq_len, embedding_dim), dtype=torch.float32, device=device)
            else:
                x = torch.randint(1, num_items, (batch_size, seq_len), dtype=torch.long, device=device)
                x[:, -5:] = 0  # 0 padding index

            out = model(x, mask=mask)

            assert out.shape == (batch_size, embedding_dim), (
                f"Shape mismatch for {label}: expected {(batch_size, embedding_dim)}, got {out.shape}"
            )
            assert not torch.isnan(out).any(), f"NaN detected in output for {label}"
            print(f"  [OK] Output shape: {out.shape}")

    print("\nAll 6 model variants passed forward pass validation successfully!\n")


if __name__ == "__main__":
    test_models()
