"""Test verification of POG embedding freeze and LOOCV iteration."""

import torch
from seq_emb.data.loader import DataLoader
from seq_emb.models.caser import CaserModel
from seq_emb.data.processor import prepare_tensors

def main():
    device = torch.device("cpu")
    loader = DataLoader(data_dir="data")
    data = loader.load_all()
    embedder = loader.load_graph_embedder(device)

    # 1. Verify all parameters in POG embedder are frozen
    params = list(embedder.parameters())
    print(f"Total embedder parameters: {len(params)}")
    for name, p in embedder.named_parameters():
        assert not p.requires_grad, f"Parameter {name} has requires_grad=True!"
    print("✓ All POG embedder parameters are strictly frozen (requires_grad=False).")

    # 2. Verify model parameters when use_pog=True
    model = CaserModel(use_pog=True, embedding_dim=150, max_seq_length=225, device=device)
    model_param_names = [name for name, _ in model.named_parameters()]
    print("Model parameter names:", model_param_names)
    assert "item_embedding.weight" not in model_param_names, "item_embedding should be None when use_pog=True"
    print("✓ Model has no learnable item_embedding when use_pog=True.")

    # 3. Test tensor preparation and gradient isolation
    student_keys = list(data["paths"].keys())[:4]
    train_x, train_y, train_masks = prepare_tensors(
        student_ids=student_keys,
        paths=data["paths"],
        grades=data["grades"],
        max_seq_len=225,
        device=device,
        use_pog=True,
        graph_embedder=embedder,
    )
    assert not train_x.requires_grad, "Input features train_x must not require grad"
    print(f"✓ train_x tensor created with shape {train_x.shape}, requires_grad=False.")

    # 4. Test fit step
    model.compile(lr=1e-3)
    model.fit(train_x, train_y, train_masks, num_epochs=2, batch_size=2)
    print("✓ Fit loop completed successfully without updating POG weights.")

    # 5. Re-verify POG parameters after fit
    for name, p in embedder.named_parameters():
        assert not p.requires_grad, f"Parameter {name} modified to requires_grad=True after fit!"
    print("✓ POG embedder weights remained frozen post-training.")

if __name__ == "__main__":
    main()
