"""Data processing and tensor preparation for sequential models."""

from typing import Dict, List, Tuple, Optional, Any
import torch


def prepare_tensors(
    student_ids: List[int],
    paths: Dict[int, List[int]],
    grades: Dict[int, int],
    max_seq_len: int,
    device: torch.device,
    use_pog: bool = False,
    graph_embedder: Optional[Any] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare x, y, and mask tensors for a set of student IDs.

    Args:
        student_ids: List of student IDs.
        paths: Mapping from student ID to resource list.
        grades: Mapping from student ID to binary grade (0: Fail, 1: Pass).
        max_seq_len: Maximum sequence length.
        device: Target torch device.
        use_pog: If True, input features are projected using graph_embedder.
        graph_embedder: RotatE embedder required when use_pog is True.

    Returns:
        (x_tensor, y_tensor, mask_tensor)
    """
    batch_size = len(student_ids)

    # 1. Labels
    y_list = [grades[uid] for uid in student_ids]
    y_tensor = torch.tensor(y_list, dtype=torch.long, device=device)

    # 2. Masks and sequence IDs
    masks = torch.ones((batch_size, max_seq_len), dtype=torch.long, device=device)
    raw_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=device)

    for i, uid in enumerate(student_ids):
        seq = paths[uid][:max_seq_len]
        seq_len = len(seq)
        if seq_len > 0:
            raw_ids[i, :seq_len] = torch.tensor(seq, dtype=torch.long, device=device)
            masks[i, :seq_len] = 0  # 0 indicates valid token, 1 indicates padding

    # 3. Input features
    if use_pog:
        if graph_embedder is None:
            raise ValueError("graph_embedder is required when use_pog=True")

        emb_dim = graph_embedder.embedding_dim
        x_tensor = torch.zeros((batch_size, max_seq_len, emb_dim), dtype=torch.float32, device=device)

        with torch.no_grad():
            for i, uid in enumerate(student_ids):
                seq = paths[uid][:max_seq_len]
                seq_len = len(seq)
                if seq_len > 0:
                    seq_tensor = torch.tensor(seq, dtype=torch.long, device=device)
                    # RotatE.transform returns shape (seq_len, emb_dim)
                    item_embs = graph_embedder.transform(seq_tensor)
                    if item_embs.dim() == 3 and item_embs.size(0) == 1:
                        item_embs = item_embs.squeeze(0)
                    x_tensor[i, :seq_len] = item_embs.detach()
    else:
        x_tensor = raw_ids

    return x_tensor, y_tensor, masks
