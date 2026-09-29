"""Classic Top-K ranking recommendation metrics."""

from typing import List, Sequence, Any
import numpy as np


def apk(y_true: Sequence[Any], y_pred: Sequence[Any], k: int = 10) -> float:
    """Average Precision at K."""
    if len(y_pred) > k:
        y_pred = y_pred[:k]

    score = 0.0
    hits = 0.0
    for i, p in enumerate(y_pred):
        if p in y_true:
            hits += 1.0
            score += hits / (i + 1.0)
    return score / min(len(y_true), k) if len(y_true) > 0 else 0.0


def ndcgk(y_true: Sequence[Any], y_pred: Sequence[Any], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain at K."""
    if len(y_pred) > k:
        y_pred = y_pred[:k]

    dcg = 0.0
    for i, p in enumerate(y_pred):
        if p in y_true:
            dcg += 1.0 / np.log2(i + 2)

    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(y_true), k)))
    return dcg / idcg if idcg > 0 else 0.0


def hit_ratio(y_true: Sequence[Any], y_pred: Sequence[Any], k: int = 10) -> float:
    """Hit Ratio at K (binary indicator if any hit occurred in top-K)."""
    if len(y_pred) > k:
        y_pred = y_pred[:k]
    return 1.0 if any(p in y_true for p in y_pred) else 0.0


def mean_reciprocal_rank(y_true: Sequence[Any], y_pred: Sequence[Any], k: int = 10) -> float:
    """Reciprocal Rank at K for a single sequence prediction."""
    if len(y_pred) > k:
        y_pred = y_pred[:k]

    for i, p in enumerate(y_pred):
        if p in y_true:
            return 1.0 / (i + 1.0)
    return 0.0
