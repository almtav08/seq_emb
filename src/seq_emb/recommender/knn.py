"""Latent k-NN recommendation algorithm based on sequence similarities."""

from collections import defaultdict
from typing import Dict, List
import torch


def torch_cosine_similarity(embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
    """Compute cosine similarity between two 1D torch tensors."""
    dot_product = torch.dot(embedding1, embedding2)
    norm1 = torch.linalg.norm(embedding1)
    norm2 = torch.linalg.norm(embedding2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    similarity = dot_product / (norm1 * norm2)
    return float(similarity.cpu().numpy().item())


def calc_recommendations(
    user_embs: Dict[int, torch.Tensor],
    target_emb: torch.Tensor,
    paths: Dict[int, List[int]],
    target_path: List[int],
    top: int = 10,
    similarity_exponent: float = 1.0,
) -> List[int]:
    """
    Generate top-k resource recommendations for a target student using
    collaborative filtering over latent sequence representations of successful peers.

    Args:
        user_embs: Dict of user ID -> sequence embedding for training/successful students.
        target_emb: Latent embedding vector of the target student's sequence.
        paths: Full interaction paths of the users.
        target_path: Current interaction sequence of the target student.
        top: Number of recommendations to generate.
        similarity_exponent: Exponent for weighting cosine similarities.

    Returns:
        Ordered list of recommended resource IDs (top-k).
    """
    if not user_embs:
        return []

    scores = []
    for user, user_emb in user_embs.items():
        score = torch_cosine_similarity(target_emb, user_emb)
        scores.append((user, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    similar_users = scores[: max(len(scores) // 2, 1)]

    target_item = target_path[-1] if target_path else None
    recommendations = defaultdict(float)
    max_preference = 0.0

    for neig_user, score in similar_users:
        weighted_similarity = score**similarity_exponent

        # Deduplicate consecutive repeat interactions in neighbor path
        neig_path = []
        for resource in paths[neig_user]:
            if len(neig_path) == 0 or neig_path[-1] != resource:
                neig_path.append(resource)

        # Find occurrence of last item of target student
        if target_item is not None and target_item in neig_path:
            last_item_idx = neig_path.index(target_item) + 1
        else:
            last_item_idx = 0

        # Accumulate score for upcoming items in neighbor's trajectory
        for idx in range(last_item_idx, len(neig_path)):
            item_id = neig_path[idx]
            recommendations[item_id] += weighted_similarity * (1.0 / (idx + 1))
            if recommendations[item_id] > max_preference:
                max_preference = recommendations[item_id]

    if max_preference > 0:
        for item in recommendations:
            recommendations[item] /= max_preference

    # Remove items already visited by the target user
    for item in target_path:
        recommendations.pop(item, None)

    sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top]
    return [item for item, _ in sorted_recs]
