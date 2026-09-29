"""Pedagogical and curricular graph metrics for learning paths."""

from typing import Dict, List, Any


def calculate_path_metrics_from_maps(
    path: List[int],
    prereq_map: Dict[int, List[int]],
    remedial_map: Dict[int, List[int]],
    total_resources: int,
    popularity_map: Dict[int, float],
    steps: int = 3,
    local_window: bool = False,
) -> Dict[str, float]:
    """
    Calculate curricular and graph transition metrics for a learning trajectory.

    Args:
        path: Ordered list of visited resource IDs.
        prereq_map: Prerequisite graph {A: [B, ...]} where A is prerequisite of B.
        remedial_map: Remedial graph {A: [B, ...]} where B is a remedial/review for A.
        total_resources: Total number of resources in the curriculum.
        popularity_map: Item popularity normalized in [0, 1].
        steps: Number of intervention steps (for novelty and local window).
        local_window: If True, transition metrics are evaluated only on the last `steps` items.

    Returns:
        Dict with keys: progress_ratio, backward_prereq_ratio, revisit_ratio,
        coverage_ratio, novelty, frac_allowed_edges.
    """
    unique_resources_visited = len(set(path))
    coverage_ratio = (
        unique_resources_visited / total_resources if total_resources > 0 else 0.0
    )

    # 1. Novelty (evaluated on the last `steps` recommended items)
    window_items = path[-steps:] if steps > 0 else path
    novelty = 0.0
    if len(window_items) > 0:
        novelty_sum = 0.0
        for item in window_items:
            try:
                item_id = int(item)
                pop_score = popularity_map.get(item_id, 0.0)
                novelty_sum += 1.0 - pop_score
            except (ValueError, TypeError):
                continue
        novelty = novelty_sum / len(window_items)

    # 2. Window selection for transition metrics
    if local_window and steps > 0:
        eval_path = path[-(steps + 1):]
    else:
        eval_path = path

    total_trans = max(len(eval_path) - 1, 0)
    if total_trans == 0:
        return {
            "progress_ratio": 0.0,
            "backward_prereq_ratio": 0.0,
            "revisit_ratio": 0.0,
            "coverage_ratio": coverage_ratio,
            "novelty": novelty,
            "frac_allowed_edges": 0.0,
        }

    forward_moves = 0
    backward_prereq_moves = 0
    revisit_moves = 0

    for i in range(total_trans):
        try:
            item_a = int(eval_path[i])
            item_b = int(eval_path[i + 1])
        except (ValueError, TypeError):
            continue

        if item_a == item_b:
            continue  # Ignore auto-loops

        # Progress move: item_a is prerequisite of item_b
        if item_b in prereq_map.get(item_a, []):
            forward_moves += 1
        # Backward move: item_b is prerequisite of item_a
        elif item_a in prereq_map.get(item_b, []):
            backward_prereq_moves += 1
        # Revisit / remedial move: item_b is a remedial resource for item_a
        elif item_b in remedial_map.get(item_a, []):
            revisit_moves += 1

    progress_ratio = forward_moves / total_trans
    backward_prereq_ratio = backward_prereq_moves / total_trans
    revisit_ratio = revisit_moves / total_trans
    allowed_moves = forward_moves + revisit_moves
    frac_allowed_edges = allowed_moves / total_trans

    return {
        "progress_ratio": progress_ratio,
        "backward_prereq_ratio": backward_prereq_ratio,
        "revisit_ratio": revisit_ratio,
        "coverage_ratio": coverage_ratio,
        "novelty": novelty,
        "frac_allowed_edges": frac_allowed_edges,
    }
