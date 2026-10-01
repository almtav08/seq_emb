"""Fold evaluation logic for both passing students (ranking) and failing students (path intervention)."""

from typing import Dict, List, Any, Optional
import numpy as np
import torch

from ..models.base import BaseSequentialModel
from ..recommender.knn import calc_recommendations
from .ranking_metrics import apk, ndcgk, hit_ratio, mean_reciprocal_rank
from .path_metrics import calculate_path_metrics_from_maps


def evaluate_fold(
    model: BaseSequentialModel,
    train_keys: List[int],
    test_keys: List[int],
    paths: Dict[int, List[int]],
    grades: Dict[int, int],
    prereq_graph: Dict[int, List[int]],
    remedial_graph: Dict[int, List[int]],
    popularity_map: Dict[int, float],
    total_resources: int,
    max_seq_len: int,
    device: torch.device,
    use_pog: bool = False,
    graph_embedder: Optional[Any] = None,
    top: int = 10,
    steps: int = 3,
    local_window: bool = False,
) -> Dict[str, float]:
    """
    Evaluate a trained model on the test partition of a fold.

    - Passing students (grades[uid] == 1): evaluated with Top-K ranking metrics (MAP, NDCG, HR, MRR).
    - Failing students (grades[uid] == 0): evaluated via multi-step path intervention simulation (Deltas).
    """
    model.eval()

    # 1. Precompute latent embeddings for successful training students
    user_embeddings: Dict[int, torch.Tensor] = {}
    user_paths: Dict[int, List[int]] = {}

    with torch.no_grad():
        for uid in train_keys:
            if grades[uid] == 0:
                continue  # Only successful students serve as recommendation prototypes

            raw_path = paths[uid][:max_seq_len]
            seq_len = len(raw_path)
            if seq_len == 0:
                continue

            mask = torch.ones((1, max_seq_len), dtype=torch.long, device=device)
            mask[0, :seq_len] = 0

            if use_pog:
                data_tensor = torch.zeros((1, max_seq_len, graph_embedder.embedding_dim), dtype=torch.float32, device=device)
                item_embs = graph_embedder.transform(torch.tensor(raw_path, dtype=torch.long, device=device))
                if item_embs.dim() == 3 and item_embs.size(0) == 1:
                    item_embs = item_embs.squeeze(0)
                data_tensor[0, :seq_len] = item_embs
            else:
                data_tensor = torch.zeros((1, max_seq_len), dtype=torch.long, device=device)
                data_tensor[0, :seq_len] = torch.tensor(raw_path, dtype=torch.long, device=device)

            emb = model(data_tensor, mask=mask)
            user_embeddings[uid] = emb.squeeze(0).cpu().detach()
            user_paths[uid] = paths[uid]

    # Metrics collectors
    all_map: List[float] = []
    all_ndcg: List[float] = []
    all_hr: List[float] = []
    all_mrr: List[float] = []

    coverage_deltas: List[float] = []
    backward_deltas: List[float] = []
    repeat_deltas: List[float] = []
    progress_deltas: List[float] = []
    frac_allowed_deltas: List[float] = []
    novelty_deltas: List[float] = []

    def get_path_embedding(path_seq: List[int]) -> torch.Tensor:
        seq = path_seq[:max_seq_len]
        s_len = len(seq)
        mask = torch.ones((1, max_seq_len), dtype=torch.long, device=device)
        mask[0, :s_len] = 0

        if use_pog:
            data = torch.zeros((1, max_seq_len, graph_embedder.embedding_dim), dtype=torch.float32, device=device)
            item_embs = graph_embedder.transform(torch.tensor(seq, dtype=torch.long, device=device))
            if item_embs.dim() == 3 and item_embs.size(0) == 1:
                item_embs = item_embs.squeeze(0)
            data[0, :s_len] = item_embs
        else:
            data = torch.zeros((1, max_seq_len), dtype=torch.long, device=device)
            data[0, :s_len] = torch.tensor(seq, dtype=torch.long, device=device)

        out = model(data, mask=mask)
        return out.squeeze(0).cpu().detach()

    user_metrics: List[Dict[str, Any]] = []
    pass_map, pass_ndcg, pass_hr, pass_mrr = [], [], [], []
    fail_cov, fail_back, fail_rep, fail_prog, fail_frac, fail_nov = [], [], [], [], [], []

    with torch.no_grad():
        for uid in test_keys:
            full_path = paths[uid]
            total_len = len(full_path)
            if total_len <= 1:
                continue

            n_interactions = total_len // 2
            target_path = full_path[:n_interactions]
            target_emb = get_path_embedding(target_path)

            # 1. Ranking evaluation (Next-Item Prediction)
            y_true = full_path[n_interactions:]
            recs = calc_recommendations(
                user_embeddings, target_emb, user_paths, target_path, top=top
            )
            true_set = y_true[: max(n_interactions // 5, 1)]

            u_map = apk(true_set, recs, k=top)
            u_ndcg = ndcgk(true_set, recs, k=top)
            u_hr = hit_ratio(true_set, recs, k=top)
            u_mrr = mean_reciprocal_rank(true_set, recs, k=top)

            all_map.append(u_map)
            all_ndcg.append(u_ndcg)
            all_hr.append(u_hr)
            all_mrr.append(u_mrr)

            if grades[uid] == 1:
                pass_map.append(u_map)
                pass_ndcg.append(u_ndcg)
                pass_hr.append(u_hr)
                pass_mrr.append(u_mrr)

            # 2. Path intervention simulation (Curricular Transitions & Deltas)
            coverage_rec: List[float] = []
            backward_rec: List[float] = []
            repeat_rec: List[float] = []
            progress_rec: List[float] = []
            frac_allowed_rec: List[float] = []
            novelty_rec: List[float] = []

            def generate_coverage_ratios(current_path: List[int], remaining_steps: int, current_emb: torch.Tensor):
                if remaining_steps == 0:
                    m = calculate_path_metrics_from_maps(
                        current_path,
                        prereq_graph,
                        remedial_graph,
                        total_resources,
                        popularity_map,
                        steps=steps,
                        local_window=local_window,
                    )
                    coverage_rec.append(m["coverage_ratio"])
                    backward_rec.append(m["backward_prereq_ratio"])
                    repeat_rec.append(m["revisit_ratio"])
                    progress_rec.append(m["progress_ratio"])
                    frac_allowed_rec.append(m["frac_allowed_edges"])
                    novelty_rec.append(m["novelty"])
                    return

                recs_sim = calc_recommendations(
                    user_embeddings, current_emb, user_paths, current_path, top=top
                )

                for rec in recs_sim:
                    new_path = current_path + [rec]
                    if len(new_path) <= max_seq_len:
                        new_emb = get_path_embedding(new_path)
                        generate_coverage_ratios(new_path, remaining_steps - 1, new_emb)

            generate_coverage_ratios(target_path, steps, target_emb)

            real_prefix = full_path[: n_interactions + steps]
            real_m = calculate_path_metrics_from_maps(
                real_prefix,
                prereq_graph,
                remedial_graph,
                total_resources,
                popularity_map,
                steps=steps,
                local_window=local_window,
            )

            u_cov = float(np.mean(coverage_rec) - real_m["coverage_ratio"]) if coverage_rec else 0.0
            u_back = float(np.mean(backward_rec) - real_m["backward_prereq_ratio"]) if backward_rec else 0.0
            u_rep = float(np.mean(repeat_rec) - real_m["revisit_ratio"]) if repeat_rec else 0.0
            u_prog = float(np.mean(progress_rec) - real_m["progress_ratio"]) if progress_rec else 0.0
            u_frac = float(np.mean(frac_allowed_rec) - real_m["frac_allowed_edges"]) if frac_allowed_rec else 0.0
            u_nov = float(np.mean(novelty_rec) - real_m["novelty"]) if novelty_rec else 0.0

            coverage_deltas.append(u_cov)
            backward_deltas.append(u_back)
            repeat_deltas.append(u_rep)
            progress_deltas.append(u_prog)
            frac_allowed_deltas.append(u_frac)
            novelty_deltas.append(u_nov)

            if grades[uid] == 0:
                fail_cov.append(u_cov)
                fail_back.append(u_back)
                fail_rep.append(u_rep)
                fail_prog.append(u_prog)
                fail_frac.append(u_frac)
                fail_nov.append(u_nov)

            user_record = {
                "user_id": uid,
                "grade": int(grades[uid]),
                "map": u_map,
                "ndcg": u_ndcg,
                "hr": u_hr,
                "mrr": u_mrr,
                "coverage_ratio": u_cov,
                "backward_ratio": u_back,
                "repeat_ratio": u_rep,
                "progress_ratio": u_prog,
                "frac_allowed_edges": u_frac,
                "novelty": u_nov,
            }
            user_metrics.append(user_record)

    return {
        "user_metrics": user_metrics,
        "pass_metrics": {
            "map": pass_map,
            "ndcg": pass_ndcg,
            "hr": pass_hr,
            "mrr": pass_mrr,
        },
        "fail_metrics": {
            "coverage_ratio": fail_cov,
            "backward_ratio": fail_back,
            "repeat_ratio": fail_rep,
            "progress_ratio": fail_prog,
            "frac_allowed_edges": fail_frac,
            "novelty": fail_nov,
        },
        "mean_map": float(np.mean(all_map)) if all_map else 0.0,
        "mean_ndcg": float(np.mean(all_ndcg)) if all_ndcg else 0.0,
        "mean_hr": float(np.mean(all_hr)) if all_hr else 0.0,
        "mean_mrr": float(np.mean(all_mrr)) if all_mrr else 0.0,
        "mean_coverage_ratio": float(np.mean(coverage_deltas)) if coverage_deltas else 0.0,
        "mean_backward_ratio": float(np.mean(backward_deltas)) if backward_deltas else 0.0,
        "mean_repeat_ratio": float(np.mean(repeat_deltas)) if repeat_deltas else 0.0,
        "mean_progress_ratio": float(np.mean(progress_deltas)) if progress_deltas else 0.0,
        "mean_frac_allowed_edges": float(np.mean(frac_allowed_deltas)) if frac_allowed_deltas else 0.0,
        "mean_novelty": float(np.mean(novelty_deltas)) if novelty_deltas else 0.0,
    }
