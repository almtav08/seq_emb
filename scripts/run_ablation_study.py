#!/usr/bin/env python3
"""Ablation Study Suite for SeqEmb using Pretrained Checkpoints.

This script rigorously evaluates the impact of:
1. Outcome-Aware Peer Intervention Ablation:
   - Factual Naive Baseline (Real unassisted trajectory without pedagogical intervention).
   - With Outcome-Aware (Passing Peers): Only students with grade=1 serve as recommendation prototypes.
   - Without Outcome-Aware (Ablation / All Peers): All students (passing and failing).
   - Anti-Outcome / Failing Peers (Negative control): Only students with grade=0.
2. Temporal Projection Mechanism (Time-Decay) and Simulation Horizon Ablation (N=1 vs N=3):
   - Factual baseline at t_split (N=0).
   - Decreasing Positional Time-Decay (w(p) = 1/(pos + 1), N=3).
   - Uniform Weighting / Without Time-Decay (w(p) = 1.0, N=3).
   - Immediate Truncation / Next-Step Only (w_0, N=3).
   - Single-Step Horizon with Time-Decay (w(p) = 1/(pos + 1), N=1).
   - Single-Step Uniform Horizon (w(p) = 1.0, N=1).
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Add src to pythonpath
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root / "src"))

from seq_emb.data.loader import DataLoader
from seq_emb.evaluation.path_metrics import calculate_path_metrics_from_maps
from seq_emb.evaluation.ranking_metrics import apk, hit_ratio, mean_reciprocal_rank, ndcgk
from seq_emb.models.factory import create_model
from seq_emb.recommender.knn import calc_recommendations


def format_table(headers: List[str], rows: List[List[str]], title: str = "") -> str:
    """Formats a plain-text table for terminal display."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    header_str = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"

    data_lines = []
    for row in rows:
        line = "| " + " | ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row)) + " |"
        data_lines.append(line)

    lines = []
    if title:
        total_w = len(sep)
        lines.append("=" * total_w)
        lines.append(f" {title} ".center(total_w, " "))
        lines.append("=" * total_w)
    lines.append(sep)
    lines.append(header_str)
    lines.append(sep)
    lines.extend(data_lines)
    lines.append(sep)
    return "\n".join(lines)


def get_checkpoint_for_fold(
    checkpoints_dir: Path,
    model_name: str,
    fold_idx: int,
) -> Optional[Path]:
    """Locates model checkpoint for a specific fold or consolidated file."""
    tag = f"{model_name}_pog"
    fold_file = checkpoints_dir / tag / f"fold_{fold_idx}.pt"
    if fold_file.exists():
        return fold_file
    consolidated = checkpoints_dir / f"{tag}.pt"
    if consolidated.exists():
        return consolidated
    return None


def run_full_ablation_suite(
    models: List[str],
    checkpoints_dir: Path,
    output_dir: Path,
    tables_dir: Optional[Path] = None,
    device_name: str = "auto",
    max_seq_len: int = 225,
    top_k: int = 10,
    steps: int = 3,
):
    """Runs both ablation studies across all specified models using their checkpoints."""
    loader = DataLoader()
    data = loader.load_all()

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print(f"\n[SeqEmb Ablation Suite] Device: {device} | Models: {[m.upper() for m in models]}")
    embedder = loader.load_graph_embedder(device)

    paths = data["paths"]
    grades = data["grades"]
    prereq_graph = data["prereq_graph"]
    remedial_graph = data["remedial_graph"]
    popularity_map = data["popularity_map"]
    total_resources = data["total_resources"]
    num_items = data["num_items"]

    student_ids = list(paths.keys())
    n_students = len(student_ids)
    fail_uids = [u for u in student_ids if grades[u] == 0]
    pass_uids = [u for u in student_ids if grades[u] == 1]

    # ==========================================================================
    # 0. REAL FACTUAL BASELINES
    # ==========================================================================
    # A. Factual trajectory of failing students with N real steps (Naive Baseline for Study 1)
    naive_metrics = []
    for uid in fail_uids:
        full_path = paths[uid]
        n_int = len(full_path) // 2
        real_prefix = full_path[: n_int + steps]
        m = calculate_path_metrics_from_maps(
            real_prefix, prereq_graph, remedial_graph, total_resources, popularity_map, steps=steps, local_window=False
        )
        naive_metrics.append(m)

    naive_base = {
        "coverage": float(np.mean([m["coverage_ratio"] for m in naive_metrics])),
        "backward": float(np.mean([m["backward_prereq_ratio"] for m in naive_metrics])),
        "repeat": float(np.mean([m["revisit_ratio"] for m in naive_metrics])),
        "frac_allowed": float(np.mean([m["frac_allowed_edges"] for m in naive_metrics])),
        "novelty": float(np.mean([m["novelty"] for m in naive_metrics])),
    }

    # B. Factual trajectory observed at t_split (N=0 for Study 2)
    t_split_metrics = []
    for uid in fail_uids:
        full_path = paths[uid]
        n_int = len(full_path) // 2
        prefix = full_path[:n_int]
        m = calculate_path_metrics_from_maps(
            prefix, prereq_graph, remedial_graph, total_resources, popularity_map, steps=0, local_window=False
        )
        t_split_metrics.append(m)

    tsplit_base = {
        "coverage": float(np.mean([m["coverage_ratio"] for m in t_split_metrics])),
        "backward": float(np.mean([m["backward_prereq_ratio"] for m in t_split_metrics])),
        "repeat": float(np.mean([m["revisit_ratio"] for m in t_split_metrics])),
        "frac_allowed": float(np.mean([m["frac_allowed_edges"] for m in t_split_metrics])),
        "novelty": float(np.mean([m["novelty"] for m in t_split_metrics])),
    }

    # Result row containers
    rows_study1: List[Dict[str, Any]] = []
    rows_study2: List[Dict[str, Any]] = []

    # Naive Baseline row for Study 1
    rows_study1.append({
        "Strategy / Model": "Naive Baseline (Real unassisted trajectory)",
        "Outcome-Aware?": "No",
        "Coverage (↑)": f"{naive_base['coverage']:.4f} (Base)",
        "Backward (↓)": f"{naive_base['backward']:.4f} (Base)",
        "Repeat (↓)": f"{naive_base['repeat']:.4f} (Base)",
        "Frac Allowed (↑)": f"{naive_base['frac_allowed']:.4f} (Base)",
        "Novelty (↑)": f"{naive_base['novelty']:.4f} (Base)",
        "Δ % Coverage": "0.00%",
        "Δ % Backward": "0.00%",
        "Δ % Repeat": "0.00%",
        "Δ % Frac Allowed": "0.00%",
        "Δ % Novelty": "0.00%",
        "MAP (Pass)": "N/A",
        "NDCG (Pass)": "N/A",
        "MRR (Pass)": "N/A",
    })

    # Factual Base N=0 row for Study 2
    rows_study2.append({
        "Model": "Factual Baseline",
        "Configuration": "Factual Base at t_split (N=0)",
        "Horizon (N)": 0,
        "Weighting w(p)": "N/A",
        "Coverage (↑)": f"{tsplit_base['coverage']:.4f} (Base)",
        "Backward (↓)": f"{tsplit_base['backward']:.4f} (Base)",
        "Repeat (↓)": f"{tsplit_base['repeat']:.4f} (Base)",
        "Frac Allowed (↑)": f"{tsplit_base['frac_allowed']:.4f} (Base)",
        "Novelty (↑)": f"{tsplit_base['novelty']:.4f} (Base)",
        "Δ % Coverage": "0.00%",
        "Δ % Backward": "0.00%",
    })

    model_display_names = {
        "caser": "Caser",
        "grurec": "GRU4Rec",
        "sasrec": "SASRec",
    }

    # ==========================================================================
    # LOOCV MODEL EVALUATION
    # ==========================================================================
    for model_name in models:
        m_display = model_display_names.get(model_name, model_name.upper())
        print(f"\n>>> Evaluating ablation studies for {m_display}...")

        model = create_model(
            model_name=model_name,
            use_pog=True,
            embedding_dim=embedder.embedding_dim,
            max_seq_length=max_seq_len,
            num_items=num_items,
            device=device,
        )

        # Metric dictionaries by condition for this model
        # Study 1:
        metrics_oa_true = {"cov": [], "back": [], "rep": [], "frac": [], "nov": [], "map": [], "ndcg": [], "mrr": []}
        metrics_oa_false = {"cov": [], "back": [], "rep": [], "frac": [], "nov": [], "map": [], "ndcg": [], "mrr": []}
        metrics_oa_failonly = {"cov": [], "back": [], "rep": [], "frac": [], "nov": [], "map": [], "ndcg": [], "mrr": []}

        # Study 2:
        study2_collectors = {
            "prop_n3": {"cov": [], "back": [], "rep": [], "frac": [], "nov": []},
            "uniform_n3": {"cov": [], "back": [], "rep": [], "frac": [], "nov": []},
            "immediate_n3": {"cov": [], "back": [], "rep": [], "frac": [], "nov": []},
            "prop_n1": {"cov": [], "back": [], "rep": [], "frac": [], "nov": []},
            "uniform_n1": {"cov": [], "back": [], "rep": [], "frac": [], "nov": []},
        }

        # Iterate over 48 folds of LOOCV
        for i in range(n_students):
            test_uid = student_ids[i]
            train_uids = [student_ids[j] for j in range(n_students) if j != i]

            ckpt_path = get_checkpoint_for_fold(checkpoints_dir, model_name, i)
            if not ckpt_path:
                print(f"Error: Checkpoint not found for {model_name} fold {i}", file=sys.stderr)
                continue

            model.load(ckpt_path)
            model.eval()

            with torch.no_grad():
                # 1. Precompute embeddings for training peers once
                train_embs = {}
                for uid in train_uids:
                    raw_p = paths[uid][:max_seq_len]
                    if len(raw_p) == 0:
                        continue
                    mask = torch.ones((1, max_seq_len), dtype=torch.long, device=device)
                    mask[0, : len(raw_p)] = 0
                    item_embs = embedder.transform(torch.tensor(raw_p, dtype=torch.long, device=device))
                    if item_embs.dim() == 3 and item_embs.size(0) == 1:
                        item_embs = item_embs.squeeze(0)
                    data_t = torch.zeros((1, max_seq_len, embedder.embedding_dim), dtype=torch.float32, device=device)
                    data_t[0, : len(raw_p)] = item_embs
                    train_embs[uid] = model(data_t, mask=mask).squeeze(0).cpu().detach()

                # 2. Embedding for test student over prefix trajectory
                full_path = paths[test_uid]
                n_int = len(full_path) // 2
                prefix = full_path[:n_int]
                pref_crop = prefix[:max_seq_len]

                mask_t = torch.ones((1, max_seq_len), dtype=torch.long, device=device)
                mask_t[0, : len(pref_crop)] = 0
                item_embs_t = embedder.transform(torch.tensor(pref_crop, dtype=torch.long, device=device))
                if item_embs_t.dim() == 3 and item_embs_t.size(0) == 1:
                    item_embs_t = item_embs_t.squeeze(0)
                data_test = torch.zeros((1, max_seq_len, embedder.embedding_dim), dtype=torch.float32, device=device)
                data_test[0, : len(pref_crop)] = item_embs_t
                target_emb = model(data_test, mask=mask_t).squeeze(0).cpu().detach()

                # Candidate sets by outcome condition
                pass_candidates = {u: emb for u, emb in train_embs.items() if grades[u] == 1}
                all_candidates = train_embs
                fail_candidates = {u: emb for u, emb in train_embs.items() if grades[u] == 0}

                # --------------------------------------------------------------
                # STUDY 1: OUTCOME-AWARE EVALUATION
                # --------------------------------------------------------------
                recs_oa_true = calc_recommendations(pass_candidates, target_emb, paths, prefix, top=top_k, weight_type="inv_pos")
                recs_oa_false = calc_recommendations(all_candidates, target_emb, paths, prefix, top=top_k, weight_type="inv_pos")
                recs_oa_fail = calc_recommendations(fail_candidates, target_emb, paths, prefix, top=top_k, weight_type="inv_pos")

                is_pass = grades[test_uid] == 1
                y_true = full_path[n_int:]
                true_set = y_true[: max(n_int // 5, 1)]

                for c_name, recs, coll in [
                    ("true", recs_oa_true, metrics_oa_true),
                    ("false", recs_oa_false, metrics_oa_false),
                    ("fail", recs_oa_fail, metrics_oa_failonly),
                ]:
                    if is_pass:
                        coll["map"].append(apk(true_set, recs, k=top_k))
                        coll["ndcg"].append(ndcgk(true_set, recs, k=top_k))
                        coll["mrr"].append(mean_reciprocal_rank(true_set, recs, k=top_k))
                    else:
                        aug_path = prefix + recs[:steps]
                        m = calculate_path_metrics_from_maps(
                            aug_path, prereq_graph, remedial_graph, total_resources, popularity_map, steps=steps, local_window=False
                        )
                        coll["cov"].append(m["coverage_ratio"])
                        coll["back"].append(m["backward_prereq_ratio"])
                        coll["rep"].append(m["revisit_ratio"])
                        coll["frac"].append(m["frac_allowed_edges"])
                        coll["nov"].append(m["novelty"])

                # --------------------------------------------------------------
                # STUDY 2: TIME-DECAY & HORIZON EVALUATION (for failing students)
                # --------------------------------------------------------------
                if not is_pass:
                    # N=3
                    recs_prop_n3 = recs_oa_true
                    recs_uni_n3 = calc_recommendations(pass_candidates, target_emb, paths, prefix, top=top_k, weight_type="uniform")
                    recs_imm_n3 = calc_recommendations(pass_candidates, target_emb, paths, prefix, top=top_k, weight_type="immediate")

                    # N=1
                    recs_prop_n1 = recs_oa_true
                    recs_uni_n1 = recs_uni_n3

                    configs_s2 = [
                        ("prop_n3", recs_prop_n3, 3),
                        ("uniform_n3", recs_uni_n3, 3),
                        ("immediate_n3", recs_imm_n3, 3),
                        ("prop_n1", recs_prop_n1, 1),
                        ("uniform_n1", recs_uni_n1, 1),
                    ]

                    for c_key, recs, n_steps in configs_s2:
                        aug = prefix + recs[:n_steps]
                        m = calculate_path_metrics_from_maps(
                            aug, prereq_graph, remedial_graph, total_resources, popularity_map, steps=n_steps, local_window=False
                        )
                        study2_collectors[c_key]["cov"].append(m["coverage_ratio"])
                        study2_collectors[c_key]["back"].append(m["backward_prereq_ratio"])
                        study2_collectors[c_key]["rep"].append(m["revisit_ratio"])
                        study2_collectors[c_key]["frac"].append(m["frac_allowed_edges"])
                        study2_collectors[c_key]["nov"].append(m["novelty"])

        # ======================================================================
        # AGGREGATE STUDY 1 RESULTS FOR THIS MODEL
        # ======================================================================
        for label_variant, is_oa, coll in [
            (f"{m_display} (With Outcome-Aware: Passing Peers)", "Yes", metrics_oa_true),
            (f"{m_display} (Without Outcome-Aware: All Peers)", "No", metrics_oa_false),
            (f"{m_display} (Negative Control: Failing Peers)", "Inverse", metrics_oa_failonly),
        ]:
            c_val = float(np.mean(coll["cov"]))
            b_val = float(np.mean(coll["back"]))
            r_val = float(np.mean(coll["rep"]))
            f_val = float(np.mean(coll["frac"]))
            n_val = float(np.mean(coll["nov"]))

            pct_cov = ((c_val - naive_base["coverage"]) / naive_base["coverage"]) * 100
            pct_back = ((b_val - naive_base["backward"]) / naive_base["backward"]) * 100
            pct_rep = ((r_val - naive_base["repeat"]) / naive_base["repeat"]) * 100
            pct_frac = ((f_val - naive_base["frac_allowed"]) / naive_base["frac_allowed"]) * 100
            pct_nov = ((n_val - naive_base["novelty"]) / naive_base["novelty"]) * 100

            map_str = f"{np.mean(coll['map']):.4f}" if coll["map"] else "N/A"
            ndcg_str = f"{np.mean(coll['ndcg']):.4f}" if coll["ndcg"] else "N/A"
            mrr_str = f"{np.mean(coll['mrr']):.4f}" if coll["mrr"] else "N/A"

            rows_study1.append({
                "Strategy / Model": label_variant,
                "Outcome-Aware?": is_oa,
                "Coverage (↑)": f"{c_val:.4f} ({pct_cov:+.2f}%)",
                "Backward (↓)": f"{b_val:.4f} ({pct_back:+.2f}%)",
                "Repeat (↓)": f"{r_val:.4f} ({pct_rep:+.2f}%)",
                "Frac Allowed (↑)": f"{f_val:.4f} ({pct_frac:+.2f}%)",
                "Novelty (↑)": f"{n_val:.4f} ({pct_nov:+.2f}%)",
                "Δ % Coverage": f"{pct_cov:+.2f}%",
                "Δ % Backward": f"{pct_back:+.2f}%",
                "Δ % Repeat": f"{pct_rep:+.2f}%",
                "Δ % Frac Allowed": f"{pct_frac:+.2f}%",
                "Δ % Novelty": f"{pct_nov:+.2f}%",
                "MAP (Pass)": map_str,
                "NDCG (Pass)": ndcg_str,
                "MRR (Pass)": mrr_str,
            })

        # ======================================================================
        # AGGREGATE STUDY 2 RESULTS FOR THIS MODEL
        # ======================================================================
        configs_s2_meta = [
            ("prop_n3", f"{m_display} Proposed (N=3, w(p)=1/(pos+1))", 3, "w(p) = 1/(pos + 1)"),
            ("uniform_n3", f"{m_display} Uniform (N=3, w=1.0)", 3, "w(p) = 1.0 (No decay)"),
            ("immediate_n3", f"{m_display} Immediate Truncation (N=3, w_0)", 3, "w(p) = 1 if pos=0 else 0"),
            ("prop_n1", f"{m_display} Single-Step (N=1, w(p)=1/(pos+1))", 1, "w(p) = 1/(pos + 1)"),
            ("uniform_n1", f"{m_display} Single-Step Uniform (N=1, w=1.0)", 1, "w(p) = 1.0 (No decay)"),
        ]

        for c_key, c_label, n_h, w_desc in configs_s2_meta:
            coll = study2_collectors[c_key]
            c_val = float(np.mean(coll["cov"]))
            b_val = float(np.mean(coll["back"]))
            r_val = float(np.mean(coll["rep"]))
            f_val = float(np.mean(coll["frac"]))
            n_val = float(np.mean(coll["nov"]))

            pct_cov = ((c_val - tsplit_base["coverage"]) / tsplit_base["coverage"]) * 100
            pct_back = ((b_val - tsplit_base["backward"]) / tsplit_base["backward"]) * 100

            rows_study2.append({
                "Model": m_display,
                "Configuration": c_label,
                "Horizon (N)": n_h,
                "Weighting w(p)": w_desc,
                "Coverage (↑)": f"{c_val:.4f} ({pct_cov:+.2f}%)",
                "Backward (↓)": f"{b_val:.4f} ({pct_back:+.2f}%)",
                "Repeat (↓)": f"{r_val:.4f}",
                "Frac Allowed (↑)": f"{f_val:.4f}",
                "Novelty (↑)": f"{n_val:.4f}",
                "Δ % Coverage": f"{pct_cov:+.2f}%",
                "Δ % Backward": f"{pct_back:+.2f}%",
            })

    # ==========================================================================
    # DISPLAY AND EXPORT TABLES
    # ==========================================================================
    df1 = pd.DataFrame(rows_study1)
    df2 = pd.DataFrame(rows_study2)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv1_path = output_dir / "ablation_1_outcome_aware_filtering.csv"
    csv2_path = output_dir / "ablation_2_time_decay_and_horizon.csv"
    df1.to_csv(csv1_path, index=False)
    df2.to_csv(csv2_path, index=False)

    print("\n" + "=" * 115)
    print("STUDY 1: OUTCOME-AWARE PEER INTERVENTION ABLATION (PASSING PEERS VS. ALL PEERS)")
    print("=" * 115)
    cols_display1 = [
        "Strategy / Model", "Outcome-Aware?", "Coverage (↑)", "Backward (↓)",
        "Repeat (↓)", "Frac Allowed (↑)", "Novelty (↑)", "MAP (Pass)", "NDCG (Pass)"
    ]
    print(format_table(cols_display1, df1[cols_display1].values.tolist()))

    print("\n" + "=" * 115)
    print("STUDY 2: TEMPORAL WEIGHTING (TIME-DECAY) AND SIMULATION HORIZON ABLATION")
    print("=" * 115)
    cols_display2 = [
        "Configuration", "Horizon (N)", "Weighting w(p)", "Coverage (↑)", "Backward (↓)", "Repeat (↓)", "Novelty (↑)"
    ]
    print(format_table(cols_display2, df2[cols_display2].values.tolist()))

    print(f"\n[OK] Results exported to '{output_dir}/':")
    print(f"  -> {csv1_path}")
    print(f"  -> {csv2_path}")

    # Export LaTeX tables in tables/ if tables_dir exists
    if tables_dir:
        tables_dir.mkdir(parents=True, exist_ok=True)
        tex1_path = tables_dir / "ablation_outcome_aware.tex"
        tex2_path = tables_dir / "ablation_time_decay.tex"

        # Generate clean LaTeX table for Study 1
        tex1_lines = [
            r"\begin{table}[ht!]",
            r"\centering",
            r"\caption{Outcome-aware peer intervention ablation under LOOCV ($K=10$, $N=3$). Comparing unassisted Naive Baseline, Outcome-Aware (passing peers only), and Without Outcome-Aware (unfiltered peers).}",
            r"\label{tab:ablation_outcome_aware}",
            r"\begin{tabular}{lcccccc}",
            r"\hline",
            r"\textbf{Strategy / Model} & \textbf{Outcome-Aware} & \textbf{CR} & \textbf{BR} & \textbf{RR} & \textbf{AE} & \textbf{N} \\",
            r"\hline",
        ]
        for _, r in df1.iterrows():
            strat = r["Strategy / Model"].replace("&", r"\&")
            oa = r["Outcome-Aware?"]
            cr = r["Coverage (↑)"].split()[0]
            br = r["Backward (↓)"].split()[0]
            rr = r["Repeat (↓)"].split()[0]
            ae = r["Frac Allowed (↑)"].split()[0]
            nov = r["Novelty (↑)"].split()[0]
            tex1_lines.append(f"{strat} & {oa} & {cr} & {br} & {rr} & {ae} & {nov} \\\\")
        tex1_lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
        tex1_path.write_text("\n".join(tex1_lines) + "\n", encoding="utf-8")

        # Generate clean LaTeX table for Study 2
        tex2_lines = [
            r"\begin{table}[ht!]",
            r"\centering",
            r"\caption{Temporal projection (Time-Decay) and simulation horizon ablation for failing student trajectories.}",
            r"\label{tab:ablation_time_decay}",
            r"\begin{tabular}{lcccc}",
            r"\hline",
            r"\textbf{Configuration} & \textbf{Horizon ($N$)} & \textbf{Decay $w(p)$} & \textbf{CR} & \textbf{BR} \\",
            r"\hline",
        ]
        for _, r in df2.iterrows():
            cfg_name = r["Configuration"].replace("&", r"\&")
            h = r["Horizon (N)"]
            w = r["Weighting w(p)"].replace("&", r"\&")
            cr = r["Coverage (↑)"].split()[0]
            br = r["Backward (↓)"].split()[0]
            tex2_lines.append(f"{cfg_name} & {h} & {w} & {cr} & {br} \\\\")
        tex2_lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
        tex2_path.write_text("\n".join(tex2_lines) + "\n", encoding="utf-8")

        print(f"[OK] LaTeX tables exported to '{tables_dir}/':")
        print(f"  -> {tex1_path}")
        print(f"  -> {tex2_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run algorithmic ablation study suite for SeqEmb using pretrained checkpoints."
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="all",
        choices=["all", "caser", "sasrec", "grurec"],
        help="Model or models to evaluate (default: all -> caser, sasrec, grurec)"
    )
    parser.add_argument(
        "-c", "--checkpoints-dir",
        type=str,
        default="checkpoints",
        help="Checkpoints directory (default: checkpoints)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="results/ablation",
        help="CSV results output directory (default: results/ablation)"
    )
    parser.add_argument(
        "-t", "--tables-dir",
        type=str,
        default="tables",
        help="LaTeX tables output directory (default: tables)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device (default: auto)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=3,
        help="Curricular recommendation horizon steps N (default: 3)"
    )

    args = parser.parse_args()

    checkpoints_dir = Path(args.checkpoints_dir)
    output_dir = Path(args.output_dir)
    tables_dir = Path(args.tables_dir) if args.tables_dir else None

    if args.model == "all":
        models_to_run = ["caser", "sasrec", "grurec"]
    else:
        models_to_run = [args.model]

    run_full_ablation_suite(
        models=models_to_run,
        checkpoints_dir=checkpoints_dir,
        output_dir=output_dir,
        tables_dir=tables_dir,
        device_name=args.device,
        steps=args.steps,
    )


if __name__ == "__main__":
    main()
