#!/usr/bin/env python3
"""Suite de Estudios de Ablación para SeqEmb.

Este script ejecuta y resume exclusivamente los dos estudios de ablación algorítmicos:
1. Ablación de Intervención Outcome-Aware:
   Baseline Factual Naive (Trayectoria real sin intervención) vs. Intervención orientada a pares aprobados.
2. Ablación de Mecanismo de Proyección Temporal y Horizonte de Simulación:
   Ponderación posicional decreciente w(p) = 1/(pos + 1) vs. Uniforme vs. 1-Paso, y Horizonte N=1 vs. N=3.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

# Añadir src al path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root / "src"))

from seq_emb.data.loader import DataLoader
from seq_emb.evaluation.path_metrics import calculate_path_metrics_from_maps


def format_table(headers: List[str], rows: List[List[str]], title: str = "") -> str:
    """Formatea una tabla en texto plano para visualización en terminal."""
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


# ==============================================================================
# ESTUDIO 1: ABLACIÓN DE INTERVENCIÓN OUTCOME-AWARE (NAIVE VS. CLOSEST PASS)
# ==============================================================================
def run_outcome_aware_ablation(
    df: pd.DataFrame,
    data_all: Dict[str, Any]
) -> Tuple[str, pd.DataFrame]:
    paths = data_all["paths"]
    grades = data_all["grades"]
    prereq = data_all["prereq_graph"]
    remedial = data_all["remedial_graph"]
    pop = data_all["popularity_map"]
    total_res = data_all["total_resources"]

    fail_uids = [u for u, g in grades.items() if g == 0]
    steps = 3

    # 1. Calcular trayectoria factual real sin intervención (Naive Baseline)
    real_metrics_list = []
    for uid in fail_uids:
        full_path = paths[uid]
        total_len = len(full_path)
        n_interactions = total_len // 2
        real_prefix = full_path[: n_interactions + steps]
        m = calculate_path_metrics_from_maps(
            real_prefix, prereq, remedial, total_res, pop, steps=steps, local_window=False
        )
        real_metrics_list.append(m)

    mean_real = {
        "coverage": float(np.mean([m["coverage_ratio"] for m in real_metrics_list])),
        "backward": float(np.mean([m["backward_prereq_ratio"] for m in real_metrics_list])),
        "repeat": float(np.mean([m["revisit_ratio"] for m in real_metrics_list])),
        "frac_allowed": float(np.mean([m["frac_allowed_edges"] for m in real_metrics_list])),
        "novelty": float(np.mean([m["novelty"] for m in real_metrics_list])),
    }

    rows = []

    # Fila 1: Naive (Trayectoria factual original sin intervención)
    rows.append({
        "Estrategia / Modelo": "Naive Baseline (Trayectoria real sin intervención)",
        "Outcome-Aware (Closest Pass)?": "No",
        "Coverage (↑)": f"{mean_real['coverage']:.4f} (Base)",
        "Backward (↓)": f"{mean_real['backward']:.4f} (Base)",
        "Repeat (↓)": f"{mean_real['repeat']:.4f} (Base)",
        "Frac Allowed (↑)": f"{mean_real['frac_allowed']:.4f} (Base)",
        "Novelty (↑)": f"{mean_real['novelty']:.4f} (Base)",
        "Δ % Coverage": "0.00%",
        "Δ % Backward": "0.00%",
        "Δ % Repeat": "0.00%",
        "Δ % Frac Allowed": "0.00%",
        "Δ % Novelty": "0.00%",
    })

    models_order = [
        ("caser", True, "Caser (con POG)"),
        ("caser", False, "Caser (sin POG)"),
        ("sasrec", True, "SASRec (con POG)"),
        ("sasrec", False, "SASRec (sin POG)"),
        ("grurec", True, "GRU4Rec (con POG)"),
        ("grurec", False, "GRU4Rec (sin POG)"),
    ]

    for model_key, use_pog, label in models_order:
        sub = df[(df["model"] == model_key) & (df["use_pog"] == use_pog) & (df["grade"] == 0)]

        delta_cov = sub["coverage_ratio"].mean()
        delta_back = sub["backward_ratio"].mean()
        delta_rep = sub["repeat_ratio"].mean()
        delta_frac = sub["frac_allowed_edges"].mean()
        delta_nov = sub["novelty"].mean()

        abs_cov = mean_real["coverage"] + delta_cov
        abs_back = mean_real["backward"] + delta_back
        abs_rep = mean_real["repeat"] + delta_rep
        abs_frac = mean_real["frac_allowed"] + delta_frac
        abs_nov = mean_real["novelty"] + delta_nov

        pct_cov = (delta_cov / mean_real["coverage"]) * 100
        pct_back = (delta_back / mean_real["backward"]) * 100
        pct_rep = (delta_rep / mean_real["repeat"]) * 100
        pct_frac = (delta_frac / mean_real["frac_allowed"]) * 100
        pct_nov = (delta_nov / mean_real["novelty"]) * 100

        rows.append({
            "Estrategia / Modelo": label,
            "Outcome-Aware (Closest Pass)?": "Sí",
            "Coverage (↑)": f"{abs_cov:.4f} ({pct_cov:+.2f}%)",
            "Backward (↓)": f"{abs_back:.4f} ({pct_back:+.2f}%)",
            "Repeat (↓)": f"{abs_rep:.4f} ({pct_rep:+.2f}%)",
            "Frac Allowed (↑)": f"{abs_frac:.4f} ({pct_frac:+.2f}%)",
            "Novelty (↑)": f"{abs_nov:.4f} ({pct_nov:+.2f}%)",
            "Δ % Coverage": f"{pct_cov:+.2f}%",
            "Δ % Backward": f"{pct_back:+.2f}%",
            "Δ % Repeat": f"{pct_rep:+.2f}%",
            "Δ % Frac Allowed": f"{pct_frac:+.2f}%",
            "Δ % Novelty": f"{pct_nov:+.2f}%",
        })

    res_df = pd.DataFrame(rows)
    headers = list(res_df.columns)
    table_rows = res_df.values.tolist()
    title = "ESTUDIO 1: ABLACIÓN DE INTERVENCIÓN OUTCOME-AWARE (NAIVE VS. CLOSEST PASS)"
    return format_table(headers, table_rows, title), res_df


# ==============================================================================
# ESTUDIO 2: ABLACIÓN DE PONDERACIÓN TEMPORAL (TIME-DECAY) Y HORIZONTE (N=1 VS N=3)
# ==============================================================================
def run_time_decay_and_horizon_ablation(data_all: Dict[str, Any]) -> Tuple[str, pd.DataFrame]:
    paths = data_all["paths"]
    grades = data_all["grades"]
    prereq = data_all["prereq_graph"]
    remedial = data_all["remedial_graph"]
    pop = data_all["popularity_map"]
    total_res = data_all["total_resources"]

    fail_uids = [u for u, g in grades.items() if g == 0]
    pass_uids = [u for u, g in grades.items() if g == 1]

    # Línea base observada en t_split (sin pasos de recomendación)
    base_metrics = []
    for uid in fail_uids:
        full_path = paths[uid]
        n_int = len(full_path) // 2
        prefix = full_path[:n_int]
        m = calculate_path_metrics_from_maps(prefix, prereq, remedial, total_res, pop, steps=0, local_window=False)
        base_metrics.append(m)

    base_cov = np.mean([m["coverage_ratio"] for m in base_metrics])
    base_back = np.mean([m["backward_prereq_ratio"] for m in base_metrics])
    base_rep = np.mean([m["revisit_ratio"] for m in base_metrics])
    base_frac = np.mean([m["frac_allowed_edges"] for m in base_metrics])
    base_nov = np.mean([m["novelty"] for m in base_metrics])

    configs = [
        ("Base Factual en t_split (N=0)", "none", 0, "N/A"),
        ("Single-Step Horizon (N=1, w=1/(pos+1))", "inv_pos", 1, "w(p) = 1/(pos + 1)"),
        ("Proposed Horizon (N=3, w=1/(pos+1))", "inv_pos", 3, "w(p) = 1/(pos + 1)"),
        ("Uniform Weighting (N=3, w=1)", "uniform", 3, "w(p) = 1.0"),
        ("Immediate Truncation (N=3, w_0)", "immediate", 3, "w(p) = 1 if pos=0 else 0"),
    ]

    rows = []
    for label, weight_type, steps, weight_desc in configs:
        if steps == 0:
            rows.append({
                "Configuración": label,
                "Horizonte (N)": 0,
                "Ponderación w(p)": weight_desc,
                "Coverage (↑)": f"{base_cov:.4f} (Base)",
                "Backward (↓)": f"{base_back:.4f} (Base)",
                "Repeat (↓)": f"{base_rep:.4f} (Base)",
                "Frac Allowed (↑)": f"{base_frac:.4f} (Base)",
                "Novelty (↑)": f"{base_nov:.4f} (Base)",
                "Δ % Coverage": "0.00%",
                "Δ % Backward": "0.00%",
            })
            continue

        aug_metrics = []
        for uid in fail_uids:
            full_path = paths[uid]
            n_int = len(full_path) // 2
            prefix = full_path[:n_int]
            target_item = prefix[-1]

            # Buscar compañero aprobado más cercano
            best_peer = None
            best_overlap = -1
            for p_uid in pass_uids:
                p_path = paths[p_uid]
                overlap = len(set(prefix).intersection(p_path))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_peer = p_uid

            peer_path = paths[best_peer]
            if target_item in peer_path:
                idx = peer_path.index(target_item) + 1
            else:
                idx = 0

            rec_scores = defaultdict(float)
            for pos, item in enumerate(peer_path[idx:]):
                if item in prefix:
                    continue
                if weight_type == "inv_pos":
                    w = 1.0 / (pos + 1)
                elif weight_type == "uniform":
                    w = 1.0
                elif weight_type == "immediate":
                    w = 1.0 if pos == 0 else 0.0
                rec_scores[item] += w

            sorted_recs = sorted(rec_scores.items(), key=lambda x: x[1], reverse=True)
            top_recs = [item for item, _ in sorted_recs[:steps]]
            aug_path = prefix + top_recs

            m = calculate_path_metrics_from_maps(
                aug_path, prereq, remedial, total_res, pop, steps=steps, local_window=False
            )
            aug_metrics.append(m)

        c = np.mean([m["coverage_ratio"] for m in aug_metrics])
        b = np.mean([m["backward_prereq_ratio"] for m in aug_metrics])
        r = np.mean([m["revisit_ratio"] for m in aug_metrics])
        f_val = np.mean([m["frac_allowed_edges"] for m in aug_metrics])
        nov = np.mean([m["novelty"] for m in aug_metrics])

        pct_c = ((c - base_cov) / base_cov) * 100
        pct_b = ((b - base_back) / base_back) * 100

        rows.append({
            "Configuración": label,
            "Horizonte (N)": steps,
            "Ponderación w(p)": weight_desc,
            "Coverage (↑)": f"{c:.4f} ({pct_c:+.2f}%)",
            "Backward (↓)": f"{b:.4f} ({pct_b:+.2f}%)",
            "Repeat (↓)": f"{r:.4f}",
            "Frac Allowed (↑)": f"{f_val:.4f}",
            "Novelty (↑)": f"{nov:.4f}",
            "Δ % Coverage": f"{pct_c:+.2f}%",
            "Δ % Backward": f"{pct_b:+.2f}%",
        })

    res_df = pd.DataFrame(rows)
    headers = list(res_df.columns)
    table_rows = res_df.values.tolist()
    title = "ESTUDIO 2: ABLACIÓN DE PONDERACIÓN TEMPORAL (TIME-DECAY) Y HORIZONTE DE SIMULACIÓN"
    return format_table(headers, table_rows, title), res_df


def main():
    parser = argparse.ArgumentParser(description="Ejecutar suite de ablación algorítmica para SeqEmb.")
    parser.add_argument("-o", "--output-dir", default="results/ablation", help="Directorio de exportación de resultados.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    csv_path = "results/folds_full_path_user_level.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} no encontrado.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)

    loader = DataLoader()
    data_all = loader.load_all()

    print("\n" + "=" * 90)
    print("EJECUTANDO SUITE DE ESTUDIOS DE ABLACIÓN (OUTCOME-AWARE & TIME-DECAY / HORIZONTE)")
    print("=" * 90 + "\n")

    # 1. Outcome-Aware Intervention Ablation
    table1, df1 = run_outcome_aware_ablation(df, data_all)
    print(table1)
    df1.to_csv(os.path.join(args.output_dir, "ablation_1_outcome_aware_filtering.csv"), index=False)
    print("\n")

    # 2. Time-Decay and Simulation Horizon Ablation
    table2, df2 = run_time_decay_and_horizon_ablation(data_all)
    print(table2)
    df2.to_csv(os.path.join(args.output_dir, "ablation_2_time_decay_and_horizon.csv"), index=False)
    print("\n")

    # Limpiar archivos obsoletos si existen
    for obsolete_file in [
        "ablation_1_pog_macro_significance.csv",
        "ablation_2_sequence_length_coldstart.csv",
        "ablation_3_naive_baseline_table3.csv",
    ]:
        fpath = os.path.join(args.output_dir, obsolete_file)
        if os.path.exists(fpath):
            os.remove(fpath)

    print(f"-> Los dos estudios de ablación se han guardado exitosamente en '{args.output_dir}/'.\n")


if __name__ == "__main__":
    main()
