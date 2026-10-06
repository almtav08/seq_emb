#!/usr/bin/env python3
"""Suite completa de Estudios de Ablación para SeqEmb.

Este script ejecuta y resume 3 estudios de ablación fundamentales:
1. Ablación Macro POG vs. Sin POG (con pruebas de significancia estadística pareadas).
2. Ablación por Longitud de Secuencia / Cold-Start (Short, Medium, Long).
3. Ablación de Intervención: Baseline Factual Naive (Trayectoria Original sin Intervención)
   vs. Intervención con Selección de Estudiante Aprobado (Closest Pass).
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Añadir src al path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root / "src"))

from seq_emb.data.loader import DataLoader
from seq_emb.evaluation.path_metrics import calculate_path_metrics_from_maps


def format_table(headers: List[str], rows: List[List[str]], title: str = "") -> str:
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
# ESTUDIO 1: ABLACIÓN MACRO (POG VS. SIN POG) + TESTS ESTADÍSTICOS
# ==============================================================================
def run_macro_pog_ablation(df: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    results = []
    models = ["caser", "sasrec", "grurec"]

    for model in models:
        pog_df = df[(df["model"] == model) & (df["use_pog"] == True)].sort_values("user_id")
        nopog_df = df[(df["model"] == model) & (df["use_pog"] == False)].sort_values("user_id")

        p_pog = pog_df[pog_df["grade"] == 1]
        p_nopog = nopog_df[nopog_df["grade"] == 1]
        f_pog = pog_df[pog_df["grade"] == 0]
        f_nopog = nopog_df[nopog_df["grade"] == 0]

        # Métricas de Pass
        for m in ["map", "ndcg", "mrr"]:
            v_pog = p_pog[m].values
            v_nopog = p_nopog[m].values
            mean_p = float(np.mean(v_pog))
            mean_np = float(np.mean(v_nopog))
            diff = mean_p - mean_np
            pct = (diff / mean_np * 100) if mean_np != 0 else 0.0

            t_p = stats.ttest_rel(v_pog, v_nopog).pvalue if len(v_pog) > 1 else 1.0
            try:
                w_p = stats.wilcoxon(v_pog, v_nopog).pvalue if len(v_pog) > 1 else 1.0
            except ValueError:
                w_p = 1.0

            results.append({
                "Model": model.upper(),
                "Cohorte": "PASS (N=14)",
                "Métrica": m.upper(),
                "Con POG": f"{mean_p:.4f}",
                "Sin POG": f"{mean_np:.4f}",
                "Δ Abs": f"{diff:+.4f}",
                "Δ %": f"{pct:+.2f}%",
                "p-val (t-test)": f"{t_p:.4f}",
                "p-val (Wilcoxon)": f"{w_p:.4f}",
                "Sig. (p<0.05)": "Sí" if t_p < 0.05 else "No"
            })

        # Métricas de Fail
        for m, name in [
            ("coverage_ratio", "Δ COVERAGE ↑"),
            ("backward_ratio", "Δ BACKWARD ↓"),
            ("repeat_ratio", "Δ REPEAT ↓"),
            ("frac_allowed_edges", "Δ FRAC ALLOWED ↑"),
            ("novelty", "Δ NOVELTY ↑"),
        ]:
            v_pog = f_pog[m].values
            v_nopog = f_nopog[m].values
            mean_p = float(np.mean(v_pog))
            mean_np = float(np.mean(v_nopog))
            diff = mean_p - mean_np

            if np.allclose(v_pog, v_nopog):
                t_p = 1.0
                w_p = 1.0
            else:
                t_p = stats.ttest_rel(v_pog, v_nopog).pvalue if len(v_pog) > 1 else 1.0
                try:
                    w_p = stats.wilcoxon(v_pog, v_nopog).pvalue if len(v_pog) > 1 else 1.0
                except ValueError:
                    w_p = 1.0

            if np.isnan(t_p):
                t_p = 1.0
            if np.isnan(w_p):
                w_p = 1.0

            results.append({
                "Model": model.upper(),
                "Cohorte": "FAIL (N=34)",
                "Métrica": name,
                "Con POG": f"{mean_p:.5f}",
                "Sin POG": f"{mean_np:.5f}",
                "Δ Abs": f"{diff:+.5f}",
                "Δ %": "-" if mean_np == 0 else f"{(diff / abs(mean_np) * 100):+.2f}%",
                "p-val (t-test)": f"{t_p:.4f}",
                "p-val (Wilcoxon)": f"{w_p:.4f}",
                "Sig. (p<0.05)": "Sí" if t_p < 0.05 else "No"
            })

    res_df = pd.DataFrame(results)
    headers = list(res_df.columns)
    rows = res_df.values.tolist()
    title = "ESTUDIO 1: ABLACIÓN MACRO POG VS. SIN POG (SIGNIFICANCIA ESTADÍSTICA)"
    return format_table(headers, rows, title), res_df


# ==============================================================================
# ESTUDIO 2: ABLACIÓN POR LONGITUD DE SECUENCIA / COLD-START
# ==============================================================================
def run_sequence_length_ablation(df: pd.DataFrame, paths: Dict[int, List[int]]) -> Tuple[str, pd.DataFrame]:
    df_len = df.copy()
    df_len["seq_len"] = df_len["user_id"].map(lambda u: len(paths.get(int(u), [])))

    # Segmentar en 3 terciles basados en la distribución real (Min: 50, Median: 204, Max: 463)
    def categorize_length(length: int) -> str:
        if length <= 175:
            return "1. Cortas (<= 175)"
        elif length <= 260:
            return "2. Medias (176 - 260)"
        else:
            return "3. Largas (> 260)"

    df_len["length_bin"] = df_len["seq_len"].apply(categorize_length)

    summary_rows = []
    models = ["caser", "sasrec", "grurec"]

    for length_bin, group in df_len.groupby("length_bin"):
        for model in models:
            p_pog = group[(group["model"] == model) & (group["use_pog"] == True) & (group["grade"] == 1)]
            p_nopog = group[(group["model"] == model) & (group["use_pog"] == False) & (group["grade"] == 1)]
            f_pog = group[(group["model"] == model) & (group["use_pog"] == True) & (group["grade"] == 0)]
            f_nopog = group[(group["model"] == model) & (group["use_pog"] == False) & (group["grade"] == 0)]

            n_p = len(p_pog)
            n_f = len(f_pog)

            # Métricas de Pass: MAP, NDCG, MRR
            for m in ["map", "ndcg", "mrr"]:
                val_p = p_pog[m].mean() if n_p > 0 else np.nan
                val_np = p_nopog[m].mean() if n_p > 0 else np.nan
                diff = val_p - val_np if not np.isnan(val_p) and not np.isnan(val_np) else np.nan
                pct = (diff / val_np * 100) if val_np and not np.isnan(val_np) and val_np != 0 else 0.0

                summary_rows.append({
                    "Longitud Secuencia": length_bin,
                    "Model": model.upper(),
                    "Cohorte": f"PASS (N={n_p})",
                    "Métrica": m.upper(),
                    "Con POG": f"{val_p:.4f}" if not np.isnan(val_p) else "N/A",
                    "Sin POG": f"{val_np:.4f}" if not np.isnan(val_np) else "N/A",
                    "Δ Abs": f"{diff:+.4f}" if not np.isnan(diff) else "N/A",
                    "Δ %": f"{pct:+.2f}%" if not np.isnan(pct) else "N/A",
                })

            # Métricas de Fail: Coverage, Backward, Repeat, Frac Allowed, Novelty
            for m, name in [
                ("coverage_ratio", "Δ COVERAGE ↑"),
                ("backward_ratio", "Δ BACKWARD ↓"),
                ("repeat_ratio", "Δ REPEAT ↓"),
                ("frac_allowed_edges", "Δ FRAC ALLOWED ↑"),
                ("novelty", "Δ NOVELTY ↑"),
            ]:
                val_p = f_pog[m].mean() if n_f > 0 else np.nan
                val_np = f_nopog[m].mean() if n_f > 0 else np.nan
                diff = val_p - val_np if not np.isnan(val_p) and not np.isnan(val_np) else np.nan
                pct = (diff / abs(val_np) * 100) if val_np and not np.isnan(val_np) and val_np != 0 else 0.0

                summary_rows.append({
                    "Longitud Secuencia": length_bin,
                    "Model": model.upper(),
                    "Cohorte": f"FAIL (N={n_f})",
                    "Métrica": name,
                    "Con POG": f"{val_p:.5f}" if not np.isnan(val_p) else "N/A",
                    "Sin POG": f"{val_np:.5f}" if not np.isnan(val_np) else "N/A",
                    "Δ Abs": f"{diff:+.5f}" if not np.isnan(diff) else "N/A",
                    "Δ %": "-" if val_np == 0 or np.isnan(pct) else f"{pct:+.2f}%",
                })

    res_df = pd.DataFrame(summary_rows)
    headers = list(res_df.columns)
    rows = res_df.values.tolist()
    title = "ESTUDIO 2: ABLACIÓN COMPLETA POR LONGITUD DE SECUENCIA (TODAS LAS MÉTRICAS Y MODELOS)"
    return format_table(headers, rows, title), res_df


# ==============================================================================
# ESTUDIO 3: ABLACIÓN DEL RECOMENDADOR NAIVE VS. CLOSEST PASS (TABLA 3 REVISOR)
# ==============================================================================
def run_naive_baseline_ablation(
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

    # 1. Calcular trayectoria real sin intervención (Naive factual baseline)
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

    # 2. Extraer deltas para cada modelo desde el df de usuarios
    rows = []

    # Fila 1: Naive (Trayectoria factual original sin intervención)
    rows.append({
        "Estrategia / Modelo": "Naive Baseline (Trayectoria real sin intervención)",
        "Closest Pass?": "No",
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

        # Valor absoluto simulado = base real + delta
        abs_cov = mean_real["coverage"] + delta_cov
        abs_back = mean_real["backward"] + delta_back
        abs_rep = mean_real["repeat"] + delta_rep
        abs_frac = mean_real["frac_allowed"] + delta_frac
        abs_nov = mean_real["novelty"] + delta_nov

        # Porcentajes de cambio relativo frente a la base real para las 5 métricas
        pct_cov = (delta_cov / mean_real["coverage"]) * 100
        pct_back = (delta_back / mean_real["backward"]) * 100
        pct_rep = (delta_rep / mean_real["repeat"]) * 100
        pct_frac = (delta_frac / mean_real["frac_allowed"]) * 100
        pct_nov = (delta_nov / mean_real["novelty"]) * 100

        rows.append({
            "Estrategia / Modelo": label,
            "Closest Pass?": "Sí",
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
    title = "ESTUDIO 3: NAIVE BASELINE (SIN CLOSEST PASS) VS. INTERVENCIÓN PROPUESTA (TABLA 3)"
    return format_table(headers, table_rows, title), res_df


def main():
    parser = argparse.ArgumentParser(description="Ejecutar estudios de ablación completos para SeqEmb.")
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
    print("EJECUTANDO SUITE COMPLETA DE ESTUDIOS DE ABLACIÓN")
    print("=" * 90 + "\n")

    # 1. Macro POG Ablation
    table1, df1 = run_macro_pog_ablation(df)
    print(table1)
    df1.to_csv(os.path.join(args.output_dir, "ablation_1_pog_macro_significance.csv"), index=False)
    print("\n")

    # 2. Sequence Length / Cold-Start Ablation
    table2, df2 = run_sequence_length_ablation(df, data_all["paths"])
    print(table2)
    df2.to_csv(os.path.join(args.output_dir, "ablation_2_sequence_length_coldstart.csv"), index=False)
    print("\n")

    # 3. Naive Baseline vs. Closest Pass (Reviewer Table 3)
    table3, df3 = run_naive_baseline_ablation(df, data_all)
    print(table3)
    df3.to_csv(os.path.join(args.output_dir, "ablation_3_naive_baseline_table3.csv"), index=False)
    print("\n")

    print(f"-> Todos los CSVs de ablación se han guardado exitosamente en '{args.output_dir}/'.\n")


if __name__ == "__main__":
    main()
