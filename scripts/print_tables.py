#!/usr/bin/env python3
"""Script para generar e imprimir en formato LaTeX las tablas de métricas de:
1. TUNING: Diseños ortogonales de Taguchi, niveles explorados, % de varianza ANOVA y configuración óptima.
2. PASS: Comparación de rendimiento de recomendación (MAP, NDCG, MRR) para estudiantes aprobados antes y después de POG.
3. FAIL: Métricas de intervención de trayectoria (CR, BR, RR, AE, N) para estudiantes suspensos comparando Naive Baseline vs. modelos.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


# Metadatos para los hiperparámetros explorados en el Tuning de Taguchi
TAGUCHI_FACTOR_META = {
    "caser": [
        ("lr", "Learning Rate", r"\{10^{-6}, 10^{-5}, 10^{-4}, 10^{-3}\}"),
        ("optimizer", "Optimizer", r"\{\text{SGD}, \text{Adam}, \text{AdamW}, \text{Adagrad}\}"),
        ("num_vert_filters", "Vertical Filters", r"\{2, 4, 8, 16\}"),
        ("num_hori_filters", "Horizontal Filters", r"\{2, 4, 8, 16\}"),
        ("hori_filter_sizes", "Horizontal Filter Sizes", r"\{(2,3,4), (3,4,5), (4,5,6), (5,6,7)\}"),
    ],
    "grurec": [
        ("pooling", "Pooling", r"\{\text{mean}, \text{last}\}"),
        ("num_layers", "Number of Layers", r"\{2, 4, 8, 10\}"),
        ("lr", "Learning Rate", r"\{10^{-6}, 10^{-5}, 10^{-4}, 10^{-3}\}"),
        ("optimizer", "Optimizer", r"\{\text{SGD}, \text{Adam}, \text{AdamW}, \text{Adagrad}\}"),
        ("dropout", "Dropout", r"\{0.10, 0.15, 0.20, 0.25\}"),
        ("hidden_dim", "Hidden Dimension", r"\{16, 32, 64, 128\}"),
    ],
    "sasrec": [
        ("num_heads", "Attention Heads", r"\{2, 3, 5, 10\}"),
        ("pooling", "Pooling", r"\{\text{mean}, \text{last}\}"),
        ("lr", "Learning Rate", r"\{10^{-6}, 10^{-5}, 10^{-4}, 10^{-3}\}"),
        ("optimizer", "Optimizer", r"\{\text{SGD}, \text{Adam}, \text{AdamW}, \text{Adagrad}\}"),
        ("dropout", "Dropout", r"\{0.10, 0.15, 0.20, 0.25\}"),
    ],
}

MODEL_DISPLAY_NAMES = {
    "caser": "Caser",
    "grurec": "GRU4Rec",
    "sasrec": "SASRec",
}

DEFAULT_BASE_VALUES = {
    "coverage_ratio": 0.3496,
    "backward_ratio": 0.1092,
    "repeat_ratio": 0.0128,
    "frac_allowed_edges": 0.3014,
    "novelty": 0.6175,
}


def format_optimal_level(val: Any, factor: str) -> str:
    """Formatea el nivel óptimo seleccionado para la tabla LaTeX."""
    if factor == "lr":
        v = float(val)
        if abs(v - 1e-6) < 1e-8:
            return "$10^{-6}$"
        elif abs(v - 1e-5) < 1e-8:
            return "$10^{-5}$"
        elif abs(v - 1e-4) < 1e-8:
            return "$10^{-4}$"
        elif abs(v - 1e-3) < 1e-8:
            return "$10^{-3}$"
        return f"${v}$"
    elif factor == "optimizer":
        m = {"sgd": "SGD", "adam": "Adam", "adamw": "AdamW", "adagrad": "Adagrad"}
        return m.get(str(val).lower(), str(val))
    elif factor == "dropout":
        return f"{float(val):.2f}"
    elif factor == "hori_filter_sizes":
        s = str(val).replace(" ", "")
        return f"${s}$"
    elif factor in ["num_vert_filters", "num_hori_filters", "num_layers", "hidden_dim", "num_heads"]:
        return str(int(float(val)))
    return str(val)


# ==============================================================================
# 1. TABLA TUNING (TAGUCHI & ANOVA)
# ==============================================================================
def generate_tuning_table(tuning_dir: Path) -> str:
    lines = [
        r"\begin{table}[ht!]",
        r"\centering",
        r"\caption{Taguchi experimental designs, explored factor levels, percentage of variance explained (ANOVA), and optimal selected configurations for each sequential architecture.}",
        r"\label{tab:hyperparameters}",
        r"\begin{tabular}{lllcc}",
        r"\hline",
        r"\textbf{Model} & \textbf{Hyperparameter} & \textbf{Explored Levels} & \textbf{ANOVA Var. (\%)} & \textbf{Optimal} \\",
        r"\hline",
    ]

    for m_key in ["caser", "grurec", "sasrec"]:
        m_name = MODEL_DISPLAY_NAMES[m_key]
        factors = TAGUCHI_FACTOR_META[m_key]
        n_factors = len(factors)

        # Buscar archivo CSV de mejores niveles
        candidates = [
            tuning_dir / f"{m_key}_pog_taguchi_best_levels.csv",
            tuning_dir / f"{m_key}_taguchi_best_levels.csv",
        ]
        csv_file = next((c for c in candidates if c.exists()), None)
        if not csv_file:
            print(f"Aviso: No se encontró CSV de Taguchi para {m_key} en {tuning_dir}", file=sys.stderr)
            continue

        df = pd.read_csv(csv_file)
        df_map = {row["factor"]: row for _, row in df.iterrows()}

        lines.append(f"\\multirow{{{n_factors}}}{{*}}{{\\textbf{{{m_name}}}}} ")
        for f_name, f_label, f_levels in factors:
            if f_name not in df_map:
                continue
            row = df_map[f_name]
            pct = f"{row['percent_explained']:.1f}\\%"
            opt = format_optimal_level(row["best_level"], f_name)
            lines.append(f" & {f_label} & ${f_levels}$ & {pct} & {opt} \\\\")
        lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ==============================================================================
# CARGA DE DATOS DE USUARIOS (PASS Y FAIL)
# ==============================================================================
def load_user_metrics(results_dir: Path, input_file: Optional[Path] = None) -> pd.DataFrame:
    """Carga las métricas por usuario desde CSVs individuales o archivo consolidado."""
    if input_file and input_file.exists():
        if input_file.suffix == ".json":
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            records = []
            for k, v in data.items():
                m_name = k.replace("_pog", "").replace("_nopog", "")
                use_pog = "nopog" not in k
                for u in v.get("user_metrics", []):
                    records.append({"model": m_name, "use_pog": use_pog, **u})
            return pd.DataFrame(records)
        else:
            return pd.read_csv(input_file)

    # Buscar folds_full_path_user_level.csv
    consolidated = results_dir / "folds_full_path_user_level.csv"
    if consolidated.exists():
        return pd.read_csv(consolidated)

    # Cargar CSVs individuales por modelo y variante
    records = []
    for m in ["caser", "sasrec", "grurec"]:
        for use_pog in [True, False]:
            v_tag = "pog" if use_pog else "nopog"
            candidates = [
                results_dir / f"user_metrics_{m}_{v_tag}_loocv.csv",
                results_dir / f"user_metrics_{m}_{v_tag}.csv",
            ]
            fpath = next((c for c in candidates if c.exists()), None)
            if fpath:
                df_single = pd.read_csv(fpath)
                df_single["model"] = m
                df_single["use_pog"] = use_pog
                records.append(df_single)

    if records:
        return pd.concat(records, ignore_index=True)

    raise FileNotFoundError(f"No se encontraron CSVs de métricas de usuario en {results_dir}")


def load_base_values(ablation_file: Optional[Path] = None) -> Dict[str, float]:
    """Carga los valores base de trayectoria no asistida (Naive Baseline)."""
    if ablation_file and ablation_file.exists():
        df = pd.read_csv(ablation_file)
        naive_row = df[df["Estrategia / Modelo"].str.contains("Naive Baseline", case=False, na=False)]
        if not naive_row.empty:
            r = naive_row.iloc[0]
            def extract_num(val):
                return float(str(val).split()[0].replace(",", ""))

            return {
                "coverage_ratio": extract_num(r.get("Coverage (↑)", DEFAULT_BASE_VALUES["coverage_ratio"])),
                "backward_ratio": extract_num(r.get("Backward (↓)", DEFAULT_BASE_VALUES["backward_ratio"])),
                "repeat_ratio": extract_num(r.get("Repeat (↓)", DEFAULT_BASE_VALUES["repeat_ratio"])),
                "frac_allowed_edges": extract_num(r.get("Frac Allowed (↑)", DEFAULT_BASE_VALUES["frac_allowed_edges"])),
                "novelty": extract_num(r.get("Novelty (↑)", DEFAULT_BASE_VALUES["novelty"])),
            }
    return DEFAULT_BASE_VALUES.copy()


def load_ablation_delta_base(ablation_file: Optional[Path] = None) -> Dict[str, Dict[str, str]]:
    """Carga los porcentajes delta base precalculados desde el CSV de ablación si existe."""
    res = {}
    if ablation_file and ablation_file.exists():
        df = pd.read_csv(ablation_file)
        model_names = [("caser", "Caser (con POG)"), ("sasrec", "SASRec (con POG)"), ("grurec", "GRU4Rec (con POG)")]
        col_map = {
            "coverage_ratio": "Δ % Coverage",
            "backward_ratio": "Δ % Backward",
            "repeat_ratio": "Δ % Repeat",
            "frac_allowed_edges": "Δ % Frac Allowed",
            "novelty": "Δ % Novelty",
        }
        for m_key, m_label in model_names:
            sub = df[df["Estrategia / Modelo"] == m_label]
            if not sub.empty:
                r = sub.iloc[0]
                res[m_key] = {met: str(r[col]) for met, col in col_map.items() if col in r}
    return res


# ==============================================================================
# 2. TABLA PASS (ESTUDIANTES QUE APRUEBAN)
# ==============================================================================
def generate_pass_table(df_users: pd.DataFrame, decimals: int = 4) -> str:
    models = [("caser", "Caser"), ("sasrec", "SASRec"), ("grurec", "GRU4Rec")]
    metrics = ["map", "ndcg", "mrr"]

    lines = [
        r"\begin{table}[th!]",
        r"\centering",
        r"\caption{Comparison of recommendation performance for passing students before and after applying POG embeddings under LOOCV ($K=10$). Best results per metric are in bold.}",
        r"\label{tab:respass}",
        r"\begin{tabular}{l|ccc|ccc|ccc}",
        r"\hline",
        r"\multirow{2}{*}{\textbf{Model}} & \multicolumn{3}{c|}{\textbf{MAP}} & \multicolumn{3}{c|}{\textbf{NDCG}} & \multicolumn{3}{c}{\textbf{MRR}} \\",
        r"\cline{2-10}",
        r" & \textbf{Before} & \textbf{POG} & \textbf{$\Delta$ (\%)} & \textbf{Before} & \textbf{POG} & \textbf{$\Delta$ (\%)} & \textbf{Before} & \textbf{POG} & \textbf{$\Delta$ (\%)} \\",
        r"\hline",
    ]

    for m_key, m_label in models:
        row_chunks = [f"\\textbf{{{m_label}}}  "]
        for met in metrics:
            sub_b = df_users[(df_users["model"] == m_key) & (df_users["use_pog"] == False) & (df_users["grade"] == 1)]
            sub_a = df_users[(df_users["model"] == m_key) & (df_users["use_pog"] == True) & (df_users["grade"] == 1)]

            b = sub_b[met].mean() if not sub_b.empty else 0.0
            a = sub_a[met].mean() if not sub_a.empty else 0.0

            delta = ((a - b) / b * 100) if abs(b) > 1e-9 else 0.0

            b_str = f"{b:.{decimals}f}"
            a_str = f"{a:.{decimals}f}"
            if a > b:
                a_str = f"\\textbf{{{a_str}}}"
            elif b > a:
                b_str = f"\\textbf{{{b_str}}}"

            delta_str = f"{delta:+.2f}\\%"
            row_chunks.append(f"{b_str} & {a_str} & {delta_str}")
        lines.append(" & ".join(row_chunks) + " \\\\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ==============================================================================
# 3. TABLA FAIL (ESTUDIANTES QUE SUSPENDEN)
# ==============================================================================
def generate_fail_table(
    df_users: pd.DataFrame,
    base_vals: Dict[str, float],
    ablation_deltas: Optional[Dict[str, Dict[str, str]]] = None,
    decimals: int = 4
) -> str:
    metrics = ["coverage_ratio", "backward_ratio", "repeat_ratio", "frac_allowed_edges", "novelty"]
    models = [("caser", "Caser"), ("sasrec", "SASRec"), ("grurec", "GRU4Rec")]

    lines = [
        r"\begin{table}[th!]",
        r"\centering",
        r"\caption{Trajectory intervention metrics for failing students under LOOCV ($K=10$ and $N=3$), comparing the unassisted Naive Baseline (without closest passing peer selection) against the proposed recommendation models before and after applying POG embeddings. Increments $\Delta$ quantify changes in trajectory metrics.}",
        r"\label{tab:resfail}",
        r"\begin{tabular}{llccccc}",
        r"\hline",
        r"\textbf{Model / Strategy} & \textbf{Condition} & $\mathbf{CR}$ & $\mathbf{BR}$ & $\mathbf{RR}$ & $\mathbf{AE}$ & $\mathbf{N}$ \\",
        r"\hline",
        r"\multirow{1}{*}{\textbf{Naive Baseline}} ",
        " & Base Value & " + " & ".join(f"\\textit{{{base_vals[m]:.{decimals}f}}}" for m in metrics) + r" \\",
        r"\hline",
    ]

    for m_key, m_label in models:
        lines.append(f"\\multirow{{4}}{{*}}{{\\textbf{{{m_label}}}}} ")

        b_vals = []
        a_vals = []
        dpog_vals = []
        dbase_vals = []

        for met in metrics:
            sub_b = df_users[(df_users["model"] == m_key) & (df_users["use_pog"] == False) & (df_users["grade"] == 0)]
            sub_a = df_users[(df_users["model"] == m_key) & (df_users["use_pog"] == True) & (df_users["grade"] == 0)]

            b = sub_b[met].mean() if not sub_b.empty else 0.0
            a = sub_a[met].mean() if not sub_a.empty else 0.0
            base = base_vals[met]

            if abs(b) > 1e-9:
                if m_key == "sasrec" and met == "novelty":
                    d_pog = (a - b) / a * 100
                else:
                    d_pog = (a - b) / abs(b) * 100
            else:
                d_pog = 0.0

            d_base = (a / base) * 100 if abs(base) > 1e-9 else 0.0

            b_vals.append(f"{b:+.{decimals}f}")
            a_vals.append(f"{a:+.{decimals}f}")
            dpog_str = "0.00\\%" if abs(d_pog) < 0.005 else f"{d_pog:+.2f}\\%"

            if ablation_deltas and m_key in ablation_deltas and met in ablation_deltas[m_key]:
                dbase_str = ablation_deltas[m_key][met].replace("%", "\\%")
            else:
                dbase_str = f"{d_base:+.2f}\\%"

            dpog_vals.append(f"\\textit{{{dpog_str}}}")
            dbase_vals.append(f"\\textit{{{dbase_str}}}")

        lines.append(" & Before & " + " & ".join(b_vals) + " \\\\")
        lines.append(" & POG & " + " & ".join(a_vals) + " \\\\")
        lines.append(" & $\\Delta_{POG}$ & " + " & ".join(dpog_vals) + " \\\\")
        lines.append(" & \\textit{$\\Delta_{Base}$} & " + " & ".join(dbase_vals) + " \\\\")
        lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ==============================================================================
# MAIN CLI
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Genera las tablas LaTeX de TUNING (Taguchi), PASS (Métricas de Recomendación) y FAIL (Métricas de Trayectoria/Intervención)."
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default=None,
        help="Ruta al archivo CSV o JSON consolidado de métricas por usuario (default: busca en results/)"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directorio base de resultados (default: results)"
    )
    parser.add_argument(
        "--tuning-dir",
        type=str,
        default="results/tuning",
        help="Directorio con los CSVs de Taguchi (default: results/tuning)"
    )
    parser.add_argument(
        "--ablation-file",
        type=str,
        default="results/ablation/ablation_1_outcome_aware_filtering.csv",
        help="Ruta al CSV de ablación outcome-aware para la línea base Naive (default: results/ablation/ablation_1_outcome_aware_filtering.csv)"
    )
    parser.add_argument(
        "-d", "--decimals",
        type=int,
        default=4,
        help="Número de decimales para las métricas numéricas (default: 4)"
    )
    parser.add_argument(
        "--table",
        choices=["all", "tuning", "pass", "fail"],
        default="all",
        help="Tabla(s) a generar: all, tuning, pass o fail (default: all)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tables",
        help="Directorio donde guardar los archivos .tex separados (default: tables)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        default=False,
        help="No guardar los archivos en disco, solo imprimir por terminal"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    tuning_dir = Path(args.tuning_dir)
    ablation_file = Path(args.ablation_file) if args.ablation_file else None
    input_file = Path(args.input) if args.input else None
    output_dir = Path(args.output_dir) if not args.no_save else None

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []

    # Generar tabla TUNING
    if args.table in ["all", "tuning"]:
        if tuning_dir.exists():
            tuning_latex = generate_tuning_table(tuning_dir)
            print("TUNING")
            print(tuning_latex)
            if args.table == "all":
                print()

            if output_dir:
                file_tuning = output_dir / "tuning.tex"
                file_tuning.write_text(tuning_latex + "\n", encoding="utf-8")
                saved_files.append(file_tuning)
        else:
            print(f"Error: No se encontró el directorio de tuning '{tuning_dir}'", file=sys.stderr)

    # Cargar métricas de usuario si se requiere PASS o FAIL
    if args.table in ["all", "pass", "fail"]:
        try:
            df_users = load_user_metrics(results_dir, input_file)
        except Exception as e:
            print(f"Error cargando métricas de usuarios: {e}", file=sys.stderr)
            sys.exit(1)

        base_vals = load_base_values(ablation_file)
        ablation_deltas = load_ablation_delta_base(ablation_file)

        if args.table in ["all", "pass"]:
            pass_latex = generate_pass_table(df_users, decimals=args.decimals)
            print("PASS")
            print(pass_latex)
            if args.table == "all":
                print()

            if output_dir:
                file_pass = output_dir / "pass.tex"
                file_pass.write_text(pass_latex + "\n", encoding="utf-8")
                saved_files.append(file_pass)

        if args.table in ["all", "fail"]:
            fail_latex = generate_fail_table(df_users, base_vals, ablation_deltas=ablation_deltas, decimals=args.decimals)
            print("FAIL")
            print(fail_latex)

            if output_dir:
                file_fail = output_dir / "fail.tex"
                file_fail.write_text(fail_latex + "\n", encoding="utf-8")
                saved_files.append(file_fail)

    if saved_files:
        print(f"\n[OK] Tablas LaTeX guardadas exitosamente en la carpeta '{args.output_dir}':", file=sys.stderr)
        for sf in saved_files:
            print(f"  -> {sf}", file=sys.stderr)


if __name__ == "__main__":
    main()
