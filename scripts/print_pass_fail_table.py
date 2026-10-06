#!/usr/bin/env python3
"""Script para imprimir en formato tabla las métricas de estudiantes PASS y FAIL.

- Estudiantes PASS: mean MAP, NDCG, MRR
- Estudiantes FAIL: mean Coverage, Backward, Repeat, Frac Allowed Edges, Novelty
"""

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional


def format_table(headers: List[str], rows: List[List[str]], title: Optional[str] = None) -> str:
    """Genera una tabla formateada con bordes limpios sin dependencias externas."""
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


def format_markdown_table(headers: List[str], rows: List[List[str]], title: Optional[str] = None) -> str:
    """Genera una tabla en formato Markdown."""
    lines = []
    if title:
        lines.append(f"### {title}\n")
    header_str = "| " + " | ".join(headers) + " |"
    sep_str = "| " + " | ".join("---" for _ in headers) + " |"
    lines.append(header_str)
    lines.append(sep_str)
    for row in rows:
        lines.append("| " + " | ".join(str(val) for val in row) + " |")
    return "\n".join(lines)


def load_from_json(filepath: str, decimals: int) -> List[Dict[str, Any]]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for model_key, model_data in data.items():
        label = model_data.get("label", model_key)
        mean_pass = model_data.get("mean_pass", {})
        mean_fail = model_data.get("mean_fail", {})

        # Si mean_pass o mean_fail no están precalculados, calcular desde user_metrics
        if not mean_pass or not mean_fail:
            user_metrics = model_data.get("user_metrics", [])
            pass_users = [u for u in user_metrics if u.get("grade") == 1]
            fail_users = [u for u in user_metrics if u.get("grade") == 0]

            def calc_mean(records: List[Dict[str, Any]], key: str) -> float:
                vals = [r.get(key, 0.0) for r in records if key in r]
                return sum(vals) / len(vals) if vals else 0.0

            mean_pass = {
                "mean_map": calc_mean(pass_users, "map"),
                "mean_ndcg": calc_mean(pass_users, "ndcg"),
                "mean_mrr": calc_mean(pass_users, "mrr"),
            }
            mean_fail = {
                "mean_coverage_ratio": calc_mean(fail_users, "coverage_ratio"),
                "mean_backward_ratio": calc_mean(fail_users, "backward_ratio"),
                "mean_repeat_ratio": calc_mean(fail_users, "repeat_ratio"),
                "mean_frac_allowed_edges": calc_mean(fail_users, "frac_allowed_edges"),
                "mean_novelty": calc_mean(fail_users, "novelty"),
            }

        results.append({
            "key": model_key,
            "label": label,
            "pass_map": f"{mean_pass.get('mean_map', 0.0):.{decimals}f}",
            "pass_ndcg": f"{mean_pass.get('mean_ndcg', 0.0):.{decimals}f}",
            "pass_mrr": f"{mean_pass.get('mean_mrr', 0.0):.{decimals}f}",
            "fail_coverage": f"{mean_fail.get('mean_coverage_ratio', 0.0):.{decimals}f}",
            "fail_backward": f"{mean_fail.get('mean_backward_ratio', 0.0):.{decimals}f}",
            "fail_repeat": f"{mean_fail.get('mean_repeat_ratio', 0.0):.{decimals}f}",
            "fail_frac_allowed": f"{mean_fail.get('mean_frac_allowed_edges', 0.0):.{decimals}f}",
            "fail_novelty": f"{mean_fail.get('mean_novelty', 0.0):.{decimals}f}",
        })
    return results


def load_from_csv(filepath: str, decimals: int) -> List[Dict[str, Any]]:
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # Agrupar por modelo
    models_dict = {}
    for r in rows:
        key = r.get("model_key") or r.get("model", "unknown")
        label = r.get("label", key)
        if key not in models_dict:
            models_dict[key] = {"label": label, "pass": [], "fail": []}

        grade = int(r.get("grade", 0))
        if grade == 1:
            models_dict[key]["pass"].append(r)
        else:
            models_dict[key]["fail"].append(r)

    results = []
    for key, data in models_dict.items():
        def calc_mean(records: List[Dict[str, Any]], field: str) -> float:
            vals = [float(rec[field]) for rec in records if field in rec and rec[field] != ""]
            return sum(vals) / len(vals) if vals else 0.0

        p = data["pass"]
        f = data["fail"]
        results.append({
            "key": key,
            "label": data["label"],
            "pass_map": f"{calc_mean(p, 'map'):.{decimals}f}",
            "pass_ndcg": f"{calc_mean(p, 'ndcg'):.{decimals}f}",
            "pass_mrr": f"{calc_mean(p, 'mrr'):.{decimals}f}",
            "fail_coverage": f"{calc_mean(f, 'coverage_ratio'):.{decimals}f}",
            "fail_backward": f"{calc_mean(f, 'backward_ratio'):.{decimals}f}",
            "fail_repeat": f"{calc_mean(f, 'repeat_ratio'):.{decimals}f}",
            "fail_frac_allowed": f"{calc_mean(f, 'frac_allowed_edges'):.{decimals}f}",
            "fail_novelty": f"{calc_mean(f, 'novelty'):.{decimals}f}",
        })
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Imprime en formato tabla las métricas de estudiantes PASS (MAP, NDCG, MRR) y FAIL (Coverage, Backward, Repeat, Frac Allowed Edges, Novelty)."
    )
    parser.add_argument(
        "-i", "--input",
        default="results/folds_full_path.json",
        help="Ruta al archivo JSON o CSV de resultados (default: results/folds_full_path.json)"
    )
    parser.add_argument(
        "-d", "--decimals",
        type=int,
        default=4,
        help="Número de decimales para las métricas (default: 4)"
    )
    parser.add_argument(
        "--format",
        choices=["grid", "markdown", "both"],
        default="grid",
        help="Estilo de la tabla: grid (ASCII), markdown o both (default: grid)"
    )
    parser.add_argument(
        "--mode",
        choices=["separate", "combined", "all"],
        default="separate",
        help="Modo de visualización: separate (2 tablas), combined (1 tabla ancha) o all (default: separate)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: No se encontró el archivo '{args.input}'", file=sys.stderr)
        sys.exit(1)

    if args.input.endswith(".csv"):
        data = load_from_csv(args.input, args.decimals)
    else:
        data = load_from_json(args.input, args.decimals)

    # Definir datos para tablas
    headers_pass = ["Modelo", "MAP (Pass)", "NDCG (Pass)", "MRR (Pass)"]
    rows_pass = [
        [d["label"], d["pass_map"], d["pass_ndcg"], d["pass_mrr"]]
        for d in data
    ]

    headers_fail = ["Modelo", "Coverage", "Backward", "Repeat", "Frac Allowed", "Novelty"]
    rows_fail = [
        [d["label"], d["fail_coverage"], d["fail_backward"], d["fail_repeat"], d["fail_frac_allowed"], d["fail_novelty"]]
        for d in data
    ]

    headers_comb = ["Modelo", "MAP (P)", "NDCG (P)", "MRR (P)", "Coverage (F)", "Backward (F)", "Repeat (F)", "Frac Allowed (F)", "Novelty (F)"]
    rows_comb = [
        [d["label"], d["pass_map"], d["pass_ndcg"], d["pass_mrr"], d["fail_coverage"], d["fail_backward"], d["fail_repeat"], d["fail_frac_allowed"], d["fail_novelty"]]
        for d in data
    ]

    def print_output(fmt_func, is_md=False):
        if args.mode in ["separate", "all"]:
            print(fmt_func(headers_pass, rows_pass, title="ESTUDIANTES PASS (Métricas de Recomendación)"))
            print("\n" if is_md else "")
            print(fmt_func(headers_fail, rows_fail, title="ESTUDIANTES FAIL (Métricas de Navegación/POG)"))
            print("\n" if is_md else "")
        if args.mode in ["combined", "all"]:
            print(fmt_func(headers_comb, rows_comb, title="RESUMEN COMPLETO (PASS + FAIL)"))
            print("\n" if is_md else "")

    if args.format in ["grid", "both"]:
        print_output(format_table, is_md=False)
    if args.format in ["markdown", "both"]:
        if args.format == "both":
            print("\n" + "=" * 80 + "\n--- FORMATO MARKDOWN ---\n" + "=" * 80 + "\n")
        print_output(format_markdown_table, is_md=True)


if __name__ == "__main__":
    main()
