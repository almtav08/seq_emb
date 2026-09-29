#!/usr/bin/env python3
"""Run full benchmark across all 6 model variants (CASER, GRU4Rec, SASRec x with/without POG)."""

import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from seq_emb.experiments.kfold_runner import KFoldRunner


BENCHMARK_CONFIGS = [
    ("caser_pog", "caser", True, "Caser (con POG)"),
    ("caser_nopog", "caser", False, "Caser (sin POG)"),
    ("grurec_pog", "grurec", True, "GRU4Rec (con POG)"),
    ("grurec_nopog", "grurec", False, "GRU4Rec (sin POG)"),
    ("sasrec_pog", "sasrec", True, "SASRec (con POG)"),
    ("sasrec_nopog", "sasrec", False, "SASRec (sin POG)"),
]


def main():
    import argparse
    parser.add_argument("--strategy", type=str, default="loocv", choices=["loocv", "kfold"],
                        help="Cross-validation strategy: 'loocv' (Leave-One-Out) or 'kfold' (Stratified K-Fold).")
    parser.add_argument("--epochs", type=int, default=500, help="Epochs per iteration (default: 500).")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4).")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility.")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', 'auto').")
    parser.add_argument("--local-window", action="store_true", default=False, help="Use local window metrics.")
    parser.add_argument("--output", type=str, default=None, help="Output path for JSON results.")

    args = parser.parse_args()

    runner = KFoldRunner(seed=args.seed, device=args.device)
    all_results = {}

    print("\n" + "=" * 110)
    print("INICIANDO BENCHMARK COMPLETO DE MODELOS SECUENCIALES")
    print(f"Estrategia: {args.strategy.upper()} | Semilla: {args.seed} | Épocas: {args.epochs} | Dispositivo: {runner.device} | Modo Ventana Local: {args.local_window}")
    print("=" * 110 + "\n")

    for key, model_name, use_pog, label in BENCHMARK_CONFIGS:
        print(f"\n>>> Ejecutando {label} ({args.strategy.upper()})...")
        try:
            res = runner.run(
                model_name=model_name,
                use_pog=use_pog,
                strategy=args.strategy,
                n_splits=5,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                local_window=args.local_window,
                verbose=True,
            )
            res["label"] = label
            all_results[key] = res
            m = res["mean"]
            print(f"    MAP: {m['mean_map']:.4f} | NDCG: {m['mean_ndcg']:.4f} | HR: {m['mean_hr']:.2f} | Coverage: {m['mean_coverage_ratio']:+.4f} | AllowedEdges: {m['mean_frac_allowed_edges']:+.4f}")
        except Exception as e:
            print(f"    ERROR en {label}: {e}")
            all_results[key] = {"error": str(e), "label": label}

    # Print summary table
    print("\n\n" + "=" * 115)
    print("RESUMEN COMPARATIVO DE RESULTADOS")
    print("=" * 115)
    header = f"{'Modelo':<22} | {'MAP':<7} {'NDCG':<7} {'HR':<5} {'MRR':<7} | {'Coverage':<8} {'Backward':<9} {'Repeat':<8} {'Progress':<9} {'FracAllowed':<11} {'Novelty':<8}"
    print(header)
    print("-" * 115)

    for key, _, _, label in BENCHMARK_CONFIGS:
        res = all_results.get(key, {})
        m = res.get("mean", {})
        if m:
            print(
                f"{label:<22} | "
                f"{m.get('mean_map', 0):.4f} "
                f"{m.get('mean_ndcg', 0):.4f} "
                f"{m.get('mean_hr', 0):.2f} "
                f"{m.get('mean_mrr', 0):.4f} | "
                f"{m.get('mean_coverage_ratio', 0):+.4f} "
                f"{m.get('mean_backward_ratio', 0):+.4f} "
                f"{m.get('mean_repeat_ratio', 0):+.4f} "
                f"{m.get('mean_progress_ratio', 0):+.4f} "
                f"{m.get('mean_frac_allowed_edges', 0):+.4f} "
                f"{m.get('mean_novelty', 0):+.4f}"
            )
        else:
            print(f"{label:<22} | ERROR: {res.get('error', 'desconocido')}")
    print("-" * 115)

    # Save output
    output_file = Path(args.output) if args.output else project_root / "results" / ("folds_local_window.json" if args.local_window else "folds_full_path.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nResultados guardados exitosamente en: {output_file}\n")


if __name__ == "__main__":
    main()
