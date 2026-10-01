#!/usr/bin/env python3
"""Run full benchmark across all 6 model variants (CASER, GRU4Rec, SASRec x with/without POG)."""

import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from seq_emb.core.config import load_config
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
    parser = argparse.ArgumentParser(description="Run full benchmark across all 6 model variants.")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML configuration file (default: configs/default.yaml).")
    parser.add_argument("--strategy", type=str, default="loocv", choices=["loocv", "kfold"],
                        help="Cross-validation strategy: 'loocv' (Leave-One-Out) or 'kfold' (Stratified K-Fold).")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs per iteration (default: from config or 500).")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default: from config or 4).")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducibility (default: from config or 42).")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('cuda', 'cpu', 'auto').")
    parser.add_argument("--local-window", action="store_true", default=False, help="Use local window metrics.")
    parser.add_argument("--output", type=str, default=None, help="Output path for JSON results.")

    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    device = args.device if args.device is not None else cfg.get("device", "auto")
    epochs = args.epochs if args.epochs is not None else cfg.get("num_epochs", 500)
    batch_size = args.batch_size if args.batch_size is not None else cfg.get("batch_size", 4)
    n_splits = cfg.get("n_splits", 5)
    models_cfg = cfg.get("models", {})

    runner = KFoldRunner(seed=seed, device=device)
    all_results = {}

    print("\n" + "=" * 110)
    print("INICIANDO BENCHMARK COMPLETO DE MODELOS SECUENCIALES")
    print(f"Estrategia: {args.strategy.upper()} | Semilla: {seed} | Épocas: {epochs} | Batch: {batch_size} | Dispositivo: {runner.device} | Modo Ventana Local: {args.local_window}")
    if args.config and Path(args.config).exists():
        print(f"Configuración cargada desde: {args.config}")
    print("=" * 110 + "\n")

    for key, model_name, use_pog, label in BENCHMARK_CONFIGS:
        print(f"\n>>> Ejecutando {label} ({args.strategy.upper()})...")
        model_custom_params = models_cfg.get(model_name, None)
        if model_custom_params:
            print(f"    Hiperparámetros ({model_name}): {model_custom_params}")
        try:
            res = runner.run(
                model_name=model_name,
                use_pog=use_pog,
                strategy=args.strategy,
                n_splits=n_splits,
                num_epochs=epochs,
                batch_size=batch_size,
                custom_params=model_custom_params,
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
    print(f"\nResultados guardados exitosamente en: {output_file}")

    # Export user-level metrics across all methods for detailed statistical analysis
    import pandas as pd
    all_user_rows = []
    for key, model_name, use_pog, label in BENCHMARK_CONFIGS:
        res = all_results.get(key, {})
        for u_rec in res.get("user_metrics", []):
            row = {
                "model_key": key,
                "model": model_name,
                "use_pog": use_pog,
                "label": label,
                **u_rec,
            }
            all_user_rows.append(row)

    if all_user_rows:
        df_users = pd.DataFrame(all_user_rows)
        users_csv = output_file.with_name(output_file.stem + "_user_level.csv")
        df_users.to_csv(users_csv, index=False)
        print(f"Métricas individuales por usuario ({len(df_users)} filas: 48 usuarios x {len(BENCHMARK_CONFIGS)} métodos) guardadas en:\n  -> {users_csv}\n")


if __name__ == "__main__":
    main()
