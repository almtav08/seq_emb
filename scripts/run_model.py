#!/usr/bin/env python3
"""Run a single sequential model experiment (with or without POG) with K-Fold cross-validation."""

import argparse
import json
import sys
from pathlib import Path

# Add src to pythonpath
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from seq_emb.core.config import load_config
from seq_emb.experiments.kfold_runner import KFoldRunner


def main():
    parser = argparse.ArgumentParser(description="Run sequential model K-Fold cross-validation.")
    parser.add_argument("--model", type=str, required=True, choices=["caser", "grurec", "sasrec"],
                        help="Model architecture to run.")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML configuration file (default: configs/default.yaml).")
    parser.add_argument("--pog", action="store_true", default=True,
                        help="Use pre-trained RotatE POG embeddings.")
    parser.add_argument("--no-pog", dest="pog", action="store_false",
                        help="Learn item embeddings from scratch (non-POG mode).")
    parser.add_argument("--strategy", type=str, default="loocv", choices=["loocv", "kfold"],
                        help="Cross-validation strategy: 'loocv' (Leave-One-Out) or 'kfold' (Stratified K-Fold).")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs per iteration.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for training.")
    parser.add_argument("--splits", type=int, default=None, help="Number of folds (when strategy='kfold').")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('cuda', 'cpu', 'auto').")
    parser.add_argument("--local-window", action="store_true", default=False,
                        help="Evaluate transition metrics on the intervention window only.")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save JSON results.")

    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    device = args.device if args.device is not None else cfg.get("device", "auto")
    epochs = args.epochs if args.epochs is not None else cfg.get("num_epochs", 500)
    batch_size = args.batch_size if args.batch_size is not None else cfg.get("batch_size", 4)
    splits = args.splits if args.splits is not None else cfg.get("n_splits", 5)
    model_custom_params = cfg.get("models", {}).get(args.model, None)

    mode_label = "Con POG" if args.pog else "Sin POG"
    print(f"\n{'='*60}")
    print(f"Modelo: {args.model.upper()} ({mode_label}) | Estrategia: {args.strategy.upper()} | Seed: {seed} | Epochs: {epochs} | Batch: {batch_size}")
    if model_custom_params:
        print(f"Hiperparámetros óptimos ({args.model}): {model_custom_params}")
    print(f"{'='*60}\n")

    runner = KFoldRunner(seed=seed, device=device)
    results = runner.run(
        model_name=args.model,
        use_pog=args.pog,
        strategy=args.strategy,
        n_splits=splits,
        num_epochs=epochs,
        batch_size=batch_size,
        custom_params=model_custom_params,
        local_window=args.local_window,
        verbose=True,
    )

    strat_label = f"LOOCV ({results.get('n_students', 48)} Iteraciones)" if args.strategy == "loocv" else f"{args.splits} Folds"
    print(f"\nResultados Promedio ({strat_label}):")
    print("-" * 50)
    for metric, val in results["mean"].items():
        std_val = results["std"].get(metric, 0.0)
        print(f"  {metric:<25}: {val:+.4f} ± {std_val:.4f}")

    import pandas as pd

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        print(f"\nResultados guardados en: {out_path}")
        csv_path = out_path.with_suffix(".csv")
    else:
        csv_path = project_root / "results" / f"user_metrics_{args.model}_{'pog' if args.pog else 'nopog'}_{args.strategy}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    user_rows = results.get("user_metrics", [])
    if user_rows:
        df_u = pd.DataFrame(user_rows)
        df_u.insert(0, "model", args.model)
        df_u.insert(1, "use_pog", args.pog)
        df_u.insert(2, "strategy", args.strategy)
        df_u.to_csv(csv_path, index=False)
        print(f"Métricas individuales por usuario ({len(df_u)} usuarios) guardadas en:\n  -> {csv_path}\n")


if __name__ == "__main__":
    main()
