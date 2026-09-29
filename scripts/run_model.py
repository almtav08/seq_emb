#!/usr/bin/env python3
"""Run a single sequential model experiment (with or without POG) with K-Fold cross-validation."""

import argparse
import json
import sys
from pathlib import Path

# Add src to pythonpath
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from seq_emb.experiments.kfold_runner import KFoldRunner


def main():
    parser = argparse.ArgumentParser(description="Run sequential model K-Fold cross-validation.")
    parser.add_argument("--model", type=str, required=True, choices=["caser", "grurec", "sasrec"],
                        help="Model architecture to run.")
    parser.add_argument("--pog", action="store_true", default=True,
                        help="Use pre-trained RotatE POG embeddings.")
    parser.add_argument("--no-pog", dest="pog", action="store_false",
                        help="Learn item embeddings from scratch (non-POG mode).")
    parser.add_argument("--strategy", type=str, default="loocv", choices=["loocv", "kfold"],
                        help="Cross-validation strategy: 'loocv' (Leave-One-Out) or 'kfold' (Stratified K-Fold).")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs per iteration.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--splits", type=int, default=5, help="Number of folds (when strategy='kfold').")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', 'auto').")
    parser.add_argument("--local-window", action="store_true", default=False,
                        help="Evaluate transition metrics on the intervention window only.")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save JSON results.")

    args = parser.parse_args()

    mode_label = "Con POG" if args.pog else "Sin POG"
    print(f"\n{'='*60}")
    print(f"Modelo: {args.model.upper()} ({mode_label}) | Estrategia: {args.strategy.upper()} | Seed: {args.seed} | Epochs: {args.epochs}")
    print(f"{'='*60}\n")

    runner = KFoldRunner(seed=args.seed, device=args.device)
    results = runner.run(
        model_name=args.model,
        use_pog=args.pog,
        strategy=args.strategy,
        n_splits=args.splits,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        local_window=args.local_window,
        verbose=True,
    )

    strat_label = f"LOOCV ({results.get('n_students', 48)} Iteraciones)" if args.strategy == "loocv" else f"{args.splits} Folds"
    print(f"\nResultados Promedio ({strat_label}):")
    print("-" * 50)
    for metric, val in results["mean"].items():
        std_val = results["std"].get(metric, 0.0)
        print(f"  {metric:<25}: {val:+.4f} ± {std_val:.4f}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        print(f"\nResultados guardados en: {out_path}")


if __name__ == "__main__":
    main()
