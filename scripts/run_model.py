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
    parser.add_argument("--model", type=str, required=True,
                        help="Model architecture to run ('caser', 'grurec', 'sasrec', or 'all' for all 3).")
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
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints (default: 'checkpoints').")
    parser.add_argument("--no-save-checkpoints", dest="save_checkpoints", action="store_false", default=True,
                        help="Disable saving model checkpoints.")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save JSON results.")

    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    device = args.device if args.device is not None else cfg.get("device", "auto")
    epochs = args.epochs if args.epochs is not None else cfg.get("num_epochs", 500)
    batch_size = args.batch_size if args.batch_size is not None else cfg.get("batch_size", 4)
    splits = args.splits if args.splits is not None else cfg.get("n_splits", 5)
    models_cfg = cfg.get("models", {})

    models_to_run = ["caser", "sasrec", "grurec"] if args.model.lower() == "all" else [args.model.lower()]

    import pandas as pd
    runner = KFoldRunner(seed=seed, device=device)

    for current_model in models_to_run:
        model_custom_params = models_cfg.get(current_model, None)
        mode_label = "With POG" if args.pog else "Without POG"
        print(f"\n{'='*60}")
        print(f"Model: {current_model.upper()} ({mode_label}) | Strategy: {args.strategy.upper()} | Seed: {seed} | Epochs: {epochs} | Batch: {batch_size}")
        if model_custom_params:
            print(f"Optimal hyperparameters ({current_model}): {model_custom_params}")
        print(f"{'='*60}\n")

        results = runner.run(
            model_name=current_model,
            use_pog=args.pog,
            strategy=args.strategy,
            n_splits=splits,
            num_epochs=epochs,
            batch_size=batch_size,
            custom_params=model_custom_params,
            local_window=args.local_window,
            verbose=True,
            save_checkpoints=args.save_checkpoints,
            checkpoints_dir=args.checkpoints_dir,
        )

        strat_label = f"LOOCV ({results.get('n_students', 48)} Iterations)" if args.strategy == "loocv" else f"{splits} Folds"
        print(f"\nAverage Results ({strat_label}) - {current_model.upper()}:")
        print("-" * 50)
        for metric, val in results["mean"].items():
            std_val = results["std"].get(metric, 0.0)
            print(f"  {metric:<25}: {val:+.4f} ± {std_val:.4f}")

        if args.output:
            out_path = Path(args.output)
            if len(models_to_run) > 1:
                out_path = out_path.with_name(f"{out_path.stem}_{current_model}{out_path.suffix}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4)
            print(f"\nResults saved to: {out_path}")
            csv_path = out_path.with_suffix(".csv")
        else:
            csv_path = project_root / "results" / f"user_metrics_{current_model}_{'pog' if args.pog else 'nopog'}_{args.strategy}.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)

        user_rows = results.get("user_metrics", [])
        if user_rows:
            df_u = pd.DataFrame(user_rows)
            df_u.insert(0, "model", current_model)
            df_u.insert(1, "use_pog", args.pog)
            df_u.insert(2, "strategy", args.strategy)
            df_u.to_csv(csv_path, index=False)
            print(f"Individual user-level metrics ({len(df_u)} users) saved to:\n  -> {csv_path}\n")


if __name__ == "__main__":
    main()
