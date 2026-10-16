#!/usr/bin/env python3
"""Run Taguchi orthogonal array tuning and statistical analysis for sequential recommendation models."""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

# Add src to pythonpath
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from seq_emb.tuning.design import get_taguchi_design
from seq_emb.tuning.runner import TaguchiRunner
from seq_emb.tuning.analysis import TaguchiAnalyzer


ALL_MODELS = ["caser", "grurec", "sasrec"]


def print_taguchi_report(analysis: dict, model_label: str):
    print("\n" + "=" * 95)
    print(f"TAGUCHI ANALYSIS REPORT — {model_label.upper()}")
    print("=" * 95)

    print("\n1. Factors and impact (ANOVA - % Explained Variance):")
    print("-" * 95)
    best_df = analysis["best_levels"]
    print(best_df.to_string(index=False))

    print("\n2. Recommended optimal levels per factor (maximum combined Z-score):")
    print("-" * 95)
    recs = []
    for _, row in best_df.iterrows():
        recs.append(f"{row['factor']} -> {row['best_level']} (mean_score={row['mean_score']:.4f}, exp={row['percent_explained']:.1f}%)")
    print(" | ".join(recs))

    pareto_df = analysis["pareto_front"]
    print(f"\n3. Multi-objective Pareto Front ({len(pareto_df)} non-dominated configurations):")
    print("-" * 95)
    pareto_cols = [c for c in pareto_df.columns if not c.startswith("z_") and c != "score_combined"]
    print(pareto_df[pareto_cols].to_string(index=False))
    print("=" * 95 + "\n")


def print_global_summary(summary_records: List[Dict[str, Any]]):
    if not summary_records:
        return
    print("\n" + "#" * 95)
    print("GLOBAL TUNING SUMMARY (ALL MODELS)")
    print("#" * 95)
    header = f"{'Model':<12} | {'Pareto Runs':<12} | {'Best Factor & Optimal Level':<60}"
    print(header)
    print("-" * 95)
    for rec in summary_records:
        print(f"{rec['model']:<12} | {rec['pareto_runs']:<12} | {rec['best_summary']:<60}")
    print("#" * 95 + "\n")


def process_single_model(
    model_name: str,
    use_pog: bool,
    strategy: str,
    epochs: int,
    batch_size: int,
    splits: int,
    seed: int,
    device: str,
    out_dir: Path,
    n_jobs: int = 1,
    analyze_only_path: Path = None,
) -> Dict[str, Any]:
    mode_tag = "pog" if use_pog else "nopog"
    label = f"{model_name}_{mode_tag}"

    if analyze_only_path is not None:
        if analyze_only_path.is_file():
            csv_file = analyze_only_path
        else:
            # Look inside directory for model csv
            candidates = [
                analyze_only_path / f"{model_name}_fine_results.csv",
                analyze_only_path / f"{label}_fine_results.csv",
            ]
            csv_file = next((c for c in candidates if c.exists()), None)
            if csv_file is None:
                print(f"Warning: No results CSV found for {model_name} in {analyze_only_path}")
                return None

        print(f"\nLoading results for {label} from {csv_file}...")
        results_df = pd.read_csv(csv_file)
    else:
        runner = TaguchiRunner(seed=seed, device=device, n_jobs=n_jobs)
        design_df = get_taguchi_design(model_name)
        results_df = runner.run_tuning(
            model_name=model_name,
            use_pog=use_pog,
            strategy=strategy,
            design_df=design_df,
            num_epochs=epochs,
            batch_size=batch_size,
            n_splits=splits,
            n_jobs=n_jobs,
            verbose=True,
        )

        raw_results_file = out_dir / f"{label}_fine_results.csv"
        results_df.to_csv(raw_results_file, index=False)
        print(f"\nRaw Taguchi results saved to: {raw_results_file}")

    # Run analysis
    analyzer = TaguchiAnalyzer(model_name=model_name, results_df=results_df)
    analysis = analyzer.analyze()

    print_taguchi_report(analysis, label)

    # Save artifact files
    zscore_file = out_dir / f"{label}_analysis_results_with_zscores.csv"
    best_file = out_dir / f"{label}_taguchi_best_levels.csv"
    pareto_file = out_dir / f"{label}_pareto_front_runs.csv"

    analysis["full_results"].to_csv(zscore_file, index=False)
    analysis["best_levels"].to_csv(best_file, index=False)
    analysis["pareto_front"].to_csv(pareto_file, index=False)

    best_recs = []
    for _, row in analysis["best_levels"].iterrows():
        best_recs.append(f"{row['factor']}={row['best_level']}")
    best_summary = ", ".join(best_recs[:4])

    return {
        "model": label,
        "pareto_runs": len(analysis["pareto_front"]),
        "best_summary": best_summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Taguchi hyperparameter tuning and statistical analysis.")
    parser.add_argument("--model", type=str, default="all", choices=["caser", "grurec", "sasrec", "all"],
                        help="Model architecture or 'all' to tune all models sequentially.")
    parser.add_argument("--pog", action="store_true", default=True, help="Use RotatE POG embeddings.")
    parser.add_argument("--no-pog", dest="pog", action="store_false", help="Use learnable item embeddings.")
    parser.add_argument("--all-variants", action="store_true", default=False,
                        help="Run both 'with POG' and 'without POG' variants for each selected model.")
    parser.add_argument("--strategy", type=str, default="loocv", choices=["loocv", "kfold"],
                        help="Cross-validation strategy: 'loocv' (Leave-One-Out) or 'kfold' (Stratified K-Fold). Default: loocv.")
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs per run (default: 500).")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4).")
    parser.add_argument("--splits", type=int, default=5, help="Number of folds per run (when strategy='kfold', default: 5).")
    parser.add_argument("--jobs", "-j", type=int, default=1,
                        help="Number of parallel worker processes for Taguchi runs (default: 1). Use -1 for auto.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility.")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', 'auto').")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save output CSVs.")
    parser.add_argument("--analyze-only", type=str, default=None,
                        help="Skip training and run analysis directly on existing results CSV or directory.")

    args = parser.parse_args()

    models_to_run = ALL_MODELS if args.model.lower() == "all" else [args.model.lower().strip()]
    modes_to_run = [True, False] if args.all_variants else [args.pog]

    out_dir = Path(args.output_dir) if args.output_dir else project_root / "results" / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    analyze_path = Path(args.analyze_only) if args.analyze_only else None

    print("\n" + "=" * 95)
    print("TAGUCHI TUNING CONFIGURATION")
    print(f"Models: {', '.join(models_to_run).upper()}")
    print(f"Modalities: {'With POG and Without POG' if args.all_variants else ('With POG' if args.pog else 'Without POG')}")
    print(f"Strategy: {args.strategy.upper()} | Epochs: {args.epochs} | Folds (if kfold): {args.splits} | Workers: {args.jobs} | Seed: {args.seed} | Device: {args.device}")
    print(f"Output directory: {out_dir}")
    print("=" * 95 + "\n")

    summary_records = []
    for model_name in models_to_run:
        for use_pog in modes_to_run:
            res = process_single_model(
                model_name=model_name,
                use_pog=use_pog,
                strategy=args.strategy,
                epochs=args.epochs,
                batch_size=args.batch_size,
                splits=args.splits,
                seed=args.seed,
                device=args.device,
                out_dir=out_dir,
                n_jobs=args.jobs,
                analyze_only_path=analyze_path,
            )
            if res is not None:
                summary_records.append(res)

    print_global_summary(summary_records)
    print(f"Tuning and analysis completed. Files saved to: {out_dir}\n")


if __name__ == "__main__":
    main()
