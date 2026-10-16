#!/usr/bin/env python3
"""End-to-end experiment pipeline runner for SeqEmb.

Executes all experimental stages in sequence:
1. Taguchi Orthogonal Hyperparameter Tuning & Pareto Analysis (scripts/run_taguchi.py)
2. Model Training and LOOCV/K-Fold Benchmark with Checkpoint Saving (scripts/run_benchmark.py)
3. Algorithmic Ablation Studies: Outcome-Aware & Time-Decay (scripts/run_ablation_study.py)
4. Publication-Ready LaTeX Table Generation (scripts/print_tables.py)
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))


def print_banner(stage_num: int, total_stages: int, title: str):
    banner = f"STAGE {stage_num}/{total_stages}: {title.upper()}"
    width = max(80, len(banner) + 8)
    print("\n" + "=" * width)
    print(f"    {banner}")
    print("=" * width + "\n", flush=True)


def run_stage(cmd: list, stage_name: str):
    print(f"[Pipeline] Executing: {' '.join(cmd)}\n", flush=True)
    start_time = time.time()
    try:
        res = subprocess.run(cmd, cwd=str(project_root), check=True)
        elapsed = time.time() - start_time
        print(f"\n[Pipeline] [OK] {stage_name} completed in {elapsed:.1f}s.\n", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[Pipeline] [ERROR] {stage_name} failed with exit code {e.returncode}.", file=sys.stderr, flush=True)
        sys.exit(e.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Run end-to-end experiments pipeline: Taguchi tuning -> Model training -> Ablation studies -> Tables."
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        choices=["all", "caser", "grurec", "sasrec"],
        help="Model architecture(s) to run (default: all -> caser, grurec, sasrec)."
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="loocv",
        choices=["loocv", "kfold"],
        help="Cross-validation strategy: 'loocv' (Leave-One-Out) or 'kfold' (default: loocv)."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Training epochs for model benchmark (default: 500)."
    )
    parser.add_argument(
        "--taguchi-epochs",
        type=int,
        default=500,
        help="Training epochs per Taguchi tuning iteration (default: 500)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for training (default: 4)."
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=5,
        help="Number of folds when strategy='kfold' (default: 5)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device ('cuda', 'cpu', 'auto'). Default: auto."
    )
    parser.add_argument(
        "--jobs", "-j",
        type=int,
        default=1,
        help="Number of parallel workers for Taguchi runs (default: 1)."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=3,
        help="Curricular recommendation horizon steps N for ablation (default: 3)."
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default="checkpoints",
        help="Directory to save/load model checkpoints (default: checkpoints)."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory for results output (default: results)."
    )
    parser.add_argument(
        "--tables-dir",
        type=str,
        default="tables",
        help="Directory to export LaTeX tables (default: tables)."
    )
    parser.add_argument(
        "--pog-only",
        action="store_true",
        default=False,
        help="In benchmark stage, run only POG variants (skip non-POG baselines)."
    )
    parser.add_argument(
        "--skip-taguchi",
        action="store_true",
        default=False,
        help="Skip Stage 1 (Taguchi tuning) and use existing tuning results."
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        default=False,
        help="Skip Stage 2 (model benchmark/training) and use existing checkpoints."
    )
    parser.add_argument(
        "--skip-ablation",
        action="store_true",
        default=False,
        help="Skip Stage 3 (ablation studies)."
    )
    parser.add_argument(
        "--no-tables",
        action="store_true",
        default=False,
        help="Skip Stage 4 (final LaTeX table generation)."
    )

    args = parser.parse_args()

    python_bin = sys.executable
    total_start = time.time()

    # Determine total active stages
    stages = []
    if not args.skip_taguchi:
        stages.append("Taguchi Tuning")
    if not args.skip_training:
        stages.append("Model Benchmark & Training")
    if not args.skip_ablation:
        stages.append("Ablation Studies")
    if not args.no_tables:
        stages.append("LaTeX Tables Generation")

    total_stages = len(stages)
    current_stage = 1

    print("\n" + "#" * 80)
    print("STARTING COMPLETE SEQEMB EXPERIMENT PIPELINE")
    print(f"Models: {args.models.upper()} | Strategy: {args.strategy.upper()} | Seed: {args.seed} | Device: {args.device}")
    print(f"Active Stages: {', '.join(stages)}")
    print("#" * 80 + "\n", flush=True)

    # --------------------------------------------------------------------------
    # STAGE 1: TAGUCHI TUNING
    # --------------------------------------------------------------------------
    if not args.skip_taguchi:
        print_banner(current_stage, total_stages, "Taguchi Hyperparameter Tuning & Pareto Analysis")
        taguchi_cmd = [
            python_bin,
            "scripts/run_taguchi.py",
            "--model", args.models,
            "--strategy", args.strategy,
            "--epochs", str(args.taguchi_epochs),
            "--batch-size", str(args.batch_size),
            "--splits", str(args.splits),
            "--jobs", str(args.jobs),
            "--seed", str(args.seed),
            "--device", args.device,
            "--output-dir", str(Path(args.results_dir) / "tuning"),
        ]
        run_stage(taguchi_cmd, "Stage 1: Taguchi Tuning")
        current_stage += 1
    else:
        print("[Pipeline] Skipping Stage 1 (Taguchi Tuning).")

    # --------------------------------------------------------------------------
    # STAGE 2: MODEL BENCHMARK / TRAINING
    # --------------------------------------------------------------------------
    if not args.skip_training:
        print_banner(current_stage, total_stages, "Model Benchmark Training & Evaluation")
        benchmark_cmd = [
            python_bin,
            "scripts/run_benchmark.py",
            "--strategy", args.strategy,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--seed", str(args.seed),
            "--device", args.device,
            "--checkpoints-dir", args.checkpoints_dir,
            "--output", str(Path(args.results_dir) / "folds_full_path.json"),
        ]
        if args.pog_only:
            benchmark_cmd.append("--pog-only")

        run_stage(benchmark_cmd, "Stage 2: Model Benchmark & Training")
        current_stage += 1
    else:
        print("[Pipeline] Skipping Stage 2 (Model Benchmark & Training).")

    # --------------------------------------------------------------------------
    # STAGE 3: ABLATION STUDIES
    # --------------------------------------------------------------------------
    if not args.skip_ablation:
        print_banner(current_stage, total_stages, "Outcome-Aware & Time-Decay Ablation Studies")
        ablation_cmd = [
            python_bin,
            "scripts/run_ablation_study.py",
            "--model", args.models,
            "--checkpoints-dir", args.checkpoints_dir,
            "--output-dir", str(Path(args.results_dir) / "ablation"),
            "--tables-dir", args.tables_dir,
            "--device", args.device,
            "--steps", str(args.steps),
        ]
        run_stage(ablation_cmd, "Stage 3: Ablation Studies")
        current_stage += 1
    else:
        print("[Pipeline] Skipping Stage 3 (Ablation Studies).")

    # --------------------------------------------------------------------------
    # STAGE 4: LATEX TABLES GENERATION
    # --------------------------------------------------------------------------
    if not args.no_tables:
        print_banner(current_stage, total_stages, "LaTeX Tables Generation")
        tables_cmd = [
            python_bin,
            "scripts/print_tables.py",
            "--table", "all",
            "--results-dir", args.results_dir,
            "--tuning-dir", str(Path(args.results_dir) / "tuning"),
            "--ablation-file", str(Path(args.results_dir) / "ablation" / "ablation_1_outcome_aware_filtering.csv"),
            "--output-dir", args.tables_dir,
        ]
        run_stage(tables_cmd, "Stage 4: LaTeX Tables Generation")
    else:
        print("[Pipeline] Skipping Stage 4 (LaTeX Tables Generation).")

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 80)
    print(f"ALL EXPERIMENTS SUCCESSFULLY COMPLETED IN {total_elapsed:.1f}s!")
    print(f"Results: {args.results_dir}/")
    print(f"Checkpoints: {args.checkpoints_dir}/")
    print(f"Tables: {args.tables_dir}/")
    print("=" * 80 + "\n", flush=True)


if __name__ == "__main__":
    main()
