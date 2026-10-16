# Sequential Recommendation with Learning Paths and Partially Ordered Graphs (seq_emb)

This project implements and evaluates sequential recommendation models for pedagogical resources along student learning paths.

## Supported Models

Each architecture supports two input modalities:
1. **With POG (*Partially Ordered Graph*)**: Initializes item representations using continuous embeddings pre-trained with **RotatE** over the curricular prerequisite and remedial graph.
2. **Without POG (*noembed*)**: Learns item representations end-to-end using an `nn.Embedding` layer.

Sequential architectures:
- **CASER**: Convolutional sequence model (capturing horizontal and vertical patterns).
- **GRU4Rec**: Recurrent neural network with GRU units and linear projections.
- **SASRec**: Autoregressive Transformer with multi-head self-attention and positional embeddings.

## Project Structure

```text
seq_emb/
├── configs/
│   └── default.yaml             # Declarative hyperparameter configuration
├── data/
│   ├── raw/                     # Learning paths, grades, prerequisites, remedials JSONs
│   └── states/                  # know.pth (Pre-trained RotatE)
├── src/seq_emb/
│   ├── core/                    # Reproducible seeds and device management
│   ├── data/                    # Typed loader and tensor generator
│   ├── models/                  # BaseSequentialModel, CaserModel, GRURecModel, SASRecModel
│   ├── recommender/             # Latent k-NN cosine recommendation
│   ├── evaluation/              # Ranking metrics (MAP, NDCG, HR, MRR) and Path metrics (DAG)
│   ├── experiments/             # Stratified, deterministic cross-validation runner
│   └── tuning/                  # Taguchi orthogonal designs, runner, and ANOVA analysis
├── scripts/
│   ├── run_experiments.py       # End-to-end pipeline: Tuning -> Benchmark -> Ablation -> Tables
│   ├── run_model.py             # CLI to train and evaluate any individual model
│   ├── run_benchmark.py         # Runs the full benchmark comparison (6 variants)
│   ├── run_taguchi.py           # Taguchi hyperparameter tuning and Pareto front analysis
│   ├── run_ablation_study.py    # Outcome-aware and time-decay ablation suite
│   └── print_tables.py          # Formats and exports LaTeX tables (Tuning, Pass, Fail)
└── pyproject.toml               # Package configuration and dependencies
```

## End-to-End Experiment Pipeline (`run_experiments.py`)

The master runner [`scripts/run_experiments.py`](scripts/run_experiments.py) orchestrates the complete experimental workflow in sequence:
1. **Taguchi Hyperparameter Tuning**: Explores orthogonal array parameter spaces and detects Pareto-optimal configurations.
2. **Model Training & Evaluation**: Trains models under LOOCV/K-Fold, saving full checkpoints (`checkpoints/`) and user-level metrics (`results/`).
3. **Ablation Studies**: Evaluates outcome-aware filtering and temporal time-decay/horizon interventions on saved checkpoints.
4. **LaTeX Tables Generation**: Compiles publication-ready tables into `tables/` (`tuning.tex`, `pass.tex`, `fail.tex`).

### Basic Usage
```bash
# Run the entire pipeline across all models (LOOCV, 500 epochs)
python scripts/run_experiments.py

# Run pipeline for a specific architecture (e.g. caser, grurec, or sasrec):
python scripts/run_experiments.py --models caser

# Fast run with custom epochs and parallel Taguchi workers:
python scripts/run_experiments.py --epochs 100 --taguchi-epochs 100 --jobs 4
```

### Stage Skipping & Modular Execution
```bash
# Skip Taguchi tuning and run training + ablation + tables:
python scripts/run_experiments.py --skip-taguchi

# Skip tuning and training, running only ablation and tables on existing checkpoints:
python scripts/run_experiments.py --skip-taguchi --skip-training

# Benchmark only models with POG (skipping non-POG baselines):
python scripts/run_experiments.py --pog-only
```

## Quick Start (Individual Scripts)

### 1. Run an Individual Model
```bash
# CASER with POG (LOOCV / K-Fold)
python scripts/run_model.py --model caser --pog

# CASER without POG
python scripts/run_model.py --model caser --no-pog

# SASRec with POG
python scripts/run_model.py --model sasrec --pog

# GRU4Rec with POG
python scripts/run_model.py --model grurec --pog
```

### 2. Run the Full Benchmark (6 Variants)
```bash
python scripts/run_benchmark.py --epochs 500 --seed 42
```
Results will be printed in a structured table and saved to `results/folds_full_path.json`.

### 3. Local Window Mode (Pedagogical Intervention)
To evaluate graph metrics exclusively over the recommendation steps ($N=3$) rather than the full trajectory:
```bash
python scripts/run_benchmark.py --local-window
```

## Hyperparameter Tuning with Taguchi (L16 and L32b)

The [`tuning`](src/seq_emb/tuning) module allows systematic exploration of the hyperparameter space using balanced Taguchi orthogonal arrays, evaluating each parameter combination through deterministic cross-validation.

### 1. Run Tuning (Individual or All Models)
```bash
# Launch tuning for ALL models (CASER + GRU4Rec + SASRec)
python scripts/run_taguchi.py --model all --pog

# Or for a specific model:
python scripts/run_taguchi.py --model caser --pog
python scripts/run_taguchi.py --model grurec --pog
python scripts/run_taguchi.py --model sasrec --pog

# To evaluate ALL variants (with POG and without POG):
python scripts/run_taguchi.py --model all --all-variants

# Run Taguchi iterations in parallel (e.g. 4 workers):
python scripts/run_taguchi.py --model caser --pog --jobs 4
```

### 2. Taguchi Statistical Analysis and Pareto Front
You can directly run the statistical analysis on existing results (individual model or all):
```bash
# Analyze all models from a directory containing CSVs:
python scripts/run_taguchi.py --model all --analyze-only results/tuning

# Or analyze a specific model:
python scripts/run_taguchi.py --model caser --analyze-only results/tuning/caser_pog_fine_results.csv
```
The analysis computes:
- Multi-objective Z-score normalization (inverting metrics where lower is better: `backward_ratio` and `repeat_ratio`).
- ANOVA variance decomposition per factor (% of explained variance).
- Identification of optimal levels per factor.
- Strict detection of non-dominated solutions on the **Pareto Front**.
- Automatic export to:
  - `results/tuning/{model}_analysis_results_with_zscores.csv`
  - `results/tuning/{model}_taguchi_best_levels.csv`
  - `results/tuning/{model}_pareto_front_runs.csv`

## Ablation Studies

The ablation suite ([`scripts/run_ablation_study.py`](scripts/run_ablation_study.py)) systematically evaluates the core components of the proposed recommendation mechanism using trained checkpoints across all models under LOOCV ($K=10$, $N=3$):

### 1. Study 1: Outcome-Aware Peer Intervention Ablation
Isolates the effect of conditioning recommendation prototypes on student academic success:
- **Naive Baseline (Factual)**: Real unassisted trajectories of failing students without pedagogical intervention ($N=3$ factual steps).
- **With Outcome-Aware (Passing Peers)**: Prototypes restricted strictly to successful students (`grade=1`).
- **Without Outcome-Aware (All Peers)**: Unconstrained peer retrieval using both passing and failing students.
- **Negative Control (Failing Peers)**: Counterfactual intervention retrieving only failing students (`grade=0`).

Evaluates ranking fidelity for passing students ($\text{MAP}$, $\text{NDCG}$, $\text{MRR}$) and curricular adherence for failing students ($\text{CR}$, $\text{BR}$, $\text{RR}$, $\text{AE}$, $\text{Novelty}$).

### 2. Study 2: Temporal Projection (Time-Decay) and Simulation Horizon
Examines how upcoming items in prototype trajectories should be aggregated and the depth of the recommendation intervention:
- **Factual Base at $t_{\text{split}}$**: Historical trajectory state before intervention ($N=0$).
- **Proposed Positional Decay**: $w(p) = \frac{1}{\text{pos} + 1}$ with horizon $N=3$.
- **Uniform Weighting (No Decay)**: $w(p) = 1.0$ with horizon $N=3$.
- **Immediate Truncation**: Next-step priority only ($w_0$) with horizon $N=3$.
- **Single-Step Horizon ($N=1$)**: Evaluating immediate transition with decay vs. uniform weighting.

### Running the Ablation Suite
```bash
# Run full ablation suite on all models (CASER, GRU4Rec, SASRec)
python scripts/run_ablation_study.py --model all --checkpoints-dir checkpoints

# Run ablation for an individual model:
python scripts/run_ablation_study.py --model caser
python scripts/run_ablation_study.py --model grurec
python scripts/run_ablation_study.py --model sasrec

# Specify custom steps horizon and export directories:
python scripts/run_ablation_study.py --steps 3 --output-dir results/ablation --tables-dir tables
```

Outputs are automatically exported to:
- `results/ablation/ablation_1_outcome_aware_filtering.csv`
- `results/ablation/ablation_2_time_decay_and_horizon.csv`
- `tables/ablation_outcome_aware.tex`
- `tables/ablation_time_decay.tex`

## LaTeX Table Generation

To generate and export publication-ready LaTeX tables matching the paper specifications:
```bash
# Generate all tables (Tuning, Pass, Fail)
python scripts/print_tables.py --table all

# Generate specific tables:
python scripts/print_tables.py --table tuning
python scripts/print_tables.py --table pass
python scripts/print_tables.py --table fail

# Generate PASS table in wide format:
python scripts/print_tables.py --table pass --pass-format wide
```
Output `.tex` files are saved to `tables/` (`tuning.tex`, `pass.tex`, `fail.tex`).

