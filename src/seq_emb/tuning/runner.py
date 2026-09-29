"""Taguchi hyperparameter tuning runner across orthogonal design runs with optional parallelization."""

from typing import Dict, Any, Optional, List
import ast
import os
import concurrent.futures
import multiprocessing as mp
import pandas as pd
import torch
from tqdm import tqdm

from ..experiments.kfold_runner import KFoldRunner
from .design import get_taguchi_design


def _build_custom_params(model_name: str, row_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model-specific hyperparameters from a Taguchi row configuration."""
    name = model_name.lower().strip()
    custom_params: Dict[str, Any] = {}

    if "lr" in row_dict and pd.notna(row_dict["lr"]):
        custom_params["lr"] = float(row_dict["lr"])
    if "optimizer" in row_dict and pd.notna(row_dict["optimizer"]):
        custom_params["optimizer"] = str(row_dict["optimizer"]).strip().lower()
    if "dropout" in row_dict and pd.notna(row_dict["dropout"]):
        custom_params["dropout"] = float(row_dict["dropout"])
    if "pooling" in row_dict and pd.notna(row_dict["pooling"]):
        custom_params["pooling"] = str(row_dict["pooling"]).strip().lower()

    # Architecture specific mappings
    if name == "caser":
        if "num_vert_filters" in row_dict and pd.notna(row_dict["num_vert_filters"]):
            custom_params["num_vert_filters"] = int(row_dict["num_vert_filters"])
        if "num_hori_filters" in row_dict and pd.notna(row_dict["num_hori_filters"]):
            custom_params["num_hori_filters"] = int(row_dict["num_hori_filters"])
        if "hori_filter_sizes" in row_dict and pd.notna(row_dict["hori_filter_sizes"]):
            h_val = row_dict["hori_filter_sizes"]
            if isinstance(h_val, str):
                h_val = ast.literal_eval(h_val)
            custom_params["hori_filter_sizes"] = tuple(int(x) for x in h_val)

    elif name == "grurec":
        if "hidden_dim" in row_dict and pd.notna(row_dict["hidden_dim"]):
            custom_params["hidden_dim"] = int(row_dict["hidden_dim"])
        if "num_layers" in row_dict and pd.notna(row_dict["num_layers"]):
            custom_params["num_layers"] = int(row_dict["num_layers"])
        elif "num_heads" in row_dict and pd.notna(row_dict["num_heads"]):
            custom_params["num_layers"] = int(row_dict["num_heads"])

    elif name == "sasrec":
        if "num_heads" in row_dict and pd.notna(row_dict["num_heads"]):
            h = int(row_dict["num_heads"])
            # Ensure embedding_dim (150) is divisible by num_heads
            if 150 % h != 0:
                if h == 4:
                    h = 3
                else:
                    divisors = [d for d in [2, 3, 5, 6, 10, 15] if 150 % d == 0]
                    h = min(divisors, key=lambda d: abs(d - h))
            custom_params["num_heads"] = h
            custom_params["num_layers"] = h  # matching original script convention

    return custom_params


def resolve_n_jobs(n_jobs: Optional[int], device: Optional[str] = None) -> int:
    """Resolve n_jobs to a concrete positive integer count of worker processes."""
    if n_jobs is None or n_jobs == 0:
        return 1
    cpu_cores = os.cpu_count() or 1
    if n_jobs < 0:
        dev_str = str(device).lower() if device else ""
        if "cuda" in dev_str or dev_str == "auto":
            # Cap parallel jobs on GPU by default to prevent VRAM overflow
            return max(1, min(cpu_cores, 4))
        return max(1, cpu_cores + 1 + n_jobs) if n_jobs < -1 else max(1, cpu_cores)
    return max(1, n_jobs)


def _execute_single_taguchi_run(task_args: Dict[str, Any]) -> Dict[str, Any]:
    """Worker function executed in separate process for a single Taguchi configuration."""
    idx = task_args["idx"]
    row_dict = task_args["row_dict"]
    model_name = task_args["model_name"]
    use_pog = task_args["use_pog"]
    strategy = task_args["strategy"]
    num_epochs = task_args["num_epochs"]
    batch_size = task_args["batch_size"]
    n_splits = task_args["n_splits"]
    local_window = task_args["local_window"]
    seed = task_args["seed"]
    device = task_args["device"]
    data_dir = task_args["data_dir"]

    custom_params = _build_custom_params(model_name, row_dict)
    run_seed = seed + idx * 1000

    runner = KFoldRunner(data_dir=data_dir, seed=run_seed, device=device)
    res = runner.run(
        model_name=model_name,
        use_pog=use_pog,
        strategy=strategy,
        n_splits=n_splits,
        num_epochs=num_epochs,
        batch_size=batch_size,
        custom_params=custom_params,
        local_window=local_window,
        verbose=False,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    record = row_dict.copy()
    record.update(res["mean"])
    return {"idx": idx, "record": record}


class TaguchiRunner:
    """Orchestrates Taguchi orthogonal array hyperparameter exploration."""

    def __init__(
        self,
        data_dir: Optional[str] = None,
        seed: int = 42,
        device: Optional[str] = "auto",
        n_jobs: int = 1,
    ):
        self.seed = seed
        self.device = device
        self.data_dir = data_dir
        self.n_jobs = n_jobs
        self.kfold_runner = KFoldRunner(data_dir=data_dir, seed=seed, device=device)

    def run_tuning(
        self,
        model_name: str,
        use_pog: bool = True,
        strategy: str = "loocv",
        design_df: Optional[pd.DataFrame] = None,
        num_epochs: int = 500,
        batch_size: int = 4,
        n_splits: int = 5,
        local_window: bool = False,
        n_jobs: Optional[int] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Execute cross-validation evaluation (LOOCV or K-Fold) for each configuration in the Taguchi orthogonal array.

        Args:
            model_name: "caser", "grurec", or "sasrec"
            use_pog: Whether to use RotatE POG embeddings
            strategy: "loocv" or "kfold"
            design_df: Taguchi design matrix DataFrame (generated if None)
            num_epochs: Number of training epochs per fold
            batch_size: Training batch size
            n_splits: Folds count for KFold
            local_window: Evaluate graph metrics on recommendation window only
            n_jobs: Number of parallel worker processes (if None, uses self.n_jobs)
            verbose: Display progress bar and summary logs

        Returns:
            DataFrame containing hyperparameter levels and averaged metrics per run.
        """
        if design_df is None:
            design_df = get_taguchi_design(model_name)

        name = model_name.lower().strip()
        effective_n_jobs = resolve_n_jobs(self.n_jobs if n_jobs is None else n_jobs, self.device)
        total_runs = len(design_df)

        if verbose:
            mode_desc = "con POG" if use_pog else "sin POG"
            worker_desc = f"{effective_n_jobs} workers en paralelo" if effective_n_jobs > 1 else "secuencial"
            print(f"\nIniciando Taguchi Tuning para {name.upper()} ({mode_desc}) - {total_runs} corridas ({worker_desc}):")

        # Case 1: Sequential execution
        if effective_n_jobs <= 1 or total_runs <= 1:
            results_rows = []
            iterator = design_df.iterrows()
            if verbose:
                iterator = tqdm(iterator, total=total_runs, desc=f"Taguchi {name.upper()}")

            for idx, row in iterator:
                row_dict = row.to_dict()
                custom_params = _build_custom_params(name, row_dict)
                res = self.kfold_runner.run(
                    model_name=name,
                    use_pog=use_pog,
                    strategy=strategy,
                    n_splits=n_splits,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    custom_params=custom_params,
                    local_window=local_window,
                    verbose=False,
                )
                record = row_dict.copy()
                record.update(res["mean"])
                results_rows.append(record)

            return pd.DataFrame(results_rows)

        # Case 2: Parallel execution across workers
        tasks = []
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

        for idx, row in design_df.iterrows():
            worker_device = self.device
            # Distribute across GPUs if multiple exist
            if num_gpus > 1 and (str(self.device).lower() in ["auto", "cuda"] or self.device is None):
                worker_device = f"cuda:{idx % num_gpus}"

            tasks.append({
                "idx": idx,
                "row_dict": row.to_dict(),
                "model_name": name,
                "use_pog": use_pog,
                "strategy": strategy,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "n_splits": n_splits,
                "local_window": local_window,
                "seed": self.seed,
                "device": worker_device,
                "data_dir": self.data_dir,
            })

        mp_ctx = mp.get_context("spawn")
        pbar = tqdm(total=total_runs, desc=f"Taguchi {name.upper()} (parallel x{effective_n_jobs})") if verbose else None
        results_dict: Dict[int, Dict[str, Any]] = {}

        with concurrent.futures.ProcessPoolExecutor(max_workers=effective_n_jobs, mp_context=mp_ctx) as executor:
            future_to_idx = {
                executor.submit(_execute_single_taguchi_run, task): task["idx"]
                for task in tasks
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                res = future.result()
                results_dict[res["idx"]] = res["record"]
                if pbar is not None:
                    pbar.update(1)

        if pbar is not None:
            pbar.close()

        # Reconstruct rows in original order
        results_rows = [results_dict[i] for i in range(total_runs)]
        return pd.DataFrame(results_rows)

