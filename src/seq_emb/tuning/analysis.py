"""Taguchi statistical analysis: Z-score normalization, ANOVA effect size, and Pareto front."""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd


METRIC_COLS = [
    "mean_map",
    "mean_ndcg",
    "mean_hr",
    "mean_mrr",
    "mean_coverage_ratio",
    "mean_backward_ratio",
    "mean_repeat_ratio",
    "mean_progress_ratio",
    "mean_frac_allowed_edges",
    "mean_novelty",
]

# Lower is better: invert these so higher is universally better
LOWER_BETTER_METRICS = [
    "mean_backward_ratio",
    "mean_repeat_ratio",
]

MODEL_FACTORS = {
    "caser": ["lr", "optimizer", "num_vert_filters", "num_hori_filters", "hori_filter_sizes"],
    "grurec": ["pooling", "num_layers", "lr", "optimizer", "dropout", "hidden_dim"],
    "sasrec": ["num_heads", "pooling", "lr", "optimizer", "dropout"],
}


class TaguchiAnalyzer:
    """Performs multi-metric Taguchi analysis, ANOVA factor effect size, and Pareto optimization."""

    def __init__(self, model_name: str, results_df: pd.DataFrame):
        self.model_name = model_name.lower().strip()
        self.df = results_df.copy()

        # Adapt grurec num_heads/num_layers column name if necessary
        if self.model_name == "grurec":
            if "num_heads" in self.df.columns and "num_layers" not in self.df.columns:
                self.df = self.df.rename(columns={"num_heads": "num_layers"})

        # Validate metrics
        for col in METRIC_COLS:
            if col not in self.df.columns:
                raise ValueError(f"Missing expected metric column '{col}' in results DataFrame.")
            self.df[col] = self.df[col].astype(float)

    def analyze(self) -> Dict[str, Any]:
        """Compute full Taguchi analysis pipeline."""
        # 1. Z-Score normalization
        z_df = pd.DataFrame(index=self.df.index)
        for col in METRIC_COLS:
            series = self.df[col].copy()
            if col in LOWER_BETTER_METRICS:
                series = -series  # invert so higher is better

            std = series.std(ddof=0)
            if std == 0:
                z_df[f"z_{col}"] = 0.0
            else:
                z_df[f"z_{col}"] = (series - series.mean()) / std

        z_df["score_combined"] = z_df.mean(axis=1)
        full_df = pd.concat([self.df, z_df], axis=1)

        # 2. Taguchi Factor Level Means and Effect Size (ANOVA)
        factors = MODEL_FACTORS.get(self.model_name, [])
        factors = [f for f in factors if f in full_df.columns]

        overall_mean = full_df["score_combined"].mean()
        ss_total = ((full_df["score_combined"] - overall_mean) ** 2).sum()

        level_means = {}
        best_levels_list = []

        for f in factors:
            # Handle list or tuple factor values as strings for groupby
            is_complex = full_df[f].apply(lambda v: isinstance(v, (list, tuple))).any()
            grp_key = full_df[f].astype(str) if is_complex else full_df[f]

            grp = (
                full_df.assign(_grp=grp_key)
                .groupby("_grp")["score_combined"]
                .agg(["mean", "count"])
                .reset_index()
                .rename(columns={"_grp": f, "mean": "mean_score", "count": "n"})
            )

            ss_between = (grp["n"] * ((grp["mean_score"] - overall_mean) ** 2)).sum()
            pct_explained = 100.0 * ss_between / ss_total if ss_total > 0 else 0.0

            grp["ss_between"] = ss_between
            grp["ss_total"] = ss_total
            grp["percent_explained"] = pct_explained
            grp = grp.sort_values(by="mean_score", ascending=False).reset_index(drop=True)

            level_means[f] = grp
            best_row = grp.iloc[0].to_dict()
            best_levels_list.append({
                "factor": f,
                "best_level": best_row[f],
                "mean_score": best_row["mean_score"],
                "n": int(best_row["n"]),
                "percent_explained": best_row["percent_explained"],
            })

        best_levels_df = pd.DataFrame(best_levels_list)

        # 3. Pareto Front (Non-dominated runs across normalized metrics)
        norm_df = pd.DataFrame(index=self.df.index)
        for col in METRIC_COLS:
            arr = self.df[col].copy().values
            if col in LOWER_BETTER_METRICS:
                arr = -arr
            denom = arr.max() - arr.min()
            norm_df[col] = (arr - arr.min()) / denom if denom > 0 else 0.5

        def is_dominated(i: int, matrix: np.ndarray) -> bool:
            for j in range(matrix.shape[0]):
                if j == i:
                    continue
                if np.all(matrix[j] >= matrix[i]) and np.any(matrix[j] > matrix[i]):
                    return True
            return False

        mat = norm_df.values
        pareto_mask = np.array([not is_dominated(i, mat) for i in range(mat.shape[0])])
        pareto_front_df = full_df[pareto_mask].copy().reset_index(drop=True)

        return {
            "full_results": full_df,
            "level_means": level_means,
            "best_levels": best_levels_df,
            "pareto_front": pareto_front_df,
            "pareto_indices": np.where(pareto_mask)[0].tolist(),
        }
