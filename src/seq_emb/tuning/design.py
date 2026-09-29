"""Taguchi orthogonal array designs (L16 and L32b) for sequential recommendation models."""

from pathlib import Path
from typing import Optional, List, Any
import numpy as np
import pandas as pd


# L16 Orthogonal Array (16 runs, up to 5 factors with 4 levels each)
L16_4x5 = np.array([
    [1, 1, 1, 1, 1],
    [1, 2, 2, 2, 2],
    [1, 3, 3, 3, 3],
    [1, 4, 4, 4, 4],
    [2, 1, 2, 3, 4],
    [2, 2, 1, 4, 3],
    [2, 3, 4, 1, 2],
    [2, 4, 3, 2, 1],
    [3, 1, 3, 4, 2],
    [3, 2, 4, 3, 1],
    [3, 3, 1, 2, 4],
    [3, 4, 2, 1, 3],
    [4, 1, 4, 2, 3],
    [4, 2, 3, 1, 4],
    [4, 3, 2, 4, 1],
    [4, 4, 1, 3, 2],
], dtype=int)

# L32b Orthogonal Array (32 runs, 1 2-level factor + up to 9 4-level factors)
L32b = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
    [1, 1, 3, 3, 3, 3, 3, 3, 3, 3],
    [1, 1, 4, 4, 4, 4, 4, 4, 4, 4],
    [1, 2, 1, 1, 2, 2, 3, 3, 4, 4],
    [1, 2, 2, 2, 1, 1, 4, 4, 3, 3],
    [1, 2, 3, 3, 4, 4, 1, 1, 2, 2],
    [1, 2, 4, 4, 3, 3, 2, 2, 1, 1],
    [1, 3, 1, 2, 3, 4, 1, 2, 3, 4],
    [1, 3, 2, 1, 4, 3, 2, 1, 4, 3],
    [1, 3, 3, 4, 1, 2, 3, 4, 1, 2],
    [1, 3, 4, 3, 2, 1, 4, 3, 2, 1],
    [1, 4, 1, 2, 4, 3, 3, 4, 2, 1],
    [1, 4, 2, 1, 3, 4, 4, 3, 1, 2],
    [1, 4, 3, 4, 2, 1, 1, 2, 4, 3],
    [1, 4, 4, 3, 1, 2, 2, 1, 3, 4],
    [2, 1, 1, 4, 1, 4, 2, 3, 2, 3],
    [2, 1, 2, 3, 2, 3, 1, 4, 1, 4],
    [2, 1, 3, 2, 3, 2, 4, 1, 4, 1],
    [2, 1, 4, 1, 4, 1, 3, 2, 3, 2],
    [2, 2, 1, 4, 2, 3, 4, 1, 3, 2],
    [2, 2, 2, 3, 1, 4, 3, 2, 4, 1],
    [2, 2, 3, 2, 4, 1, 2, 3, 1, 4],
    [2, 2, 4, 1, 3, 2, 1, 4, 2, 3],
    [2, 3, 1, 3, 3, 1, 2, 4, 4, 2],
    [2, 3, 2, 4, 4, 2, 1, 3, 3, 1],
    [2, 3, 3, 1, 1, 3, 4, 2, 2, 4],
    [2, 3, 4, 2, 2, 4, 3, 1, 1, 3],
    [2, 4, 1, 3, 4, 2, 4, 2, 1, 3],
    [2, 4, 2, 4, 3, 1, 3, 1, 2, 4],
    [2, 4, 3, 1, 2, 4, 2, 4, 3, 1],
    [2, 4, 4, 2, 1, 3, 1, 3, 4, 2],
], dtype=int)


def _map_levels(col: np.ndarray, values: List[Any]) -> List[Any]:
    return [values[int(v) - 1] for v in col]


def build_caser_design() -> pd.DataFrame:
    """Build Taguchi L16 design for CASER."""
    lr_list = [1e-6, 1e-5, 1e-4, 1e-3]
    optimizer_list = ["sgd", "adam", "adamw", "adagrad"]
    num_vert_filters = [2, 4, 8, 16]
    num_hori_filters = [2, 4, 8, 16]
    hori_filter_sizes = [(2, 3, 4), (3, 4, 5), (4, 5, 6), (5, 6, 7)]

    df = pd.DataFrame({
        "run_id": np.arange(1, L16_4x5.shape[0] + 1),
        "lr": _map_levels(L16_4x5[:, 0], lr_list),
        "optimizer": _map_levels(L16_4x5[:, 1], optimizer_list),
        "num_vert_filters": _map_levels(L16_4x5[:, 2], num_vert_filters),
        "num_hori_filters": _map_levels(L16_4x5[:, 3], num_hori_filters),
        "hori_filter_sizes": _map_levels(L16_4x5[:, 4], hori_filter_sizes),
    })
    return df


def build_grurec_design() -> pd.DataFrame:
    """Build Taguchi L32b design for GRU4Rec."""
    pooling_list = ["mean", "last"]
    num_layers_list = [2, 4, 8, 10]
    lr_list = [1e-6, 1e-5, 1e-4, 1e-3]
    optimizer_list = ["sgd", "adam", "adamw", "adagrad"]
    dropout_list = [0.1, 0.15, 0.2, 0.25]
    hidden_dim_list = [128, 64, 32, 16]

    df = pd.DataFrame({
        "run_id": np.arange(1, L32b.shape[0] + 1),
        "pooling": _map_levels(L32b[:, 0], pooling_list),
        "num_layers": _map_levels(L32b[:, 1], num_layers_list),
        "lr": _map_levels(L32b[:, 2], lr_list),
        "optimizer": _map_levels(L32b[:, 3], optimizer_list),
        "dropout": _map_levels(L32b[:, 4], dropout_list),
        "hidden_dim": _map_levels(L32b[:, 5], hidden_dim_list),
    })
    return df


def build_sasrec_design(seed: int = 0) -> pd.DataFrame:
    """Build balanced L32 design for SASRec.
    Note: num_heads must divide embedding_dim (150). Divisors used: 2, 3, 5, 10.
    """
    num_heads_list = [2, 3, 5, 10]
    pooling_list = ["mean", "last"]
    lr_list = [1e-6, 1e-5, 1e-4, 1e-3]
    optimizer_list = ["sgd", "adam", "adamw", "adagrad"]
    dropout_list = [0.1, 0.15, 0.2, 0.25]

    n_runs = 32
    rng = np.random.RandomState(seed)

    def balanced_col(levels: List[Any]) -> List[Any]:
        n_levels = len(levels)
        repeats = n_runs // n_levels
        rem = n_runs % n_levels
        col = []
        for i, lvl in enumerate(levels):
            col += [lvl] * (repeats + (1 if i < rem else 0))
        arr = np.array(col)
        rng.shuffle(arr)
        return list(arr)

    df = pd.DataFrame({
        "run_id": np.arange(1, n_runs + 1),
        "num_heads": [int(x) for x in balanced_col(num_heads_list)],
        "pooling": balanced_col(pooling_list),
        "lr": [float(x) for x in balanced_col(lr_list)],
        "optimizer": balanced_col(optimizer_list),
        "dropout": [float(x) for x in balanced_col(dropout_list)],
    })
    return df


def get_taguchi_design(model_name: str, csv_path: Optional[str] = None) -> pd.DataFrame:
    """Get Taguchi experimental design DataFrame for given model."""
    name = model_name.lower().strip()

    if csv_path and Path(csv_path).exists():
        df = pd.read_csv(csv_path)
        # Parse tuple representations if any
        if "hori_filter_sizes" in df.columns and isinstance(df["hori_filter_sizes"].iloc[0], str):
            import ast
            df["hori_filter_sizes"] = df["hori_filter_sizes"].apply(ast.literal_eval)
        return df

    # Check default data/tuning path
    default_csv = Path(__file__).resolve().parents[3] / "data" / "tuning" / f"{name}.csv"
    if default_csv.exists():
        df = pd.read_csv(default_csv)
        if "hori_filter_sizes" in df.columns and isinstance(df["hori_filter_sizes"].iloc[0], str):
            import ast
            df["hori_filter_sizes"] = df["hori_filter_sizes"].apply(ast.literal_eval)
        return df

    if name == "caser":
        return build_caser_design()
    elif name == "grurec":
        return build_grurec_design()
    elif name == "sasrec":
        return build_sasrec_design()
    else:
        raise ValueError(f"Unknown model name '{model_name}'. Choose from: 'caser', 'grurec', 'sasrec'.")
