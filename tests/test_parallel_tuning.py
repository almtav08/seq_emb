"""Unit tests to verify parameter resolution and parallel execution configuration for Taguchi tuning."""

import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from seq_emb.tuning.design import get_taguchi_design
from seq_emb.tuning.runner import TaguchiRunner, resolve_n_jobs, _build_custom_params


def test_resolve_n_jobs():
    assert resolve_n_jobs(1) == 1
    assert resolve_n_jobs(None) == 1
    assert resolve_n_jobs(0) == 1
    assert resolve_n_jobs(4) == 4
    assert resolve_n_jobs(-1, device="cpu") >= 1
    assert resolve_n_jobs(-1, device="cuda") <= 4
    print("[OK] test_resolve_n_jobs passed.")


def test_build_custom_params():
    # Caser
    caser_row = {
        "lr": 0.001,
        "optimizer": "adam",
        "dropout": 0.2,
        "pooling": "mean",
        "num_vert_filters": 8,
        "num_hori_filters": 16,
        "hori_filter_sizes": "(2, 3, 4)",
    }
    params = _build_custom_params("caser", caser_row)
    assert params["lr"] == 0.001
    assert params["optimizer"] == "adam"
    assert params["num_vert_filters"] == 8
    assert params["hori_filter_sizes"] == (2, 3, 4)

    # GRURec
    gru_row = {
        "lr": 0.005,
        "optimizer": "sgd",
        "dropout": 0.1,
        "pooling": "last",
        "hidden_dim": 128,
        "num_layers": 2,
    }
    params_gru = _build_custom_params("grurec", gru_row)
    assert params_gru["hidden_dim"] == 128
    assert params_gru["num_layers"] == 2

    # SASRec (checking head divisibility into 150)
    sas_row = {
        "lr": 0.001,
        "optimizer": "adam",
        "dropout": 0.3,
        "pooling": "mean",
        "num_heads": 4,  # 150 % 4 != 0, so should be mapped to 3
    }
    params_sas = _build_custom_params("sasrec", sas_row)
    assert 150 % params_sas["num_heads"] == 0
    assert params_sas["num_heads"] == 3
    assert params_sas["num_layers"] == 3
    print("[OK] test_build_custom_params passed.")


def test_runner_initialization():
    runner = TaguchiRunner(seed=42, device="cpu", n_jobs=4)
    assert runner.n_jobs == 4
    assert runner.seed == 42
    print("[OK] test_runner_initialization passed.")


if __name__ == "__main__":
    test_resolve_n_jobs()
    test_build_custom_params()
    test_runner_initialization()
    print("\nAll unit tests for parallel Taguchi tuning passed successfully!\n")
