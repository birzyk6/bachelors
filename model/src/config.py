"""Central configuration for recommender experiments."""

from __future__ import annotations

from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "processed"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = ARTIFACTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TOP_K = 10
AVENGERS_TITLE = "Avengers: Infinity War"
USE_SAMPLE = True  # flip to False to train on all available rows
CF_MAX_ROWS = 1_000_000
CF_MIN_INTERACTIONS = 40
CF_SAMPLE_USERS = 4_000
CF_MAX_GRID_ROWS = 2_000_000
CF_MAX_GRID_COMBOS = 12
CF_GRID_N_JOBS = 1
TWO_TOWER_MAX_ROWS = 1_000_000
TWO_TOWER_MAX_GRID_COMBOS = 8
TWO_TOWER_EARLY_STOPPING_PATIENCE = 10
TWO_TOWER_NEGATIVE_SAMPLES = 2  # number of negative samples per positive
GRID_EVAL_USERS = 500  # subset for ranking metrics during grid search

device_name = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
)
DEVICE = torch.device(device_name)

# CF_PARAM_GRID = {
#     "n_factors": [100, 150, 200],
#     "lr_all": [0.003, 0.005],
#     "reg_all": [0.01, 0.02, 0.05],
#     "n_epochs": [30, 40],
# }

CF_PARAM_GRID = {
    "n_factors": [150],
    "lr_all": [0.005],
    "reg_all": [0.02],
    "n_epochs": [100],
}

TWO_TOWER_PARAM_GRID = {
    "latent_dim": [256, 384],
    "dropout": [0.1, 0.15],
    "lr": [5e-4, 8e-4],
    "weight_decay": [1e-5, 5e-5],
    "batch_size": [256, 512],
    "epochs": [40],
}

TWO_TOWER_DEFAULT_PARAMS = {
    "latent_dim": 256,
    "dropout": 0.2,
    "lr": 8e-4,
    "weight_decay": 1e-4,
    "batch_size": 512,
    "epochs": 30,
}

PLOT_COLORS = {
    "cf": "#1f77b4",
    "content": "#2ca02c",
    "two_tower": "#ff7f0e",
}

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
LOG_LEVEL = "INFO"
