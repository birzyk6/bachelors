"""
Generate all thesis plots from evaluation results.

This script loads results and creates all visualizations needed for the thesis.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

from model.src.visualization.plots import plot_ndcg_comparison, plot_recall_at_k, plot_rmse_comparison
from model.src.visualization.predictions_comparison import plot_model_predictions_comparison

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
METRICS_DIR = PROJECT_ROOT / "model" / "metrics"
PLOTS_DIR = PROJECT_ROOT / "model" / "plots"
DATA_DIR = PROJECT_ROOT / "model" / "data" / "processed"
SAVED_MODELS_DIR = PROJECT_ROOT / "model" / "saved_models"


def load_individual_results() -> dict:
    """Load evaluation results from individual model JSON files."""
    results = {
        "rating_prediction": {},
        "ranking": {},
    }

    model_files = {
        "collaborative": "collaborative_results.json",
        "content_based": "content_based_results.json",
        "knn": "knn_results.json",
        "ncf": "ncf_results.json",
        "two_tower": "two_tower_results.json",
    }

    for model_name, filename in model_files.items():
        filepath = METRICS_DIR / filename
        if filepath.exists():
            with open(filepath) as f:
                data = json.load(f)

            test_metrics = data.get("test_metrics", {})
            val_metrics = data.get("val_metrics", {})

            metrics = test_metrics if test_metrics else val_metrics

            if metrics:
                results["rating_prediction"][model_name] = {
                    "rmse": metrics.get("rmse"),
                    "mae": metrics.get("mae"),
                }

            print(f"  ✓ Loaded {model_name} results")
        else:
            print(f"  ⚠ {filename} not found")

    return results


def load_results() -> dict:
    """Load evaluation results from JSON."""
    results_file = METRICS_DIR / "results.json"

    if results_file.exists():
        with open(results_file, "r") as f:
            results = json.load(f)
        print(f"✓ Loaded combined results from {results_file}")
        return results

    print("Loading individual model results...")
    return load_individual_results()


def generate_model_comparison_plots(results: dict):
    """Generate model comparison plots."""
    print("\nGenerating model comparison plots...")

    if not results.get("rating_prediction"):
        print("  ⚠ No rating prediction results found, skipping comparison plots")
        return

    plot_rmse_comparison(results, PLOTS_DIR / "models" / "rmse_comparison.png")

    if results.get("ranking"):
        plot_ndcg_comparison(results, PLOTS_DIR / "ndcg_comparison.png", k=10)
        plot_recall_at_k(results, PLOTS_DIR / "models" / "recall_at_k.png", k_values=[5, 10, 20])


def generate_model_predictions_plot():
    """Generate model predictions comparison for a sample user."""
    print("\nGenerating model predictions comparison...")

    try:
        plot_model_predictions_comparison(
            data_dir=DATA_DIR,
            saved_models_dir=SAVED_MODELS_DIR,
            metrics_dir=METRICS_DIR,
            output_dir=PLOTS_DIR,
            sample_user_id=42,
            num_movies=10,
        )
    except Exception as e:
        print(f"  ✗ Could not generate predictions comparison: {e}")
        import traceback

        traceback.print_exc()


def generate_two_tower_plots():
    """Generate Two-Tower specific visualizations."""
    print("\nGenerating Two-Tower model plots...")

    try:
        import tensorflow as tf

        embeddings_file = SAVED_MODELS_DIR / "movie_embeddings.npy"
        metadata_file = SAVED_MODELS_DIR / "movie_embeddings_metadata.json"

        if not embeddings_file.exists() or not metadata_file.exists():
            print("  ⚠ Two-Tower embeddings not found, skipping Two-Tower plots")
            return

        embeddings_dict = np.load(embeddings_file, allow_pickle=True).item()
        with open(metadata_file) as f:
            metadata = json.load(f)

        movie_ids = metadata["movie_ids"]

        embeddings = np.array([embeddings_dict[mid] for mid in movie_ids if mid in embeddings_dict])
        movie_ids = [mid for mid in movie_ids if mid in embeddings_dict]

        movie_file = DATA_DIR / "movies.parquet"
        if movie_file.exists():
            movies_df = pl.read_parquet(movie_file).to_pandas()
        else:
            print("  ⚠ Movies data not found")
            return

        title_col = "title_ml" if "title_ml" in movies_df.columns else "title"
        id_to_title = dict(zip(movies_df["movieId"], movies_df[title_col]))
        id_to_genres = dict(zip(movies_df["movieId"], movies_df["genres"]))

    except Exception as e:
        print(f"  ✗ Could not generate Two-Tower plots: {e}")
        import traceback

        traceback.print_exc()


def generate_all_plots():
    """
    Main function to generate all thesis plots.

    Generates comprehensive visualizations organized into subdirectories:
    - eda/: Exploratory data analysis plots
    - embeddings/: Embedding space visualizations
    - models/: Model evaluation and comparison plots
    - two_tower/: Two-Tower model specific plots
    - recommender/: Recommendation engine test results
    """
    print("=" * 80)
    print("Generowanie wykresów do pracy magisterskiej")
    print("=" * 80)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for subdir in ["eda", "embeddings", "models", "two_tower", "recommender"]:
        (PLOTS_DIR / subdir).mkdir(exist_ok=True)

    results = load_results()

    if results:
        generate_model_comparison_plots(results)

    generate_model_predictions_plot()

    generate_two_tower_plots()

    try:
        from model.src.visualization.eda_plots import generate_all_eda_plots

        print("\n" + "-" * 40)
        generate_all_eda_plots(data_dir=DATA_DIR, output_dir=PLOTS_DIR)
    except Exception as e:
        print(f"  ⚠ Could not generate EDA plots: {e}")

    try:
        from model.src.visualization.embedding_plots import generate_all_embedding_plots

        print("\n" + "-" * 40)
        generate_all_embedding_plots(
            data_dir=DATA_DIR,
            saved_models_dir=SAVED_MODELS_DIR,
            output_dir=PLOTS_DIR,
        )
    except Exception as e:
        print(f"  ⚠ Could not generate embedding plots: {e}")
        import traceback

        traceback.print_exc()

    try:
        from model.src.visualization.model_plots import generate_all_model_plots

        print("\n" + "-" * 40)
        generate_all_model_plots(metrics_dir=METRICS_DIR, output_dir=PLOTS_DIR)
    except Exception as e:
        print(f"  ⚠ Could not generate model plots: {e}")

    try:
        from model.src.visualization.two_tower_plots import generate_all_two_tower_plots

        print("\n" + "-" * 40)
        generate_all_two_tower_plots(
            data_dir=DATA_DIR,
            saved_models_dir=SAVED_MODELS_DIR,
            output_dir=PLOTS_DIR,
        )
    except Exception as e:
        print(f"  ⚠ Could not generate Two-Tower plots: {e}")

    print("\n" + "=" * 80)
    print(f"✓ Wszystkie wykresy zapisane w {PLOTS_DIR}")
    print("=" * 80)

    print("\nWygenerowane wykresy:")
    for subdir in ["", "eda", "embeddings", "models", "two_tower", "recommender"]:
        dir_path = PLOTS_DIR / subdir if subdir else PLOTS_DIR
        plots = sorted(dir_path.glob("*.png"))
        if plots:
            if subdir:
                print(f"\n  {subdir}/")
            for plot_file in plots:
                prefix = "    " if subdir else "  "
                print(f"{prefix}- {plot_file.name}")


if __name__ == "__main__":
    generate_all_plots()
