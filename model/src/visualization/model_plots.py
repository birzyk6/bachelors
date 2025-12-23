"""
Model evaluation plots for the movie recommender system.

Generates visualizations for comparing model performance including:
- RMSE/MAE comparison
- NDCG@K and Recall@K curves
- Precision@K comparison
- Learning curves
- Error analysis by genre

All labels are in Polish for academic thesis presentation.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

from .plots import COLORS, MODEL_COLORS, POLISH_LABELS, get_model_name_pl, set_thesis_style

# Extended Polish labels for model evaluation
MODEL_LABELS = {
    **POLISH_LABELS,
    # Titles
    "rmse_comparison": "Porównanie błędu RMSE między modelami",
    "mae_comparison": "Porównanie błędu MAE między modelami",
    "ndcg_comparison": "Porównanie jakości rankingu (NDCG@K)",
    "recall_curves": "Krzywe Recall@K dla różnych modeli",
    "precision_curves": "Krzywe Precision@K dla różnych modeli",
    "learning_curves": "Krzywe uczenia - {model}",
    "metrics_radar": "Porównanie wielowymiarowe modeli",
    "error_by_genre": "Analiza błędów predykcji według gatunku",
    "rating_vs_predicted": "Oceny rzeczywiste vs przewidywane",
    "error_distribution": "Rozkład błędów predykcji",
    # Axis labels
    "k_value": "Liczba rekomendacji (K)",
    "metric_value": "Wartość metryki",
    "prediction_error": "Błąd predykcji",
    "actual_rating": "Ocena rzeczywista",
    "predicted_rating": "Ocena przewidywana",
    # Legend
    "lower_better": "(im niżej, tym lepiej)",
    "higher_better": "(im wyżej, tym lepiej)",
}


def load_all_results(metrics_dir: Path) -> Dict:
    """Load results from all model result files."""
    results = {
        "rating_prediction": {},
        "ranking": {},
        "training_history": {},
    }

    model_files = {
        "collaborative": "collaborative_results.json",
        "content_based": "content_based_results.json",
        "knn": "knn_results.json",
        "ncf": "ncf_results.json",
        "two_tower": "two_tower_results.json",
    }

    for model_name, filename in model_files.items():
        filepath = metrics_dir / filename
        if filepath.exists():
            with open(filepath) as f:
                data = json.load(f)

            # Extract metrics
            test_metrics = data.get("test_metrics", {})
            val_metrics = data.get("val_metrics", {})
            metrics = test_metrics if test_metrics else val_metrics

            if metrics:
                results["rating_prediction"][model_name] = {
                    "rmse": metrics.get("rmse"),
                    "mae": metrics.get("mae"),
                }

                # Extract ranking metrics if available
                ranking_metrics = {}
                for key in [
                    "ndcg@5",
                    "ndcg@10",
                    "ndcg@20",
                    "recall@5",
                    "recall@10",
                    "recall@20",
                    "precision@5",
                    "precision@10",
                    "precision@20",
                    "mrr",
                    "hit_rate@10",
                ]:
                    if key in metrics:
                        ranking_metrics[key] = metrics[key]

                if ranking_metrics:
                    results["ranking"][model_name] = ranking_metrics

            # Extract training history if available
            if "training_history" in data:
                results["training_history"][model_name] = data["training_history"]
            elif "history" in data:
                results["training_history"][model_name] = data["history"]

    return results


def plot_rmse_mae_comparison(
    results: Dict,
    output_dir: Path,
):
    """
    Plot RMSE and MAE comparison bar charts side by side.
    """
    set_thesis_style()

    print("  Tworzenie wykresu porównania RMSE/MAE...")

    if not results.get("rating_prediction"):
        print("    ⚠ Brak wyników predykcji ocen")
        return

    # Extract metrics
    models = []
    rmse_values = []
    mae_values = []

    for model_name, metrics in results["rating_prediction"].items():
        if metrics.get("rmse") is not None:
            models.append(get_model_name_pl(model_name))
            rmse_values.append(metrics["rmse"])
            mae_values.append(metrics.get("mae", 0))

    if not models:
        print("    ⚠ Brak metryk RMSE do wykreślenia")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(models))
    width = 0.6

    # RMSE plot
    ax1 = axes[0]
    colors = [
        MODEL_COLORS.get(m.lower().replace(" ", "_").replace("-", "_"), COLORS[0]) for m in results["rating_prediction"].keys()
    ]
    bars1 = ax1.bar(x, rmse_values, width, color=colors, edgecolor="black", alpha=0.8)

    for bar, val in zip(bars1, rmse_values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax1.set_xlabel(MODEL_LABELS["model"], fontweight="bold")
    ax1.set_ylabel("RMSE", fontweight="bold")
    ax1.set_title(MODEL_LABELS["rmse_comparison"] + f"\n{MODEL_LABELS['lower_better']}", fontweight="bold", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha="right")
    ax1.set_ylim(0, max(rmse_values) * 1.15)
    ax1.grid(axis="y", alpha=0.3)

    # Highlight best model
    best_idx = np.argmin(rmse_values)
    bars1[best_idx].set_edgecolor("gold")
    bars1[best_idx].set_linewidth(3)

    # MAE plot
    ax2 = axes[1]
    bars2 = ax2.bar(x, mae_values, width, color=colors, edgecolor="black", alpha=0.8)

    for bar, val in zip(bars2, mae_values):
        if val > 0:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    ax2.set_xlabel(MODEL_LABELS["model"], fontweight="bold")
    ax2.set_ylabel("MAE", fontweight="bold")
    ax2.set_title(MODEL_LABELS["mae_comparison"] + f"\n{MODEL_LABELS['lower_better']}", fontweight="bold", pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha="right")
    ax2.set_ylim(0, max(mae_values) * 1.15 if max(mae_values) > 0 else 1)
    ax2.grid(axis="y", alpha=0.3)

    # Highlight best model
    if max(mae_values) > 0:
        best_idx = np.argmin([v if v > 0 else float("inf") for v in mae_values])
        bars2[best_idx].set_edgecolor("gold")
        bars2[best_idx].set_linewidth(3)

    plt.tight_layout()
    plt.savefig(output_dir / "porownanie_rmse_mae.png")
    plt.close()

    print(f"  ✓ Zapisano porównanie RMSE/MAE")


def plot_ndcg_comparison(
    results: Dict,
    output_path: Path,
    k_values: List[int] = [5, 10, 20],
):
    """
    Plot NDCG@K comparison for different K values.
    """
    set_thesis_style()

    print("  Tworzenie wykresu porównania NDCG...")

    if not results.get("ranking"):
        print("    ⚠ Brak wyników rankingu")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(k_values))
    width = 0.15

    for i, (model_name, metrics) in enumerate(results["ranking"].items()):
        ndcg_values = [metrics.get(f"ndcg@{k}", 0) for k in k_values]

        offset = width * (i - len(results["ranking"]) / 2 + 0.5)
        color = MODEL_COLORS.get(model_name, COLORS[i % len(COLORS)])

        bars = ax.bar(
            x + offset, ndcg_values, width, label=get_model_name_pl(model_name), color=color, edgecolor="black", alpha=0.8
        )

        # Add value labels
        for bar, val in zip(bars, ndcg_values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                )

    ax.set_xlabel(MODEL_LABELS["k_value"], fontweight="bold")
    ax.set_ylabel("NDCG@K", fontweight="bold")
    ax.set_title(MODEL_LABELS["ndcg_comparison"] + f"\n{MODEL_LABELS['higher_better']}", fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in k_values])
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", frameon=True, shadow=True)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"  ✓ Zapisano porównanie NDCG do {output_path}")


def plot_recall_precision_curves(
    results: Dict,
    output_dir: Path,
    k_values: List[int] = [5, 10, 20],
):
    """
    Plot Recall@K and Precision@K curves.
    """
    set_thesis_style()

    print("  Tworzenie krzywych Recall/Precision...")

    if not results.get("ranking"):
        print("    ⚠ Brak wyników rankingu")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Recall@K curves
    ax1 = axes[0]
    for model_name, metrics in results["ranking"].items():
        recalls = [metrics.get(f"recall@{k}", 0) for k in k_values]
        color = MODEL_COLORS.get(model_name, COLORS[0])
        ax1.plot(k_values, recalls, "o-", label=get_model_name_pl(model_name), color=color, linewidth=2, markersize=8)

    ax1.set_xlabel(MODEL_LABELS["k_value"], fontweight="bold")
    ax1.set_ylabel("Recall@K", fontweight="bold")
    ax1.set_title(MODEL_LABELS["recall_curves"], fontweight="bold")
    ax1.set_xticks(k_values)
    ax1.set_ylim(0, 1.0)
    ax1.legend(loc="best", frameon=True, shadow=True)
    ax1.grid(alpha=0.3)

    # Precision@K curves
    ax2 = axes[1]
    for model_name, metrics in results["ranking"].items():
        precisions = [metrics.get(f"precision@{k}", 0) for k in k_values]
        color = MODEL_COLORS.get(model_name, COLORS[0])
        ax2.plot(k_values, precisions, "s-", label=get_model_name_pl(model_name), color=color, linewidth=2, markersize=8)

    ax2.set_xlabel(MODEL_LABELS["k_value"], fontweight="bold")
    ax2.set_ylabel("Precision@K", fontweight="bold")
    ax2.set_title(MODEL_LABELS["precision_curves"], fontweight="bold")
    ax2.set_xticks(k_values)
    ax2.set_ylim(0, 1.0)
    ax2.legend(loc="best", frameon=True, shadow=True)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "krzywe_recall_precision.png")
    plt.close()

    print(f"  ✓ Zapisano krzywe Recall/Precision")


def plot_learning_curves_all(
    results: Dict,
    output_dir: Path,
):
    """
    Plot learning curves for all models with training history.
    """
    set_thesis_style()

    print("  Tworzenie krzywych uczenia...")

    if not results.get("training_history"):
        print("    ⚠ Brak historii trenowania")
        return

    models_with_history = [
        (name, hist)
        for name, hist in results["training_history"].items()
        if hist and isinstance(hist, dict) and "loss" in hist
    ]

    if not models_with_history:
        print("    ⚠ Żaden model nie ma historii uczenia")
        return

    n_models = len(models_with_history)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, history) in zip(axes, models_with_history):
        epochs = range(1, len(history["loss"]) + 1)

        # Training loss
        ax.plot(epochs, history["loss"], "o-", label="Strata treningowa", color=COLORS[0], linewidth=2, markersize=4)

        # Validation loss
        if "val_loss" in history:
            ax.plot(epochs, history["val_loss"], "s-", label="Strata walidacyjna", color=COLORS[1], linewidth=2, markersize=4)

        ax.set_xlabel(MODEL_LABELS["epoch"], fontweight="bold")
        ax.set_ylabel(MODEL_LABELS["loss"], fontweight="bold")
        ax.set_title(MODEL_LABELS["learning_curves"].format(model=get_model_name_pl(model_name)), fontweight="bold")
        ax.legend(loc="best", frameon=True)
        ax.grid(alpha=0.3)

        # Mark minimum validation loss
        if "val_loss" in history:
            min_idx = np.argmin(history["val_loss"])
            ax.axvline(min_idx + 1, color="red", linestyle="--", alpha=0.5)
            ax.scatter([min_idx + 1], [history["val_loss"][min_idx]], color="red", s=100, zorder=5, marker="*")

    plt.tight_layout()
    plt.savefig(output_dir / "krzywe_uczenia.png")
    plt.close()

    print(f"  ✓ Zapisano krzywe uczenia")


def plot_metrics_radar(
    results: Dict,
    output_path: Path,
):
    """
    Plot radar chart comparing multiple metrics across models.
    """
    set_thesis_style()

    print("  Tworzenie wykresu radarowego...")

    if not results.get("ranking"):
        print("    ⚠ Brak wyników rankingu")
        return

    # Metrics to include (normalize to 0-1 scale)
    metrics_to_plot = ["ndcg@10", "recall@10", "precision@10", "mrr", "hit_rate@10"]
    metric_labels = ["NDCG@10", "Recall@10", "Precision@10", "MRR", "Hit Rate@10"]

    # Filter to available metrics
    available_metrics = []
    available_labels = []
    for m, l in zip(metrics_to_plot, metric_labels):
        if any(m in metrics for metrics in results["ranking"].values()):
            available_metrics.append(m)
            available_labels.append(l)

    if len(available_metrics) < 3:
        print("    ⚠ Za mało metryk do wykresu radarowego")
        return

    # Number of variables
    num_vars = len(available_metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for model_name, metrics in results["ranking"].items():
        values = [metrics.get(m, 0) for m in available_metrics]
        values += values[:1]  # Complete the loop

        color = MODEL_COLORS.get(model_name, COLORS[0])
        ax.plot(angles, values, "o-", linewidth=2, label=get_model_name_pl(model_name), color=color)
        ax.fill(angles, values, alpha=0.25, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(available_labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title(MODEL_LABELS["metrics_radar"], fontweight="bold", pad=20, fontsize=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), frameon=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Zapisano wykres radarowy do {output_path}")


def plot_model_comparison_summary(
    results: Dict,
    output_path: Path,
):
    """
    Create comprehensive summary table/heatmap of all metrics.
    """
    set_thesis_style()

    print("  Tworzenie podsumowania modeli...")

    # Collect all metrics
    all_metrics = {}

    # Rating prediction metrics
    for model_name, metrics in results.get("rating_prediction", {}).items():
        if model_name not in all_metrics:
            all_metrics[model_name] = {}
        if metrics.get("rmse"):
            all_metrics[model_name]["RMSE"] = metrics["rmse"]
        if metrics.get("mae"):
            all_metrics[model_name]["MAE"] = metrics["mae"]

    # Ranking metrics
    for model_name, metrics in results.get("ranking", {}).items():
        if model_name not in all_metrics:
            all_metrics[model_name] = {}
        for key in ["ndcg@10", "recall@10", "precision@10", "mrr"]:
            if key in metrics:
                all_metrics[model_name][key.upper().replace("@", "@")] = metrics[key]

    if not all_metrics:
        print("    ⚠ Brak metryk do podsumowania")
        return

    # Create DataFrame
    df = pd.DataFrame(all_metrics).T
    df.index = [get_model_name_pl(m) for m in df.index]

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 6))

    # Normalize for visualization (reverse for RMSE/MAE where lower is better)
    df_norm = df.copy()
    for col in df_norm.columns:
        if col in ["RMSE", "MAE"]:
            # Invert so higher is better (for consistent color scheme)
            df_norm[col] = 1 - (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min() + 1e-8)
        else:
            df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min() + 1e-8)

    # Create annotations with original values
    annot = df.round(4).astype(str)

    sns.heatmap(
        df_norm,
        annot=annot,
        fmt="",
        cmap="RdYlGn",
        ax=ax,
        cbar_kws={"label": "Znormalizowana wydajność"},
        linewidths=0.5,
    )

    ax.set_title("Porównanie wszystkich modeli - podsumowanie metryk", fontweight="bold", pad=20)
    ax.set_xlabel("Metryka", fontweight="bold")
    ax.set_ylabel("Model", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"  ✓ Zapisano podsumowanie modeli do {output_path}")


def generate_all_model_plots(
    metrics_dir: Path,
    output_dir: Path,
):
    """
    Generate all model evaluation plots.

    Args:
        metrics_dir: Path to metrics directory
        output_dir: Path to save plots
    """
    print("\n" + "=" * 80)
    print("Generowanie wykresów ewaluacji modeli")
    print("=" * 80)

    # Create output directory
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load all results
    print("\nŁadowanie wyników...")
    results = load_all_results(metrics_dir)

    if not results["rating_prediction"] and not results["ranking"]:
        print("⚠ Nie znaleziono wyników modeli")
        return

    print(f"  Załadowano wyniki dla {len(results['rating_prediction'])} modeli")

    # Generate plots
    print("\nGenerowanie wykresów...")

    # 1. RMSE/MAE comparison
    plot_rmse_mae_comparison(results, model_dir)

    # 2. NDCG comparison
    if results.get("ranking"):
        plot_ndcg_comparison(results, model_dir / "porownanie_ndcg.png")

    # 3. Recall/Precision curves
    if results.get("ranking"):
        plot_recall_precision_curves(results, model_dir)

    # 4. Learning curves
    if results.get("training_history"):
        plot_learning_curves_all(results, model_dir)

    # 5. Metrics radar chart
    if results.get("ranking"):
        plot_metrics_radar(results, model_dir / "wykres_radarowy.png")

    # 6. Summary heatmap
    plot_model_comparison_summary(results, model_dir / "podsumowanie_modeli.png")

    print("\n" + "=" * 80)
    print(f"✓ Wykresy modeli zapisane w {model_dir}")
    print("=" * 80)


if __name__ == "__main__":
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
    METRICS_DIR = PROJECT_ROOT / "model" / "metrics"
    PLOTS_DIR = PROJECT_ROOT / "model" / "plots"

    generate_all_model_plots(METRICS_DIR, PLOTS_DIR)
