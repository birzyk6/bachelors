"""
Two-Tower model specific visualizations.

Generates visualizations specific to the Two-Tower retrieval model including:
- Model architecture diagram
- User vs movie embedding space comparison
- Similarity score distributions
- Sample recommendations visualization
- Cold-start analysis for new movies

All labels are in Polish for academic thesis presentation.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .plots import COLORS, MODEL_COLORS, POLISH_LABELS, set_thesis_style

# Polish labels for Two-Tower plots
TWO_TOWER_LABELS = {
    **POLISH_LABELS,
    # Titles
    "architecture": "Architektura modelu Two-Tower",
    "user_movie_space": "Przestrzeń osadzeń: użytkownicy vs filmy",
    "similarity_distribution": "Rozkład wyników podobieństwa użytkownik-film",
    "sample_recommendations": "Przykładowe rekomendacje dla wybranych użytkowników",
    "cold_start_analysis": "Analiza problemu zimnego startu - nowe filmy TMDB",
    "embedding_alignment": "Wyrównanie przestrzeni osadzeń",
    "top_similar_movies": "Najbardziej podobne filmy według modelu",
    # Axis labels
    "similarity_score": "Wynik podobieństwa (dot product)",
    "user_id": "ID użytkownika",
    "movie_title": "Tytuł filmu",
    "embedding_dim": "Wymiar osadzenia",
    # Legend and annotations
    "user_tower": "Wieża użytkownika",
    "movie_tower": "Wieża filmu",
    "dot_product": "Iloczyn skalarny",
    "l2_normalize": "Normalizacja L2",
    "input_layer": "Warstwa wejściowa",
    "embedding_layer": "Warstwa osadzeń",
    "dense_layers": "Warstwy gęste",
    "output_embedding": "Osadzenie wyjściowe",
    # Cold start
    "ml_movies": "Filmy MovieLens",
    "tmdb_only": "Tylko TMDB (cold-start)",
    "new_movies": "Nowe filmy",
}


def plot_two_tower_architecture(output_path: Path):
    """
    Create a diagram of the Two-Tower model architecture.
    """
    set_thesis_style()

    print("  Tworzenie diagramu architektury Two-Tower...")

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Colors
    user_color = "#3498db"  # Blue
    movie_color = "#e74c3c"  # Red
    shared_color = "#2ecc71"  # Green
    arrow_color = "#7f8c8d"

    # User Tower (left side)
    # Input
    user_input = FancyBboxPatch((1, 8), 3, 1, boxstyle="round,pad=0.1", facecolor=user_color, edgecolor="black", alpha=0.7)
    ax.add_patch(user_input)
    ax.text(2.5, 8.5, "User ID\n(wejście)", ha="center", va="center", fontsize=10, fontweight="bold")

    # Embedding layer
    user_emb = FancyBboxPatch((1, 6), 3, 1.2, boxstyle="round,pad=0.1", facecolor=user_color, edgecolor="black", alpha=0.5)
    ax.add_patch(user_emb)
    ax.text(2.5, 6.6, "Osadzenie\nużytkownika\n(128D)", ha="center", va="center", fontsize=9)

    # Dense layers
    user_dense = FancyBboxPatch((1, 4), 3, 1.5, boxstyle="round,pad=0.1", facecolor=user_color, edgecolor="black", alpha=0.3)
    ax.add_patch(user_dense)
    ax.text(2.5, 4.75, "Warstwy gęste\n256 → 128", ha="center", va="center", fontsize=9)

    # Output embedding
    user_out = FancyBboxPatch((1, 2), 3, 1.2, boxstyle="round,pad=0.1", facecolor=user_color, edgecolor="black", alpha=0.7)
    ax.add_patch(user_out)
    ax.text(2.5, 2.6, "Osadzenie wyjściowe\n(128D, L2 norm)", ha="center", va="center", fontsize=9, fontweight="bold")

    # Movie Tower (right side)
    # Inputs
    movie_id_input = FancyBboxPatch(
        (11, 8.5), 2.5, 0.8, boxstyle="round,pad=0.1", facecolor=movie_color, edgecolor="black", alpha=0.7
    )
    ax.add_patch(movie_id_input)
    ax.text(12.25, 8.9, "Movie ID", ha="center", va="center", fontsize=9, fontweight="bold")

    movie_text_input = FancyBboxPatch(
        (11, 7.5), 2.5, 0.8, boxstyle="round,pad=0.1", facecolor=movie_color, edgecolor="black", alpha=0.7
    )
    ax.add_patch(movie_text_input)
    ax.text(12.25, 7.9, "BERT (tekst)", ha="center", va="center", fontsize=9, fontweight="bold")

    # Embedding layers
    movie_emb = FancyBboxPatch(
        (11, 5.5), 2.5, 1.5, boxstyle="round,pad=0.1", facecolor=movie_color, edgecolor="black", alpha=0.5
    )
    ax.add_patch(movie_emb)
    ax.text(12.25, 6.25, "Osadzenia\nID + BERT\n(128 + 512)", ha="center", va="center", fontsize=9)

    # Dense layers
    movie_dense = FancyBboxPatch(
        (11, 3.5), 2.5, 1.5, boxstyle="round,pad=0.1", facecolor=movie_color, edgecolor="black", alpha=0.3
    )
    ax.add_patch(movie_dense)
    ax.text(12.25, 4.25, "Warstwy gęste\n640 → 256 → 128", ha="center", va="center", fontsize=9)

    # Output embedding
    movie_out = FancyBboxPatch(
        (11, 1.5), 2.5, 1.2, boxstyle="round,pad=0.1", facecolor=movie_color, edgecolor="black", alpha=0.7
    )
    ax.add_patch(movie_out)
    ax.text(12.25, 2.1, "Osadzenie wyjściowe\n(128D, L2 norm)", ha="center", va="center", fontsize=9, fontweight="bold")

    # Dot product in the middle
    dot_product = plt.Circle((7.5, 2), 1, facecolor=shared_color, edgecolor="black", alpha=0.7)
    ax.add_patch(dot_product)
    ax.text(7.5, 2, "⊙\nDot\nProduct", ha="center", va="center", fontsize=10, fontweight="bold")

    # Arrows
    # User tower arrows
    ax.annotate("", xy=(2.5, 7.2), xytext=(2.5, 8), arrowprops=dict(arrowstyle="->", color=arrow_color, lw=2))
    ax.annotate("", xy=(2.5, 5.5), xytext=(2.5, 6), arrowprops=dict(arrowstyle="->", color=arrow_color, lw=2))
    ax.annotate("", xy=(2.5, 3.2), xytext=(2.5, 4), arrowprops=dict(arrowstyle="->", color=arrow_color, lw=2))

    # Movie tower arrows
    ax.annotate("", xy=(12.25, 7), xytext=(12.25, 7.5), arrowprops=dict(arrowstyle="->", color=arrow_color, lw=2))
    ax.annotate("", xy=(12.25, 5), xytext=(12.25, 5.5), arrowprops=dict(arrowstyle="->", color=arrow_color, lw=2))
    ax.annotate("", xy=(12.25, 2.7), xytext=(12.25, 3.5), arrowprops=dict(arrowstyle="->", color=arrow_color, lw=2))

    # Arrows to dot product
    ax.annotate("", xy=(6.5, 2), xytext=(4, 2.6), arrowprops=dict(arrowstyle="->", color=user_color, lw=3))
    ax.annotate("", xy=(8.5, 2), xytext=(11, 2.1), arrowprops=dict(arrowstyle="->", color=movie_color, lw=3))

    # Output from dot product
    ax.annotate("", xy=(7.5, 0.5), xytext=(7.5, 1), arrowprops=dict(arrowstyle="->", color=shared_color, lw=3))

    # Output box
    output_box = FancyBboxPatch(
        (6, -0.5), 3, 1, boxstyle="round,pad=0.1", facecolor=shared_color, edgecolor="black", alpha=0.7
    )
    ax.add_patch(output_box)
    ax.text(7.5, 0, "Wynik podobieństwa\n[-1, 1]", ha="center", va="center", fontsize=10, fontweight="bold")

    # Tower labels
    ax.text(2.5, 9.5, "WIEŻA UŻYTKOWNIKA", ha="center", va="center", fontsize=14, fontweight="bold", color=user_color)
    ax.text(12.25, 9.5, "WIEŻA FILMU", ha="center", va="center", fontsize=14, fontweight="bold", color=movie_color)

    # Title
    ax.text(7.5, 10.5, TWO_TOWER_LABELS["architecture"], ha="center", va="center", fontsize=16, fontweight="bold")

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=user_color, alpha=0.7, label="Wieża użytkownika"),
        mpatches.Patch(facecolor=movie_color, alpha=0.7, label="Wieża filmu"),
        mpatches.Patch(facecolor=shared_color, alpha=0.7, label="Warstwa wyjściowa"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Zapisano diagram architektury do {output_path}")


def plot_similarity_distribution(
    embeddings_dict: Dict[int, np.ndarray],
    user_profiles: np.ndarray,
    user_ids: List[int],
    movie_ids: List[int],
    output_path: Path,
    n_samples: int = 10000,
):
    """
    Plot distribution of user-movie similarity scores.
    """
    set_thesis_style()

    print("  Obliczanie rozkładu podobieństw...")

    # Sample random user-movie pairs
    if len(user_ids) * len(movie_ids) > n_samples:
        user_sample = np.random.choice(len(user_ids), min(100, len(user_ids)), replace=False)
        movie_sample = np.random.choice(len(movie_ids), min(100, len(movie_ids)), replace=False)
    else:
        user_sample = range(len(user_ids))
        movie_sample = range(len(movie_ids))

    # Compute similarities
    similarities = []
    for ui in user_sample:
        if ui < len(user_profiles):
            user_vec = user_profiles[ui]
            for mi in movie_sample:
                mid = movie_ids[mi]
                if mid in embeddings_dict:
                    movie_vec = embeddings_dict[mid]
                    sim = np.dot(user_vec, movie_vec)
                    similarities.append(sim)

    if not similarities:
        print("    ⚠ Nie można obliczyć podobieństw")
        return

    similarities = np.array(similarities)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Histogram
    ax1 = axes[0]
    ax1.hist(similarities, bins=50, color=COLORS[4], edgecolor="black", alpha=0.7, density=True)
    ax1.axvline(similarities.mean(), color="red", linestyle="--", linewidth=2, label=f"Średnia: {similarities.mean():.3f}")
    ax1.axvline(0, color="gray", linestyle=":", linewidth=1, alpha=0.7)

    ax1.set_xlabel(TWO_TOWER_LABELS["similarity_score"], fontweight="bold")
    ax1.set_ylabel(POLISH_LABELS["density"], fontweight="bold")
    ax1.set_title("Histogram wyników podobieństwa", fontweight="bold")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Right: KDE plot with percentiles
    ax2 = axes[1]
    sns.kdeplot(similarities, ax=ax2, color=COLORS[4], fill=True, alpha=0.5)

    # Add percentile markers
    percentiles = [10, 25, 50, 75, 90]
    for p in percentiles:
        val = np.percentile(similarities, p)
        ax2.axvline(val, color="gray", linestyle="--", alpha=0.5)
        ax2.text(val, ax2.get_ylim()[1] * 0.9, f"{p}%", ha="center", fontsize=8)

    ax2.set_xlabel(TWO_TOWER_LABELS["similarity_score"], fontweight="bold")
    ax2.set_ylabel(POLISH_LABELS["density"], fontweight="bold")
    ax2.set_title("Rozkład gęstości z percentylami", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    # Stats box
    stats_text = (
        f"n = {len(similarities):,}\n"
        f"Min: {similarities.min():.3f}\n"
        f"Max: {similarities.max():.3f}\n"
        f"Std: {similarities.std():.3f}"
    )
    ax2.text(
        0.02,
        0.98,
        stats_text,
        transform=ax2.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.suptitle(TWO_TOWER_LABELS["similarity_distribution"], fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"  ✓ Zapisano rozkład podobieństw do {output_path}")


def plot_sample_recommendations(
    embeddings_dict: Dict[int, np.ndarray],
    movie_ids: List[int],
    train_df: pd.DataFrame,
    id_to_title: Dict[int, str],
    output_path: Path,
    n_users: int = 5,
    top_k: int = 10,
):
    """
    Plot top recommendations for sample users.
    """
    set_thesis_style()

    print("  Generowanie przykładowych rekomendacji...")

    # Get active users
    user_rating_counts = train_df.groupby("userId").size()
    active_users = user_rating_counts[user_rating_counts >= 50].index.tolist()[:n_users]

    if not active_users:
        active_users = train_df["userId"].unique()[:n_users].tolist()

    if len(active_users) == 0:
        print("    ⚠ Brak użytkowników do analizy")
        return

    fig, axes = plt.subplots(1, len(active_users), figsize=(5 * len(active_users), 8))
    if len(active_users) == 1:
        axes = [axes]

    for ax, user_id in zip(axes, active_users):
        # Get user's highly rated movies
        user_train = train_df[train_df["userId"] == user_id]
        top_rated = user_train[user_train["rating"] >= 4.0].nlargest(5, "rating")

        if len(top_rated) == 0:
            continue

        # Build user profile from highly rated movies
        user_profile = []
        for mid in top_rated["movieId"]:
            if mid in embeddings_dict:
                user_profile.append(embeddings_dict[mid])

        if not user_profile:
            continue

        # Average to create user preference vector
        user_vec = np.mean(user_profile, axis=0)
        user_vec = user_vec / (np.linalg.norm(user_vec) + 1e-8)

        # Find most similar movies
        seen_movies = set(user_train["movieId"])
        similarities = []

        for mid in movie_ids:
            if mid in seen_movies:
                continue
            if mid in embeddings_dict:
                sim = np.dot(user_vec, embeddings_dict[mid])
                similarities.append((mid, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        top_recs = similarities[:top_k]

        # Plot
        titles = [str(id_to_title.get(mid, f"Film {mid}"))[:30] for mid, _ in top_recs]
        scores = [score for _, score in top_recs]

        y_pos = np.arange(len(titles))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(titles)))
        bars = ax.barh(y_pos, scores, color=colors)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(titles, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Wynik dopasowania", fontweight="bold")

        # Get top rated movie title for subtitle
        top_movie = id_to_title.get(top_rated.iloc[0]["movieId"], "Unknown")[:25]
        ax.set_title(f"Użytkownik {user_id}\nUlubiony: {top_movie}...", fontsize=10, fontweight="bold")

        # Add score labels (inside bar if > 75% of max, outside otherwise)
        max_score = max(scores) if scores else 1
        for bar, score in zip(bars, scores):
            if score > 0.75 * max_score:
                ax.text(
                    bar.get_width() * 0.5,
                    bar.get_y() + bar.get_height() / 2,
                    f"{score:.3f}",
                    va="center",
                    ha="center",
                    fontsize=7,
                    fontweight="bold",
                    color="black",
                )
            else:
                ax.text(
                    bar.get_width() + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{score:.3f}",
                    va="center",
                    ha="left",
                    fontsize=7,
                    fontweight="bold",
                )

    plt.suptitle(TWO_TOWER_LABELS["sample_recommendations"], fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Zapisano przykładowe rekomendacje do {output_path}")


def plot_cold_start_analysis(
    combined_embeddings_file: Path,
    combined_metadata_file: Path,
    movies_df: pd.DataFrame,
    output_path: Path,
    n_samples: int = 3000,
):
    """
    Analyze cold-start problem - compare ML movies vs TMDB-only movies.
    """
    set_thesis_style()

    print("  Analiza problemu zimnego startu...")

    # Load combined embeddings
    if not combined_embeddings_file.exists():
        print(f"    ⚠ Nie znaleziono osadzeń combined: {combined_embeddings_file}")
        return

    combined_emb = np.load(combined_embeddings_file, allow_pickle=True)
    with open(combined_metadata_file) as f:
        combined_meta = json.load(f)

    # Separate ML movies from TMDB-only movies
    ml_movie_ids = set(movies_df["movieId"].tolist())

    combined_ids = combined_meta.get("movie_ids", [])
    ml_mask = [mid in ml_movie_ids for mid in combined_ids]
    tmdb_mask = [not m for m in ml_mask]

    n_ml = sum(ml_mask)
    n_tmdb = sum(tmdb_mask)

    print(f"    Filmy MovieLens: {n_ml:,}")
    print(f"    Filmy tylko TMDB: {n_tmdb:,}")

    if n_ml == 0 or n_tmdb == 0:
        print("    ⚠ Brak filmów do porównania")
        return

    # Sample for visualization
    ml_indices = np.where(ml_mask)[0]
    tmdb_indices = np.where(tmdb_mask)[0]

    n_ml_sample = min(n_samples // 2, len(ml_indices))
    n_tmdb_sample = min(n_samples // 2, len(tmdb_indices))

    ml_sample = np.random.choice(ml_indices, n_ml_sample, replace=False)
    tmdb_sample = np.random.choice(tmdb_indices, n_tmdb_sample, replace=False)

    # Combine samples
    all_indices = np.concatenate([ml_sample, tmdb_sample])
    sample_embeddings = combined_emb[all_indices]
    sample_labels = ["MovieLens"] * len(ml_sample) + ["TMDB (cold-start)"] * len(tmdb_sample)

    # Compute t-SNE
    print("    Obliczanie t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    embeddings_2d = tsne.fit_transform(sample_embeddings)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: t-SNE visualization
    ax1 = axes[0]

    ml_2d = embeddings_2d[: len(ml_sample)]
    tmdb_2d = embeddings_2d[len(ml_sample) :]

    ax1.scatter(ml_2d[:, 0], ml_2d[:, 1], c="#3498db", alpha=0.5, s=10, label=f"MovieLens ({n_ml_sample})")
    ax1.scatter(tmdb_2d[:, 0], tmdb_2d[:, 1], c="#e74c3c", alpha=0.5, s=10, label=f"TMDB cold-start ({n_tmdb_sample})")

    ax1.set_xlabel("t-SNE wymiar 1", fontweight="bold")
    ax1.set_ylabel("t-SNE wymiar 2", fontweight="bold")
    ax1.set_title("Porównanie przestrzeni osadzeń", fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.2)

    # Right: Embedding norm comparison
    ax2 = axes[1]

    ml_norms = np.linalg.norm(combined_emb[ml_sample], axis=1)
    tmdb_norms = np.linalg.norm(combined_emb[tmdb_sample], axis=1)

    data = [ml_norms, tmdb_norms]
    positions = [1, 2]
    parts = ax2.violinplot(data, positions, showmeans=True, showmedians=True)

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(["#3498db", "#e74c3c"][i])
        pc.set_alpha(0.7)

    ax2.set_xticks(positions)
    ax2.set_xticklabels(["MovieLens", "TMDB\n(cold-start)"])
    ax2.set_ylabel("Norma wektora osadzenia (L2)", fontweight="bold")
    ax2.set_title("Porównanie norm osadzeń", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    # Add stats
    stats_text = (
        f"MovieLens:\n  μ={ml_norms.mean():.3f}, σ={ml_norms.std():.3f}\n"
        f"TMDB:\n  μ={tmdb_norms.mean():.3f}, σ={tmdb_norms.std():.3f}"
    )
    ax2.text(
        0.02,
        0.98,
        stats_text,
        transform=ax2.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.suptitle(TWO_TOWER_LABELS["cold_start_analysis"], fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Zapisano analizę cold-start do {output_path}")


def plot_movie_similarity_examples(
    embeddings_dict: Dict[int, np.ndarray],
    movie_ids: List[int],
    id_to_title: Dict[int, str],
    id_to_genres: Dict[int, str],
    output_path: Path,
    query_movies: List[str] = None,
    top_k: int = 8,
):
    """
    Plot similar movies for selected query movies.
    """
    set_thesis_style()

    print("  Generowanie przykładów podobnych filmów...")

    # Default query movies (popular ones)
    if query_movies is None:
        query_movies = ["Matrix", "Inception", "Toy Story", "Godfather"]

    # Find movie IDs by title
    title_to_id = {str(v).lower(): k for k, v in id_to_title.items()}

    query_ids = []
    query_titles = []
    for q in query_movies:
        for title, mid in title_to_id.items():
            if q.lower() in title and mid in embeddings_dict:
                query_ids.append(mid)
                query_titles.append(id_to_title.get(mid, q))
                break
        if len(query_ids) == len(query_titles):
            continue

    if not query_ids:
        print("    ⚠ Nie znaleziono filmów zapytania")
        return

    n_queries = min(4, len(query_ids))
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for ax, qid, qtitle in zip(axes[:n_queries], query_ids[:n_queries], query_titles[:n_queries]):
        # Get query embedding
        query_emb = embeddings_dict[qid]
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)

        # Find similar movies
        similarities = []
        for mid in movie_ids:
            if mid == qid:
                continue
            if mid in embeddings_dict:
                movie_emb = embeddings_dict[mid]
                movie_emb = movie_emb / (np.linalg.norm(movie_emb) + 1e-8)
                sim = np.dot(query_emb, movie_emb)
                similarities.append((mid, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        top_similar = similarities[:top_k]

        # Plot
        titles = [f"{str(id_to_title.get(mid, 'Unknown'))[:25]}" for mid, _ in top_similar]
        genres = [str(id_to_genres.get(mid, ""))[:20] for mid, _ in top_similar]
        scores = [score for _, score in top_similar]

        y_pos = np.arange(len(titles))
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(titles)))
        bars = ax.barh(y_pos, scores, color=colors)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{t}\n({g})" for t, g in zip(titles, genres)], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Podobieństwo kosinusowe", fontweight="bold")
        ax.set_title(f"Podobne do: {str(qtitle)[:35]}", fontsize=11, fontweight="bold")

        # Add score labels (inside bar if > 75% of max, outside otherwise)
        max_score = max(scores) if scores else 1
        for bar, score in zip(bars, scores):
            if score > 0.75 * max_score:
                ax.text(
                    bar.get_width() * 0.5,
                    bar.get_y() + bar.get_height() / 2,
                    f"{score:.3f}",
                    va="center",
                    ha="center",
                    fontsize=8,
                    fontweight="bold",
                    color="black",
                )
            else:
                ax.text(
                    bar.get_width() + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{score:.3f}",
                    va="center",
                    ha="left",
                    fontsize=8,
                    fontweight="bold",
                )

    # Hide unused axes
    for ax in axes[n_queries:]:
        ax.set_visible(False)

    plt.suptitle(TWO_TOWER_LABELS["top_similar_movies"], fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Zapisano przykłady podobnych filmów do {output_path}")


def plot_two_tower_similarity_heatmap(
    embeddings_dict: Dict,
    movie_ids: List[int],
    id_to_title: Dict,
    movies_df: pd.DataFrame,
    output_path: Path,
    n_movies: int = 15,
):
    """
    Plot lower triangular cosine similarity heatmap for diverse popular movies.

    Args:
        embeddings_dict: Dictionary mapping movie IDs to embeddings
        movie_ids: List of movie IDs
        id_to_title: Mapping from movie ID to title
        movies_df: DataFrame with movie metadata including popularity
        output_path: Path to save plot
        n_movies: Number of movies to include
    """
    set_thesis_style()

    print("  Tworzenie heatmapy podobieństwa kosinusowego...")

    # Filter for movies with embeddings
    valid_movies = [(mid, id_to_title.get(mid, f"Film {mid}")) for mid in movie_ids if mid in embeddings_dict]

    # Get popularity scores
    if "popularity" in movies_df.columns:
        movie_to_pop = dict(zip(movies_df["movieId"], movies_df["popularity"]))
    else:
        # Fallback: use position in list as proxy for popularity
        movie_to_pop = {mid: len(valid_movies) - i for i, (mid, _) in enumerate(valid_movies)}

    # Sort by popularity
    valid_with_pop = [(mid, title, movie_to_pop.get(mid, 0)) for mid, title in valid_movies]
    valid_with_pop.sort(key=lambda x: x[2], reverse=True)

    # Define iconic movies from different genres to show diversity
    desired_movies = [
        # Sci-Fi Action
        "Star Wars",
        "Matrix",
        "Terminator 2",
        "Blade Runner",
        # Fantasy/Adventure
        "Lord of the Rings",
        "Harry Potter and the Sorcerer",
        # Animated Family
        "Toy Story",
        "Lion King",
        "Finding Nemo",
        "Spirited Away",
        # Classic Drama
        "Godfather",
        "Shawshank Redemption",
        "Forrest Gump",
        # Thriller/Dark
        "Dark Knight",
        "Pulp Fiction",
        "Inception",
        "Fight Club",
        # Horror
        "Alien",
        "Shining",
        # Romance
        "Titanic",
        "Notebook",
        # Comedy
        "Groundhog Day",
        "Big Lebowski",
        "Back to the Future",
    ]

    # Try to find these movies in our dataset
    selected_movies = []
    used_ids = set()

    for desired in desired_movies:
        if len(selected_movies) >= n_movies:
            break
        # Search for movie in valid_with_pop
        for mid, title, pop in valid_with_pop:
            if isinstance(title, str) and mid not in used_ids and desired.lower() in title.lower():
                selected_movies.append((mid, title, pop))
                used_ids.add(mid)
                print(f"    Found: {title}")
                break

    # If we don't have enough, fill with popular diverse movies
    if len(selected_movies) < n_movies:
        print(f"    Filling remaining slots with popular movies...")
        top_candidates = [m for m in valid_with_pop if m[0] not in used_ids][:100]

        if top_candidates:
            candidate_ids = [x[0] for x in top_candidates]
            candidate_embeddings = np.array([embeddings_dict[mid] for mid in candidate_ids])

            # Normalize embeddings
            norms = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
            normalized = candidate_embeddings / (norms + 1e-8)

            # Compute similarity matrix
            similarity_matrix = np.dot(normalized, normalized.T)

            # Greedy diversity selection
            selected_from_top = [0]  # Start with most popular unused

            for _ in range(n_movies - len(selected_movies) - 1):
                min_sims = []
                for i in range(len(candidate_ids)):
                    if i not in selected_from_top:
                        sims_to_selected = [similarity_matrix[i, j] for j in selected_from_top]
                        min_sim = min(sims_to_selected)
                        min_sims.append((i, min_sim))

                if min_sims:
                    next_idx = min(min_sims, key=lambda x: x[1])[0]
                    selected_from_top.append(next_idx)

            # Add these to selected_movies
            for idx in selected_from_top:
                selected_movies.append(top_candidates[idx])

    # Extract final data
    final_ids = [x[0] for x in selected_movies[:n_movies]]
    final_titles = [x[1][:30] for x in selected_movies[:n_movies]]
    final_embeddings = np.array([embeddings_dict[mid] for mid in final_ids])

    # Compute final similarity matrix
    norms = np.linalg.norm(final_embeddings, axis=1, keepdims=True)
    normalized = final_embeddings / (norms + 1e-8)
    similarity_matrix = np.dot(normalized, normalized.T)

    # Create lower triangular mask (k=1 to show diagonal)
    mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Use masked array for lower triangle + diagonal
    masked_sim = np.ma.array(similarity_matrix, mask=mask)

    sns.heatmap(
        masked_sim,
        xticklabels=final_titles,
        yticklabels=final_titles,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        square=True,
        ax=ax,
        vmin=-1,
        vmax=1,
        center=0,
        annot_kws={"size": 8},
        cbar_kws={"label": "Podobieństwo kosinusowe"},
        mask=mask,
        linewidths=0,
        linecolor="none",
    )

    # Remove grid
    ax.grid(False)

    ax.set_title(
        "Macierz podobieństwa kosinusowego - Model Two-Tower",
        fontweight="bold",
        fontsize=12,
        pad=15,
    )
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Zapisano heatmapę podobieństwa do {output_path}")


def generate_all_two_tower_plots(
    data_dir: Path,
    saved_models_dir: Path,
    output_dir: Path,
):
    """
    Generate all Two-Tower specific plots.

    Args:
        data_dir: Path to processed data directory
        saved_models_dir: Path to saved models directory
        output_dir: Path to save plots
    """
    print("\n" + "=" * 80)
    print("Generowanie wykresów modelu Two-Tower")
    print("=" * 80)

    # Create output directory
    tt_dir = output_dir / "two_tower"
    tt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Architecture diagram (no data needed)
    plot_two_tower_architecture(tt_dir / "architektura_modelu.png")

    # Load data
    movies_file = data_dir / "movies.parquet"
    train_file = data_dir / "train.parquet"
    embeddings_file = saved_models_dir / "movie_embeddings.npy"
    metadata_file = saved_models_dir / "movie_embeddings_metadata.json"

    if not all(f.exists() for f in [movies_file, embeddings_file, metadata_file]):
        print("⚠ Brak wymaganych plików dla pozostałych wykresów")
        return

    # Load data
    movies_df = pl.read_parquet(movies_file).to_pandas()
    embeddings_dict = np.load(embeddings_file, allow_pickle=True).item()
    with open(metadata_file) as f:
        metadata = json.load(f)
    movie_ids = metadata["movie_ids"]

    title_col = "title_ml" if "title_ml" in movies_df.columns else "title"
    id_to_title = dict(zip(movies_df["movieId"], movies_df[title_col]))
    id_to_genres = dict(zip(movies_df["movieId"], movies_df["genres"]))

    print(f"  Załadowano {len(embeddings_dict):,} osadzeń filmów")

    # 2. Sample recommendations
    if train_file.exists():
        train_df = pl.read_parquet(train_file).to_pandas()
        plot_sample_recommendations(embeddings_dict, movie_ids, train_df, id_to_title, tt_dir / "rekomendacje_przykladowe.png")

    # 3. Movie similarity examples
    plot_movie_similarity_examples(embeddings_dict, movie_ids, id_to_title, id_to_genres, tt_dir / "podobne_filmy.png")

    # 4. Cosine similarity heatmap
    plot_two_tower_similarity_heatmap(embeddings_dict, movie_ids, id_to_title, movies_df, tt_dir / "macierz_podobienstwa.png")

    # 5. Cold-start analysis
    combined_emb_file = saved_models_dir / "combined_movie_embeddings.npy"
    combined_meta_file = saved_models_dir / "combined_movie_embeddings_metadata.json"

    if combined_emb_file.exists() and combined_meta_file.exists():
        plot_cold_start_analysis(combined_emb_file, combined_meta_file, movies_df, tt_dir / "analiza_cold_start.png")

    print("\n" + "=" * 80)
    print(f"✓ Wykresy Two-Tower zapisane w {tt_dir}")
    print("=" * 80)


if __name__ == "__main__":
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "model" / "data" / "processed"
    SAVED_MODELS_DIR = PROJECT_ROOT / "model" / "saved_models"
    PLOTS_DIR = PROJECT_ROOT / "model" / "plots"

    generate_all_two_tower_plots(DATA_DIR, SAVED_MODELS_DIR, PLOTS_DIR)
