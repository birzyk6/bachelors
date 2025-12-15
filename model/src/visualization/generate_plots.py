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

from model.src.visualization.plots import (
    plot_genre_distribution,
    plot_ndcg_comparison,
    plot_rating_distribution,
    plot_recall_at_k,
    plot_rmse_comparison,
)
from model.src.visualization.predictions_comparison import plot_model_predictions_comparison

# Paths
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

    # Model file mapping
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

            # Extract metrics
            test_metrics = data.get("test_metrics", {})
            val_metrics = data.get("val_metrics", {})

            # Use test metrics if available, else val metrics
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

    # Fall back to loading individual files
    print("Loading individual model results...")
    return load_individual_results()


def generate_model_comparison_plots(results: dict):
    """Generate model comparison plots."""
    print("\nGenerating model comparison plots...")

    if not results.get("rating_prediction"):
        print("  ⚠ No rating prediction results found, skipping comparison plots")
        return

    # RMSE comparison
    plot_rmse_comparison(results, PLOTS_DIR / "rmse_comparison.png")

    # NDCG@10 comparison (if ranking results exist)
    if results.get("ranking"):
        plot_ndcg_comparison(results, PLOTS_DIR / "ndcg_comparison.png", k=10)
        plot_recall_at_k(results, PLOTS_DIR / "recall_at_k.png", k_values=[5, 10, 20])


def generate_data_exploration_plots():
    """Generate data exploration plots."""
    print("\nGenerating data exploration plots...")

    # Load training data
    train_file = DATA_DIR / "train.parquet"
    if not train_file.exists():
        print(f"  ⚠ Training data not found at {train_file}")
        print("  Skipping data exploration plots.")
        return

    train_data = pl.read_parquet(train_file).to_pandas()

    # Rating distribution
    plot_rating_distribution(train_data["rating"], PLOTS_DIR / "rating_distribution.png")

    # Genre distribution
    if "genres" in train_data.columns:
        plot_genre_distribution(train_data["genres"], PLOTS_DIR / "genre_distribution.png")


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

        # Check if embeddings exist
        embeddings_file = SAVED_MODELS_DIR / "movie_embeddings.npy"
        metadata_file = SAVED_MODELS_DIR / "movie_embeddings_metadata.json"

        if not embeddings_file.exists() or not metadata_file.exists():
            print("  ⚠ Two-Tower embeddings not found, skipping Two-Tower plots")
            return

        # Load embeddings (saved as pickled dict) and metadata
        embeddings_dict = np.load(embeddings_file, allow_pickle=True).item()
        with open(metadata_file) as f:
            metadata = json.load(f)

        movie_ids = metadata["movie_ids"]

        # Convert dict to array
        embeddings = np.array([embeddings_dict[mid] for mid in movie_ids if mid in embeddings_dict])
        movie_ids = [mid for mid in movie_ids if mid in embeddings_dict]

        # Load movie info
        movie_file = DATA_DIR / "movies.parquet"
        if movie_file.exists():
            movies_df = pl.read_parquet(movie_file).to_pandas()
        else:
            print("  ⚠ Movies data not found")
            return

        # Create movie id to title mapping (title_ml is the column name)
        title_col = "title_ml" if "title_ml" in movies_df.columns else "title"
        id_to_title = dict(zip(movies_df["movieId"], movies_df[title_col]))
        id_to_genres = dict(zip(movies_df["movieId"], movies_df["genres"]))

        # 1. Plot embedding space visualization (t-SNE)
        plot_embedding_space(embeddings, movie_ids, id_to_title, id_to_genres, PLOTS_DIR)

        # 2. Plot top recommendations for sample users
        plot_two_tower_recommendations(embeddings_dict, movie_ids, DATA_DIR, PLOTS_DIR, id_to_title)

        # 3. Plot similarity heatmap for popular movies
        plot_movie_similarity_heatmap(embeddings, movie_ids, id_to_title, movies_df, PLOTS_DIR)

        # 4. Plot query-based recommendations (semantic search)
        plot_query_recommendations(DATA_DIR, PLOTS_DIR, id_to_title, id_to_genres, movies_df)

        # 5. Plot genre clusters in embedding space
        plot_genre_clusters(embeddings, movie_ids, id_to_genres, PLOTS_DIR)

    except Exception as e:
        print(f"  ✗ Could not generate Two-Tower plots: {e}")
        import traceback

        traceback.print_exc()


def plot_embedding_space(embeddings, movie_ids, id_to_title, id_to_genres, output_dir):
    """Plot 2D visualization of movie embedding space using t-SNE."""
    from sklearn.manifold import TSNE

    print("  Creating embedding space visualization (t-SNE)...")

    # Sample for visualization (t-SNE is slow on large datasets)
    max_movies = 500
    if len(embeddings) > max_movies:
        indices = np.random.choice(len(embeddings), max_movies, replace=False)
        sample_embeddings = embeddings[indices]
        sample_ids = [movie_ids[i] for i in indices]
    else:
        sample_embeddings = embeddings
        sample_ids = movie_ids

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(sample_embeddings)

    # Extract PRIMARY genre for coloring (first genre only, comma-separated)
    genre_list = []
    for mid in sample_ids:
        genres = id_to_genres.get(mid, "Unknown")
        if isinstance(genres, str) and genres:
            # Genres are comma-separated, take the first one
            primary_genre = genres.split(",")[0].strip()
        else:
            primary_genre = "Unknown"
        genre_list.append(primary_genre)

    # Count genre frequencies and keep top 10
    from collections import Counter

    genre_counts = Counter(genre_list)
    top_genres = [g for g, _ in genre_counts.most_common(10)]

    # Map genres to colors, "Other" for rare genres
    genre_to_color = {g: i for i, g in enumerate(top_genres)}
    colors = [genre_to_color.get(g, len(top_genres)) for g in genre_list]

    # Replace rare genres with "Other" for legend
    display_genres = top_genres + ["Other"]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, cmap="tab10", alpha=0.6, s=30)

    # Add legend with proper handles
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label=g, markerfacecolor=plt.cm.tab10(i / 10), markersize=10)
        for i, g in enumerate(display_genres)
    ]
    ax.legend(handles=legend_elements, title="Primary Genre", loc="upper right", fontsize=8)

    ax.set_xlabel("t-SNE Dimension 1", fontweight="bold")
    ax.set_ylabel("t-SNE Dimension 2", fontweight="bold")
    ax.set_title("Two-Tower Model: Movie Embedding Space", fontweight="bold", fontsize=14)

    plt.tight_layout()
    output_path = output_dir / "two_tower_embedding_space.png"
    plt.savefig(output_path)
    plt.close()

    print(f"  ✓ Saved embedding space to {output_path}")


def plot_two_tower_recommendations(embeddings_dict, movie_ids, data_dir, output_dir, id_to_title):
    """Plot top recommendations from Two-Tower for sample users."""
    print("  Creating Two-Tower recommendations comparison...")

    # Load test data to get sample users
    test_file = data_dir / "test.parquet"
    train_file = data_dir / "train.parquet"

    if not test_file.exists() or not train_file.exists():
        print("    ⚠ Test/train data not found")
        return

    test_data = pl.read_parquet(test_file).to_pandas()
    train_data = pl.read_parquet(train_file).to_pandas()

    # Get active users (with many ratings)
    user_rating_counts = train_data.groupby("userId").size()
    active_users = user_rating_counts[user_rating_counts >= 50].index.tolist()[:3]

    if not active_users:
        active_users = train_data["userId"].unique()[:3].tolist()

    # For each user, find similar movies to their highly rated ones
    fig, axes = plt.subplots(1, len(active_users), figsize=(6 * len(active_users), 8))
    if len(active_users) == 1:
        axes = [axes]

    for ax, user_id in zip(axes, active_users):
        # Get user's highly rated movies
        user_train = train_data[train_data["userId"] == user_id]
        top_rated = user_train[user_train["rating"] >= 4.0].nlargest(3, "rating")

        if len(top_rated) == 0:
            continue

        # Get embeddings of top rated movies
        user_profile = []
        for mid in top_rated["movieId"]:
            if mid in embeddings_dict:
                user_profile.append(embeddings_dict[mid])

        if not user_profile:
            continue

        # Average to create user preference vector (DON'T normalize - keep raw)
        user_vec = np.mean(user_profile, axis=0)

        # Find most similar movies using dot product (embeddings are L2-normalized)
        similarities = []
        seen_movies = set(user_train["movieId"])

        for mid in movie_ids:
            if mid in seen_movies:
                continue
            if mid in embeddings_dict:
                # Dot product is cosine similarity for normalized vectors
                sim = np.dot(user_vec, embeddings_dict[mid])
                similarities.append((mid, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        top_recs = similarities[:10]

        # Normalize scores for visualization (min-max scale to [0, 1])
        if top_recs:
            min_score = min(s for _, s in top_recs)
            max_score = max(s for _, s in top_recs)
            score_range = max_score - min_score if max_score != min_score else 1

        # Plot recommendations
        titles = [id_to_title.get(mid, f"Movie {mid}")[:25] for mid, _ in top_recs]
        scores = [score for _, score in top_recs]

        y_pos = np.arange(len(titles))
        bars = ax.barh(y_pos, scores, color=sns.color_palette("Set2")[4])

        ax.set_yticks(y_pos)
        ax.set_yticklabels(titles, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Affinity Score (Dot Product)", fontweight="bold")
        ax.set_title(
            f"User {user_id}\nTop Rated: {id_to_title.get(top_rated.iloc[0]['movieId'], '')[:20]}...",
            fontsize=11,
            fontweight="bold",
        )

        # Add score labels
        for bar, score in zip(bars, scores):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"{score:.3f}", va="center", fontsize=8)

    plt.suptitle("Two-Tower Model: Personalized Recommendations", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    output_path = output_dir / "two_tower_recommendations.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved recommendations to {output_path}")


def plot_movie_similarity_heatmap(embeddings, movie_ids, id_to_title, movies_df, output_dir):
    """Plot similarity heatmap for popular movies."""
    print("  Creating movie similarity heatmap...")

    # Get popular movies
    if "popularity" in movies_df.columns:
        popular = movies_df.nlargest(15, "popularity")["movieId"].tolist()
    else:
        # Use movies with most embeddings (likely popular)
        popular = movie_ids[:15]

    # Filter to movies in embeddings
    popular = [mid for mid in popular if mid in movie_ids][:12]

    if len(popular) < 5:
        print("    ⚠ Not enough popular movies with embeddings")
        return

    # Get embeddings for popular movies
    movie_id_to_idx = {mid: i for i, mid in enumerate(movie_ids)}
    pop_embeddings = np.array([embeddings[movie_id_to_idx[mid]] for mid in popular])

    # Compute similarity matrix
    norms = np.linalg.norm(pop_embeddings, axis=1, keepdims=True)
    normalized = pop_embeddings / (norms + 1e-8)
    similarity_matrix = np.dot(normalized, normalized.T)

    # Get titles
    titles = [id_to_title.get(mid, f"Movie {mid}")[:20] for mid in popular]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        similarity_matrix,
        xticklabels=titles,
        yticklabels=titles,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        square=True,
        ax=ax,
        vmin=0,
        vmax=1,
        annot_kws={"size": 8},
    )

    ax.set_title(
        "Two-Tower Model: Movie Similarity Matrix\n(Cosine Similarity of Learned Embeddings)", fontweight="bold", fontsize=12
    )
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    plt.tight_layout()
    output_path = output_dir / "two_tower_similarity_heatmap.png"
    plt.savefig(output_path)
    plt.close()

    print(f"  ✓ Saved similarity heatmap to {output_path}")


def plot_query_recommendations(data_dir, output_dir, id_to_title, id_to_genres, movies_df):
    """Plot recommendations for semantic text queries using movie title/genre matching."""
    print("  Creating query-based recommendations...")

    try:
        import tensorflow as tf
        import tensorflow_hub as hub

        # Load saved movie embeddings (these are 64-dim from the Two-Tower output)
        embeddings_file = SAVED_MODELS_DIR / "movie_embeddings.npy"
        metadata_file = SAVED_MODELS_DIR / "movie_embeddings_metadata.json"

        if not embeddings_file.exists():
            print("    ⚠ Movie embeddings not found")
            return

        embeddings_dict = np.load(embeddings_file, allow_pickle=True).item()
        with open(metadata_file) as f:
            metadata = json.load(f)
        movie_ids = metadata["movie_ids"]

        # Build movie embeddings matrix
        valid_ids = [mid for mid in movie_ids if mid in embeddings_dict]
        movie_embeddings = np.array([embeddings_dict[mid] for mid in valid_ids])
        embedding_dim = movie_embeddings.shape[1]

        print(f"    Movie embeddings shape: {movie_embeddings.shape}")

        # Sample queries for different tastes - we'll find movies by genre/title matching
        queries = [
            ("Action & Sci-Fi", ["Action", "Sci-Fi"]),
            ("Romantic Comedy", ["Romance", "Comedy"]),
            ("Horror & Thriller", ["Horror", "Thriller"]),
            ("Animation & Family", ["Animation", "Children"]),
        ]

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        for ax, (query_name, target_genres) in zip(axes, queries):
            # Find movies matching these genres
            matching_movies = []
            for mid in valid_ids:
                movie_genres = id_to_genres.get(mid, "")
                if isinstance(movie_genres, str):
                    movie_genres_list = [g.strip() for g in movie_genres.split(",")]
                    # Check if any target genre matches
                    if any(tg in movie_genres_list for tg in target_genres):
                        matching_movies.append(mid)

            if len(matching_movies) < 3:
                continue

            # Get embeddings for matching movies and compute their centroid
            matching_embeddings = np.array([embeddings_dict[mid] for mid in matching_movies[:100]])
            query_centroid = np.mean(matching_embeddings, axis=0, keepdims=True)
            query_centroid = query_centroid / (np.linalg.norm(query_centroid) + 1e-8)

            # Compute similarities with all movies
            similarities = np.dot(movie_embeddings, query_centroid.T).flatten()

            # Get top 10 recommendations (excluding movies used to form the query)
            top_indices = np.argsort(similarities)[::-1]
            top_movies = []
            for idx in top_indices:
                mid = valid_ids[idx]
                if mid not in matching_movies[:100]:  # Exclude query movies
                    top_movies.append((mid, similarities[idx]))
                if len(top_movies) >= 10:
                    break

            # Plot
            titles = [f"{id_to_title.get(mid, f'Movie {mid}')[:28]}" for mid, _ in top_movies]
            genres = [id_to_genres.get(mid, "")[:25] for mid, _ in top_movies]
            scores = [score for _, score in top_movies]

            y_pos = np.arange(len(titles))
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(titles)))
            bars = ax.barh(y_pos, scores, color=colors)

            ax.set_yticks(y_pos)
            ax.set_yticklabels([f"{t}\n({g})" for t, g in zip(titles, genres)], fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("Similarity Score", fontweight="bold")
            ax.set_title(f'Query: "{query_name}"', fontsize=11, fontweight="bold")

            # Add score labels
            for bar, score in zip(bars, scores):
                ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2, f"{score:.3f}", va="center", fontsize=7)

        plt.suptitle(
            "Two-Tower Model: Genre-Based Query Recommendations\n(Finding similar movies based on genre embedding centroids)",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()

        output_path = output_dir / "two_tower_query_recommendations.png"
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close()

        print(f"  ✓ Saved query recommendations to {output_path}")

    except Exception as e:
        print(f"    ⚠ Could not create query recommendations: {e}")
        import traceback

        traceback.print_exc()


def plot_genre_clusters(embeddings, movie_ids, id_to_genres, output_dir):
    """Plot genre distribution analysis in embedding space."""
    print("  Creating genre cluster analysis...")

    from collections import Counter, defaultdict

    from sklearn.manifold import TSNE

    # Sample movies
    max_movies = 1000
    if len(embeddings) > max_movies:
        indices = np.random.choice(len(embeddings), max_movies, replace=False)
        sample_embeddings = embeddings[indices]
        sample_ids = [movie_ids[i] for i in indices]
    else:
        sample_embeddings = embeddings
        sample_ids = movie_ids

    # Extract all genres for each movie
    movie_genres = {}
    genre_movies = defaultdict(list)

    for i, mid in enumerate(sample_ids):
        genres_str = id_to_genres.get(mid, "Unknown")
        if isinstance(genres_str, str) and genres_str:
            genres = [g.strip() for g in genres_str.split(",")]
        else:
            genres = ["Unknown"]
        movie_genres[mid] = genres
        for g in genres:
            genre_movies[g].append(i)

    # Get top 6 genres by frequency
    genre_counts = Counter()
    for mid in sample_ids:
        for g in movie_genres.get(mid, []):
            genre_counts[g] += 1

    top_genres = [g for g, _ in genre_counts.most_common(6) if g not in ["Unknown", "(no genres listed)"]][:6]

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(sample_embeddings)

    # Create figure with 2x3 grid for each genre
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for ax, genre in zip(axes, top_genres):
        # Get indices of movies with this genre
        genre_indices = set(genre_movies[genre])

        # Color by whether movie has this genre
        colors = ["#FF6B6B" if i in genre_indices else "#E0E0E0" for i in range(len(sample_ids))]
        alphas = [0.8 if i in genre_indices else 0.2 for i in range(len(sample_ids))]

        # Plot non-genre movies first (background)
        non_genre_mask = [i not in genre_indices for i in range(len(sample_ids))]
        ax.scatter(
            embeddings_2d[non_genre_mask, 0], embeddings_2d[non_genre_mask, 1], c="#E0E0E0", alpha=0.2, s=15, label="Other"
        )

        # Plot genre movies on top
        genre_mask = [i in genre_indices for i in range(len(sample_ids))]
        ax.scatter(embeddings_2d[genre_mask, 0], embeddings_2d[genre_mask, 1], c="#FF6B6B", alpha=0.7, s=25, label=genre)

        ax.set_title(f"{genre}\n({len(genre_indices)} movies)", fontweight="bold", fontsize=11)
        ax.set_xlabel("t-SNE 1", fontsize=9)
        ax.set_ylabel("t-SNE 2", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")

    plt.suptitle(
        "Two-Tower Model: Genre Distribution in Embedding Space\n(Movies colored by genre membership)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    output_path = output_dir / "two_tower_genre_clusters.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()

    print(f"  ✓ Saved genre clusters to {output_path}")


def generate_all_plots():
    """
    Main function to generate all thesis plots.
    """
    print("=" * 80)
    print("Thesis Plot Generation")
    print("=" * 80)

    # Create plots directory
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load results
    results = load_results()

    if results:
        # Generate comparison plots
        generate_model_comparison_plots(results)

    # Generate data exploration plots
    generate_data_exploration_plots()

    # Generate model predictions comparison
    generate_model_predictions_plot()

    # Generate Two-Tower specific plots
    generate_two_tower_plots()

    print("\n" + "=" * 80)
    print(f"✓ All plots saved to {PLOTS_DIR}")
    print("=" * 80)

    # List generated plots
    print("\nGenerated plots:")
    for plot_file in sorted(PLOTS_DIR.glob("*.png")):
        print(f"  - {plot_file.name}")


if __name__ == "__main__":
    generate_all_plots()
