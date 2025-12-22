"""
Test the recommender engine and generate visualization plots.

This script tests various functionalities of the recommendation engine
and creates plots showing the results for thesis presentation.

All labels are in Polish for academic thesis presentation.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.src.visualization.plots import COLORS, set_thesis_style

# Polish labels for recommender plots
RECOMMENDER_LABELS = {
    "search_results": "Wyniki wyszukiwania dla zapytań",
    "similar_movies": "Rekomendacje podobnych filmów",
    "user_profiles": "Profile użytkowników i ich rekomendacje",
    "method_comparison": "Porównanie metod rekomendacji",
    "genre_search": "Wyszukiwanie według gatunku",
    "title_search": "Wyszukiwanie według tytułu",
    "semantic_search": "Wyszukiwanie semantyczne",
    "similarity_score": "Wynik podobieństwa",
    "movie_title": "Tytuł filmu",
    "query": "Zapytanie",
    "top_recommendations": "Top rekomendacje",
    "genre": "Gatunek",
}


def load_recommender_engine():
    """Load the recommendation engine."""
    from model.src.cli.embedding_store import EmbeddingStore
    from model.src.cli.engine import RecommendationEngine
    from model.src.cli.movie_catalog import MovieCatalog

    # Paths
    data_dir = PROJECT_ROOT / "model" / "data" / "processed"
    models_dir = PROJECT_ROOT / "model" / "saved_models"

    # Load components
    print("  Ładowanie katalogu filmów...")
    catalog = MovieCatalog(data_dir / "movies.parquet")
    catalog.load()  # Explicit load

    print("  Ładowanie osadzeń...")
    embeddings = EmbeddingStore(models_dir / "movie_embeddings.npy", models_dir / "movie_embeddings_metadata.json")
    embeddings.load()  # Explicit load

    print("  Inicjalizacja silnika rekomendacji...")
    engine = RecommendationEngine(embeddings, catalog)

    return engine, catalog


def test_search_queries(engine, output_path: Path):
    """
    Test various search queries and visualize results.
    """
    set_thesis_style()

    print("\n  Testowanie zapytań wyszukiwania...")

    # Test queries with different types
    test_queries = [
        ("Matrix", "Tytuł"),
        ("action sci-fi", "Gatunek"),
        ("romantic comedy", "Gatunek"),
        ("space adventure", "Opis"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for ax, (query, query_type) in zip(axes, test_queries):
        results, explanation = engine.smart_search(query, top_k=10)

        if not results:
            ax.text(0.5, 0.5, f"Brak wyników dla: {query}", ha="center", va="center")
            ax.set_title(f'Zapytanie: "{query}" ({query_type})')
            continue

        # Extract data
        titles = []
        scores = []
        genres = []

        for movie_id, score, source in results[:10]:
            title = engine.catalog.get_title(movie_id)
            genre = engine.catalog.get_genres(movie_id)
            titles.append(str(title)[:30] if title else f"Film {movie_id}")
            scores.append(score)
            genres.append(str(genre)[:20] if genre else "")

        # Plot
        y_pos = np.arange(len(titles))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(titles)))
        bars = ax.barh(y_pos, scores, color=colors)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{t}\n({g})" for t, g in zip(titles, genres)], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(RECOMMENDER_LABELS["similarity_score"], fontweight="bold")
        ax.set_title(f'Zapytanie: "{query}" ({query_type})\n{explanation}', fontsize=10, fontweight="bold")

        # Add score labels
        for bar, score in zip(bars, scores):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"{score:.3f}", va="center", fontsize=7)

    plt.suptitle(RECOMMENDER_LABELS["search_results"], fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Zapisano wyniki wyszukiwania do {output_path}")


def test_similar_movies(engine, output_path: Path):
    """
    Test similar movie recommendations for popular movies.
    """
    set_thesis_style()

    print("\n  Testowanie rekomendacji podobnych filmów...")

    # Popular movies to find similar ones for
    query_titles = ["Matrix", "Toy Story", "Titanic", "Star Wars"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for ax, query_title in zip(axes, query_titles):
        # Find the movie
        movie_ids = engine.catalog.search_by_title(query_title, limit=1)

        if not movie_ids:
            ax.text(0.5, 0.5, f"Nie znaleziono: {query_title}", ha="center", va="center")
            continue

        query_id = movie_ids[0]
        full_title = engine.catalog.get_title(query_id)

        # Get similar movies using the correct method
        similar = engine.recommend_similar_to_movie(query_id, top_k=10)

        if not similar:
            ax.text(0.5, 0.5, f"Brak podobnych dla: {full_title}", ha="center", va="center")
            continue

        # Extract data - format is (movie_id, score, info)
        titles = []
        scores = []
        genres = []

        for movie_id, score, info in similar:
            title = engine.catalog.get_title(movie_id)
            genre = engine.catalog.get_genres(movie_id)
            titles.append(str(title)[:28] if title else f"Film {movie_id}")
            scores.append(score)
            genres.append(str(genre)[:18] if genre else "")

        # Plot
        y_pos = np.arange(len(titles))
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(titles)))
        bars = ax.barh(y_pos, scores, color=colors)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{t}\n({g})" for t, g in zip(titles, genres)], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(RECOMMENDER_LABELS["similarity_score"], fontweight="bold")
        ax.set_title(f"Podobne do: {str(full_title)[:35]}", fontsize=10, fontweight="bold")

        # Add score labels
        for bar, score in zip(bars, scores):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"{score:.3f}", va="center", fontsize=7)

    plt.suptitle(RECOMMENDER_LABELS["similar_movies"], fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Zapisano podobne filmy do {output_path}")


def test_genre_recommendations(engine, output_path: Path):
    """
    Test genre-based recommendations.
    """
    set_thesis_style()

    print("\n  Testowanie rekomendacji według gatunku...")

    # Test different genres
    test_genres = [
        ["Action", "Sci-Fi"],
        ["Comedy", "Romance"],
        ["Horror", "Thriller"],
        ["Animation", "Family"],
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for ax, genres in zip(axes, test_genres):
        # Get genre recommendations using the correct method
        results = engine.recommend_by_genres(genres, top_k=10)

        if not results:
            ax.text(0.5, 0.5, f"Brak wyników dla: {', '.join(genres)}", ha="center", va="center")
            continue

        # Extract data - format is (movie_id, score, info)
        titles = []
        scores = []
        movie_genres = []

        for movie_id, score, info in results:
            title = engine.catalog.get_title(movie_id)
            genre = engine.catalog.get_genres(movie_id)
            titles.append(str(title)[:28] if title else f"Film {movie_id}")
            scores.append(score)
            movie_genres.append(str(genre)[:18] if genre else "")

        # Plot
        y_pos = np.arange(len(titles))
        colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(titles)))
        bars = ax.barh(y_pos, scores, color=colors)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{t}\n({g})" for t, g in zip(titles, movie_genres)], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(RECOMMENDER_LABELS["similarity_score"], fontweight="bold")
        ax.set_title(f'Gatunek: {", ".join(genres)}', fontsize=10, fontweight="bold")

        # Add score labels
        for bar, score in zip(bars, scores):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"{score:.3f}", va="center", fontsize=7)

    plt.suptitle(RECOMMENDER_LABELS["genre_search"], fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Zapisano rekomendacje gatunkowe do {output_path}")


def test_user_profiles(engine, output_path: Path):
    """
    Test user profile-based recommendations.
    """
    set_thesis_style()

    print("\n  Testowanie profili użytkowników...")

    # Create sample user profiles based on movie preferences
    sample_profiles = [
        {
            "name": "Fan science fiction",
            "liked_movies": ["Matrix", "Inception", "Interstellar"],
        },
        {
            "name": "Fan komedii romantycznych",
            "liked_movies": ["Notting Hill", "Pretty Woman", "When Harry Met Sally"],
        },
        {
            "name": "Fan filmów akcji",
            "liked_movies": ["Die Hard", "Terminator", "Mad Max"],
        },
        {
            "name": "Fan animacji",
            "liked_movies": ["Toy Story", "Shrek", "Finding Nemo"],
        },
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for ax, profile in zip(axes, sample_profiles):
        # Find movie IDs for liked movies
        liked_ids = []
        liked_titles = []
        for movie_title in profile["liked_movies"]:
            ids = engine.catalog.search_by_title(movie_title, limit=1)
            if ids:
                liked_ids.append(ids[0])
                liked_titles.append(engine.catalog.get_title(ids[0]))

        if len(liked_ids) < 2:
            ax.text(0.5, 0.5, f"Za mało filmów dla profilu: {profile['name']}", ha="center", va="center")
            continue

        # Get recommendations based on profile using correct method
        results = engine.recommend_similar_to_movies(liked_ids, top_k=8)

        if not results:
            ax.text(0.5, 0.5, f"Brak rekomendacji dla: {profile['name']}", ha="center", va="center")
            continue

        # Extract data - format is (movie_id, score, info)
        titles = []
        scores = []
        genres = []

        for movie_id, score, info in results:
            title = engine.catalog.get_title(movie_id)
            genre = engine.catalog.get_genres(movie_id)
            titles.append(str(title)[:28] if title else f"Film {movie_id}")
            scores.append(score)
            genres.append(str(genre)[:18] if genre else "")

        # Plot
        y_pos = np.arange(len(titles))
        colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(titles)))
        bars = ax.barh(y_pos, scores, color=colors)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{t}\n({g})" for t, g in zip(titles, genres)], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(RECOMMENDER_LABELS["similarity_score"], fontweight="bold")

        # Create subtitle with liked movies
        liked_str = ", ".join([str(t)[:15] for t in liked_titles[:2]])
        ax.set_title(f'{profile["name"]}\nLubi: {liked_str}...', fontsize=10, fontweight="bold")

        # Add score labels
        for bar, score in zip(bars, scores):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"{score:.3f}", va="center", fontsize=7)

    plt.suptitle(RECOMMENDER_LABELS["user_profiles"], fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Zapisano profile użytkowników do {output_path}")


def test_method_comparison(engine, output_path: Path):
    """
    Compare different recommendation methods for the same query.
    """
    set_thesis_style()

    print("\n  Porównanie metod rekomendacji...")

    # Test movie
    test_movie = "Matrix"
    movie_ids = engine.catalog.search_by_title(test_movie, limit=1)

    if not movie_ids:
        print(f"    ⚠ Nie znaleziono filmu: {test_movie}")
        return

    query_id = movie_ids[0]
    full_title = engine.catalog.get_title(query_id)
    query_genres = engine.catalog.get_genres(query_id)

    # Get recommendations from different methods
    methods_results = {}

    # 1. Similar movies (embedding similarity)
    similar = engine.recommend_similar_to_movie(query_id, top_k=5)
    # Convert to (movie_id, score) tuples
    methods_results["Podobieństwo\nosadzeń"] = [(mid, score) for mid, score, _ in similar]

    # 2. Genre-based
    if query_genres:
        genre_list = query_genres.split(",") if "," in query_genres else query_genres.split("|")
        genre_list = [g.strip() for g in genre_list[:2]]
        genre_results = engine.recommend_by_genres(genre_list, top_k=5)
        methods_results["Według\ngatunku"] = [(mid, score) for mid, score, _ in genre_results]

    # 3. Smart search
    search_results, _ = engine.smart_search(test_movie, top_k=5)
    # Convert to same format - smart_search returns (mid, score, info)
    search_formatted = [(mid, abs(score)) for mid, score, _ in search_results]  # abs() because some may be negative
    methods_results["Inteligentne\nwyszukiwanie"] = search_formatted

    # Create comparison plot
    fig, ax = plt.subplots(figsize=(14, 8))

    n_methods = len(methods_results)
    n_movies = 5
    bar_width = 0.25

    x = np.arange(n_movies)

    for i, (method_name, results) in enumerate(methods_results.items()):
        if not results:
            continue

        titles = []
        scores = []
        for movie_id, score in results[:n_movies]:
            title = engine.catalog.get_title(movie_id)
            titles.append(str(title)[:20] if title else f"Film {movie_id}")
            scores.append(score if score > 0 else 0.5)  # Default for title matches

        # Pad if needed
        while len(scores) < n_movies:
            titles.append("")
            scores.append(0)

        offset = bar_width * (i - n_methods / 2 + 0.5)
        bars = ax.bar(x + offset, scores, bar_width, label=method_name, color=COLORS[i])

    ax.set_xlabel(RECOMMENDER_LABELS["movie_title"], fontweight="bold")
    ax.set_ylabel(RECOMMENDER_LABELS["similarity_score"], fontweight="bold")
    ax.set_title(f'{RECOMMENDER_LABELS["method_comparison"]}\nFilm bazowy: {full_title}', fontweight="bold", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"#{i+1}" for i in range(n_movies)])
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"  ✓ Zapisano porównanie metod do {output_path}")


def generate_all_recommender_tests(output_dir: Path):
    """
    Run all recommender engine tests and generate plots.

    Args:
        output_dir: Path to save plots
    """
    print("\n" + "=" * 80)
    print("Testowanie silnika rekomendacji")
    print("=" * 80)

    # Create output directory
    rec_dir = output_dir / "recommender"
    rec_dir.mkdir(parents=True, exist_ok=True)

    # Load engine
    print("\nŁadowanie silnika rekomendacji...")
    try:
        engine, catalog = load_recommender_engine()
    except Exception as e:
        print(f"⚠ Nie można załadować silnika rekomendacji: {e}")
        return

    print(f"  Załadowano {catalog.num_movies:,} filmów")

    # Run tests
    print("\nUruchamianie testów...")

    # 1. Search queries
    test_search_queries(engine, rec_dir / "wyniki_wyszukiwania.png")

    # 2. Similar movies
    test_similar_movies(engine, rec_dir / "podobne_filmy.png")

    # 3. Genre recommendations
    test_genre_recommendations(engine, rec_dir / "rekomendacje_gatunkowe.png")

    # 4. User profiles
    test_user_profiles(engine, rec_dir / "profile_uzytkownikow.png")

    # 5. Method comparison
    test_method_comparison(engine, rec_dir / "porownanie_metod.png")

    print("\n" + "=" * 80)
    print(f"✓ Wykresy testów silnika zapisane w {rec_dir}")
    print("=" * 80)


if __name__ == "__main__":
    PLOTS_DIR = PROJECT_ROOT / "model" / "plots"
    generate_all_recommender_tests(PLOTS_DIR)
