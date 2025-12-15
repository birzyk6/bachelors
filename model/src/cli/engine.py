"""
Recommendation engine using Two-Tower embeddings.

Provides high-level recommendation methods combining embeddings and metadata.
"""

from typing import List, Optional, Set, Tuple

import numpy as np

from .embedding_store import EmbeddingStore
from .movie_catalog import MovieCatalog


class RecommendationEngine:
    """
    High-level recommendation engine combining embeddings and movie catalog.

    Provides semantic search, genre-based queries, and similar movie finding.
    """

    def __init__(self, embedding_store: EmbeddingStore, movie_catalog: MovieCatalog):
        """
        Initialize recommendation engine.

        Args:
            embedding_store: Loaded embedding store
            movie_catalog: Loaded movie catalog
        """
        self.embeddings = embedding_store
        self.catalog = movie_catalog

    def smart_search(self, query: str, top_k: int = 10) -> Tuple[List[Tuple[int, float, str]], str]:
        """
        Intelligent search that interprets any query.

        Works like a search engine - accepts any text and finds relevant movies.

        Strategy:
        1. Search for movies matching the query text (title match)
        2. Parse query for genre keywords
        3. If title matches found, use their embeddings to find similar movies
        4. If genres found, include genre-based recommendations
        5. Combine and rank results

        Args:
            query: Any text query (title, genres, description, etc.)
            top_k: Number of recommendations

        Returns:
            Tuple of (results list, explanation string)
        """
        query = query.strip()
        if not query:
            return [], "Empty query"

        # Find title matches
        title_matches = self.catalog.search_by_title(query, limit=20)
        title_matches_with_embeddings = [mid for mid in title_matches if mid in self.embeddings.embeddings]

        # Parse for genres
        genres = self.parse_genre_query(query)

        # Determine search strategy
        explanation_parts = []
        all_results = {}  # movie_id -> (score, source)

        # Strategy 1: If we found exact or partial title matches
        if title_matches_with_embeddings:
            # Use matched movies to find similar ones
            matched_titles = [self.catalog.get_title(mid) for mid in title_matches_with_embeddings[:3]]
            explanation_parts.append(f"Found: {', '.join(matched_titles[:2])}")

            # Get similar movies based on the matches
            centroid = self.embeddings.compute_centroid(title_matches_with_embeddings[:5])
            if centroid is not None:
                similar = self.embeddings.find_similar(
                    centroid, top_k=top_k * 2, exclude_ids=set(title_matches_with_embeddings[:5])
                )
                for mid, score in similar:
                    if mid not in all_results or score > all_results[mid][0]:
                        all_results[mid] = (score, "similar")

        # Strategy 2: If we found genre keywords
        if genres:
            explanation_parts.append(f"Genres: {', '.join(genres)}")

            # Get genre-matching movies
            genre_movies = self.catalog.get_movies_by_genres(genres, match_all=False)
            genre_movies_with_embeddings = [mid for mid in genre_movies if mid in self.embeddings.embeddings][:100]

            if genre_movies_with_embeddings:
                centroid = self.embeddings.compute_centroid(genre_movies_with_embeddings)
                if centroid is not None:
                    similar = self.embeddings.find_similar(
                        centroid, top_k=top_k * 2, exclude_ids=set(genre_movies_with_embeddings)
                    )
                    for mid, score in similar:
                        # Boost score if also matched by title
                        boost = 0.1 if mid in all_results else 0
                        new_score = score + boost
                        if mid not in all_results or new_score > all_results[mid][0]:
                            all_results[mid] = (new_score, "genre")

        # Strategy 3: If no matches, try fuzzy/partial matching
        if not title_matches_with_embeddings and not genres:
            # Try word-by-word matching
            words = query.lower().split()
            partial_matches = set()

            for word in words:
                if len(word) >= 3:  # Skip very short words
                    matches = self.catalog.search_by_title(word, limit=10)
                    partial_matches.update(matches)

            partial_with_embeddings = [mid for mid in partial_matches if mid in self.embeddings.embeddings]

            if partial_with_embeddings:
                explanation_parts.append(f"Partial matches for: {query}")
                centroid = self.embeddings.compute_centroid(partial_with_embeddings[:10])
                if centroid is not None:
                    similar = self.embeddings.find_similar(
                        centroid, top_k=top_k * 2, exclude_ids=set(partial_with_embeddings[:5])
                    )
                    for mid, score in similar:
                        if mid not in all_results or score > all_results[mid][0]:
                            all_results[mid] = (score, "partial")

        # Build explanation
        if explanation_parts:
            explanation = " | ".join(explanation_parts)
        else:
            explanation = f"No direct matches for '{query}'"

        # Sort and format results
        sorted_results = sorted(all_results.items(), key=lambda x: x[1][0], reverse=True)[:top_k]

        results = []
        for movie_id, (score, source) in sorted_results:
            info = self.catalog.format_movie(movie_id)
            results.append((movie_id, score, info))

        # If still no results, return some popular/random movies as fallback
        if not results:
            explanation = f"No matches found for '{query}' - showing sample movies"
            sample_ids = self.embeddings.movie_ids[:top_k]
            for mid in sample_ids:
                info = self.catalog.format_movie(mid)
                results.append((mid, 0.0, info))

        return results, explanation

    def recommend_by_genres(
        self, genres: List[str], top_k: int = 10, exclude_query_movies: bool = True
    ) -> List[Tuple[int, float, str]]:
        """
        Get recommendations based on genre query.

        Finds movies matching the genres, computes their centroid embedding,
        then finds similar movies.

        Args:
            genres: List of genres to match
            top_k: Number of recommendations
            exclude_query_movies: Whether to exclude movies used in query

        Returns:
            List of (movie_id, score, formatted_info) tuples
        """
        # Find movies matching genres
        matching_ids = self.catalog.get_movies_by_genres(genres, match_all=False)

        # Filter to movies with embeddings
        matching_ids = [mid for mid in matching_ids if mid in self.embeddings.embeddings]

        if not matching_ids:
            return []

        # Compute centroid of matching movies (use up to 100)
        sample_ids = matching_ids[:100]
        centroid = self.embeddings.compute_centroid(sample_ids)

        if centroid is None:
            return []

        # Find similar movies
        exclude = set(sample_ids) if exclude_query_movies else set()
        similar = self.embeddings.find_similar(centroid, top_k=top_k, exclude_ids=exclude)

        # Format results
        results = []
        for movie_id, score in similar:
            info = self.catalog.format_movie(movie_id)
            results.append((movie_id, score, info))

        return results

    def recommend_similar_to_movie(self, movie_id: int, top_k: int = 10) -> List[Tuple[int, float, str]]:
        """
        Find movies similar to a given movie.

        Args:
            movie_id: Source movie ID
            top_k: Number of recommendations

        Returns:
            List of (movie_id, score, formatted_info) tuples
        """
        similar = self.embeddings.find_similar_to_movie(movie_id, top_k=top_k)

        results = []
        for mid, score in similar:
            info = self.catalog.format_movie(mid)
            results.append((mid, score, info))

        return results

    def recommend_similar_to_movies(
        self, movie_ids: List[int], top_k: int = 10, exclude_source: bool = True
    ) -> List[Tuple[int, float, str]]:
        """
        Find movies similar to a set of movies (e.g., user favorites).

        Args:
            movie_ids: List of source movie IDs
            top_k: Number of recommendations
            exclude_source: Whether to exclude source movies from results

        Returns:
            List of (movie_id, score, formatted_info) tuples
        """
        centroid = self.embeddings.compute_centroid(movie_ids)

        if centroid is None:
            return []

        exclude = set(movie_ids) if exclude_source else set()
        similar = self.embeddings.find_similar(centroid, top_k=top_k, exclude_ids=exclude)

        results = []
        for mid, score in similar:
            info = self.catalog.format_movie(mid)
            results.append((mid, score, info))

        return results

    def search_movies(self, query: str, limit: int = 20) -> List[Tuple[int, str]]:
        """
        Search movies by title.

        Args:
            query: Title search query
            limit: Maximum results

        Returns:
            List of (movie_id, formatted_info) tuples
        """
        movie_ids = self.catalog.search_by_title(query, limit=limit)

        results = []
        for mid in movie_ids:
            info = self.catalog.format_movie(mid)
            has_embedding = mid in self.embeddings.embeddings
            status = "✓" if has_embedding else "✗"
            results.append((mid, f"{status} {info}"))

        return results

    def parse_genre_query(self, query: str) -> List[str]:
        """
        Parse a genre query string into genre names.

        Handles various formats:
        - "action sci-fi" -> ["Action", "Sci-Fi"]
        - "romantic comedy" -> ["Romance", "Comedy"]
        - "horror, thriller" -> ["Horror", "Thriller"]

        Args:
            query: Genre query string

        Returns:
            List of matched genre names
        """
        available_genres = self.catalog.genres

        # Common aliases
        aliases = {
            "scifi": "Sci-Fi",
            "sci-fi": "Sci-Fi",
            "science fiction": "Sci-Fi",
            "romantic": "Romance",
            "romcom": "Romance",
            "kids": "Children",
            "animated": "Animation",
            "cartoons": "Animation",
            "scary": "Horror",
            "suspense": "Thriller",
            "funny": "Comedy",
            "docs": "Documentary",
            "docu": "Documentary",
            "war": "War",
            "noir": "Film-Noir",
            "film noir": "Film-Noir",
            "western": "Western",
            "westerns": "Western",
            "fantasy": "Fantasy",
            "adventures": "Adventure",
            "mysteries": "Mystery",
            "dramas": "Drama",
            "thrillers": "Thriller",
            "horrors": "Horror",
            "comedies": "Comedy",
            "musicals": "Musical",
        }

        # Normalize query
        query_lower = query.lower()

        # Replace common separators
        for sep in [",", "&", "+", "and", "/"]:
            query_lower = query_lower.replace(sep, " ")

        # Split into words
        words = query_lower.split()

        matched_genres = []

        for word in words:
            word = word.strip()
            if not word:
                continue

            # Check aliases first
            if word in aliases:
                genre = aliases[word]
                if genre not in matched_genres:
                    matched_genres.append(genre)
                continue

            # Check direct match (case-insensitive)
            for genre in available_genres:
                if genre.lower() == word or genre.lower().startswith(word):
                    if genre not in matched_genres:
                        matched_genres.append(genre)
                    break

        return matched_genres
