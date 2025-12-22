"""
Recommendation engine using Two-Tower embeddings.

Provides high-level recommendation methods combining embeddings and metadata.
"""

import math
from typing import List, Optional, Set, Tuple

import numpy as np

from .embedding_store import EmbeddingStore
from .movie_catalog import MovieCatalog


class RecommendationEngine:
    """
    High-level recommendation engine combining embeddings and movie catalog.

    Provides semantic search, genre-based queries, and similar movie finding.
    Uses popularity and ratings to improve relevance.
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

    def _compute_popularity_boost(self, movie_id: int) -> float:
        """
        Compute a popularity boost factor for ranking.

        Uses log-scaled popularity and vote count to avoid domination by
        extremely popular movies while still boosting well-known films.

        Returns a value between 0 and ~0.3 to add to similarity scores.
        """
        popularity = self.catalog.get_popularity(movie_id)
        vote_count = self.catalog.get_vote_count(movie_id)
        vote_avg = self.catalog.get_vote_average(movie_id)

        # Log-scale popularity (typical range 0-500, some outliers to 1000+)
        # log10(100) = 2, log10(1000) = 3
        pop_score = math.log10(max(popularity, 1) + 1) / 4.0  # Normalize to ~0-0.15

        # Vote count factor (more votes = more reliable)
        # log10(1000) = 3, log10(10000) = 4
        vote_score = math.log10(max(vote_count, 1) + 1) / 5.0  # Normalize to ~0-0.1

        # Rating factor (only boost good movies, 7+ out of 10)
        rating_boost = max(0, (vote_avg - 6.0)) / 20.0  # 0-0.2 for ratings 6-10

        # Combine (max boost ~0.3)
        return min(0.3, pop_score + vote_score + rating_boost)

    def smart_search(self, query: str, top_k: int = 10) -> Tuple[List[Tuple[int, float, str]], str]:
        """
        Intelligent search that interprets any query.

        Works like a search engine - accepts any text and finds relevant movies.

        Strategy:
        1. Search for movies matching the query text (title match)
        2. Parse query for genre keywords
        3. If title matches found, include them AND find similar movies
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

            # FIRST: Include the matched movies themselves with high scores
            for i, mid in enumerate(title_matches_with_embeddings[:top_k]):
                # Give matched movies a high score (1.0 for best match, decreasing)
                match_score = 1.0 - (i * 0.02)  # Slight decrease for less exact matches
                all_results[mid] = (match_score, "match")

            # THEN: Get similar movies based on the matches
            centroid = self.embeddings.compute_centroid(title_matches_with_embeddings[:5])
            if centroid is not None:
                similar = self.embeddings.find_similar(
                    centroid, top_k=top_k * 2, exclude_ids=set(title_matches_with_embeddings[:10])
                )
                for mid, score in similar:
                    # Similar movies get their actual similarity score (< 1.0)
                    if mid not in all_results:
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

        # Separate title matches from similar movies
        # Title matches ALWAYS come first, sorted by popularity among themselves
        # Similar movies come after, sorted by score + popularity boost
        title_matches = []
        similar_movies = []

        for movie_id, (score, source) in all_results.items():
            pop_boost = self._compute_popularity_boost(movie_id)
            raw_popularity = self.catalog.get_popularity(movie_id)
            if source == "match":
                # Title matches: use raw popularity as primary sort key
                title_matches.append((movie_id, score, pop_boost, raw_popularity))
            else:
                # Similar movies: add popularity boost to similarity score
                final_score = score + pop_boost
                similar_movies.append((movie_id, final_score, source))

        # Sort title matches by raw popularity (highest first)
        title_matches.sort(key=lambda x: x[3], reverse=True)

        # Sort similar movies by boosted score
        similar_movies.sort(key=lambda x: x[1], reverse=True)

        # Combine: title matches first, then similar movies
        results = []

        # Add title matches (up to top_k)
        # Use negative score to indicate "match" (display will show ★ instead of similarity)
        for movie_id, score, pop_boost, raw_pop in title_matches[:top_k]:
            info = self.catalog.format_movie(movie_id)
            # Use -1 as marker for "title match", store popularity for display
            results.append((movie_id, -raw_pop, info))

        # Fill remaining slots with similar movies
        remaining = top_k - len(results)
        for movie_id, score, source in similar_movies[:remaining]:
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

    def recommend_for_user(
        self,
        user_embedding: np.ndarray,
        exclude_movie_ids: Optional[Set[int]] = None,
        top_k: int = 10,
    ) -> List[Tuple[int, float, str]]:
        """
        Get recommendations for a user based on their embedding.

        Uses the user's learned embedding from the Two-Tower model to find
        movies with similar embeddings (dot product / cosine similarity).

        Args:
            user_embedding: User's 128-dim embedding vector
            exclude_movie_ids: Movie IDs to exclude (e.g., already watched)
            top_k: Number of recommendations

        Returns:
            List of (movie_id, score, formatted_info) tuples
        """
        if user_embedding is None:
            return []

        exclude_ids = exclude_movie_ids or set()

        # Find movies similar to user embedding
        similar = self.embeddings.find_similar(
            user_embedding, top_k=top_k + len(exclude_ids), exclude_ids=exclude_ids  # Get extra in case we exclude many
        )

        # Add popularity boost
        results_with_boost = []
        for movie_id, score in similar:
            pop_boost = self._compute_popularity_boost(movie_id)
            final_score = score + pop_boost
            results_with_boost.append((movie_id, final_score))

        # Re-sort by boosted score
        results_with_boost.sort(key=lambda x: x[1], reverse=True)

        # Format results
        results = []
        for movie_id, score in results_with_boost[:top_k]:
            info = self.catalog.format_movie(movie_id)
            results.append((movie_id, score, info))

        return results

    def recommend_for_user_with_theme(
        self,
        user_embedding: np.ndarray,
        theme_query: str,
        exclude_movie_ids: Optional[Set[int]] = None,
        top_k: int = 10,
    ) -> List[Tuple[int, float, str]]:
        """
        Get themed recommendations combining user taste with a theme/genre.

        Strategy: Find ALL movies matching the theme, then rank them by
        how well they match the user's taste. This ensures we always get
        theme-relevant results.

        Args:
            user_embedding: User's 128-dim embedding vector
            theme_query: Theme or genre query (e.g., "vampires", "space", "comedy")
            exclude_movie_ids: Movie IDs to exclude
            top_k: Number of recommendations

        Returns:
            List of (movie_id, score, formatted_info) tuples
        """
        if user_embedding is None:
            return []

        exclude_ids = exclude_movie_ids or set()

        # Find movies matching the theme from multiple sources
        theme_matches = set()

        # 1. Title matches (search for theme keyword in titles)
        title_matches = self.catalog.search_by_title(theme_query, limit=100)
        for mid in title_matches:
            if mid in self.embeddings.embeddings and mid not in exclude_ids:
                theme_matches.add(mid)

        # 2. Overview/description matches
        overview_matches = self.catalog.search_by_overview(theme_query, limit=100)
        for mid in overview_matches:
            if mid in self.embeddings.embeddings and mid not in exclude_ids:
                theme_matches.add(mid)

        # 3. Genre matches (if theme is a genre like "comedy", "horror")
        genres = self.parse_genre_query(theme_query)
        if genres:
            genre_movies = list(self.catalog.get_movies_by_genres(genres, match_all=False))
            for mid in genre_movies[:500]:
                if mid in self.embeddings.embeddings and mid not in exclude_ids:
                    theme_matches.add(mid)

        # 4. Find semantically similar movies to the theme matches (expand the pool)
        if theme_matches:
            theme_centroid = self.embeddings.compute_centroid(list(theme_matches)[:50])
            if theme_centroid is not None:
                similar_to_theme = self.embeddings.find_similar(theme_centroid, top_k=200, exclude_ids=exclude_ids)
                for mid, sim in similar_to_theme:
                    if sim > 0.5:  # Only add highly similar movies
                        theme_matches.add(mid)

        theme_matches_list = list(theme_matches)

        if not theme_matches_list:
            # Fallback to general user recommendations
            return self.recommend_for_user(user_embedding, exclude_ids, top_k)

        # Score theme movies by user similarity
        results = []
        for movie_id in theme_matches_list:
            movie_emb = self.embeddings.embeddings.get(movie_id)
            if movie_emb is not None:
                # How well does this movie match the user's taste?
                user_similarity = float(np.dot(user_embedding, movie_emb))
                pop_boost = self._compute_popularity_boost(movie_id)
                final_score = user_similarity + pop_boost
                results.append((movie_id, final_score))

        # Sort by how well theme movies match user taste
        results.sort(key=lambda x: x[1], reverse=True)

        # Format results
        formatted = []
        for movie_id, score in results[:top_k]:
            info = self.catalog.format_movie(movie_id)
            formatted.append((movie_id, score, info))

        return formatted

    def recommend_popular_for_user(
        self,
        user_embedding: np.ndarray,
        exclude_movie_ids: Optional[Set[int]] = None,
        top_k: int = 10,
        min_popularity: float = 50.0,
    ) -> List[Tuple[int, float, str]]:
        """
        Get popular movies that match user's taste.

        Recommends well-known, highly-rated movies the user would likely enjoy.

        Args:
            user_embedding: User's 128-dim embedding vector
            exclude_movie_ids: Movie IDs to exclude
            top_k: Number of recommendations
            min_popularity: Minimum popularity score to consider

        Returns:
            List of (movie_id, score, formatted_info) tuples
        """
        if user_embedding is None:
            return []

        exclude_ids = exclude_movie_ids or set()

        # Get all movies with high popularity
        popular_movies = []
        for movie_id in self.embeddings.movie_ids:
            if movie_id in exclude_ids:
                continue
            pop = self.catalog.get_popularity(movie_id)
            vote_avg = self.catalog.get_vote_average(movie_id)
            if pop >= min_popularity and vote_avg >= 6.5:
                popular_movies.append(movie_id)

        if not popular_movies:
            return self.recommend_for_user(user_embedding, exclude_ids, top_k)

        # Score popular movies by similarity to user
        results = []
        for movie_id in popular_movies:
            movie_emb = self.embeddings.embeddings.get(movie_id)
            if movie_emb is not None:
                similarity = np.dot(user_embedding, movie_emb)
                pop = self.catalog.get_popularity(movie_id)
                # Weight more heavily by popularity for this mode
                pop_boost = math.log10(max(pop, 1) + 1) / 3.0
                final_score = similarity + pop_boost
                results.append((movie_id, final_score))

        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)

        # Format results
        formatted = []
        for movie_id, score in results[:top_k]:
            info = self.catalog.format_movie(movie_id)
            formatted.append((movie_id, score, info))

        return formatted

    def recommend_based_on_movie(
        self,
        source_movie_id: int,
        user_embedding: Optional[np.ndarray] = None,
        exclude_movie_ids: Optional[Set[int]] = None,
        top_k: int = 10,
    ) -> List[Tuple[int, float, str]]:
        """
        Get recommendations based on a specific movie, optionally personalized.

        If user_embedding provided, blends movie similarity with user taste.

        Args:
            source_movie_id: Movie to base recommendations on
            user_embedding: Optional user embedding for personalization
            exclude_movie_ids: Movie IDs to exclude
            top_k: Number of recommendations

        Returns:
            List of (movie_id, score, formatted_info) tuples
        """
        exclude_ids = set(exclude_movie_ids) if exclude_movie_ids else set()
        exclude_ids.add(source_movie_id)

        movie_emb = self.embeddings.embeddings.get(source_movie_id)
        if movie_emb is None:
            return []

        # Optionally blend with user embedding
        if user_embedding is not None:
            # 60% movie, 40% user taste
            blended = 0.6 * movie_emb + 0.4 * user_embedding
            blended = blended / np.linalg.norm(blended)
            search_emb = blended
        else:
            search_emb = movie_emb

        similar = self.embeddings.find_similar(search_emb, top_k=top_k, exclude_ids=exclude_ids)

        results = []
        for mid, score in similar:
            pop_boost = self._compute_popularity_boost(mid)
            info = self.catalog.format_movie(mid)
            results.append((mid, score + pop_boost, info))

        return results

    def recommend_discover(
        self,
        user_embedding: np.ndarray,
        user_top_genres: List[str],
        exclude_movie_ids: Optional[Set[int]] = None,
        top_k: int = 10,
    ) -> List[Tuple[int, float, str]]:
        """
        Recommend movies outside the user's comfort zone.

        Finds highly-rated movies in genres the user doesn't usually watch,
        but that still have some similarity to their taste.

        Args:
            user_embedding: User's embedding vector
            user_top_genres: User's most-watched genres (to avoid)
            exclude_movie_ids: Movie IDs to exclude
            top_k: Number of recommendations

        Returns:
            List of (movie_id, score, formatted_info) tuples
        """
        if user_embedding is None:
            return []

        exclude_ids = exclude_movie_ids or set()

        # Find movies NOT in user's top genres but still good quality
        candidates = []
        for movie_id in self.embeddings.movie_ids:
            if movie_id in exclude_ids:
                continue

            # Check genres
            movie_genres = self.catalog.get_genres_list(movie_id)
            if not movie_genres:
                continue

            # Skip if movie is primarily in user's top genres
            overlap = set(movie_genres) & set(user_top_genres)
            if len(overlap) >= len(movie_genres) * 0.5:
                continue

            # Must be decent quality (relaxed criteria)
            vote_avg = self.catalog.get_vote_average(movie_id)
            vote_count = self.catalog.get_vote_count(movie_id)
            pop = self.catalog.get_popularity(movie_id)
            # Accept movies with good ratings OR good popularity
            if (vote_avg >= 6.5 and vote_count >= 50) or pop >= 20:
                candidates.append(movie_id)

        if not candidates:
            # Fallback
            return self.recommend_for_user(user_embedding, exclude_ids, top_k)

        # Score by similarity (want some relevance, but not too much)
        results = []
        for movie_id in candidates[:1000]:  # Increased limit for more diversity
            movie_emb = self.embeddings.embeddings.get(movie_id)
            if movie_emb is not None:
                similarity = np.dot(user_embedding, movie_emb)
                # We want movies that are somewhat relevant (relaxed range)
                # Not too similar (already in comfort zone) or too different (irrelevant)
                if 0.15 <= similarity <= 0.85:
                    # Boost higher-rated movies
                    vote_avg = self.catalog.get_vote_average(movie_id)
                    quality_boost = (vote_avg - 5.0) / 10.0
                    final_score = similarity + quality_boost
                    results.append((movie_id, final_score))

        results.sort(key=lambda x: x[1], reverse=True)

        formatted = []
        for movie_id, score in results[:top_k]:
            info = self.catalog.format_movie(movie_id)
            formatted.append((movie_id, score, info))

        return formatted

    def recommend_hidden_gems(
        self,
        user_embedding: np.ndarray,
        exclude_movie_ids: Optional[Set[int]] = None,
        top_k: int = 10,
    ) -> List[Tuple[int, float, str]]:
        """
        Recommend lesser-known but highly-rated movies matching user taste.

        Finds movies with good ratings but low popularity (hidden gems).

        Args:
            user_embedding: User's embedding vector
            exclude_movie_ids: Movie IDs to exclude
            top_k: Number of recommendations

        Returns:
            List of (movie_id, score, formatted_info) tuples
        """
        if user_embedding is None:
            return []

        exclude_ids = exclude_movie_ids or set()

        # Find hidden gems: decent rating, low popularity
        # Use popularity percentile since vote_count is often 0 in combined dataset
        candidates = []

        # Get popularity distribution for threshold
        popularities = [self.catalog.get_popularity(mid) for mid in self.embeddings.movie_ids[:5000]]
        pop_median = np.median(popularities)
        pop_threshold = max(pop_median * 0.8, 15.0)  # Below median or 15, whichever is higher

        for movie_id in self.embeddings.movie_ids:
            if movie_id in exclude_ids:
                continue

            pop = self.catalog.get_popularity(movie_id)
            vote_avg = self.catalog.get_vote_average(movie_id)

            # Hidden gem criteria: good rating, below-average popularity
            # Skip movies with no ratings data (vote_avg = 0)
            if vote_avg >= 6.5 and 0 < pop < pop_threshold:
                candidates.append(movie_id)

        if not candidates:
            # Try even more relaxed criteria
            for movie_id in self.embeddings.movie_ids[:10000]:
                if movie_id in exclude_ids:
                    continue
                pop = self.catalog.get_popularity(movie_id)
                vote_avg = self.catalog.get_vote_average(movie_id)
                if vote_avg >= 6.0 and 0 < pop < pop_threshold * 2:
                    candidates.append(movie_id)

        if not candidates:
            return self.recommend_for_user(user_embedding, exclude_ids, top_k)

        # Score by similarity to user
        results = []
        for movie_id in candidates[:1000]:
            movie_emb = self.embeddings.embeddings.get(movie_id)
            if movie_emb is not None:
                similarity = np.dot(user_embedding, movie_emb)
                if similarity > 0.2:
                    vote_avg = self.catalog.get_vote_average(movie_id)
                    pop = self.catalog.get_popularity(movie_id)
                    # Boost quality but also uniqueness (lower popularity = better gem)
                    quality_boost = (vote_avg - 5.0) / 10.0
                    rarity_boost = (pop_threshold - min(pop, pop_threshold)) / (pop_threshold * 5)
                    results.append((movie_id, similarity + quality_boost + rarity_boost))

        results.sort(key=lambda x: x[1], reverse=True)

        formatted = []
        for movie_id, score in results[:top_k]:
            info = self.catalog.format_movie(movie_id)
            formatted.append((movie_id, score, info))

        return formatted
