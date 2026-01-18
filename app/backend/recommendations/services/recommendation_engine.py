"""Main recommendation engine combining all services."""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from core.models import Movie
from django.conf import settings
from users.models import AppUser, Rating

from .qdrant_service import QdrantService
from .tf_serving_client import TFServingClient
from .tmdb_client import TMDBClient

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """
    Main recommendation engine that combines:
    - Two-Tower model (via TF Serving for user embeddings)
    - Qdrant (for movie embeddings and similarity search)
    - TMDB (for movie metadata)
    """

    _ml_to_tmdb_cache = None
    _tmdb_to_ml_cache = None

    def __init__(self):
        self.qdrant = QdrantService()
        self.tf_client = TFServingClient()
        self.tmdb = TMDBClient()
        self._user_id_map = None
        self._movie_id_map = None

    @classmethod
    def _load_ml_to_tmdb_mapping(cls) -> Dict[int, int]:
        """Load MovieLens ID to TMDB ID mapping from links.csv."""
        if cls._ml_to_tmdb_cache is not None:
            return cls._ml_to_tmdb_cache

        cls._ml_to_tmdb_cache = {}
        model_data_path = Path("/data/model_data")

        for dataset in ["ml-32m", "ml-latest-small"]:
            links_path = model_data_path / "raw" / dataset / "links.csv"
            if links_path.exists():
                logger.info(f"Loading ID mapping from {links_path}")
                with open(links_path, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        ml_id = int(row["movieId"])
                        tmdb_id = row.get("tmdbId", "")
                        if tmdb_id:
                            cls._ml_to_tmdb_cache[ml_id] = int(tmdb_id)
                logger.info(f"Loaded {len(cls._ml_to_tmdb_cache)} ID mappings")
                break

        return cls._ml_to_tmdb_cache

    def _ml_to_tmdb(self, ml_id: int) -> Optional[int]:
        """Convert MovieLens ID to TMDB ID."""
        mapping = self._load_ml_to_tmdb_mapping()
        return mapping.get(ml_id)

    @classmethod
    def _load_tmdb_to_ml_mapping(cls) -> Dict[int, int]:
        """Load TMDB ID to MovieLens ID reverse mapping."""
        if cls._tmdb_to_ml_cache is not None:
            return cls._tmdb_to_ml_cache

        ml_to_tmdb = cls._load_ml_to_tmdb_mapping()
        cls._tmdb_to_ml_cache = {v: k for k, v in ml_to_tmdb.items()}
        return cls._tmdb_to_ml_cache

    def _tmdb_to_ml(self, tmdb_id: int) -> Optional[int]:
        """Convert TMDB ID to MovieLens ID."""
        mapping = self._load_tmdb_to_ml_mapping()
        return mapping.get(tmdb_id)

    def _load_id_mappings(self):
        """Load user and movie ID mappings."""
        if self._user_id_map is not None:
            return

        embeddings_path = Path(settings.EMBEDDINGS_PATH)

        metadata_path = embeddings_path / "movie_embeddings_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                movie_ids = metadata.get("movie_ids", [])
                self._movie_id_map = {mid: idx for idx, mid in enumerate(movie_ids)}
        else:
            self._movie_id_map = {}

        self._user_id_map = {}

    def get_for_you_recommendations(
        self,
        user: AppUser,
        limit: int = 20,
        exclude_rated: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Get personalized recommendations for a user.

        Uses the Two-Tower model:
        1. Get user embedding from TF Serving
        2. Search Qdrant for similar movies
        """
        self._load_id_mappings()

        exclude_ids = []
        if exclude_rated:
            exclude_ids = list(user.ratings.values_list("tmdb_id", flat=True))

        if user.movielens_user_id and not user.is_cold_start:
            user_embedding = self.tf_client.get_user_embedding(user.movielens_user_id)
        else:
            user_embedding = self._get_cold_start_embedding(user)

        if not user_embedding:
            return self._get_popular_fallback(limit, exclude_ids)

        results = self.qdrant.search_similar(
            query_vector=user_embedding,
            limit=limit,
            exclude_ids=exclude_ids,
        )

        return self._enrich_with_metadata(results)

    def _get_cold_start_embedding(self, user: AppUser) -> Optional[List[float]]:
        """
        Generate embedding for cold-start user based on preferences and ratings.

        Strategy:
        1. Get embeddings of rated movies
        2. Weight by rating
        3. Average to create user profile
        """
        ratings = user.ratings.all()

        if not ratings.exists():
            preferences = user.preferences.filter(preference_type="genre")
            if not preferences.exists():
                return None

            return None

        movie_ids = [r.tmdb_id for r in ratings]
        embeddings = self.qdrant.get_multiple_embeddings(movie_ids)

        if not embeddings:
            return None

        weighted_sum = np.zeros(128)
        total_weight = 0

        for rating in ratings:
            if rating.tmdb_id in embeddings:
                weight = rating.rating / 5.0
                weighted_sum += np.array(embeddings[rating.tmdb_id]) * weight
                total_weight += weight

        if total_weight == 0:
            return None

        user_embedding = weighted_sum / total_weight
        user_embedding = user_embedding / np.linalg.norm(user_embedding)

        return user_embedding.tolist()

    def get_similar_movies(
        self,
        movie_id: int,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get movies similar to a given movie using vector similarity."""
        ml_id = self._tmdb_to_ml(movie_id)

        if not ml_id:
            logger.info(f"No MovieLens ID found for TMDB ID {movie_id}, falling back to TMDB")
            return self.tmdb.get_similar_movies(movie_id)

        movie_embedding = self.qdrant.get_movie_embedding(ml_id)

        if not movie_embedding:
            logger.info(f"No embedding found for ML ID {ml_id}, falling back to TMDB")
            return self.tmdb.get_similar_movies(movie_id)

        results = self.qdrant.search_similar(
            query_vector=movie_embedding,
            limit=limit + 1,
            exclude_ids=[ml_id],
        )

        return self._enrich_with_metadata(results[:limit])

    def get_because_you_watched(
        self,
        user: AppUser,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get recommendations based on user's recently watched movies.

        Returns movies similar to the most recently watched/rated movie.
        """
        recent_rating = user.ratings.order_by("-created_at").first()

        if not recent_rating:
            return []

        similar = self.get_similar_movies(recent_rating.tmdb_id, limit=limit)

        return {
            "based_on": recent_rating.tmdb_id,
            "movies": similar,
        }

    def get_by_genre(
        self,
        genre: str,
        user: Optional[AppUser] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Get recommendations filtered by genre."""
        exclude_ids = []
        if user:
            exclude_ids = list(user.ratings.values_list("tmdb_id", flat=True))

        genre_map = {
            "action": 28,
            "adventure": 12,
            "animation": 16,
            "comedy": 35,
            "crime": 80,
            "documentary": 99,
            "drama": 18,
            "family": 10751,
            "fantasy": 14,
            "history": 36,
            "horror": 27,
            "music": 10402,
            "mystery": 9648,
            "romance": 10749,
            "science fiction": 878,
            "thriller": 53,
            "war": 10752,
            "western": 37,
        }

        genre_id = genre_map.get(genre.lower())
        if not genre_id:
            return []

        return self.tmdb.get_movies_by_genre(genre_id)

    def get_trending(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get trending/popular movies."""
        return self.tmdb.get_popular_movies()[:limit]

    def _get_popular_fallback(
        self,
        limit: int,
        exclude_ids: List[int],
    ) -> List[Dict[str, Any]]:
        """Fallback to popular movies when personalization fails."""
        popular = self.tmdb.get_popular_movies()

        filtered = [m for m in popular if m["tmdb_id"] not in exclude_ids]

        return filtered[:limit]

    def _get_year(self, release_date) -> Optional[int]:
        """Extract year from release_date (handles both string and date objects)."""
        if not release_date:
            return None
        if isinstance(release_date, str):
            try:
                return int(release_date[:4])
            except (ValueError, IndexError):
                return None
        if hasattr(release_date, "year"):
            return release_date.year
        return None

    def _enrich_with_metadata(
        self,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Enrich Qdrant results with movie metadata."""
        enriched = []

        for result in results:
            ml_id = result["tmdb_id"]
            tmdb_id = self._ml_to_tmdb(ml_id)

            if not tmdb_id:
                logger.warning(f"No TMDB mapping for MovieLens ID {ml_id}")
                continue

            try:
                movie = Movie.objects.get(tmdb_id=tmdb_id)
                enriched.append(
                    {
                        "tmdb_id": movie.tmdb_id,
                        "title": movie.title,
                        "poster_url": f"https://image.tmdb.org/t/p/w342{movie.poster_path}" if movie.poster_path else None,
                        "year": self._get_year(movie.release_date),
                        "vote_average": movie.vote_average,
                        "genres": movie.genres,
                        "score": result.get("score"),
                    }
                )
            except Movie.DoesNotExist:
                movie_data = self.tmdb.get_movie(tmdb_id)
                if movie_data:
                    movie, _ = Movie.objects.get_or_create(
                        tmdb_id=movie_data["id"],
                        defaults={
                            "title": movie_data.get("title", ""),
                            "overview": movie_data.get("overview"),
                            "poster_path": movie_data.get("poster_path"),
                            "backdrop_path": movie_data.get("backdrop_path"),
                            "release_date": movie_data.get("release_date") or None,
                            "vote_average": movie_data.get("vote_average", 0),
                            "genres": [g["name"] for g in movie_data.get("genres", [])],
                            "popularity": movie_data.get("popularity", 0),
                        },
                    )
                    enriched.append(
                        {
                            "tmdb_id": movie.tmdb_id,
                            "title": movie.title,
                            "poster_url": f"https://image.tmdb.org/t/p/w342{movie.poster_path}" if movie.poster_path else None,
                            "year": self._get_year(movie.release_date),
                            "vote_average": movie.vote_average,
                            "genres": movie.genres,
                            "score": result.get("score"),
                        }
                    )
                else:
                    logger.warning(f"Could not fetch TMDB data for ID {tmdb_id} (ML ID: {ml_id})")

        return enriched

    def get_seed_movies_for_onboarding(self, count: int = 20) -> List[Dict[str, Any]]:
        """Get diverse, popular movies for new user onboarding."""
        popular = self.tmdb.get_popular_movies()
        top_rated = self.tmdb.get_top_rated_movies()

        seen = set()
        movies = []

        for movie in popular + top_rated:
            if movie["tmdb_id"] not in seen:
                seen.add(movie["tmdb_id"])
                movies.append(movie)

            if len(movies) >= count:
                break

        return movies
