"""TMDB API client."""

import logging
from typing import Any, Dict, List, Optional

import requests
from django.conf import settings

logger = logging.getLogger(__name__)


class TMDBClient:
    """Client for The Movie Database (TMDB) API."""

    BASE_URL = "https://api.themoviedb.org/3"
    IMAGE_BASE_URL = "https://image.tmdb.org/t/p"

    def __init__(self):
        self.api_key = settings.TMDB_API_KEY
        if not self.api_key:
            logger.warning("TMDB_API_KEY not set")

    def _get(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make GET request to TMDB API."""
        if not self.api_key:
            return None

        url = f"{self.BASE_URL}{endpoint}"
        params = params or {}
        params["api_key"] = self.api_key

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"TMDB API error: {e}")
            return None

    def get_movie(self, movie_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed movie information."""
        data = self._get(f"/movie/{movie_id}", params={"append_to_response": "keywords,credits"})
        return data

    def search_movies(self, query: str, page: int = 1) -> List[Dict[str, Any]]:
        """Search for movies by title."""
        data = self._get("/search/movie", params={"query": query, "page": page})

        if not data:
            return []

        results = []
        for movie in data.get("results", [])[:20]:
            results.append(
                {
                    "tmdb_id": movie["id"],
                    "title": movie["title"],
                    "overview": movie.get("overview", ""),
                    "poster_path": movie.get("poster_path"),
                    "release_date": movie.get("release_date"),
                    "vote_average": movie.get("vote_average", 0),
                    "poster_url": self.get_poster_url(movie.get("poster_path")),
                }
            )

        return results

    def get_popular_movies(self, page: int = 1) -> List[Dict[str, Any]]:
        """Get popular movies."""
        data = self._get("/movie/popular", params={"page": page})

        if not data:
            return []

        results = []
        for movie in data.get("results", []):
            results.append(
                {
                    "tmdb_id": movie["id"],
                    "title": movie["title"],
                    "overview": movie.get("overview", ""),
                    "poster_path": movie.get("poster_path"),
                    "release_date": movie.get("release_date"),
                    "vote_average": movie.get("vote_average", 0),
                    "poster_url": self.get_poster_url(movie.get("poster_path")),
                }
            )

        return results

    def get_top_rated_movies(self, page: int = 1) -> List[Dict[str, Any]]:
        """Get top rated movies."""
        data = self._get("/movie/top_rated", params={"page": page})

        if not data:
            return []

        return [
            {
                "tmdb_id": m["id"],
                "title": m["title"],
                "overview": m.get("overview", ""),
                "poster_path": m.get("poster_path"),
                "release_date": m.get("release_date"),
                "vote_average": m.get("vote_average", 0),
                "poster_url": self.get_poster_url(m.get("poster_path")),
            }
            for m in data.get("results", [])
        ]

    def get_movies_by_genre(self, genre_id: int, page: int = 1) -> List[Dict[str, Any]]:
        """Get movies by genre."""
        data = self._get("/discover/movie", params={"with_genres": genre_id, "sort_by": "popularity.desc", "page": page})

        if not data:
            return []

        return [
            {
                "tmdb_id": m["id"],
                "title": m["title"],
                "overview": m.get("overview", ""),
                "poster_path": m.get("poster_path"),
                "release_date": m.get("release_date"),
                "vote_average": m.get("vote_average", 0),
                "poster_url": self.get_poster_url(m.get("poster_path")),
            }
            for m in data.get("results", [])
        ]

    def get_similar_movies(self, movie_id: int) -> List[Dict[str, Any]]:
        """Get similar movies from TMDB (content-based)."""
        data = self._get(f"/movie/{movie_id}/similar")

        if not data:
            return []

        return [
            {
                "tmdb_id": m["id"],
                "title": m["title"],
                "overview": m.get("overview", ""),
                "poster_path": m.get("poster_path"),
                "release_date": m.get("release_date"),
                "vote_average": m.get("vote_average", 0),
                "poster_url": self.get_poster_url(m.get("poster_path")),
            }
            for m in data.get("results", [])[:10]
        ]

    def get_genres(self) -> List[Dict[str, Any]]:
        """Get list of movie genres."""
        data = self._get("/genre/movie/list")

        if not data:
            return []

        return data.get("genres", [])

    def get_poster_url(self, poster_path: str, size: str = "w342") -> Optional[str]:
        """Get full poster URL."""
        if poster_path:
            return f"{self.IMAGE_BASE_URL}/{size}{poster_path}"
        return None

    def get_backdrop_url(self, backdrop_path: str, size: str = "original") -> Optional[str]:
        """Get full backdrop URL."""
        if backdrop_path:
            return f"{self.IMAGE_BASE_URL}/{size}{backdrop_path}"
        return None
