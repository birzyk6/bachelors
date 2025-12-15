"""
Movie catalog for metadata lookup.

Handles loading and querying movie information.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set

import polars as pl


class MovieCatalog:
    """
    Catalog of movie metadata for display and search.

    Provides lookup by ID, title search, and genre filtering.
    """

    def __init__(self, movies_path: Path):
        """
        Initialize movie catalog.

        Args:
            movies_path: Path to movies.parquet file
        """
        self.movies_path = movies_path
        self._movies_df: Optional[pl.DataFrame] = None
        self._id_to_title: Dict[int, str] = {}
        self._id_to_genres: Dict[int, str] = {}
        self._id_to_year: Dict[int, Optional[int]] = {}
        self._id_to_overview: Dict[int, str] = {}
        self._genre_to_ids: Dict[str, Set[int]] = {}
        self._all_genres: List[str] = []
        self._loaded: bool = False

    def load(self) -> None:
        """Load movie data from disk."""
        if self._loaded:
            return

        if not self.movies_path.exists():
            raise FileNotFoundError(f"Movies file not found: {self.movies_path}")

        self._movies_df = pl.read_parquet(self.movies_path)

        # Determine title column
        title_col = "title_ml" if "title_ml" in self._movies_df.columns else "title"

        # Build lookup dicts
        for row in self._movies_df.iter_rows(named=True):
            movie_id = row["movieId"]
            self._id_to_title[movie_id] = row.get(title_col, f"Movie {movie_id}")
            self._id_to_genres[movie_id] = row.get("genres", "")
            self._id_to_year[movie_id] = row.get("year")
            self._id_to_overview[movie_id] = row.get("overview", "")

            # Build genre index
            genres_str = row.get("genres", "")
            if genres_str:
                for genre in genres_str.split(","):
                    genre = genre.strip()
                    if genre and genre != "(no genres listed)":
                        if genre not in self._genre_to_ids:
                            self._genre_to_ids[genre] = set()
                        self._genre_to_ids[genre].add(movie_id)

        # Sort genres by count
        self._all_genres = sorted(self._genre_to_ids.keys(), key=lambda g: len(self._genre_to_ids[g]), reverse=True)

        self._loaded = True

    @property
    def genres(self) -> List[str]:
        """Get all available genres sorted by frequency."""
        self.load()
        return self._all_genres

    @property
    def num_movies(self) -> int:
        """Get total number of movies."""
        self.load()
        return len(self._id_to_title)

    def get_title(self, movie_id: int) -> str:
        """Get movie title by ID."""
        self.load()
        return self._id_to_title.get(movie_id, f"Unknown Movie ({movie_id})")

    def get_genres(self, movie_id: int) -> str:
        """Get movie genres string by ID."""
        self.load()
        return self._id_to_genres.get(movie_id, "")

    def get_genres_list(self, movie_id: int) -> List[str]:
        """Get movie genres as list."""
        genres_str = self.get_genres(movie_id)
        if not genres_str:
            return []
        return [g.strip() for g in genres_str.split(",")]

    def get_year(self, movie_id: int) -> Optional[int]:
        """Get movie release year by ID."""
        self.load()
        return self._id_to_year.get(movie_id)

    def get_overview(self, movie_id: int) -> str:
        """Get movie overview/description by ID."""
        self.load()
        return self._id_to_overview.get(movie_id, "")

    def get_movies_by_genre(self, genre: str) -> Set[int]:
        """Get all movie IDs with a given genre."""
        self.load()
        return self._genre_to_ids.get(genre, set())

    def get_movies_by_genres(self, genres: List[str], match_all: bool = False) -> Set[int]:
        """
        Get movies matching given genres.

        Args:
            genres: List of genre names
            match_all: If True, movie must have ALL genres. If False, ANY genre.

        Returns:
            Set of matching movie IDs
        """
        self.load()

        if not genres:
            return set()

        genre_sets = [self._genre_to_ids.get(g, set()) for g in genres]

        if match_all:
            result = genre_sets[0].copy()
            for s in genre_sets[1:]:
                result &= s
        else:
            result = set()
            for s in genre_sets:
                result |= s

        return result

    def search_by_title(self, query: str, limit: int = 20) -> List[int]:
        """
        Search movies by title substring (case-insensitive).

        Args:
            query: Search query
            limit: Maximum results to return

        Returns:
            List of matching movie IDs
        """
        self.load()

        query_lower = query.lower()
        matches = []

        for movie_id, title in self._id_to_title.items():
            if query_lower in title.lower():
                matches.append(movie_id)
                if len(matches) >= limit:
                    break

        return matches

    def format_movie(self, movie_id: int, include_overview: bool = False) -> str:
        """
        Format movie info for display.

        Args:
            movie_id: Movie ID
            include_overview: Whether to include overview text

        Returns:
            Formatted string
        """
        self.load()

        title = self.get_title(movie_id)
        genres = self.get_genres(movie_id)
        year = self.get_year(movie_id)

        year_str = f" ({year})" if year else ""
        genres_str = f" [{genres}]" if genres else ""

        result = f"{title}{year_str}{genres_str}"

        if include_overview:
            overview = self.get_overview(movie_id)
            if overview:
                # Truncate long overviews
                if len(overview) > 200:
                    overview = overview[:197] + "..."
                result += f"\n    {overview}"

        return result
