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
        self._id_to_popularity: Dict[int, float] = {}
        self._id_to_vote_average: Dict[int, float] = {}
        self._id_to_vote_count: Dict[int, int] = {}
        self._id_to_source: Dict[int, str] = {}  # 'movielens' or 'tmdb'
        self._id_to_normalized_title: Dict[int, str] = {}  # Pre-computed for fast search
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

        # Extract columns as lists for fast iteration (much faster than iter_rows)
        movie_ids = self._movies_df["movieId"].to_list()
        titles = self._movies_df[title_col].to_list() if title_col in self._movies_df.columns else [None] * len(movie_ids)
        genres_list = self._movies_df["genres"].to_list() if "genres" in self._movies_df.columns else [None] * len(movie_ids)
        years = self._movies_df["year"].to_list() if "year" in self._movies_df.columns else [None] * len(movie_ids)
        overviews = self._movies_df["overview"].to_list() if "overview" in self._movies_df.columns else [None] * len(movie_ids)
        popularities = (
            self._movies_df["popularity"].to_list() if "popularity" in self._movies_df.columns else [0.0] * len(movie_ids)
        )
        vote_avgs = (
            self._movies_df["vote_average"].to_list() if "vote_average" in self._movies_df.columns else [0.0] * len(movie_ids)
        )
        vote_counts = (
            self._movies_df["vote_count"].to_list() if "vote_count" in self._movies_df.columns else [0] * len(movie_ids)
        )
        sources = self._movies_df["source"].to_list() if "source" in self._movies_df.columns else ["unknown"] * len(movie_ids)

        # Build lookup dicts using fast iteration
        for i, movie_id in enumerate(movie_ids):
            title = titles[i] or f"Movie {movie_id}"
            self._id_to_title[movie_id] = title
            self._id_to_genres[movie_id] = genres_list[i] or ""
            self._id_to_year[movie_id] = years[i]
            self._id_to_overview[movie_id] = overviews[i] or ""
            self._id_to_popularity[movie_id] = float(popularities[i] or 0.0)
            self._id_to_vote_average[movie_id] = float(vote_avgs[i] or 0.0)
            self._id_to_vote_count[movie_id] = int(vote_counts[i] or 0)
            self._id_to_source[movie_id] = sources[i] or "unknown"

            # Pre-compute normalized title for fast search
            self._id_to_normalized_title[movie_id] = self._normalize_for_search(title)

            # Build genre index
            genres_str = genres_list[i] or ""
            if genres_str:
                for genre in genres_str.split(","):
                    genre = genre.strip()
                    if genre and genre != "(no genres listed)":
                        if genre not in self._genre_to_ids:
                            self._genre_to_ids[genre] = set()
                        self._genre_to_ids[genre].add(movie_id)

        # Sort genres by count
        self._all_genres = sorted(self._genre_to_ids.keys(), key=lambda g: len(self._genre_to_ids[g]), reverse=True)

        # Free the dataframe memory - we've extracted what we need
        self._movies_df = None

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

    def get_popularity(self, movie_id: int) -> float:
        """Get movie popularity score by ID (from TMDB)."""
        self.load()
        return self._id_to_popularity.get(movie_id, 0.0)

    def get_vote_average(self, movie_id: int) -> float:
        """Get movie average vote (rating) by ID (from TMDB, 0-10 scale)."""
        self.load()
        return self._id_to_vote_average.get(movie_id, 0.0)

    def get_vote_count(self, movie_id: int) -> int:
        """Get movie vote count by ID (from TMDB)."""
        self.load()
        return self._id_to_vote_count.get(movie_id, 0)

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

    def _normalize_for_search(self, text: str) -> str:
        """
        Normalize text for search by removing special characters.

        - Converts to lowercase
        - Removes punctuation (commas, colons, apostrophes, etc.)
        - Handles "Title, The" -> "the title" normalization
        """
        import re

        text = text.lower()

        # Handle "Title, The" format -> "the title"
        if ", the" in text:
            text = "the " + text.replace(", the", "")
        if ", a " in text:
            text = "a " + text.replace(", a ", " ")
        if ", an " in text:
            text = "an " + text.replace(", an ", " ")

        # Remove special characters (keep only alphanumeric and spaces)
        text = re.sub(r"[^\w\s]", " ", text)

        # Normalize whitespace
        text = " ".join(text.split())

        return text

    def get_source(self, movie_id: int) -> str:
        """Get movie source ('movielens' or 'tmdb')."""
        self.load()
        return self._id_to_source.get(movie_id, "unknown")

    def _get_dedup_key(self, movie_id: int) -> str:
        """
        Get deduplication key for a movie based on normalized title + year.

        Used to identify duplicate entries from different sources.
        Handles common naming variations between MovieLens and TMDB.
        """
        import re

        title = self._id_to_title.get(movie_id, "")
        year = self._id_to_year.get(movie_id)
        normalized = self._normalize_for_search(title)

        # Remove year from title if present (common in ML titles like "Toy Story (1995)")
        normalized = re.sub(r"\b\d{4}\b", "", normalized).strip()

        # Remove common suffixes that vary between datasets
        # "Part I", "Part II", "Part 1", "Part 2", etc.
        normalized = re.sub(r"\s+part\s+(i{1,3}|iv|v|vi|[1-6])\s*$", "", normalized, flags=re.IGNORECASE)
        # "- Part I", "- Part II" in the middle
        normalized = re.sub(r"\s+part\s+(i{1,3}|iv|v|vi|[1-6])\s*", " ", normalized, flags=re.IGNORECASE)

        # Normalize whitespace again after removals
        normalized = " ".join(normalized.split())

        year_str = str(int(year)) if year else ""
        return f"{normalized}|{year_str}"

    def search_by_title(self, query: str, limit: int = 20, deduplicate: bool = True) -> List[int]:
        """
        Search movies by title substring (case-insensitive).

        Normalizes both query and titles by:
        - Stripping special characters
        - Handling article placement ("The Avengers" matches "Avengers, The")

        Args:
            query: Search query
            limit: Maximum results to return
            deduplicate: If True, remove duplicate entries preferring MovieLens

        Returns:
            List of matching movie IDs sorted by popularity
        """
        self.load()

        query_normalized = self._normalize_for_search(query)
        matches = []

        # Use pre-computed normalized titles for fast search
        for movie_id, title_normalized in self._id_to_normalized_title.items():
            if query_normalized in title_normalized:
                popularity = self._id_to_popularity.get(movie_id, 0.0)
                has_genres = bool(self._id_to_genres.get(movie_id))
                matches.append((movie_id, popularity, has_genres))

        # Sort by popularity (highest first)
        matches.sort(key=lambda x: x[1], reverse=True)

        if deduplicate:
            # Deduplicate by normalized title + year, preferring entries with genres (MovieLens)
            seen_keys: Dict[str, int] = {}
            deduplicated = []

            for movie_id, popularity, has_genres in matches:
                dedup_key = self._get_dedup_key(movie_id)

                if dedup_key not in seen_keys:
                    # First time seeing this movie
                    seen_keys[dedup_key] = movie_id
                    deduplicated.append((movie_id, popularity))
                else:
                    # Already seen - check if this one is better (has genres)
                    existing_id = seen_keys[dedup_key]
                    existing_has_genres = bool(self._id_to_genres.get(existing_id))

                    if has_genres and not existing_has_genres:
                        # Replace with better entry
                        deduplicated = [
                            (mid, pop) if mid != existing_id else (movie_id, popularity) for mid, pop in deduplicated
                        ]
                        seen_keys[dedup_key] = movie_id

            return [mid for mid, _ in deduplicated[:limit]]
        else:
            return [mid for mid, _, _ in matches[:limit]]

    def search_by_overview(self, query: str, limit: int = 50) -> List[int]:
        """
        Search movies by keyword in overview/description.

        Args:
            query: Search query
            limit: Maximum results to return

        Returns:
            List of matching movie IDs sorted by popularity
        """
        self.load()

        query_lower = query.lower()
        matches = []

        for movie_id, overview in self._id_to_overview.items():
            if overview and query_lower in overview.lower():
                popularity = self._id_to_popularity.get(movie_id, 0.0)
                matches.append((movie_id, popularity))

        # Sort by popularity (highest first) and return IDs
        matches.sort(key=lambda x: x[1], reverse=True)
        return [mid for mid, _ in matches[:limit]]

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
