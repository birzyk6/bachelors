"""
User store for loading user data and generating user embeddings.

Handles user impersonation and personalized recommendations.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import tensorflow as tf


class UserStore:
    """
    Store for user data and the Two-Tower user model.

    Provides:
    - User watch history lookup
    - User embedding generation via the trained user tower
    - Sample user suggestions
    """

    def __init__(
        self,
        ratings_path: Path,
        user_tower_path: Path,
        movies_catalog: Optional["MovieCatalog"] = None,
    ):
        """
        Initialize user store.

        Args:
            ratings_path: Path to ratings parquet file (train.parquet)
            user_tower_path: Path to saved user tower model
            movies_catalog: Optional MovieCatalog for formatting
        """
        self.ratings_path = ratings_path
        self.user_tower_path = user_tower_path
        self.movies_catalog = movies_catalog

        # Data - keep ratings in Polars for fast filtering
        self._ratings_df: Optional[pl.DataFrame] = None
        self._all_user_ids: List[int] = []
        self._user_tower: Optional[tf.keras.Model] = None
        self._user_id_map: Dict[int, int] = {}  # user_id -> embedding index
        self._user_id_set: set = set()  # Fast lookup

        self._loaded = False
        self._model_loaded = False

    def load(self) -> None:
        """Load user ratings data from disk."""
        if self._loaded:
            return

        if not self.ratings_path.exists():
            raise FileNotFoundError(f"Ratings file not found: {self.ratings_path}")

        print("Loading user ratings...")

        # Load ratings with relevant columns and sort by timestamp
        self._ratings_df = pl.read_parquet(self.ratings_path, columns=["userId", "movieId", "rating", "timestamp"]).sort(
            "timestamp"
        )

        # Get unique user IDs (fast operation)
        self._all_user_ids = sorted(self._ratings_df["userId"].unique().to_list())
        self._user_id_set = set(self._all_user_ids)

        # Create user ID mapping for model
        self._user_id_map = {uid: idx for idx, uid in enumerate(self._all_user_ids)}

        self._loaded = True
        print(f"  ✓ Loaded ratings for {len(self._all_user_ids):,} users")

    def load_user_tower(self) -> bool:
        """
        Load the trained user tower model.

        Returns:
            True if loaded successfully, False otherwise
        """
        if self._model_loaded:
            return True

        if not self.user_tower_path.exists():
            print(f"User tower not found: {self.user_tower_path}")
            return False

        try:
            print("Loading user tower model...")
            self._user_tower = tf.keras.models.load_model(self.user_tower_path)
            self._model_loaded = True
            print("  ✓ User tower loaded")
            return True
        except Exception as e:
            print(f"  ✗ Failed to load user tower: {e}")
            return False

    @property
    def num_users(self) -> int:
        """Get total number of users."""
        self.load()
        return len(self._all_user_ids)

    def user_exists(self, user_id: int) -> bool:
        """Check if user exists in the dataset."""
        self.load()
        return user_id in self._user_id_set

    def get_user_history(
        self, user_id: int, min_rating: float = 0.0, limit: Optional[int] = None
    ) -> List[Tuple[int, float, int]]:
        """
        Get user's watch history.

        Args:
            user_id: User ID
            min_rating: Minimum rating to include
            limit: Maximum number of movies (most recent first)

        Returns:
            List of (movie_id, rating, timestamp) tuples, sorted by timestamp (newest first)
        """
        self.load()

        if user_id not in self._user_id_set:
            return []

        # Filter ratings for this user using Polars (fast!)
        user_df = self._ratings_df.filter(pl.col("userId") == user_id)

        if min_rating > 0:
            user_df = user_df.filter(pl.col("rating") >= min_rating)

        # Sort by timestamp descending (most recent first)
        user_df = user_df.sort("timestamp", descending=True)

        if limit:
            user_df = user_df.head(limit)

        # Convert to list of tuples
        return list(zip(user_df["movieId"].to_list(), user_df["rating"].to_list(), user_df["timestamp"].to_list()))

    def get_user_embedding(self, user_id: int) -> Optional[np.ndarray]:
        """
        Get embedding for a user using the trained user tower.

        Args:
            user_id: User ID

        Returns:
            128-dim embedding vector, or None if user/model not available
        """
        if not self._model_loaded:
            if not self.load_user_tower():
                return None

        if user_id not in self._user_id_map:
            return None

        # Map user_id to embedding index
        user_idx = self._user_id_map[user_id]

        # Generate embedding using user tower
        user_input = np.array([[user_idx]], dtype=np.int32)
        embedding = self._user_tower.predict(user_input, verbose=0)

        return embedding[0]

    def get_sample_users(self, n: int = 10, min_ratings: int = 50) -> List[int]:
        """
        Get a sample of active users.

        Args:
            n: Number of users to return
            min_ratings: Minimum number of ratings for user to be considered active

        Returns:
            List of user IDs
        """
        self.load()

        # Find users with enough ratings using Polars aggregation
        user_counts = self._ratings_df.group_by("userId").len()
        active_users = user_counts.filter(pl.col("len") >= min_ratings)["userId"].to_list()

        # Return evenly distributed sample
        if len(active_users) <= n:
            return active_users

        step = len(active_users) // n
        return [active_users[i * step] for i in range(n)]

    def get_user_stats(self, user_id: int) -> Dict:
        """
        Get statistics for a user.

        Args:
            user_id: User ID

        Returns:
            Dict with user statistics
        """
        self.load()

        if user_id not in self._user_id_set:
            return {}

        # Get user's ratings using Polars
        user_df = self._ratings_df.filter(pl.col("userId") == user_id)

        if len(user_df) == 0:
            return {}

        ratings = user_df["rating"]

        return {
            "user_id": user_id,
            "num_ratings": len(user_df),
            "avg_rating": ratings.mean(),
            "min_rating": ratings.min(),
            "max_rating": ratings.max(),
        }

    def get_top_genres_for_user(self, user_id: int, top_k: int = 5) -> List[Tuple[str, int]]:
        """
        Get user's most watched genres.

        Args:
            user_id: User ID
            top_k: Number of genres to return

        Returns:
            List of (genre, count) tuples
        """
        if not self.movies_catalog:
            return []

        history = self.get_user_history(user_id)

        genre_counts: Dict[str, int] = {}
        for movie_id, _, _ in history:
            genres = self.movies_catalog.get_genres_list(movie_id)
            for genre in genres:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1

        sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_genres[:top_k]
