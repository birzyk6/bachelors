"""
K-Nearest Neighbors (KNN) Collaborative Filtering.

Item-based KNN using cosine similarity between movies based on user ratings.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import norm as sparse_norm
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseRecommender


class KNNModel(BaseRecommender):
    """
    K-Nearest Neighbors collaborative filtering.

    Memory-efficient implementation using scipy sparse matrices.
    Only stores top-K neighbors per item instead of full N×N matrix.

    Args:
        k: Number of neighbors to consider
        similarity: Similarity metric ('cosine')
        user_based: If True, use user-based KNN; if False, use item-based
        min_support: Minimum number of common users/items for similarity
        max_movies: Maximum number of movies to consider (for memory efficiency)
    """

    def __init__(
        self,
        k: int = 40,
        similarity: str = "cosine",
        user_based: bool = False,
        min_support: int = 1,
        max_movies: int = 20000,
    ):
        super().__init__(name="KNN")
        self.k = k
        self.similarity = similarity
        self.user_based = user_based
        self.min_support = min_support
        self.max_movies = max_movies

        self.item_matrix = None
        self.user_id_map = {}
        self.item_id_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}
        self.neighbor_indices = None
        self.neighbor_sims = None
        self.item_means = None
        self.global_mean = 3.5

        self.train_data = None
        self.user_rated_items = {}

    def fit(self, train_data: pd.DataFrame, verbose: bool = True) -> "KNNModel":
        """
        Train KNN model on rating data using memory-efficient sparse matrices.

        Args:
            train_data: DataFrame with [userId, movieId, rating]
            verbose: Whether to print progress

        Returns:
            self
        """
        import gc

        if verbose:
            mode = "user-based" if self.user_based else "item-based"
            print(f"Training {self.name} model ({mode})...")
            print(f"  Data shape: {train_data.shape}")
            print(f"  k={self.k}, similarity={self.similarity}")

        movie_counts = train_data["movieId"].value_counts()
        if len(movie_counts) > self.max_movies:
            if verbose:
                print(f"  ⚠️  Limiting to top {self.max_movies:,} movies (was {len(movie_counts):,})")
            top_movies = set(movie_counts.head(self.max_movies).index)
            train_data = train_data[train_data["movieId"].isin(top_movies)].copy()
            if verbose:
                print(f"  Filtered data shape: {train_data.shape}")

        unique_users = train_data["userId"].unique()
        unique_items = train_data["movieId"].unique()

        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}
        self.reverse_user_map = {idx: uid for uid, idx in self.user_id_map.items()}
        self.reverse_item_map = {idx: iid for iid, idx in self.item_id_map.items()}

        n_users = len(unique_users)
        n_items = len(unique_items)

        if verbose:
            print(f"  Building sparse matrix: {n_users:,} users × {n_items:,} items")

        row_indices = train_data["userId"].map(self.user_id_map).values
        col_indices = train_data["movieId"].map(self.item_id_map).values
        ratings = train_data["rating"].values

        self.item_matrix = sparse.csr_matrix(
            (ratings, (row_indices, col_indices)),
            shape=(n_users, n_items),
            dtype=np.float32,
        )

        item_sums = np.array(self.item_matrix.sum(axis=0)).flatten()
        item_counts = np.array((self.item_matrix != 0).sum(axis=0)).flatten()
        self.item_means = np.divide(
            item_sums, item_counts, out=np.full(n_items, self.global_mean, dtype=np.float32), where=item_counts > 0
        )
        self.global_mean = float(train_data["rating"].mean())

        if verbose:
            print(f"  Computing similarities in batches (memory-efficient)...")

        self.neighbor_indices = np.zeros((n_items, self.k), dtype=np.int32)
        self.neighbor_sims = np.zeros((n_items, self.k), dtype=np.float32)

        item_user_matrix = self.item_matrix.T.tocsr()

        norms = sparse_norm(item_user_matrix, axis=1)
        norms[norms == 0] = 1
        item_user_normed = item_user_matrix.multiply(1 / norms.reshape(-1, 1)).tocsr()

        batch_size = 1000
        n_batches = (n_items + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            if verbose and batch_idx % 5 == 0:
                print(f"    Batch {batch_idx + 1}/{n_batches}...")

            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_items)

            batch_matrix = item_user_normed[start_idx:end_idx]
            sim_batch = batch_matrix.dot(item_user_normed.T).toarray()

            for i, sim_row in enumerate(sim_batch):
                item_idx = start_idx + i
                sim_row[item_idx] = -1

                top_k_indices = np.argpartition(sim_row, -self.k)[-self.k :]
                top_k_indices = top_k_indices[np.argsort(sim_row[top_k_indices])[::-1]]

                self.neighbor_indices[item_idx] = top_k_indices
                self.neighbor_sims[item_idx] = sim_row[top_k_indices]

            gc.collect()

        self.train_data = train_data
        self.user_rated_items = train_data.groupby("userId")["movieId"].apply(set).to_dict()

        self.is_fitted = True

        if verbose:
            print(f"  ✓ Model trained on {len(train_data):,} ratings")
            print(f"  ✓ {n_items:,} items with {self.k} neighbors each")

        return self

    def predict(self, user_movie_pairs: pd.DataFrame) -> np.ndarray:
        """
        Predict ratings using KNN with weighted average of neighbor ratings.

        Args:
            user_movie_pairs: DataFrame with [userId, movieId]

        Returns:
            Array of predicted ratings
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling predict()")

        predictions = []

        for _, row in user_movie_pairs.iterrows():
            user_id = row["userId"]
            movie_id = row["movieId"]

            if user_id not in self.user_id_map or movie_id not in self.item_id_map:
                predictions.append(self.global_mean)
                continue

            user_idx = self.user_id_map[user_id]
            item_idx = self.item_id_map[movie_id]

            user_ratings = self.item_matrix[user_idx].toarray().flatten()

            neighbor_idxs = self.neighbor_indices[item_idx]
            neighbor_sims = self.neighbor_sims[item_idx]

            numerator = 0.0
            denominator = 0.0

            for n_idx, sim in zip(neighbor_idxs, neighbor_sims):
                if sim > 0 and user_ratings[n_idx] > 0:
                    numerator += sim * user_ratings[n_idx]
                    denominator += abs(sim)

            if denominator > 0:
                pred = numerator / denominator
            else:
                pred = self.item_means[item_idx]

            pred = max(0.5, min(5.0, pred))
            predictions.append(pred)

        return np.array(predictions)

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_seen: bool = True,
        candidate_movies: List[int] | None = None,
    ) -> List[Tuple[int, float]]:
        """
        Generate top-K recommendations using KNN.

        Args:
            user_id: User ID
            top_k: Number of recommendations
            exclude_seen: Whether to exclude rated movies
            candidate_movies: Optional list of candidate movies

        Returns:
            List of (movie_id, predicted_rating) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling recommend()")

        if candidate_movies is None:
            candidate_movies = list(self.item_id_map.keys())

        if exclude_seen and user_id in self.user_rated_items:
            seen_movies = self.user_rated_items[user_id]
            candidate_movies = [m for m in candidate_movies if m not in seen_movies]

        pairs_df = pd.DataFrame({"userId": [user_id] * len(candidate_movies), "movieId": candidate_movies})

        pred_ratings = self.predict(pairs_df)

        predictions = list(zip(candidate_movies, pred_ratings))

        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:top_k]

    def get_params(self) -> dict:
        """Get model hyperparameters."""
        return {
            "name": self.name,
            "k": self.k,
            "similarity": self.similarity,
            "user_based": self.user_based,
            "min_support": self.min_support,
            "max_movies": self.max_movies,
        }

    def get_neighbors(self, item_id: int, k: int | None = None) -> List[Tuple[int, float]]:
        """
        Get K nearest neighbors for an item (only for item-based KNN).

        Args:
            item_id: Movie ID
            k: Number of neighbors (uses self.k if None)

        Returns:
            List of (neighbor_item_id, similarity) tuples
        """
        if self.user_based:
            raise ValueError("get_neighbors() only works for item-based KNN")

        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        k = k or self.k

        if item_id not in self.item_id_map:
            return []

        item_idx = self.item_id_map[item_id]

        neighbor_idxs = self.neighbor_indices[item_idx][:k]
        neighbor_sims = self.neighbor_sims[item_idx][:k]

        result = []
        for n_idx, sim in zip(neighbor_idxs, neighbor_sims):
            neighbor_raw_id = self.reverse_item_map[n_idx]
            result.append((neighbor_raw_id, float(sim)))

        return result

    def save(self, path: Path):
        """Save model to disk."""
        import json

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        sparse.save_npz(path / "item_matrix.npz", self.item_matrix)

        np.savez_compressed(
            path / "neighbors.npz",
            indices=self.neighbor_indices,
            sims=self.neighbor_sims,
            item_means=self.item_means,
        )

        np.savez(
            path / "mappings.npz",
            user_ids=np.array(list(self.user_id_map.keys())),
            item_ids=np.array(list(self.item_id_map.keys())),
        )

        with open(path / "params.json", "w") as f:
            json.dump(
                {
                    "name": self.name,
                    "k": self.k,
                    "similarity": self.similarity,
                    "user_based": self.user_based,
                    "min_support": self.min_support,
                    "max_movies": self.max_movies,
                    "global_mean": self.global_mean,
                },
                f,
            )

    def load(self, path: Path):
        """Load model from disk."""
        import json

        path = Path(path)

        self.item_matrix = sparse.load_npz(path / "item_matrix.npz")

        neighbors_data = np.load(path / "neighbors.npz")
        self.neighbor_indices = neighbors_data["indices"]
        self.neighbor_sims = neighbors_data["sims"]
        self.item_means = neighbors_data["item_means"]

        mappings = np.load(path / "mappings.npz")
        user_ids = mappings["user_ids"]
        item_ids = mappings["item_ids"]

        self.user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
        self.item_id_map = {iid: idx for idx, iid in enumerate(item_ids)}
        self.reverse_user_map = {idx: uid for uid, idx in self.user_id_map.items()}
        self.reverse_item_map = {idx: iid for iid, idx in self.item_id_map.items()}

        with open(path / "params.json") as f:
            params = json.load(f)
            self.k = params.get("k", self.k)
            self.similarity = params.get("similarity", self.similarity)
            self.user_based = params.get("user_based", self.user_based)
            self.min_support = params.get("min_support", self.min_support)
            self.max_movies = params.get("max_movies", self.max_movies)
            self.global_mean = params.get("global_mean", 3.5)

        self.is_fitted = True
