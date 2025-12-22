"""
Movie embedding store for efficient similarity search.

Handles loading and querying pre-computed Two-Tower embeddings.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class EmbeddingStore:
    """
    Store for movie embeddings with similarity search capabilities.

    Attributes:
        embeddings: Dict mapping movie_id -> embedding vector
        movie_ids: List of all movie IDs with embeddings
        embedding_matrix: Numpy array of all embeddings (for batch operations)
        embedding_dim: Dimension of embeddings
    """

    def __init__(self, embeddings_path: Path, metadata_path: Path):
        """
        Initialize embedding store from saved files.

        Args:
            embeddings_path: Path to .npy file with embeddings dict
            metadata_path: Path to .json file with metadata
        """
        self.embeddings_path = embeddings_path
        self.metadata_path = metadata_path

        self._embeddings: Dict[int, np.ndarray] = {}
        self._movie_ids: List[int] = []
        self._embedding_matrix: Optional[np.ndarray] = None
        self._embedding_dim: int = 0
        self._loaded: bool = False

    def load(self) -> None:
        """Load embeddings from disk."""
        if self._loaded:
            return

        if not self.embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings not found: {self.embeddings_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {self.metadata_path}")

        # Load embeddings dict
        self._embeddings = np.load(self.embeddings_path, allow_pickle=True).item()

        # Load metadata
        with open(self.metadata_path) as f:
            metadata = json.load(f)

        self._embedding_dim = metadata["embedding_dim"]

        # Build movie ID list - prefer from embeddings dict (handles combined embeddings)
        if "movie_ids" in metadata:
            # Use metadata list (filter to only those with embeddings)
            self._movie_ids = [mid for mid in metadata["movie_ids"] if mid in self._embeddings]
        else:
            # Extract from embeddings dict (for combined embeddings without full ID list)
            self._movie_ids = list(self._embeddings.keys())

        # Build embedding matrix for fast batch operations
        self._embedding_matrix = np.array([self._embeddings[mid] for mid in self._movie_ids])

        self._loaded = True

    @property
    def embeddings(self) -> Dict[int, np.ndarray]:
        """Get embeddings dict."""
        self.load()
        return self._embeddings

    @property
    def movie_ids(self) -> List[int]:
        """Get list of movie IDs with embeddings."""
        self.load()
        return self._movie_ids

    @property
    def embedding_matrix(self) -> np.ndarray:
        """Get embedding matrix (num_movies x embedding_dim)."""
        self.load()
        return self._embedding_matrix

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        self.load()
        return self._embedding_dim

    @property
    def num_movies(self) -> int:
        """Get number of movies with embeddings."""
        self.load()
        return len(self._movie_ids)

    def get_embedding(self, movie_id: int) -> Optional[np.ndarray]:
        """Get embedding for a single movie."""
        self.load()
        return self._embeddings.get(movie_id)

    def find_similar(
        self, query_embedding: np.ndarray, top_k: int = 10, exclude_ids: Optional[set] = None
    ) -> List[Tuple[int, float]]:
        """
        Find movies most similar to query embedding.

        Args:
            query_embedding: Query vector (embedding_dim,) or (1, embedding_dim)
            top_k: Number of results to return
            exclude_ids: Set of movie IDs to exclude from results

        Returns:
            List of (movie_id, similarity_score) tuples, sorted by score descending
        """
        self.load()

        # Ensure query is 1D
        query = query_embedding.flatten()

        # Normalize query (embeddings are already L2-normalized)
        query_norm = np.linalg.norm(query)
        if query_norm > 0:
            query = query / query_norm

        # Compute dot product similarities
        similarities = np.dot(self._embedding_matrix, query)

        # Get top indices
        top_indices = np.argsort(similarities)[::-1]

        # Filter and collect results
        results = []
        exclude_ids = exclude_ids or set()

        for idx in top_indices:
            movie_id = self._movie_ids[idx]
            if movie_id not in exclude_ids:
                results.append((movie_id, float(similarities[idx])))
                if len(results) >= top_k:
                    break

        return results

    def find_similar_to_movie(self, movie_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Find movies similar to a given movie.

        Args:
            movie_id: Source movie ID
            top_k: Number of similar movies to return

        Returns:
            List of (movie_id, similarity_score) tuples
        """
        embedding = self.get_embedding(movie_id)
        if embedding is None:
            return []

        return self.find_similar(embedding, top_k=top_k, exclude_ids={movie_id})

    def compute_centroid(self, movie_ids: List[int]) -> Optional[np.ndarray]:
        """
        Compute centroid embedding for a set of movies.

        Args:
            movie_ids: List of movie IDs

        Returns:
            Centroid embedding or None if no valid embeddings
        """
        self.load()

        embeddings = []
        for mid in movie_ids:
            emb = self._embeddings.get(mid)
            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            return None

        centroid = np.mean(embeddings, axis=0)
        return centroid
