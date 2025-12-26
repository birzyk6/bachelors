"""Qdrant vector database service."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from django.conf import settings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchAny, MatchValue, PointStruct, VectorParams

logger = logging.getLogger(__name__)


class QdrantService:
    """Service for interacting with Qdrant vector database."""

    COLLECTION_NAME = "movies"
    EMBEDDING_DIM = 128

    def __init__(self):
        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
        )
        self._ensure_collection()

    def _ensure_collection(self):
        """Ensure the movies collection exists."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.COLLECTION_NAME not in collection_names:
            logger.info(f"Creating collection: {self.COLLECTION_NAME}")
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.EMBEDDING_DIM,
                    distance=Distance.DOT,
                ),
            )

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        info = self.client.get_collection(self.COLLECTION_NAME)
        return {
            "points_count": info.points_count,
            "status": info.status.value if hasattr(info.status, "value") else str(info.status),
        }

    def load_embeddings(self, embeddings_path: str, metadata_path: Optional[str] = None):
        """Load movie embeddings into Qdrant."""
        logger.info(f"Loading embeddings from {embeddings_path}")

        # Load embeddings
        embeddings_data = np.load(embeddings_path, allow_pickle=True)

        if isinstance(embeddings_data, np.ndarray) and embeddings_data.dtype == object:
            # Dictionary format: {movie_id: embedding}
            embeddings_dict = embeddings_data.item()
        else:
            # Assume it's the npz format with separate arrays
            embeddings_dict = dict(zip(embeddings_data["movie_ids"], embeddings_data["embeddings"]))

        # Load metadata if provided
        metadata = {}
        if metadata_path and Path(metadata_path).exists():
            with open(metadata_path) as f:
                metadata = json.load(f)

        # Prepare points
        points = []
        for movie_id, embedding in embeddings_dict.items():
            movie_id = int(movie_id)
            point = PointStruct(
                id=movie_id,
                vector=embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                payload={
                    "tmdb_id": movie_id,
                },
            )
            points.append(point)

        # Upload in batches
        batch_size = 1000
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=batch,
            )
            logger.info(f"Uploaded {min(i + batch_size, len(points))}/{len(points)} points")

        logger.info(f"Successfully loaded {len(points)} embeddings")
        return len(points)

    def search_similar(
        self,
        query_vector: List[float],
        limit: int = 20,
        score_threshold: float = None,
        filter_genres: List[str] = None,
        exclude_ids: List[int] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar movies using vector similarity."""
        query_filter = None
        if filter_genres:
            query_filter = Filter(must=[FieldCondition(key="genres", match=MatchAny(any=filter_genres))])

        # Use query_points for newer qdrant-client versions
        results = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_vector,
            limit=limit + len(exclude_ids or []),
            score_threshold=score_threshold,
            query_filter=query_filter,
        ).points

        # Post-process to exclude IDs
        output = []
        for hit in results:
            if exclude_ids and hit.id in exclude_ids:
                continue
            output.append(
                {
                    "tmdb_id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload,
                }
            )
            if len(output) >= limit:
                break

        return output

    def get_movie_embedding(self, tmdb_id: int) -> Optional[List[float]]:
        """Get embedding for a specific movie."""
        results = self.client.retrieve(
            collection_name=self.COLLECTION_NAME,
            ids=[tmdb_id],
            with_vectors=True,
        )

        if results:
            return results[0].vector
        return None

    def get_multiple_embeddings(self, tmdb_ids: List[int]) -> Dict[int, List[float]]:
        """Get embeddings for multiple movies."""
        results = self.client.retrieve(
            collection_name=self.COLLECTION_NAME,
            ids=tmdb_ids,
            with_vectors=True,
        )

        return {point.id: point.vector for point in results}

    def update_movie_metadata(self, tmdb_id: int, metadata: Dict[str, Any]):
        """Update metadata for a movie point."""
        self.client.set_payload(
            collection_name=self.COLLECTION_NAME,
            payload=metadata,
            points=[tmdb_id],
        )

    def delete_movie(self, tmdb_id: int):
        """Delete a movie from the collection."""
        self.client.delete(
            collection_name=self.COLLECTION_NAME,
            points_selector=[tmdb_id],
        )

    def count_movies(self) -> int:
        """Get total number of movies in collection."""
        info = self.client.get_collection(self.COLLECTION_NAME)
        return getattr(info, "points_count", 0) or 0
