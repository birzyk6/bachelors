"""
Content-Based Filtering using BERT embeddings of movie overviews.

Uses pre-trained BERT from TensorFlow Hub to encode movie plot summaries into dense vectors,
then computes similarity for recommendations.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseRecommender


class ContentBasedModel(BaseRecommender):
    """
    Content-Based Filtering using BERT embeddings.

    This model encodes movie overviews using BERT, then recommends
    movies similar to those the user has liked in the past.

    Args:
        embedding_model: TensorFlow Hub URL for the BERT encoder
        similarity_metric: Similarity metric ('cosine', 'dot')
    """

    def __init__(
        self,
        embedding_model: str = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2",
        similarity_metric: str = "cosine",
    ):
        super().__init__(name="ContentBased")
        self.embedding_model_name = embedding_model
        self.similarity_metric = similarity_metric

        print(f"Loading TensorFlow Hub BERT model...")
        self.bert_encoder = hub.KerasLayer(self.embedding_model_name, trainable=False)
        self.bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")

        self.movie_embeddings = {}
        self.movie_metadata = None
        self.user_profiles = {}

    def _encode_text(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode texts using TensorFlow Hub BERT.

        Args:
            texts: List of text strings
            batch_size: Batch size for encoding

        Returns:
            Array of shape (len(texts), embedding_dim)
        """
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            preprocessed = self.bert_preprocess(batch_texts)
            outputs = self.bert_encoder(preprocessed)

            cls_embeddings = outputs["pooled_output"].numpy()

            embeddings.append(cls_embeddings)

        return np.vstack(embeddings)

    def fit(self, movie_data: pd.DataFrame, train_ratings: pd.DataFrame | None = None, verbose: bool = True):
        """
        Pre-compute movie embeddings from overviews.

        First tries to load pre-computed BERT embeddings from disk.
        Falls back to computing them if not found.

        Args:
            movie_data: DataFrame with columns [movieId, overview, title, genres]
            train_ratings: Optional ratings data for building user profiles
            verbose: Whether to print progress

        Returns:
            self
        """
        if verbose:
            print(f"Training {self.name} model...")
            print(f"  Movies: {len(movie_data)}")

        self.movie_metadata = movie_data.copy()

        try:
            from config import MODELS_DIR
        except ImportError:
            MODELS_DIR = Path(__file__).parent.parent.parent / "saved_models"

        precomputed_path = MODELS_DIR / "bert_embeddings.npz"

        if precomputed_path.exists():
            if verbose:
                print(f"  Loading pre-computed BERT embeddings from {precomputed_path}...")
            data = np.load(precomputed_path)
            saved_embeddings = data["embeddings"]
            saved_movie_ids = data["movie_ids"]

            for mid, emb in zip(saved_movie_ids, saved_embeddings):
                self.movie_embeddings[int(mid)] = emb

            if verbose:
                print(f"  ✓ Loaded {len(self.movie_embeddings):,} pre-computed embeddings")
        else:
            overviews = movie_data["overview"].fillna("").tolist()
            movie_ids = movie_data["movieId"].tolist()

            if verbose:
                print(f"  Encoding {len(overviews)} movie overviews with BERT...")
                print(f"  (Tip: Run 'python precompute_embeddings.py' first to speed this up)")

            embeddings = self._encode_text(overviews)

            for movie_id, embedding in zip(movie_ids, embeddings):
                self.movie_embeddings[movie_id] = embedding

            if verbose:
                print(f"  ✓ Encoded {len(self.movie_embeddings)} movies")

        if train_ratings is not None:
            if verbose:
                print(f"  Building user preference profiles...")
            self._build_user_profiles(train_ratings)

        self.is_fitted = True

        return self

    def _build_user_profiles(self, ratings: pd.DataFrame):
        """
        Build user preference vectors by aggregating embeddings of liked movies.

        Args:
            ratings: DataFrame with [userId, movieId, rating]
        """
        liked_movies = ratings[ratings["rating"] >= 4.0]

        for user_id in liked_movies["userId"].unique():
            user_movies = liked_movies[liked_movies["userId"] == user_id]["movieId"].tolist()

            user_embeddings = [self.movie_embeddings[mid] for mid in user_movies if mid in self.movie_embeddings]

            if user_embeddings:
                self.user_profiles[user_id] = np.mean(user_embeddings, axis=0)

    def predict(self, user_movie_pairs: pd.DataFrame) -> np.ndarray:
        """
        Predict ratings based on content similarity.

        Args:
            user_movie_pairs: DataFrame with [userId, movieId]

        Returns:
            Array of predicted similarity scores (scaled to rating range)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling predict()")

        predictions = []

        for _, row in user_movie_pairs.iterrows():
            user_id = row["userId"]
            movie_id = row["movieId"]

            if user_id not in self.user_profiles or movie_id not in self.movie_embeddings:
                predictions.append(2.5)
                continue

            user_vector = self.user_profiles[user_id].reshape(1, -1)
            movie_vector = self.movie_embeddings[movie_id].reshape(1, -1)

            if self.similarity_metric == "cosine":
                similarity = cosine_similarity(user_vector, movie_vector)[0, 0]
            else:
                similarity = np.dot(user_vector, movie_vector.T)[0, 0]

            min_sim = 0.7
            max_sim = 1.0
            similarity_normalized = (similarity - min_sim) / (max_sim - min_sim)
            similarity_normalized = np.clip(similarity_normalized, 0, 1)
            rating = 0.5 + 4.5 * similarity_normalized
            rating = np.clip(rating, 0.5, 5.0)

            predictions.append(rating)

        return np.array(predictions)

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_seen: bool = True,
        candidate_movies: List[int] | None = None,
    ) -> List[Tuple[int, float]]:
        """
        Recommend movies similar to user's preference profile.

        Args:
            user_id: User ID
            top_k: Number of recommendations
            exclude_seen: Whether to exclude rated movies
            candidate_movies: Optional list of candidate movie IDs

        Returns:
            List of (movie_id, similarity_score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling recommend()")

        if user_id not in self.user_profiles:
            return self._recommend_popular(top_k)

        user_vector = self.user_profiles[user_id]

        if candidate_movies is None:
            candidate_movies = list(self.movie_embeddings.keys())

        similarities = []
        for movie_id in candidate_movies:
            if movie_id not in self.movie_embeddings:
                continue

            movie_vector = self.movie_embeddings[movie_id]

            if self.similarity_metric == "cosine":
                sim = cosine_similarity(
                    user_vector.reshape(1, -1),
                    movie_vector.reshape(1, -1),
                )[0, 0]
            else:
                sim = np.dot(user_vector, movie_vector)

            similarities.append((movie_id, float(sim)))

        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def _recommend_popular(self, top_k: int) -> List[Tuple[int, float]]:
        """Fallback: recommend popular movies for cold-start users."""
        if self.movie_metadata is None or "popularity" not in self.movie_metadata.columns:
            return [(mid, 1.0) for mid in list(self.movie_embeddings.keys())[:top_k]]

        popular = self.movie_metadata.nlargest(top_k, "popularity")[["movieId", "popularity"]].values.tolist()
        return [(int(mid), float(pop)) for mid, pop in popular]

    def get_params(self) -> dict:
        """Get model hyperparameters."""
        return {
            "name": self.name,
            "embedding_model": self.embedding_model_name,
            "similarity_metric": self.similarity_metric,
        }

    def save(self, path: Path):
        """Save model to disk."""
        import json

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.movie_embeddings:
            movie_ids = list(self.movie_embeddings.keys())
            embeddings = np.array([self.movie_embeddings[mid] for mid in movie_ids])
            np.save(path / "movie_embeddings.npy", embeddings)
            np.save(path / "movie_ids.npy", np.array(movie_ids))

        if self.user_profiles:
            user_ids = list(self.user_profiles.keys())
            profiles = np.array([self.user_profiles[uid] for uid in user_ids])
            np.save(path / "user_profiles.npy", profiles)
            np.save(path / "user_ids.npy", np.array(user_ids))

        with open(path / "params.json", "w") as f:
            json.dump(self.get_params(), f)

    def load(self, path: Path):
        """Load model from disk."""
        import json

        path = Path(path)

        embeddings_path = path / "movie_embeddings.npy"
        movie_ids_path = path / "movie_ids.npy"
        if embeddings_path.exists() and movie_ids_path.exists():
            embeddings = np.load(embeddings_path)
            movie_ids = np.load(movie_ids_path)
            self.movie_embeddings = {int(mid): emb for mid, emb in zip(movie_ids, embeddings)}

        profiles_path = path / "user_profiles.npy"
        user_ids_path = path / "user_ids.npy"
        if profiles_path.exists() and user_ids_path.exists():
            profiles = np.load(profiles_path)
            user_ids = np.load(user_ids_path)
            self.user_profiles = {int(uid): prof for uid, prof in zip(user_ids, profiles)}

        params_path = path / "params.json"
        if params_path.exists():
            with open(params_path) as f:
                params = json.load(f)
                self.embedding_model_name = params.get("embedding_model", self.embedding_model_name)
                self.similarity_metric = params.get("similarity_metric", self.similarity_metric)

        self.is_fitted = True
