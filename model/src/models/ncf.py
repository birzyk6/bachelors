"""
Neural Collaborative Filtering (NCF) implementation.

Combines Generalized Matrix Factorization (GMF) with Multi-Layer Perceptron (MLP)
for deep learning-based collaborative filtering.

Reference:
    He, X., et al. (2017). Neural Collaborative Filtering. WWW 2017.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .base import BaseRecommender


class NeuralCollaborativeFiltering(BaseRecommender):
    """
    Neural Collaborative Filtering (NCF) model.

    Architecture:
        - GMF Path: Element-wise product of user and item embeddings
        - MLP Path: Concatenation of embeddings through dense layers
        - Fusion: Concatenate GMF and MLP outputs -> final prediction

    Args:
        num_users: Number of unique users
        num_movies: Number of unique movies
        embedding_dim: Dimension of embeddings
        mlp_layers: List of hidden layer sizes for MLP path
        dropout_rate: Dropout rate for regularization
        learning_rate: Learning rate for Adam optimizer
    """

    def __init__(
        self,
        num_users: int,
        num_movies: int,
        embedding_dim: int = 64,
        mlp_layers: List[int] | None = None,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
    ):
        super().__init__(name="NeuralCollaborativeFiltering")

        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_dim = embedding_dim
        self.mlp_layers = mlp_layers or [128, 64, 32]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

        self.model = self._build_model()

        self.user_id_map = {}
        self.movie_id_map = {}
        self.reverse_movie_map = {}

        self.train_data = None
        self.user_rated_items = {}

    def _build_model(self) -> keras.Model:
        """
        Build NCF model architecture.

        Returns:
            Compiled Keras model
        """
        user_input = layers.Input(shape=(1,), name="user_id")
        movie_input = layers.Input(shape=(1,), name="movie_id")

        gmf_user_embedding = layers.Embedding(
            input_dim=self.num_users,
            output_dim=self.embedding_dim,
            name="gmf_user_embedding",
        )(user_input)
        gmf_user_flat = layers.Flatten()(gmf_user_embedding)

        gmf_movie_embedding = layers.Embedding(
            input_dim=self.num_movies,
            output_dim=self.embedding_dim,
            name="gmf_movie_embedding",
        )(movie_input)
        gmf_movie_flat = layers.Flatten()(gmf_movie_embedding)

        gmf_vector = layers.Multiply()([gmf_user_flat, gmf_movie_flat])

        mlp_user_embedding = layers.Embedding(
            input_dim=self.num_users,
            output_dim=self.embedding_dim,
            name="mlp_user_embedding",
        )(user_input)
        mlp_user_flat = layers.Flatten()(mlp_user_embedding)

        mlp_movie_embedding = layers.Embedding(
            input_dim=self.num_movies,
            output_dim=self.embedding_dim,
            name="mlp_movie_embedding",
        )(movie_input)
        mlp_movie_flat = layers.Flatten()(mlp_movie_embedding)

        mlp_vector = layers.Concatenate()([mlp_user_flat, mlp_movie_flat])

        for i, units in enumerate(self.mlp_layers):
            mlp_vector = layers.Dense(
                units,
                activation="relu",
                name=f"mlp_layer_{i}",
            )(mlp_vector)
            mlp_vector = layers.Dropout(self.dropout_rate)(mlp_vector)

        fusion = layers.Concatenate()([gmf_vector, mlp_vector])

        output = layers.Dense(1, activation="linear", name="rating")(fusion)

        model = keras.Model(
            inputs=[user_input, movie_input],
            outputs=output,
            name="NCF",
        )

        return model

    def compile(self, optimizer=None, loss="mse", metrics=None):
        """
        Compile the model.

        Args:
            optimizer: Keras optimizer (defaults to Adam)
            loss: Loss function
            metrics: List of metrics to track
        """
        if optimizer is None:
            optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)

        if metrics is None:
            metrics = ["mae"]

        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
        )

    def fit(
        self,
        train_data: pd.DataFrame,
        validation_data: pd.DataFrame | None = None,
        epochs: int = 10,
        batch_size: int = 256,
        verbose: int = 1,
        callbacks: List | None = None,
    ):
        """
        Train the NCF model.

        Args:
            train_data: DataFrame with [userId, movieId, rating]
            validation_data: Optional validation DataFrame
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity mode
            callbacks: List of Keras callbacks

        Returns:
            Training history
        """
        print(f"Training {self.name} model...")

        unique_users = sorted(train_data["userId"].unique())
        unique_movies = sorted(train_data["movieId"].unique())

        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.movie_id_map = {mid: idx for idx, mid in enumerate(unique_movies)}
        self.reverse_movie_map = {idx: mid for mid, idx in self.movie_id_map.items()}

        X_train = {
            "user_id": np.array([self.user_id_map[uid] for uid in train_data["userId"]]),
            "movie_id": np.array([self.movie_id_map[mid] for mid in train_data["movieId"]]),
        }
        y_train = train_data["rating"].values

        val_data = None
        if validation_data is not None:
            val_filtered = validation_data[
                validation_data["userId"].isin(self.user_id_map) & validation_data["movieId"].isin(self.movie_id_map)
            ]

            X_val = {
                "user_id": np.array([self.user_id_map[uid] for uid in val_filtered["userId"]]),
                "movie_id": np.array([self.movie_id_map[mid] for mid in val_filtered["movieId"]]),
            }
            y_val = val_filtered["rating"].values
            val_data = (X_val, y_val)

        self.train_data = train_data
        self.user_rated_items = train_data.groupby("userId")["movieId"].apply(set).to_dict()

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks,
        )

        self.is_fitted = True

        return history

    def predict(self, user_movie_pairs: pd.DataFrame) -> np.ndarray:
        """
        Predict ratings for user-movie pairs.

        Args:
            user_movie_pairs: DataFrame with [userId, movieId]

        Returns:
            Array of predicted ratings
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling predict()")

        valid_pairs = user_movie_pairs[
            user_movie_pairs["userId"].isin(self.user_id_map) & user_movie_pairs["movieId"].isin(self.movie_id_map)
        ].copy()

        if len(valid_pairs) == 0:
            return np.full(len(user_movie_pairs), 2.5)

        X = {
            "user_id": np.array([self.user_id_map[uid] for uid in valid_pairs["userId"]]),
            "movie_id": np.array([self.movie_id_map[mid] for mid in valid_pairs["movieId"]]),
        }

        valid_predictions = self.model.predict(X, verbose=0).flatten()

        valid_predictions = np.clip(valid_predictions, 0.5, 5.0)

        all_predictions = np.full(len(user_movie_pairs), 2.5)
        valid_mask = user_movie_pairs["userId"].isin(self.user_id_map) & user_movie_pairs["movieId"].isin(self.movie_id_map)
        all_predictions[valid_mask] = valid_predictions

        return all_predictions

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_seen: bool = True,
        candidate_movies: List[int] | None = None,
    ) -> List[Tuple[int, float]]:
        """
        Generate top-K recommendations for a user.

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

        if user_id not in self.user_id_map:
            return []

        if candidate_movies is None:
            candidate_movies = list(self.movie_id_map.keys())

        if exclude_seen and user_id in self.user_rated_items:
            seen = self.user_rated_items[user_id]
            candidate_movies = [m for m in candidate_movies if m not in seen]

        candidate_movies = [m for m in candidate_movies if m in self.movie_id_map]

        if not candidate_movies:
            return []

        user_ids = [user_id] * len(candidate_movies)
        pairs = pd.DataFrame({"userId": user_ids, "movieId": candidate_movies})

        predictions = self.predict(pairs)

        recommendations = list(zip(candidate_movies, predictions))
        recommendations.sort(key=lambda x: x[1], reverse=True)

        return recommendations[:top_k]

    def save(self, path: Path):
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.model.save(path / "ncf_model.keras")

        import json

        with open(path / "mappings.json", "w") as f:
            json.dump(
                {
                    "user_id_map": {int(k): int(v) for k, v in self.user_id_map.items()},
                    "movie_id_map": {int(k): int(v) for k, v in self.movie_id_map.items()},
                },
                f,
            )

    def load(self, path: Path):
        """Load model from disk."""
        import json

        import keras

        path = Path(path)

        with open(path / "mappings.json") as f:
            mappings = json.load(f)
            self.user_id_map = {int(k): int(v) for k, v in mappings["user_id_map"].items()}
            self.movie_id_map = {int(k): int(v) for k, v in mappings["movie_id_map"].items()}
            self.reverse_movie_map = {v: k for k, v in self.movie_id_map.items()}

        self.model = keras.models.load_model(path / "ncf_model.keras")
        self.is_fitted = True

    def get_params(self) -> dict:
        """Get model hyperparameters."""
        return {
            "name": self.name,
            "num_users": self.num_users,
            "num_movies": self.num_movies,
            "embedding_dim": self.embedding_dim,
            "mlp_layers": self.mlp_layers,
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
        }
