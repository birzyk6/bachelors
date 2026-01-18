"""
Two-Tower Model for Recommendation.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_recommenders as tfrs
import tensorflow_text as text
from tensorflow import keras
from tensorflow.keras import layers

from .base import BaseRecommender


class L2Normalize(layers.Layer):
    """Custom layer for L2 normalization."""

    def call(self, inputs):
        return tf.nn.l2_normalize(inputs, axis=1)


class TwoTowerModel(BaseRecommender):
    """
    Two-Tower Recommendation Model.

    Learns separate user and movie embeddings optimized for retrieval.

    Args:
        user_ids: List of unique user IDs
        movie_ids: List of unique movie IDs
        embedding_dim: Dimension of final user/movie embeddings
        use_text_features: Whether to use BERT embeddings for movie overviews
        bert_model: BERT model name (if use_text_features=True)
        tower_layers: List of hidden layer sizes for both towers
        learning_rate: Learning rate for optimizer
        temperature: Temperature for softmax in retrieval loss (lower = sharper)
        dropout_rate: Dropout rate for regularization
    """

    def __init__(
        self,
        user_ids: List[int],
        movie_ids: List[int],
        embedding_dim: int = 128,
        use_text_features: bool = True,
        bert_model: str = "bert-base-uncased",
        tower_layers: List[int] | None = None,
        learning_rate: float = 0.0005,
        temperature: float = 0.1,
        dropout_rate: float = 0.1,
    ):
        super().__init__(name="TwoTower")

        self.user_ids = user_ids
        self.movie_ids = movie_ids
        self.embedding_dim = embedding_dim
        self.use_text_features = use_text_features
        self.bert_model_name = bert_model
        self.tower_layers = tower_layers or [256, 128]
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.dropout_rate = dropout_rate

        self.bert_preprocess = None
        self.bert_encoder = None
        if use_text_features:
            print("Loading TensorFlow Hub BERT for text encoding...")
            bert_encoder_url = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2"
            self.bert_encoder = hub.KerasLayer(bert_encoder_url, trainable=False)
            bert_preprocess_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
            self.bert_preprocess = hub.KerasLayer(bert_preprocess_url)

        self.user_tower = self._build_user_tower()
        self.movie_tower = self._build_movie_tower()

        self.content_only_tower = self._build_content_only_movie_tower()

        self.bert_embedding_dim = 512

        self.model = self._build_retrieval_model()

        self.movie_features = {}
        self.movie_embeddings_cache = None

        self.train_data = None

    def _build_user_tower(self) -> keras.Model:
        """
        Build Query Tower (User Model).

        Returns:
            Keras model mapping user features to embeddings
        """
        user_id_input = layers.Input(shape=(1,), dtype=tf.int32, name="user_id")

        user_embedding = layers.Embedding(
            input_dim=len(self.user_ids) + 1,
            output_dim=self.embedding_dim,
            name="user_embedding",
        )(user_id_input)
        user_embedding = layers.Flatten()(user_embedding)

        x = user_embedding
        for i, units in enumerate(self.tower_layers):
            x = layers.Dense(units, activation="relu", name=f"user_dense_{i}")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout_rate)(x)

        user_output = layers.Dense(
            self.embedding_dim,
            activation=None,
            name="user_output_embedding",
        )(x)

        user_output = L2Normalize(name="user_l2_normalize")(user_output)

        return keras.Model(inputs=user_id_input, outputs=user_output, name="user_tower")

    def _build_movie_tower(self) -> keras.Model:
        """
        Build Candidate Tower (Movie Model).

        Returns:
            Keras model mapping movie features to embeddings
        """
        inputs = {}

        movie_id_input = layers.Input(shape=(1,), dtype=tf.int32, name="movie_id")
        inputs["movie_id"] = movie_id_input

        movie_embedding = layers.Embedding(
            input_dim=len(self.movie_ids) + 1,
            output_dim=self.embedding_dim,
            name="movie_embedding",
        )(movie_id_input)
        movie_embedding = layers.Flatten()(movie_embedding)

        features = [movie_embedding]

        if self.use_text_features:
            text_input = layers.Input(shape=(512,), dtype=tf.float32, name="text_embedding")
            inputs["text_embedding"] = text_input
            features.append(text_input)

        if len(features) > 1:
            x = layers.Concatenate()(features)
        else:
            x = features[0]

        for i, units in enumerate(self.tower_layers):
            x = layers.Dense(units, activation="relu", name=f"movie_dense_{i}")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout_rate)(x)

        movie_output = layers.Dense(
            self.embedding_dim,
            activation=None,
            name="movie_output_embedding",
        )(x)

        movie_output = L2Normalize(name="movie_l2_normalize")(movie_output)

        return keras.Model(inputs=inputs, outputs=movie_output, name="movie_tower")

    def _build_content_only_movie_tower(self) -> keras.Model:
        """
        Build Content-Only Movie Tower for cold-start movies.

        This tower uses ONLY BERT text embeddings (no movie ID), enabling
        embedding generation for any movie with text, including those
        not in the training set (cold-start).

        Returns:
            Keras model mapping text embedding to movie embedding
        """
        text_input = layers.Input(shape=(512,), dtype=tf.float32, name="text_embedding")

        x = text_input
        for i, units in enumerate(self.tower_layers):
            x = layers.Dense(units, activation="relu", name=f"content_dense_{i}")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout_rate)(x)

        content_output = layers.Dense(
            self.embedding_dim,
            activation=None,
            name="content_output_embedding",
        )(x)

        content_output = L2Normalize(name="content_l2_normalize")(content_output)

        return keras.Model(inputs=text_input, outputs=content_output, name="content_only_tower")

    def _build_retrieval_model(self) -> tfrs.Model:
        """
        Build full two-tower retrieval model using TFRS.

        Returns:
            TFRS retrieval model
        """
        temperature = self.temperature

        class TwoTowerRetrievalModel(tfrs.Model):
            def __init__(self, user_model, movie_model, movie_dataset, temp):
                super().__init__()
                self.user_model = user_model
                self.movie_model = movie_model
                self.temp = temp

                self.task = tfrs.tasks.Retrieval(temperature=temp)

            def compute_loss(self, features, training=False):
                user_embeddings = self.user_model(features["user_id"])

                movie_inputs = {"movie_id": features["movie_id"]}
                if "text_embedding" in features:
                    movie_inputs["text_embedding"] = features["text_embedding"]

                movie_embeddings = self.movie_model(movie_inputs)

                return self.task(user_embeddings, movie_embeddings)

        dummy_movie_data = tf.data.Dataset.from_tensor_slices({"movie_id": tf.constant([0], dtype=tf.int32)})

        if self.use_text_features:
            dummy_movie_data = dummy_movie_data.map(
                lambda x: {
                    "movie_id": x["movie_id"],
                    "text_embedding": tf.zeros((512,), dtype=tf.float32),
                }
            )

        model = TwoTowerRetrievalModel(
            user_model=self.user_tower,
            movie_model=self.movie_tower,
            movie_dataset=dummy_movie_data,
            temp=temperature,
        )

        return model

    def _precompute_text_embeddings(self, movie_data: pd.DataFrame) -> Dict[int, np.ndarray]:
        """
        Pre-compute BERT embeddings for all movies.

        First tries to load pre-computed embeddings from disk.
        Falls back to computing them if not found.

        Args:
            movie_data: DataFrame with [movieId, overview]

        Returns:
            Dictionary mapping movie_id to text embedding
        """
        if not self.use_text_features:
            return {}

        from pathlib import Path

        try:
            from config import MODELS_DIR
        except ImportError:
            MODELS_DIR = Path(__file__).parent.parent.parent / "saved_models"

        precomputed_path = MODELS_DIR / "bert_embeddings.npz"

        if precomputed_path.exists():
            print(f"Loading pre-computed BERT embeddings from {precomputed_path}...")
            data = np.load(precomputed_path)
            saved_embeddings = data["embeddings"]
            saved_movie_ids = data["movie_ids"]

            embeddings = {}
            for mid, emb in zip(saved_movie_ids, saved_embeddings):
                embeddings[int(mid)] = emb

            print(f"  ✓ Loaded {len(embeddings):,} pre-computed text embeddings")
            return embeddings

        print("Pre-computing BERT embeddings for movies...")
        print("  (Tip: Run 'python precompute_embeddings.py' first to avoid this step)")

        embeddings = {}
        overviews = movie_data["overview"].fillna("").tolist()
        movie_ids = movie_data["movieId"].tolist()

        batch_size = 32
        for i in range(0, len(overviews), batch_size):
            batch_texts = overviews[i : i + batch_size]
            batch_ids = movie_ids[i : i + batch_size]

            preprocessed = self.bert_preprocess(batch_texts)
            outputs = self.bert_encoder(preprocessed)

            cls_embeddings = outputs["pooled_output"].numpy()

            for movie_id, embedding in zip(batch_ids, cls_embeddings):
                embeddings[movie_id] = embedding

        print(f"  ✓ Pre-computed {len(embeddings)} text embeddings")

        return embeddings

    def fit(
        self,
        train_data: pd.DataFrame,
        movie_data: pd.DataFrame,
        validation_data: pd.DataFrame | None = None,
        epochs: int = 5,
        batch_size: int = 8192,
        verbose: int = 1,
        use_generator: bool = True,
        chunk_size: int = 1_000_000,
    ):
        """
        Train the two-tower model.

        Args:
            train_data: DataFrame with [userId, movieId, rating] OR Path to parquet file
            movie_data: DataFrame with movie features [movieId, overview, ...]
            validation_data: Optional validation DataFrame
            epochs: Number of epochs
            batch_size: Batch size
            verbose: Verbosity
            use_generator: If True, use memory-efficient generator for large datasets
            chunk_size: Number of samples per chunk when using generator

        Returns:
            Training history
        """
        print(f"Training {self.name} model...")

        text_embeddings = {}
        if self.use_text_features:
            text_embeddings = self._precompute_text_embeddings(movie_data)
            self.movie_features = text_embeddings

        user_id_map = {uid: idx for idx, uid in enumerate(self.user_ids)}
        movie_id_map = {mid: idx for idx, mid in enumerate(self.movie_ids)}

        num_samples = len(train_data)
        use_generator = use_generator and num_samples > 5_000_000

        if use_generator:
            print(f"Using memory-efficient generator for {num_samples:,} samples...")
            return self._fit_with_generator(
                train_data=train_data,
                movie_data=movie_data,
                text_embeddings=text_embeddings,
                user_id_map=user_id_map,
                movie_id_map=movie_id_map,
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose,
                chunk_size=chunk_size,
            )

        print("Preparing training data...")
        user_indices = train_data["userId"].map(user_id_map).fillna(0).astype(np.int32).values
        movie_indices = train_data["movieId"].map(movie_id_map).fillna(0).astype(np.int32).values
        movie_ids_list = train_data["movieId"].values
        print(f"  {num_samples:,} training samples")

        if self.use_text_features:
            num_movies = len(self.movie_ids)
            text_emb_matrix = np.zeros((num_movies, 512), dtype=np.float32)
            for mid, emb in text_embeddings.items():
                if mid in movie_id_map:
                    text_emb_matrix[movie_id_map[mid]] = emb
            print(f"  Text embedding matrix: {text_emb_matrix.shape}")

            def create_features(user_id, movie_id):
                text_emb = tf.gather(text_emb_matrix, movie_id)
                return {
                    "user_id": user_id,
                    "movie_id": movie_id,
                    "text_embedding": text_emb,
                }

            train_dataset = tf.data.Dataset.from_tensor_slices((user_indices, movie_indices))
            train_dataset = train_dataset.map(create_features, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            train_dataset = tf.data.Dataset.from_tensor_slices(
                {
                    "user_id": user_indices,
                    "movie_id": movie_indices,
                }
            )

        train_dataset = train_dataset.shuffle(100_000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        self.model.compile(optimizer=tf.keras.optimizers.legacy.Adam(self.learning_rate))

        print("Starting training...")
        history = self.model.fit(
            train_dataset,
            epochs=epochs,
            verbose=verbose,
        )

        self.train_data = train_data[["userId", "movieId"]].copy()
        self.is_fitted = True

        self._precompute_movie_embeddings(movie_data, text_embeddings)

        return history

    def _fit_with_generator(
        self,
        train_data: pd.DataFrame,
        movie_data: pd.DataFrame,
        text_embeddings: Dict,
        user_id_map: Dict,
        movie_id_map: Dict,
        epochs: int,
        batch_size: int,
        verbose: int,
        chunk_size: int,
    ):
        """
        Memory-efficient training using a generator that processes data in chunks.
        """
        import gc

        num_samples = len(train_data)
        print(f"  {num_samples:,} training samples")

        text_emb_matrix = None
        if self.use_text_features:
            num_movies = len(self.movie_ids)
            text_emb_matrix = np.zeros((num_movies, 512), dtype=np.float32)
            for mid, emb in text_embeddings.items():
                if mid in movie_id_map:
                    text_emb_matrix[movie_id_map[mid]] = emb
            print(f"  Text embedding matrix: {text_emb_matrix.shape}")

        steps_per_epoch = (num_samples + batch_size - 1) // batch_size

        def data_generator():
            """Generator that yields individual samples by processing chunks."""
            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            for chunk_start in range(0, num_samples, chunk_size):
                chunk_end = min(chunk_start + chunk_size, num_samples)
                chunk_indices = indices[chunk_start:chunk_end]

                chunk_data = train_data.iloc[chunk_indices]

                user_indices = chunk_data["userId"].map(user_id_map).fillna(0).astype(np.int32).values
                movie_indices = chunk_data["movieId"].map(movie_id_map).fillna(0).astype(np.int32).values

                for i in range(len(user_indices)):
                    if self.use_text_features:
                        yield (user_indices[i], movie_indices[i], text_emb_matrix[movie_indices[i]])
                    else:
                        yield (user_indices[i], movie_indices[i])

                del chunk_data, user_indices, movie_indices
                gc.collect()

        if self.use_text_features:
            output_signature = (
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(512,), dtype=tf.float32),
            )

            def process_sample(user_id, movie_id, text_emb):
                return {
                    "user_id": user_id,
                    "movie_id": movie_id,
                    "text_embedding": text_emb,
                }

        else:
            output_signature = (
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
            )

            def process_sample(user_id, movie_id):
                return {
                    "user_id": user_id,
                    "movie_id": movie_id,
                }

        train_dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_signature=output_signature,
        )

        train_dataset = (
            train_dataset.map(process_sample, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        )

        self.model.compile(optimizer=tf.keras.optimizers.legacy.Adam(self.learning_rate))

        print("Starting training...")
        all_history = {"loss": []}

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            epoch_dataset = (
                tf.data.Dataset.from_generator(
                    data_generator,
                    output_signature=output_signature,
                )
                .map(process_sample, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )

            history = self.model.fit(
                epoch_dataset,
                epochs=1,
                verbose=verbose,
                steps_per_epoch=steps_per_epoch,
            )

            all_history["loss"].append(history.history["loss"][0])

            for key in history.history:
                if key != "loss":
                    if key not in all_history:
                        all_history[key] = []
                    all_history[key].append(history.history[key][0])

            gc.collect()

        class HistoryWrapper:
            def __init__(self, history_dict):
                self.history = history_dict

        self.train_data = train_data[["userId", "movieId"]].copy()
        self.is_fitted = True

        self._precompute_movie_embeddings(movie_data, text_embeddings)

        return HistoryWrapper(all_history)

    def _precompute_movie_embeddings(self, movie_data: pd.DataFrame, text_embeddings: Dict):
        """Pre-compute embeddings for all movies."""
        print("Pre-computing movie embeddings...")

        movie_id_map = {mid: idx for idx, mid in enumerate(self.movie_ids)}

        movie_ids = []
        embeddings = []

        for _, row in movie_data.iterrows():
            movie_id = row["movieId"]
            if movie_id not in movie_id_map:
                continue

            inputs = {"movie_id": np.array([[movie_id_map[movie_id]]], dtype=np.int32)}

            if self.use_text_features and movie_id in text_embeddings:
                inputs["text_embedding"] = text_embeddings[movie_id].reshape(1, -1)

            embedding = self.movie_tower(inputs, training=False).numpy()[0]

            movie_ids.append(movie_id)
            embeddings.append(embedding)

        self.movie_embeddings_cache = {mid: emb for mid, emb in zip(movie_ids, embeddings)}

        print(f"  ✓ Cached {len(self.movie_embeddings_cache)} movie embeddings")

    def predict(self, user_movie_pairs: pd.DataFrame) -> np.ndarray:
        """
        Predict affinity scores for user-movie pairs.

        Args:
            user_movie_pairs: DataFrame with [userId, movieId]

        Returns:
            Array of affinity scores (dot product of embeddings)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predict()")

        raise NotImplementedError("Use recommend() for two-tower retrieval")

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_seen: bool = True,
        candidate_movies: List[int] | None = None,
    ) -> List[Tuple[int, float]]:
        """
        Retrieve top-K movies for a user using ANN search.

        Args:
            user_id: User ID
            top_k: Number of recommendations
            exclude_seen: Whether to exclude rated movies
            candidate_movies: Optional candidate list

        Returns:
            List of (movie_id, score) tuples
        """
        if not self.is_fitted or self.movie_embeddings_cache is None:
            raise ValueError("Model must be fitted before recommend()")

        user_id_map = {uid: idx for idx, uid in enumerate(self.user_ids)}
        if user_id not in user_id_map:
            return []

        user_idx = user_id_map[user_id]
        user_emb = self.user_tower(np.array([[user_idx]]), training=False).numpy()[0]

        if candidate_movies is None:
            candidate_movies = list(self.movie_embeddings_cache.keys())

        scores = []
        for movie_id in candidate_movies:
            if movie_id not in self.movie_embeddings_cache:
                continue

            movie_emb = self.movie_embeddings_cache[movie_id]
            score = np.dot(user_emb, movie_emb)
            scores.append((movie_id, float(score)))

        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]

    def save_user_tower(self, path: Path):
        """Save user tower for deployment."""
        self.user_tower.save(Path(path))
        print(f"✓ User tower saved to {path}")

    def save_movie_tower(self, path: Path):
        """Save movie tower for deployment."""
        self.movie_tower.save(Path(path))
        print(f"✓ Movie tower saved to {path}")

    def save_content_only_tower(self, path: Path):
        """Save content-only tower for cold-start inference."""
        self.content_only_tower.save(Path(path))
        print(f"✓ Content-only tower saved to {path}")

    def save_movie_embeddings(self, path: Path):
        """Save pre-computed movie embeddings for Qdrant."""
        if self.movie_embeddings_cache is None:
            raise ValueError("Movie embeddings not computed")

        np.save(path, self.movie_embeddings_cache)
        print(f"✓ Movie embeddings saved to {path}")

    def get_cold_start_embedding(self, text_embedding: np.ndarray) -> np.ndarray:
        """
        Get embedding for a cold-start movie using only its text features.

        This enables recommendations for movies not in the training set,
        using the content-only tower that processes BERT embeddings.

        Args:
            text_embedding: 512-dim BERT embedding of movie text (title + overview)

        Returns:
            Movie embedding in the same space as trained movies
        """
        if text_embedding.ndim == 1:
            text_embedding = text_embedding.reshape(1, -1)

        return self.content_only_tower(text_embedding, training=False).numpy()[0]

    def get_cold_start_embeddings_batch(self, text_embeddings: np.ndarray) -> np.ndarray:
        """
        Get embeddings for multiple cold-start movies in batch.

        Args:
            text_embeddings: (N, 512) array of BERT embeddings

        Returns:
            (N, embedding_dim) array of movie embeddings
        """
        return self.content_only_tower(text_embeddings, training=False).numpy()

    def get_params(self) -> dict:
        """Get model hyperparameters."""
        return {
            "name": self.name,
            "embedding_dim": self.embedding_dim,
            "use_text_features": self.use_text_features,
            "bert_model": self.bert_model_name,
            "tower_layers": self.tower_layers,
            "learning_rate": self.learning_rate,
            "temperature": self.temperature,
            "dropout_rate": self.dropout_rate,
        }
