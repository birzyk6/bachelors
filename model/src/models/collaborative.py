"""
Collaborative Filtering model using SVD (Singular Value Decomposition).

This is a baseline model using matrix factorization via the Surprise library.
Implements User-User collaborative filtering through latent factor models.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate

from .base import BaseRecommender


class CollaborativeFilteringModel(BaseRecommender):
    """
    Collaborative Filtering using SVD from the Surprise library.

    This model decomposes the user-item rating matrix into latent factors,
    learning user and item embeddings that capture preferences.

    Args:
        n_factors: Number of latent factors (embedding dimension)
        n_epochs: Number of training epochs
        lr_all: Learning rate for all parameters
        reg_all: Regularization term for all parameters
        random_state: Random seed for reproducibility
    """

    def __init__(
        self,
        n_factors: int = 100,
        n_epochs: int = 20,
        lr_all: float = 0.005,
        reg_all: float = 0.02,
        random_state: int = 42,
    ):
        super().__init__(name="CollaborativeFiltering")
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.random_state = random_state

        # Initialize Surprise SVD model
        self.model = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
            random_state=random_state,
        )

        self.train_data = None
        self.user_rated_items = {}  # Cache for excluding seen items

    def fit(self, train_data: pd.DataFrame, verbose: bool = True) -> "CollaborativeFilteringModel":
        """
        Train the SVD model on rating data.

        Args:
            train_data: DataFrame with columns [userId, movieId, rating]
            verbose: Whether to print training progress

        Returns:
            self
        """
        if verbose:
            print(f"Training {self.name} model...")
            print(f"  Data shape: {train_data.shape}")
            print(f"  Factors: {self.n_factors}, Epochs: {self.n_epochs}")

        # Convert to Surprise Dataset format
        reader = Reader(rating_scale=(0.5, 5.0))
        dataset = Dataset.load_from_df(
            train_data[["userId", "movieId", "rating"]],
            reader,
        )

        # Build full trainset
        trainset = dataset.build_full_trainset()

        # Train model
        self.model.fit(trainset)

        # Cache training data for excluding seen items
        self.train_data = train_data
        self.user_rated_items = train_data.groupby("userId")["movieId"].apply(set).to_dict()

        self.is_fitted = True

        if verbose:
            print(f"  ✓ Model trained on {trainset.n_ratings} ratings")

        return self

    def predict(self, user_movie_pairs: pd.DataFrame) -> np.ndarray:
        """
        Predict ratings for user-movie pairs.

        Args:
            user_movie_pairs: DataFrame with columns [userId, movieId]

        Returns:
            Array of predicted ratings
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling predict()")

        predictions = []
        for _, row in user_movie_pairs.iterrows():
            pred = self.model.predict(
                uid=row["userId"],
                iid=row["movieId"],
                verbose=False,
            )
            predictions.append(pred.est)

        return np.array(predictions)

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_seen: bool = True,
        candidate_movies: List[int] | None = None,
    ) -> List[Tuple[int, float]]:
        """
        Generate top-K movie recommendations for a user.

        Args:
            user_id: User ID
            top_k: Number of recommendations
            exclude_seen: Whether to exclude rated movies
            candidate_movies: Optional list of movies to rank. If None, uses all movies.

        Returns:
            List of (movie_id, predicted_rating) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling recommend()")

        # Get all movies if candidates not specified
        if candidate_movies is None:
            candidate_movies = self.train_data["movieId"].unique()

        # Exclude already rated movies
        if exclude_seen and user_id in self.user_rated_items:
            seen_movies = self.user_rated_items[user_id]
            candidate_movies = [m for m in candidate_movies if m not in seen_movies]

        # Predict ratings for all candidates
        predictions = []
        for movie_id in candidate_movies:
            pred = self.model.predict(user_id, movie_id, verbose=False)
            predictions.append((movie_id, pred.est))

        # Sort by predicted rating descending
        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:top_k]

    def get_params(self) -> dict:
        """Get model hyperparameters."""
        return {
            "name": self.name,
            "n_factors": self.n_factors,
            "n_epochs": self.n_epochs,
            "lr_all": self.lr_all,
            "reg_all": self.reg_all,
        }

    def cross_validate(self, data: pd.DataFrame, cv: int = 5) -> dict:
        """
        Perform cross-validation on the model.

        Args:
            data: Full dataset DataFrame
            cv: Number of folds

        Returns:
            Dictionary with mean RMSE and MAE
        """
        reader = Reader(rating_scale=(0.5, 5.0))
        dataset = Dataset.load_from_df(
            data[["userId", "movieId", "rating"]],
            reader,
        )

        results = cross_validate(
            self.model,
            dataset,
            measures=["RMSE", "MAE"],
            cv=cv,
            verbose=True,
        )

        return {
            "rmse_mean": results["test_rmse"].mean(),
            "rmse_std": results["test_rmse"].std(),
            "mae_mean": results["test_mae"].mean(),
            "mae_std": results["test_mae"].std(),
        }

    def save(self, path: Path):
        """Save model to disk."""
        import pickle

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save Surprise model
        with open(path / "model.pkl", "wb") as f:
            pickle.dump(self.model, f)

        # Save training data for user_rated_items
        if self.train_data is not None:
            self.train_data.to_parquet(path / "train_data.parquet")

        # Save hyperparameters
        import json

        with open(path / "params.json", "w") as f:
            json.dump(self.get_params(), f)

    def load(self, path: Path):
        """Load model from disk."""
        import json
        import pickle

        path = Path(path)

        # Load Surprise model
        with open(path / "model.pkl", "rb") as f:
            self.model = pickle.load(f)

        # Load training data if available
        train_data_path = path / "train_data.parquet"
        if train_data_path.exists():
            self.train_data = pd.read_parquet(train_data_path)
            self.user_rated_items = self.train_data.groupby("userId")["movieId"].apply(set).to_dict()

        # Load params
        with open(path / "params.json") as f:
            params = json.load(f)
            self.n_factors = params.get("n_factors", self.n_factors)
            self.n_epochs = params.get("n_epochs", self.n_epochs)
            self.lr_all = params.get("lr_all", self.lr_all)
            self.reg_all = params.get("reg_all", self.reg_all)

        self.is_fitted = True
