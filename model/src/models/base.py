"""
Base recommender class defining the interface for all models.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


class BaseRecommender(ABC):
    """
    Abstract base class for all recommendation models.

    All models must implement:
    - fit(): Train the model
    - predict(): Predict ratings for user-item pairs
    - recommend(): Generate top-K recommendations for users
    """

    def __init__(self, name: str):
        """
        Initialize base recommender.

        Args:
            name: Model name for logging and tracking
        """
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, train_data: pd.DataFrame, **kwargs) -> "BaseRecommender":
        """
        Train the recommendation model.

        Args:
            train_data: Training DataFrame with columns [userId, movieId, rating]
            **kwargs: Additional model-specific parameters

        Returns:
            self (for method chaining)
        """
        pass

    @abstractmethod
    def predict(self, user_movie_pairs: pd.DataFrame) -> np.ndarray:
        """
        Predict ratings for user-movie pairs.

        Args:
            user_movie_pairs: DataFrame with columns [userId, movieId]

        Returns:
            Array of predicted ratings
        """
        pass

    @abstractmethod
    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_seen: bool = True,
    ) -> List[Tuple[int, float]]:
        """
        Generate top-K movie recommendations for a user.

        Args:
            user_id: User ID to generate recommendations for
            top_k: Number of recommendations to return
            exclude_seen: Whether to exclude movies the user has already rated

        Returns:
            List of (movie_id, predicted_score) tuples, sorted by score descending
        """
        pass

    def save(self, path: Path) -> None:
        """
        Save model to disk.

        Args:
            path: Directory to save model artifacts
        """
        raise NotImplementedError(f"{self.name} does not implement save()")

    @classmethod
    def load(cls, path: Path) -> "BaseRecommender":
        """
        Load model from disk.

        Args:
            path: Directory containing model artifacts

        Returns:
            Loaded model instance
        """
        raise NotImplementedError(f"{cls.__name__} does not implement load()")

    def get_params(self) -> Dict[str, Any]:
        """
        Get model hyperparameters.

        Returns:
            Dictionary of parameter names and values
        """
        return {"name": self.name}

    def __repr__(self) -> str:
        params = self.get_params()
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{self.__class__.__name__}({param_str})"
