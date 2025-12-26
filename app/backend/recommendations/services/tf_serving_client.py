"""TensorFlow Serving client for user embeddings."""

import logging
from typing import List, Optional

import numpy as np
import requests
from django.conf import settings

logger = logging.getLogger(__name__)


class TFServingClient:
    """Client for TensorFlow Serving to get user embeddings."""

    def __init__(self):
        self.base_url = f"http://{settings.TF_SERVING_HOST}:{settings.TF_SERVING_PORT}"
        self.model_name = "user_tower"

    def get_user_embedding(self, user_index: int) -> Optional[List[float]]:
        """
        Get embedding for a user by their index.

        Args:
            user_index: The index of the user in the model's vocabulary

        Returns:
            128-dimensional user embedding or None if error
        """
        url = f"{self.base_url}/v1/models/{self.model_name}:predict"

        payload = {"instances": [{"user_id": [[user_index]]}]}

        try:
            response = requests.post(url, json=payload, timeout=5)
            response.raise_for_status()

            result = response.json()
            predictions = result.get("predictions", [])

            if predictions:
                embedding = predictions[0]
                # Normalize to unit vector
                embedding = np.array(embedding)
                embedding = embedding / np.linalg.norm(embedding)
                return embedding.tolist()

            return None

        except requests.exceptions.RequestException as e:
            logger.error(f"TF Serving request failed: {e}")
            return None

    def get_batch_user_embeddings(self, user_indices: List[int]) -> List[Optional[List[float]]]:
        """
        Get embeddings for multiple users.

        Args:
            user_indices: List of user indices

        Returns:
            List of embeddings (None for failed requests)
        """
        url = f"{self.base_url}/v1/models/{self.model_name}:predict"

        payload = {"instances": [{"user_id": [[idx]]} for idx in user_indices]}

        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()

            result = response.json()
            predictions = result.get("predictions", [])

            embeddings = []
            for pred in predictions:
                embedding = np.array(pred)
                embedding = embedding / np.linalg.norm(embedding)
                embeddings.append(embedding.tolist())

            return embeddings

        except requests.exceptions.RequestException as e:
            logger.error(f"TF Serving batch request failed: {e}")
            return [None] * len(user_indices)

    def health_check(self) -> bool:
        """Check if TensorFlow Serving is healthy."""
        url = f"{self.base_url}/v1/models/{self.model_name}"

        try:
            response = requests.get(url, timeout=2)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def get_model_status(self) -> dict:
        """Get model status from TensorFlow Serving."""
        url = f"{self.base_url}/v1/models/{self.model_name}"

        try:
            response = requests.get(url, timeout=2)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
