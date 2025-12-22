"""
Global configuration for the recommendation system.

Set TEST_MODE=True to use the small MovieLens dataset (1MB, 100k ratings)
for faster development and testing.

Set TEST_MODE=False for full thesis experiments with 32M ratings.
"""

import os
from pathlib import Path

# =============================================================================
# TEST MODE - Set this to switch between small and large datasets
# =============================================================================
TEST_MODE = os.environ.get("TEST_MODE", "false").lower() == "true"

# =============================================================================
# Dataset Configuration
# =============================================================================

if TEST_MODE:
    # Small dataset for testing/development
    DATASET_NAME = "ml-latest-small"
    DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    DATASET_SIZE = "1 MB"
    NUM_RATINGS = "100,000"
    NUM_MOVIES = "9,000"
    NUM_USERS = "600"
    print("⚠️  TEST MODE ENABLED - Using ml-latest-small dataset")
else:
    # Full dataset for thesis
    DATASET_NAME = "ml-32m"
    DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"
    DATASET_SIZE = "900 MB"
    NUM_RATINGS = "32,000,000"
    NUM_MOVIES = "87,585"
    NUM_USERS = "200,948"

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "model" / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# MovieLens paths
ML_DIR = RAW_DIR / DATASET_NAME
ML_RATINGS = ML_DIR / "ratings.csv"
ML_MOVIES = ML_DIR / "movies.csv"
ML_TAGS = ML_DIR / "tags.csv"
ML_LINKS = ML_DIR / "links.csv"

# TMDB paths
TMDB_DIR = RAW_DIR / "tmdb"
TMDB_FILE = TMDB_DIR / "TMDB_movie_dataset_v11.csv"

# Output paths - use /test directory in TEST_MODE
if TEST_MODE:
    OUTPUT_BASE = PROJECT_ROOT / "test"
else:
    OUTPUT_BASE = PROJECT_ROOT / "model"

MODELS_DIR = OUTPUT_BASE / "saved_models"
METRICS_DIR = OUTPUT_BASE / "metrics"
PLOTS_DIR = OUTPUT_BASE / "plots"

# =============================================================================
# Model Hyperparameters (adjusted for TEST_MODE)
# =============================================================================

if TEST_MODE:
    # Improved training for testing
    COLLABORATIVE_PARAMS = {
        "n_factors": 100,  # More latent factors
        "n_epochs": 50,  # More epochs
        "lr_all": 0.005,
        "reg_all": 0.02,
    }

    KNN_PARAMS = {
        "k": 50,  # More neighbors
        "similarity": "cosine",
        "user_based": False,
    }

    NCF_PARAMS = {
        "embedding_dim": 64,  # Larger embeddings
        "mlp_layers": [128, 64, 32],  # Deeper network
        "dropout_rate": 0.3,  # More regularization
        "learning_rate": 0.0005,  # Lower LR for stability
    }
    NCF_EPOCHS = 50  # More epochs
    NCF_BATCH_SIZE = 64  # Smaller batches for better gradients

    TWO_TOWER_PARAMS = {
        "embedding_dim": 128,  # Larger capacity to prevent collapse
        "use_text_features": True,  # CRITICAL: Enable content differentiation
        "tower_layers": [256, 128],  # Deeper towers
        "learning_rate": 0.0005,  # Much lower LR to prevent embedding collapse
        "temperature": 0.1,  # Sharper softmax for retrieval loss
        "dropout_rate": 0.1,  # Less aggressive dropout
    }
    TWO_TOWER_EPOCHS = 30  # More epochs for convergence
    TWO_TOWER_BATCH_SIZE = 2048  # Smaller batches for better gradients
    TWO_TOWER_CHUNK_SIZE = 500_000  # Chunk size for memory-efficient generator training (smaller for test)

    CONTENT_BASED_PARAMS = {
        "embedding_model": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2",
        "similarity_metric": "cosine",
    }
else:
    # Full hyperparameters for thesis
    COLLABORATIVE_PARAMS = {
        "n_factors": 100,
        "n_epochs": 20,
        "lr_all": 0.005,
        "reg_all": 0.02,
    }

    KNN_PARAMS = {
        "k": 40,
        "similarity": "cosine",
        "user_based": False,
        "max_movies": 20000,  # Limit movies for memory efficiency
    }

    NCF_PARAMS = {
        "embedding_dim": 64,
        "mlp_layers": [128, 64, 32],
        "dropout_rate": 0.2,
        "learning_rate": 0.001,
    }
    NCF_EPOCHS = 10
    NCF_BATCH_SIZE = 256

    TWO_TOWER_PARAMS = {
        "embedding_dim": 128,  # Larger capacity to prevent collapse
        "use_text_features": True,  # Content differentiation
        "tower_layers": [256, 128],  # Deeper towers
        "learning_rate": 0.0005,  # Much lower LR to prevent embedding collapse
        "temperature": 0.1,  # Sharper softmax for retrieval loss
        "dropout_rate": 0.1,  # Less aggressive dropout
    }
    TWO_TOWER_EPOCHS = 20  # More epochs for convergence
    TWO_TOWER_BATCH_SIZE = 4096  # Smaller batches for better gradients
    TWO_TOWER_CHUNK_SIZE = 2_000_000  # Chunk size for memory-efficient generator training

    CONTENT_BASED_PARAMS = {
        "embedding_model": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2",
        "similarity_metric": "cosine",
    }


# =============================================================================
# Display Configuration
# =============================================================================


def print_config():
    """Print current configuration."""
    print("=" * 80)
    print("Configuration")
    print("=" * 80)
    print(f"Mode: {'TEST (ml-latest-small)' if TEST_MODE else 'PRODUCTION (ml-32m)'}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Size: {DATASET_SIZE}")
    print(f"Ratings: {NUM_RATINGS}")
    print(f"Movies: {NUM_MOVIES}")
    print(f"Users: {NUM_USERS}")
    print("=" * 80)


if __name__ == "__main__":
    print_config()
