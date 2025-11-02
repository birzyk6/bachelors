# Data Preprocessing Summary

## Datasets

-   **MovieLens 32M**: User ratings dataset with 32 million ratings
    -   Source: `justsahil/movielens-32m`
    -   Files: `ratings.csv`, `links.csv`, `movies.csv`, `tags.csv`
-   **TMDB Movies Dataset**: Movie metadata with 930K movies
    -   Source: `asaniczka/tmdb-movies-dataset-2023-930k-movies`
    -   File: `TMDB_movie_dataset_v11.csv`

## Processing Pipeline

### 1. Data Fetching

-   Downloaded datasets using `kagglehub` API
-   Retrieved both MovieLens and TMDB datasets

### 2. Data Loading

-   Used **Polars** for efficient data manipulation. The datasets consisted of 32 million rows, so polars came in handy, reducing the time to load datasets and manipulate them by around 10x.
-   Loaded three main files:
    -   Movie metadata from TMDB
    -   Ratings from MovieLens
    -   Links mapping MovieLens IDs to TMDB IDs

### 3. Column Selection

Selected relevant columns from TMDB dataset:

-   `id`, `title`, `overview`, `tagline`
-   `genres`, `keywords`
-   `vote_average`, `vote_count`, `runtime`
-   `release_date`, `original_language`

### 4. Data Cleaning

#### Invalid Ratings Handling

-   Identified ratings < 0.5 as invalid (representing "no rating")
-   Converted invalid ratings to `null` values

#### Missing Value Imputation

-   `overview`, `tagline`: Filled with empty strings
-   `genres`, `keywords`: Filled with empty strings
-   `runtime`: Filled with median runtime
-   `vote_average`, `vote_count`: Filled with 0
-   Dropped rows with missing `movieId` or `title`

### 5. Dataset Merging

-   Joined ratings with links on `movieId`
-   Joined result with movie metadata on `tmdbId`/`id`
-   Created unified dataset with both user interactions and movie features

## Output Datasets

### Dataset 1: Collaborative Filtering

**File**: `data/processed/ratings_cf.parquet`

**Purpose**: User-item interaction matrix for collaborative filtering models

**Features**:

-   `userId`: User identifier
-   `movieId`: MovieLens movie identifier
-   `tmdbId`: TMDB movie identifier
-   `rating`: User rating (0.5-5.0 scale)
-   `timestamp`: When the rating was made

**Statistics**:

-   Filtered to include only non-null ratings
-   Calculated matrix sparsity = `0.998`

### Dataset 2: Content-Based Filtering

**Files**:

-   `data/processed/movies_cb.parquet` (movie metadata)
-   `data/processed/bert_embeddings_cb.npy` (BERT embeddings)
-   `data/processed/tfidf_embeddings_cb.npz` (TF-IDF embeddings)
-   `data/processed/tfidf_vectorizer.pkl` (fitted vectorizer)

**Purpose**: Movie content features and embeddings for content-based recommendations

**Text Processing**:

1. **Feature Merging**: Combined multiple text columns into `merged_text`:
    - `title` + `overview` + `tagline` + `genres` + `keywords`
2. **Emoji Removal**: Cleaned non-ASCII characters from text
3. **Whitespace Normalization**: Removed extra spaces

**Embeddings Generated**:

1. **TF-IDF Embeddings**:

    - Method: Sklearn's `TfidfVectorizer`
    - Configuration:
        - `max_features=10000`: Top 10K features
        - `ngram_range=(1, 2)`: Unigrams and bigrams
        - `min_df=2`: Minimum document frequency of 2
        - `max_df=0.8`: Maximum document frequency of 80%
        - `stop_words='english'`: English stop words removed
    - Output: Sparse matrix format for memory efficiency

2. **BERT Embeddings**:
    - Model: `sentence-transformers/all-MiniLM-L6-v2`
    - Batch size: 32
    - Output: Dense embeddings (384 dimensions)
    - Format: NumPy array

**Movie Features**:

-   `id`, `title`, `overview`, `tagline`, `genres`
-   `vote_average`, `vote_count`, `runtime`
-   `release_date`, `original_language`
-   `merged_text`: Combined text features

### Dataset 3: Two-Tower Model

**File**: `data/processed/two_tower_train.parquet`

**Purpose**: Training data for neural two-tower (dual-encoder) architecture

**Features**:

-   **User Tower**: `userId`, `rating`, `timestamp`
-   **Item Tower**: `tmdbId`, `merged_text`, `genres`, `vote_average`, `vote_count`, `runtime`

**Structure**:

-   Combines user interactions with movie features
-   Each row represents a user-movie interaction with full context
-   Suitable for learning separate user and item embeddings

## Metadata File

**File**: `data/processed/metadata.json`

Contains comprehensive statistics for all three datasets:

```json
{
    "collaborative_filtering": {
        "num_ratings": 31922307,
        "num_users": 200948,
        "num_movies": 83323,
        "avg_rating": 3.539802261158631,
        "sparsity": 0.998093461043237
    },
    "content_based": {
        "num_movies": 1312963,
        "tfidf_dim": 10000,
        "bert_dim": 384,
        "bert_model": "sentence-transformers/all-MiniLM-L6-v2"
    },
    "two_tower": {
        "num_samples": 31922307,
        "num_users": 200948,
        "num_movies": 83323
    }
}
```

## Stack

-   **Data Processing**: Polars
-   **Embedding Generation**:
    -   TF-IDF: Scikit-learn
    -   BERT: sentence-transformers
-   **Storage**:
    -   Parquet (efficient columnar format)
    -   NumPy/SciPy (embeddings)
    -   Joblib (model serialization)

## Next Steps

The processed datasets are ready for:

1. Training collaborative filtering models (e.g., Matrix Factorization, ALS)
2. Building content-based recommenders using TF-IDF or BERT embeddings
3. Training neural two-tower models for hybrid recommendations
