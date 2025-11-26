"""Two-tower neural recommender."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import ParameterGrid
from src import config
from src.data_utils import MovieIndex
from src.metrics import RankingMetrics, aggregate_ranking_metrics, mae, rmse
from src.plotting import plot_loss_accuracy
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

MIN_RATING = 0.5
MAX_RATING = 5.0
RANGE = MAX_RATING - MIN_RATING


def normalize(ratings: np.ndarray) -> np.ndarray:
    return (ratings - MIN_RATING) / RANGE


def denormalize(ratings: np.ndarray) -> np.ndarray:
    return ratings * RANGE + MIN_RATING


class InteractionDataset(Dataset):
    def __init__(
        self,
        user_idx: np.ndarray,
        movie_idx: np.ndarray,
        ratings: np.ndarray,
        num_movies: int = 0,
        num_negatives: int = 0,
    ) -> None:
        self.user_idx = torch.from_numpy(user_idx.astype(np.int64))
        self.movie_idx = torch.from_numpy(movie_idx.astype(np.int64))
        self.ratings = torch.from_numpy(ratings.astype(np.float32))
        self.num_movies = num_movies
        self.num_negatives = num_negatives

        # Build user interaction sets for negative sampling
        if num_negatives > 0:
            self.user_items: Dict[int, set[int]] = {}
            for u, i in zip(user_idx, movie_idx):
                if u not in self.user_items:
                    self.user_items[u] = set()
                self.user_items[u].add(i)

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user = self.user_idx[idx]
        pos_item = self.movie_idx[idx]
        rating = self.ratings[idx]

        if self.num_negatives > 0:
            # Return positive sample plus negative samples
            neg_items = []
            user_set = self.user_items.get(int(user), set())
            while len(neg_items) < self.num_negatives:
                neg = random.randint(0, self.num_movies - 1)
                if neg not in user_set and neg not in neg_items:
                    neg_items.append(neg)

            # Return: user, [pos_item, neg_items...], [rating, MIN_RATING-normalized for negs]
            # Negative samples get minimum normalized rating (0.0 after normalization)
            items = torch.tensor([int(pos_item)] + neg_items, dtype=torch.long)
            neg_rating_norm = normalize(np.array([MIN_RATING]))[0]
            ratings = torch.tensor([float(rating)] + [neg_rating_norm] * self.num_negatives, dtype=torch.float32)
            return user, items, ratings

        return user, pos_item, rating


class TwoTowerModel(nn.Module):
    def __init__(self, num_users: int, movie_matrix: np.ndarray, latent_dim: int, dropout: float) -> None:
        super().__init__()
        self.register_buffer("movie_features", torch.tensor(movie_matrix, dtype=torch.float32))
        movie_dim = movie_matrix.shape[1]
        self.user_emb = nn.Embedding(num_users, latent_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)

        # Simpler user tower
        self.user_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Simpler item tower
        self.item_proj = nn.Sequential(
            nn.Linear(movie_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, user_idx: torch.Tensor, movie_idx: torch.Tensor) -> torch.Tensor:
        user_vec = self.user_proj(self.user_emb(user_idx))
        movie_vec = self.item_proj(self.movie_features[movie_idx])
        # L2 normalize embeddings for stable dot product
        user_vec = nn.functional.normalize(user_vec, p=2, dim=1)
        movie_vec = nn.functional.normalize(movie_vec, p=2, dim=1)
        # Dot product gives similarity in [-1, 1], scale to [0, 1]
        score = (user_vec * movie_vec).sum(dim=1)
        # Scale from [-1, 1] to [0, 1] for normalized ratings
        return (score + 1.0) / 2.0


def prepare_tensors(
    ratings: pd.DataFrame,
    movie_index: MovieIndex,
    user_map: Dict[int, int] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, int]]:
    filtered = ratings[ratings["tmdbId"].isin(movie_index.id_to_idx.keys())].copy()
    if user_map is None:
        user_ids = sorted(filtered["userId"].unique())
        user_map = {uid: idx for idx, uid in enumerate(user_ids)}
    filtered = filtered[filtered["userId"].isin(user_map.keys())]
    filtered["user_idx"] = filtered["userId"].map(user_map)
    filtered["movie_idx"] = filtered["tmdbId"].map(movie_index.id_to_idx)
    filtered = filtered.dropna(subset=["user_idx", "movie_idx"])
    ratings_norm = normalize(filtered["rating"].to_numpy().astype(np.float32))
    return (
        filtered["user_idx"].to_numpy().astype(np.int64),
        filtered["movie_idx"].to_numpy().astype(np.int64),
        ratings_norm,
        user_map,
    )


@dataclass
class TwoTowerResults:
    metrics: Dict[str, float]
    ranking: RankingMetrics
    loss_plot: str


class TwoTowerTrainer:
    def __init__(
        self,
        movie_index: MovieIndex,
        param_grid: Dict | None = None,
        default_params: Dict | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.movie_index = movie_index
        self.param_grid = param_grid or config.TWO_TOWER_PARAM_GRID
        self.default_params = default_params or config.TWO_TOWER_DEFAULT_PARAMS
        self.model: TwoTowerModel | None = None
        self.user_map: Dict[int, int] | None = None
        self.loss_plot: str | None = None
        self.best_params: Dict | None = None
        self.global_mean: float = 3.0
        self.logger = logger or logging.getLogger(__name__)

    def _train_single(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        params: Dict,
    ) -> Tuple[float, List[float], List[float], Dict[int, int], Dict[str, float], Dict[str, float]]:
        user_idx, movie_idx, ratings_norm, user_map = prepare_tensors(train_df, self.movie_index)
        val_user_idx, val_movie_idx, val_ratings_norm, _ = prepare_tensors(val_df, self.movie_index, user_map)
        num_movies = self.movie_index.matrix.shape[0]
        train_dataset = InteractionDataset(
            user_idx, movie_idx, ratings_norm, num_movies=num_movies, num_negatives=config.TWO_TOWER_NEGATIVE_SAMPLES
        )
        val_dataset = InteractionDataset(val_user_idx, val_movie_idx, val_ratings_norm)
        train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params["batch_size"], shuffle=False)
        model = TwoTowerModel(
            num_users=len(user_map),
            movie_matrix=self.movie_index.matrix,
            latent_dim=params["latent_dim"],
            dropout=params["dropout"],
        )
        model.to(config.DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        criterion = nn.MSELoss()
        best_val = float("inf")
        best_state = None
        best_epoch = 0
        patience_counter = 0
        loss_history: List[float] = []
        acc_history: List[float] = []
        epoch_iter = tqdm(range(params["epochs"]), desc="Two-tower epochs", leave=False)
        for epoch in epoch_iter:
            model.train()
            epoch_loss = 0.0
            for batch in train_loader:
                u_batch, items_batch, r_batch = batch
                u_batch = u_batch.to(config.DEVICE)
                items_batch = items_batch.to(config.DEVICE)
                r_batch = r_batch.to(config.DEVICE)

                # For negative sampling, items_batch has shape [batch_size, 1+num_negatives]
                # We need to expand u_batch to match
                batch_size = u_batch.size(0)
                num_items_per_user = items_batch.size(1) if len(items_batch.shape) > 1 else 1

                if num_items_per_user > 1:
                    # Expand user indices to match [batch_size * num_items_per_user]
                    u_expanded = u_batch.unsqueeze(1).expand(-1, num_items_per_user).reshape(-1)
                    i_flat = items_batch.reshape(-1)
                    r_flat = r_batch.reshape(-1)
                else:
                    u_expanded = u_batch
                    i_flat = items_batch
                    r_flat = r_batch

                optimizer.zero_grad()
                preds = model(u_expanded, i_flat)
                loss = criterion(preds, r_flat)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item() * batch_size
            loss_history.append(epoch_loss / len(train_dataset))
            model.eval()
            val_loss = 0.0
            correct = 0
            count = 0
            with torch.no_grad():
                for u_batch, i_batch, r_batch in val_loader:
                    u_batch = u_batch.to(config.DEVICE)
                    i_batch = i_batch.to(config.DEVICE)
                    r_batch = r_batch.to(config.DEVICE)
                    preds = model(u_batch, i_batch)
                    val_loss += criterion(preds, r_batch).item() * len(u_batch)
                    denorm_pred = denormalize(preds.cpu().numpy())
                    denorm_truth = denormalize(r_batch.cpu().numpy())
                    correct += np.sum(np.abs(denorm_pred - denorm_truth) <= 0.5)
                    count += len(denorm_truth)
            acc_history.append(correct / count if count else 0.0)
            current_val = val_loss / len(val_dataset)
            scheduler.step(current_val)

            # Early stopping logic
            if current_val < best_val:
                best_val = current_val
                best_state = model.state_dict()
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.TWO_TOWER_EARLY_STOPPING_PATIENCE:
                    if self.logger:
                        self.logger.info(f"Early stopping at epoch {epoch+1}/{params['epochs']} (best: {best_epoch+1})")
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        preds_norm = []
        targets_norm = []
        with torch.no_grad():
            for u_batch, i_batch, r_batch in val_loader:
                preds = model(u_batch.to(config.DEVICE), i_batch.to(config.DEVICE)).cpu().numpy()
                preds_norm.extend(preds.tolist())
                targets_norm.extend(r_batch.numpy().tolist())
        preds = denormalize(np.array(preds_norm))
        targets = denormalize(np.array(targets_norm))
        metrics_dict = {"rmse": rmse(targets, preds), "mae": mae(targets, preds)}
        return best_val, loss_history, acc_history, user_map, model.state_dict(), metrics_dict

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame, use_grid: bool = True) -> None:
        if not train_df.empty:
            self.global_mean = float(train_df["rating"].mean())
        best_score = float("inf")
        best_payload = None
        full_grid = list(ParameterGrid(self.param_grid))
        if not full_grid:
            raise ValueError("Parameter grid for two-tower model is empty")

        if use_grid:
            param_list = full_grid
            if config.TWO_TOWER_MAX_GRID_COMBOS and config.TWO_TOWER_MAX_GRID_COMBOS < len(full_grid):
                rng = random.Random(config.SEED)
                rng.shuffle(param_list)
                param_list = param_list[: config.TWO_TOWER_MAX_GRID_COMBOS]
                self.logger.info(
                    "Two-tower grid truncated to %d combos (from %d)",
                    len(param_list),
                    len(full_grid),
                )
        else:
            chosen = dict(self.default_params or full_grid[0])
            self.logger.info("Two-tower full-data run: skipping grid search, using params=%s", chosen)
            param_list = [chosen]
        for params in tqdm(param_list, desc="Two-tower grid"):
            val_loss, loss_history, acc_history, user_map, state_dict, metrics_dict = self._train_single(
                train_df, val_df, params
            )
            if val_loss < best_score:
                best_score = val_loss
                best_payload = (params, loss_history, acc_history, user_map, state_dict, metrics_dict)
                self.logger.info("Two-tower grid improved -> %s (val_loss=%.4f)", params, val_loss)
        if best_payload is None:
            raise RuntimeError("Grid search failed to train")
        params, loss_history, acc_history, user_map, state_dict, _ = best_payload
        self.best_params = params
        self.user_map = user_map
        self.model = TwoTowerModel(
            num_users=len(user_map),
            movie_matrix=self.movie_index.matrix,
            latent_dim=params["latent_dim"],
            dropout=params["dropout"],
        )
        self.model.load_state_dict(state_dict)
        self.model.to(config.DEVICE)
        self.loss_plot = str(plot_loss_accuracy(loss_history, acc_history, "two_tower_loss_accuracy.png"))

    def predict(self, user_id: int, movie_id: int) -> float:
        if self.model is None or self.user_map is None:
            raise RuntimeError("Model not trained")
        if user_id not in self.user_map:
            return self.global_mean
        movie_idx = self.movie_index.id_to_idx.get(movie_id)
        if movie_idx is None:
            return self.global_mean
        self.model.eval()
        with torch.no_grad():
            u_tensor = torch.tensor([self.user_map[user_id]], device=config.DEVICE)
            m_tensor = torch.tensor([movie_idx], device=config.DEVICE)
            pred = self.model(u_tensor, m_tensor).cpu().numpy()[0]
        return float(denormalize(np.array([pred]))[0])

    def recommend(
        self,
        user_id: int,
        history: set[int],
        topk: int = config.TOP_K,
    ) -> List[int]:
        if self.model is None or self.user_map is None:
            return []
        if user_id not in self.user_map:
            return []
        self.model.eval()
        preds = np.empty(len(self.movie_index.movie_ids), dtype=np.float32)
        with torch.no_grad():
            user_idx = torch.full((len(self.movie_index.movie_ids),), self.user_map[user_id], dtype=torch.long)
            movie_idx = torch.arange(len(self.movie_index.movie_ids), dtype=torch.long)
            preds = self.model(user_idx.to(config.DEVICE), movie_idx.to(config.DEVICE)).cpu().numpy()
        ratings = denormalize(preds)
        mask_idx = [self.movie_index.id_to_idx[mid] for mid in history if mid in self.movie_index.id_to_idx]
        if mask_idx:
            ratings[mask_idx] = MIN_RATING - 1
        top_idx = np.argsort(-ratings)[:topk]
        return [self.movie_index.movie_ids[idx] for idx in top_idx]

    def evaluate(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        candidate_ids: Sequence[int],
        topk: int = config.TOP_K,
    ) -> TwoTowerResults:
        preds = np.array([self.predict(row.userId, row.tmdbId) for row in test_df.itertuples()])
        targets = test_df["rating"].to_numpy()
        mask = np.isfinite(preds)
        error_metrics = {
            "rmse": rmse(targets[mask], preds[mask]) if mask.any() else float("inf"),
            "mae": mae(targets[mask], preds[mask]) if mask.any() else float("inf"),
        }
        sampled_users = (
            test_df["userId"]
            .drop_duplicates()
            .sample(min(config.GRID_EVAL_USERS, test_df["userId"].nunique()), random_state=config.SEED)
        )
        user_truth = {uid: set(test_df.loc[test_df["userId"] == uid, "tmdbId"].tolist()) for uid in sampled_users}
        user_histories = {uid: set(train_df.loc[train_df["userId"] == uid, "tmdbId"].tolist()) for uid in sampled_users}
        user_rankings = {uid: self.recommend(uid, user_histories.get(uid, set()), topk) for uid in sampled_users}
        ranking_metrics = aggregate_ranking_metrics(user_truth, user_rankings, topk, catalog_size=len(candidate_ids))
        self.logger.info(
            "Two-tower eval | rmse=%.4f mae=%.4f precision=%.4f",
            error_metrics["rmse"],
            error_metrics["mae"],
            ranking_metrics.precision,
        )
        return TwoTowerResults(
            metrics=error_metrics,
            ranking=ranking_metrics,
            loss_plot=self.loss_plot or "",
        )
