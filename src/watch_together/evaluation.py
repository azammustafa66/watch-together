"""Measure explicit rating error and full-catalog ranking on chronological holdouts."""

import numpy as np
import pandas as pd
import torch

from watch_together.model import RatingModel


def rating_error(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Return RMSE and MAE after clipping predictions to the dataset's rating scale."""
    error = np.clip(predicted, 0.5, 5) - np.asarray(actual)
    return {'rmse': float(np.sqrt(np.mean(error**2))), 'mae': float(np.mean(np.abs(error)))}


@torch.inference_mode()
def predict_pairs(model: RatingModel, frame: pd.DataFrame) -> np.ndarray:
    """Predict aligned CPU user/movie pairs without gradients in evaluation mode."""
    model.eval()
    users = torch.tensor(frame.user_idx.to_numpy(), dtype=torch.long)
    items = torch.tensor(frame.item_idx.to_numpy(), dtype=torch.long)
    return model(users, items).numpy()


def ranking_metrics(order: np.ndarray, relevant: set[int], k: int = 10) -> dict[str, float]:
    """Calculate binary Recall@k and NDCG@k for a nonempty relevant-item set."""
    if not relevant:
        raise ValueError('Ranking metrics require at least one relevant item.')
    hits = np.array([item in relevant for item in order[:k]], dtype=float)
    discount = 1 / np.log2(np.arange(len(hits)) + 2)
    ideal = np.sum(1 / np.log2(np.arange(min(k, len(relevant))) + 2))
    return {
        f'recall@{k}': float(hits.sum() / len(relevant)),
        f'ndcg@{k}': float(np.sum(hits * discount) / ideal),
    }


@torch.inference_mode()
def evaluate_rankings(
    models: dict[str, RatingModel],
    catalog: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> dict:
    """Rank the full training catalog for users with test ratings of at least four.

    Training and validation movies are excluded as previously seen. Missing
    interactions are not verified dislikes. No group or discovery heuristic is
    applied. Popularity uses the training-only shrinkage-adjusted mean rating.
    """
    known = pd.concat([train, validation]).groupby('user_idx').item_idx.agg(set).to_dict()
    relevant = test.loc[test.rating >= 4].groupby('user_idx').item_idx.agg(set)
    totals = {name: [] for name in ['popularity', *models]}
    items = torch.arange(len(catalog))
    for user, targets in relevant.items():
        targets = targets - known.get(user, set())
        if not targets:
            continue
        scores = {'popularity': catalog.popular_score.to_numpy().copy()}
        for name, model in models.items():
            model.eval()
            scores[name] = model(torch.full_like(items, user), items).clamp(0.5, 5).numpy()
        for name, values in scores.items():
            values[list(known.get(user, set()))] = -np.inf
            order = np.argsort(-values, kind='stable')[:10]
            totals[name].append(ranking_metrics(order, targets))
    return {
        name: {
            'users': len(rows),
            **{
                key: float(np.mean([row[key] for row in rows])) if rows else None
                for key in ('recall@10', 'ndcg@10')
            },
        }
        for name, rows in totals.items()
    }
