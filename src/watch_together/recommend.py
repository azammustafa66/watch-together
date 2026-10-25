"""Fit temporary viewer preferences and combine scores for a movie-night shortlist.

Group satisfaction and discovery are explicit ranking rules, not separately
trained models. Viewer ratings stay in memory and do not modify shared weights.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from watch_together.data import digest
from watch_together.model import RatingModel


@dataclass
class Recommender:
    """Bundle one frozen trained scorer, its movie catalog, and artifact identity."""

    model: RatingModel
    catalog: pd.DataFrame
    model_hash: str


def load_recommender(root: Path) -> Recommender:
    """Load the validation-selected model and reject a mismatched catalog.

    Returns a CPU evaluation-mode model whose parameters cannot be updated.
    Local viewer embeddings created by fit_viewer remain trainable.
    """
    checkpoint = torch.load(root / 'artifacts/selected.pt', map_location='cpu', weights_only=True)
    if checkpoint['catalog_sha256'] != digest(root / 'artifacts/catalog.csv'):
        raise ValueError('The catalog has changed. Run watch-together train again.')
    model = RatingModel(**checkpoint['config'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.requires_grad_(False)
    catalog = (
        pd.read_csv(root / 'artifacts/catalog.csv').sort_values('item_idx').reset_index(drop=True)
    )
    if catalog.item_idx.tolist() != list(range(len(catalog))):
        raise ValueError('Catalog indices must match model item rows.')
    return Recommender(model, catalog, digest(root / 'artifacts/selected.pt'))


def fit_viewer(model: RatingModel, ratings: dict[int, float], steps: int = 120) -> np.ndarray:
    """Predict all catalog ratings after fitting a temporary new viewer embedding.

    Args:
        model: Frozen CPU model; its movie embeddings and neural head stay fixed.
        ratings: Mapping of movie item_idx values to explicit 0.5–5 ratings.
        steps: Number of Adam steps for this viewer only.

    Returns:
        One clipped prediction per catalog row. Empty preferences use a mean
        user embedding. Ratings influence a local embedding and bias only;
        this is an approximate cold-start strategy, not full model retraining.
    """
    if any(not 0 <= index < model.items.num_embeddings for index in ratings):
        raise ValueError('Unknown catalog item index.')
    if any(not np.isfinite(value) or not 0.5 <= value <= 5 for value in ratings.values()):
        raise ValueError('Ratings must be finite values between 0.5 and 5.')
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError('Freeze the shared model before fitting a viewer.')
    prior = model.users.weight.detach().mean(dim=0, keepdim=True)
    vector = prior.clone().requires_grad_(True)
    bias = torch.zeros(1, requires_grad=True)
    if ratings:
        items = torch.tensor(list(ratings), dtype=torch.long)
        targets = torch.tensor(list(ratings.values()), dtype=torch.float32)
        optimizer = torch.optim.Adam([vector, bias], lr=0.05)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            predictions = model.score_vectors(vector, bias, items)
            loss = ((predictions - targets) ** 2).mean()
            loss = loss + 0.05 * ((vector - prior) ** 2).mean() + 0.05 * bias.square().mean()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        all_items = torch.arange(model.items.num_embeddings)
        return model.score_vectors(vector, bias, all_items).clamp(0.5, 5).numpy()


def group_score(predictions: np.ndarray, compromise: float) -> np.ndarray:
    """Mix mean enjoyment with the lowest member score, using compromise in [0, 1]."""
    values = np.asarray(predictions, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError('Expected finite scores with shape (members, movies).')
    if not 0 <= compromise <= 1:
        raise ValueError('compromise must be between zero and one.')
    return (1 - compromise) * values.mean(axis=0) + compromise * values.min(axis=0)


def rank_movies(
    catalog: pd.DataFrame,
    predictions: np.ndarray,
    seen: set[int],
    compromise: float = 0.5,
    discovery: float = 0.0,
    genres: tuple[str, ...] = (),
    years: tuple[int, int] | None = None,
    minimum_ratings: int = 10,
    limit: int = 10,
) -> pd.DataFrame:
    """Rank eligible movies and include per-viewer predicted ratings for transparency.

    Excludes anything any member rated. Genre filters match any selected genre;
    a year filter excludes movies with unknown release years. Discovery mixes
    the preference score with a training-count novelty score, up to 30% weight.
    It is an adjustable heuristic, not evidence of improved recommendation quality.
    """
    if not 0 <= discovery <= 1:
        raise ValueError('discovery must be between zero and one.')
    predictions = np.asarray(predictions)
    if predictions.shape[1:] != (len(catalog),):
        raise ValueError('Predictions must align with every catalog row.')
    scores = group_score(predictions, compromise)
    result = catalog.copy()
    result['predicted_average'] = predictions.mean(axis=0)
    result['predicted_lowest'] = predictions.min(axis=0)
    novelty_weight = 0.3 * discovery
    result['rank_score'] = (1 - novelty_weight) * (
        (scores - 0.5) / 4.5
    ) + novelty_weight * result.novelty
    for member, values in enumerate(predictions, start=1):
        result[f'viewer_{member}'] = values
    eligible = ~result.item_idx.isin(seen) & (result['count'] >= minimum_ratings)
    if genres:
        eligible &= result.genres.apply(lambda value: bool(set(value.split('|')) & set(genres)))
    if years is not None:
        eligible &= result.year.between(*years)
    return (
        result.loc[eligible]
        .sort_values(
            ['rank_score', 'count', 'movieId'],
            ascending=[False, False, True],
            kind='stable',
        )
        .head(limit)
        .reset_index(drop=True)
    )
