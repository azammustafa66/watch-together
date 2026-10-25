"""Check data isolation, model learning, viewer adaptation, and group ranking rules."""

import numpy as np
import pandas as pd
import pytest
import torch

from watch_together.data import chronological_split, prepare
from watch_together.evaluation import ranking_metrics
from watch_together.model import RatingModel
from watch_together.recommend import fit_viewer, group_score, load_recommender, rank_movies


def test_temporal_split_is_disjoint_and_chronological():
    """Each user's later interactions are held out, without duplicate pairs."""
    frame = pd.DataFrame(
        [
            {'userId': user, 'movieId': item, 'rating': 4, 'timestamp': item}
            for user in (1, 2)
            for item in range(20)
        ]
    )
    train, val, test = chronological_split(frame.sample(frac=1, random_state=4))
    keys = [set(zip(part.userId, part.movieId)) for part in (train, val, test)]
    assert not keys[0] & keys[1] and not keys[0] & keys[2] and not keys[1] & keys[2]
    assert len(train) == 32 and len(val) == len(test) == 4
    for user in (1, 2):
        assert (
            train.loc[train.userId == user, 'timestamp'].max()
            < val.loc[val.userId == user, 'timestamp'].min()
        )
        assert (
            val.loc[val.userId == user, 'timestamp'].max()
            < test.loc[test.userId == user, 'timestamp'].min()
        )


@pytest.mark.parametrize('kind', ['mf', 'neural'])
def test_rating_model_learns(kind):
    """An optimizer step updates embedding weights for both scorer variants."""
    torch.manual_seed(42)
    model = RatingModel(3, 4, 3.5, kind=kind)
    before = model.items.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    prediction = model(torch.tensor([0, 1]), torch.tensor([1, 2]))
    ((prediction - torch.tensor([5.0, 1.0])) ** 2).mean().backward()
    optimizer.step()
    assert prediction.shape == (2,)
    assert not torch.equal(before, model.items.weight)
    if kind == 'neural':
        assert model.interaction[0].weight.grad is not None


@pytest.mark.parametrize('kind', ['mf', 'neural'])
def test_new_viewer_adapts_without_mutating_shared_model(kind):
    """Different preferences produce different scores while model weights remain fixed."""
    torch.manual_seed(12)
    model = RatingModel(3, 4, 3.5, kind=kind).eval().requires_grad_(False)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    likes = fit_viewer(model, {0: 5.0, 1: 4.5}, steps=80)
    dislikes = fit_viewer(model, {0: 0.5, 1: 1.0}, steps=80)
    assert likes[0] > dislikes[0]
    assert np.isfinite(likes).all() and likes.min() >= 0.5 and likes.max() <= 5
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])
    with pytest.raises(ValueError):
        fit_viewer(model, {0: float('nan')})


def test_compromise_changes_ranking():
    """Protecting the lowest-scoring member can overturn an average-score winner."""
    scores = np.array([[5.0, 3.5], [2.5, 3.5]])
    assert group_score(scores, 0).argmax() == 0
    assert group_score(scores, 1).argmax() == 1


def test_filters_exclude_any_members_seen_movies():
    """Seen, wrong-genre, and insufficiently rated items cannot enter the shortlist."""
    catalog = pd.DataFrame(
        {
            'item_idx': [0, 1, 2, 3],
            'movieId': [10, 11, 12, 13],
            'genres': ['Comedy', 'Drama', 'Comedy', 'Comedy'],
            'year': [2000, 2001, 2002, 2003],
            'count': [20, 20, 20, 1],
            'novelty': [0.1, 0.2, 0.3, 0.9],
        }
    )
    result = rank_movies(
        catalog,
        np.array([[5, 4, 3, 5], [4, 4, 4, 5]]),
        {0},
        genres=('Comedy',),
        years=(2000, 2003),
        minimum_ratings=10,
    )
    assert result.item_idx.tolist() == [2]


def test_ranking_metrics_against_hand_calculation():
    """Recall counts retrieved positives; NDCG discounts a lower-ranked positive."""
    report = ranking_metrics(np.array([9, 2, 7]), {2, 3}, k=3)
    assert report['recall@3'] == 0.5
    expected = (1 / np.log2(3)) / (1 + 1 / np.log2(3))
    assert report['ndcg@3'] == pytest.approx(expected)


def test_catalog_statistics_do_not_use_holdout_ratings(tmp_path):
    """Holdout-only movies are excluded and held-out ratings cannot change training means."""
    raw = tmp_path / 'data/raw'
    raw.mkdir(parents=True)
    ratings = pd.DataFrame(
        [
            {'userId': user, 'movieId': item, 'rating': 1 if item <= 8 else 5, 'timestamp': item}
            for user in (1, 2)
            for item in range(1, 11)
        ]
    )
    movies = pd.DataFrame(
        {
            'movieId': range(1, 11),
            'title': [f'Movie {item} (2000)' for item in range(1, 11)],
            'genres': ['Comedy'] * 10,
        }
    )
    ratings.to_csv(raw / 'ratings.csv', index=False)
    movies.to_csv(raw / 'movies.csv', index=False)
    info = prepare(tmp_path)
    catalog = pd.read_csv(tmp_path / 'artifacts/catalog.csv')
    assert info['global_mean'] == 1
    assert catalog['mean'].eq(1).all()
    assert catalog.movieId.tolist() == list(range(1, 9))
    assert info['splits']['test']['unseen_movie_rows_excluded'] == 2


def test_loading_rejects_catalog_from_other_weights(tmp_path):
    """Mismatched model/catalog artifacts must not silently produce recommendations."""
    artifacts = tmp_path / 'artifacts'
    artifacts.mkdir()
    (artifacts / 'catalog.csv').write_text('unexpected catalog')
    torch.save({'catalog_sha256': 'different-hash'}, artifacts / 'selected.pt')
    with pytest.raises(ValueError, match='catalog has changed'):
        load_recommender(tmp_path)
