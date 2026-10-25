# Code guide

Read `data.py`, `model.py`, `pipeline.py`, `evaluation.py`, `recommend.py`, then `app.py`.

The CLI creates artifacts; the dashboard loads them and adapts temporary viewers.
Dataset and trained model weights remain local. Configuration files are described in the README.

## `src/watch_together/__init__.py`

Learn movie preferences and combine them into individual or group recommendations.

data prepares MovieLens, model defines the scorers, pipeline trains and evaluates,
and recommend fits new viewers without changing the shared model.

## `src/watch_together/data.py`

Download MovieLens Small and make per-user chronological rating partitions.

All movie statistics and model mappings are computed from training ratings only.
The downloaded archive and its upstream license notices remain local.

### `write_json`

Create parent directories and write an indented UTF-8 JSON document.

### `digest`

Return a SHA-256 identity for a local artifact.

### `download`

Download the official ZIP and copy only explicitly named CSV and notice files.

Args:
    root: Project directory under which data/raw is created.

Existing archives are reused. ZIP paths are never extracted to the filesystem.
The archive hash is recorded for provenance; it is not an upstream signature.

### `chronological_split`

Split each user's time-ordered ratings into approximately 80/10/10 partitions.

Users need at least five ratings. Equal timestamps are ordered by movie ID.
The validation and test sets each get at least one rating per retained user.
Returns training, validation, and test frames with no shared user/movie pairs.

### `prepare`

Create split CSVs and a training-only movie catalog with popularity statistics.

Writes data/processed/{train,validation,test}.csv, artifacts/catalog.csv,
and artifacts/data.json. Holdout movies absent from training are omitted
from model evaluation and their counts are reported explicitly.

## `src/watch_together/model.py`

Define biased matrix factorization and a small nonlinear rating model.

Both models learn user/movie embeddings. Matrix factorization uses a dot
product; the neural variant adds an MLP interaction term to that same baseline.

### `RatingModel`

Predict explicit ratings from user and movie embeddings.

Attributes:
    users: User embeddings of shape (number_of_users, dimensions).
    items: Movie embeddings of shape (number_of_movies, dimensions).
    user_bias: Learned per-user rating offsets.
    item_bias: Learned per-movie rating offsets.
    interaction: Optional MLP over user, movie, and elementwise product features.

Scores are unbounded during training and clipped to [0.5, 5] for reporting.
The neural residual makes the scorer nonlinear; the MF variant alone is
a latent-factor model, not a deep network.

### `RatingModel.__init__`

Initialize embeddings, zero biases, global mean, and an optional neural head.

### `RatingModel.score_vectors`

Score known items for arbitrary user vectors, including newly fitted viewers.

Args:
    user_vectors: Embedding tensor shaped (N, D) or (1, D) for broadcasting.
    user_bias: User offsets broadcastable to a length-N vector.
    item_indices: Length-N tensor of movie row indices.

Returns:
    Length-N raw predicted ratings. This path permits optimizing a new
    viewer embedding while leaving the trained model unchanged.

### `RatingModel.forward`

Return one rating prediction for each aligned user/movie index pair.

## `src/watch_together/evaluation.py`

Measure explicit rating error and full-catalog ranking on chronological holdouts.

### `rating_error`

Return RMSE and MAE after clipping predictions to the dataset's rating scale.

### `predict_pairs`

Predict aligned CPU user/movie pairs without gradients in evaluation mode.

### `ranking_metrics`

Calculate binary Recall@k and NDCG@k for a nonempty relevant-item set.

### `evaluate_rankings`

Rank the full training catalog for users with test ratings of at least four.

Training and validation movies are excluded as previously seen. Missing
interactions are not verified dislikes. No group or discovery heuristic is
applied. Popularity uses the training-only shrinkage-adjusted mean rating.

## `src/watch_together/recommend.py`

Fit temporary viewer preferences and combine scores for a movie-night shortlist.

Group satisfaction and discovery are explicit ranking rules, not separately
trained models. Viewer ratings stay in memory and do not modify shared weights.

### `Recommender`

Bundle one frozen trained scorer, its movie catalog, and artifact identity.

### `load_recommender`

Load the validation-selected model and reject a mismatched catalog.

Returns a CPU evaluation-mode model whose parameters cannot be updated.
Local viewer embeddings created by fit_viewer remain trainable.

### `fit_viewer`

Predict all catalog ratings after fitting a temporary new viewer embedding.

Args:
    model: Frozen CPU model; its movie embeddings and neural head stay fixed.
    ratings: Mapping of movie item_idx values to explicit 0.5–5 ratings.
    steps: Number of Adam steps for this viewer only.

Returns:
    One clipped prediction per catalog row. Empty preferences use a mean
    user embedding. Ratings influence a local embedding and bias only;
    this is an approximate cold-start strategy, not full model retraining.

### `group_score`

Mix mean enjoyment with the lowest member score, using compromise in [0, 1].

### `rank_movies`

Rank eligible movies and include per-viewer predicted ratings for transparency.

Excludes anything any member rated. Genre filters match any selected genre;
a year filter excludes movies with unknown release years. Discovery mixes
the preference score with a training-count novelty score, up to 30% weight.
It is an adjustable heuristic, not evidence of improved recommendation quality.

## `src/watch_together/pipeline.py`

CLI for preparing MovieLens, training two small scorers, and evaluating them.

Model and epoch selection use validation RMSE only. Test data is read for final
reporting after selection, and never used for early stopping or model selection.

### `load_splits`

Read the training, validation, and supported-test processed CSV files.

### `load_checkpoint`

Restore CPU tensor weights and metadata without loading arbitrary pickle objects.

### `train`

Train MF and neural variants and save the best validation epoch for each.

Args:
    root: Project directory containing prepared data and catalog artifacts.
    epochs: Maximum training passes; early stopping has patience five.
    seed: Seed for initialization and shuffled minibatches.

Writes mf.pt, neural.pt, selected.pt, and training.json. Selection is between
learned scorers by validation RMSE. The baseline is reported separately.
All training uses CPU, Adam, batch size 1024, and MSE loss.

### `evaluate`

Evaluate the fixed models against movie-mean and global-mean baselines.

Reports RMSE/MAE and full-catalog Recall@10/NDCG@10. Scores measure existing
users with training histories; they do not validate new-viewer adaptation,
group enjoyment, or the discovery slider.

### `main`

Parse validated CLI options and dispatch download, prepare, train, evaluate, or all.

## `app.py`

Streamlit app for individual and group movie recommendations.

Loads an existing local model, collects a few explicit ratings per viewer,
and fits temporary viewer vectors. No external API or streaming service is used.

### `resources`

Cache a frozen scorer and catalog, invalidating on artifact content changes.

### `seed_demo`

Populate editable example ratings for three viewers with different tastes.

### `main`

Render preference entry, ranking controls, per-viewer scores, and evaluation.

## `tests/test_core.py`

Check data isolation, model learning, viewer adaptation, and group ranking rules.

### `test_temporal_split_is_disjoint_and_chronological`

Each user's later interactions are held out, without duplicate pairs.

### `test_rating_model_learns`

An optimizer step updates embedding weights for both scorer variants.

### `test_new_viewer_adapts_without_mutating_shared_model`

Different preferences produce different scores while model weights remain fixed.

### `test_compromise_changes_ranking`

Protecting the lowest-scoring member can overturn an average-score winner.

### `test_filters_exclude_any_members_seen_movies`

Seen, wrong-genre, and insufficiently rated items cannot enter the shortlist.

### `test_ranking_metrics_against_hand_calculation`

Recall counts retrieved positives; NDCG discounts a lower-ranked positive.

### `test_catalog_statistics_do_not_use_holdout_ratings`

Holdout-only movies are excluded and held-out ratings cannot change training means.

### `test_loading_rejects_catalog_from_other_weights`

Mismatched model/catalog artifacts must not silently produce recommendations.
