# Watch Together

**What should we watch together?** A movie recommender for individuals and groups of
two or three people. Rate a few movies, then generate a shortlist balancing average
predicted enjoyment with the least satisfied viewer's predicted rating.

This is a fresh-graduate learning project inspired by collaborative filtering in
Andrew Ng's Machine Learning Specialization. It includes real model training,
a popularity baseline, a small neural comparison, evaluation, and a dark Streamlit UI.

## Quick start

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). From this project directory:

```bash
uv sync
uv run watch-together all
uv run streamlit run app.py --server.port 8503
```

The pipeline downloads MovieLens Small, prepares chronological splits, trains two
small models on CPU, and reports held-out metrics. Installation and the first dataset
download need internet access. Training and recommendations run locally without API keys
or LLM calls. If artifacts already exist, start only the Streamlit command.

Run stages independently:

```bash
uv run watch-together download
uv run watch-together prepare
uv run watch-together train --epochs 30 --seed 42
uv run watch-together evaluate
```

`--epochs` is a maximum; training stops after five epochs without a validation improvement.
The CLI accepts `--root PATH` for a different data/artifact directory. The dashboard reads
the artifacts in its own checkout. After replacing data or rebuilding the catalog, retrain
and reevaluate. Do not run training concurrently with dashboard use.

## Docker

From this project directory, with Docker Desktop or a running Docker engine:

```bash
docker build -t watch-together:local .
mkdir -p data artifacts
docker run --rm -p 127.0.0.1:8503:8501 \
  --mount type=bind,src="$(pwd)/data",dst=/app/data \
  --mount type=bind,src="$(pwd)/artifacts",dst=/app/artifacts \
  watch-together:local
```

Open <http://localhost:8503>. Stop the existing local dashboard first if that port is in use,
or choose another host port before `:8501`. The image runs Streamlit as a non-root user.
Mount the existing folders to reuse your data and models. An empty artifact folder shows
setup instructions instead of recommendations.

To prepare MovieLens and train **inside the container**:

```bash
docker run --rm \
  --mount type=bind,src="$(pwd)/data",dst=/app/data \
  --mount type=bind,src="$(pwd)/artifacts",dst=/app/artifacts \
  watch-together:local watch-together all
```

The first preparation run downloads the dataset. Training and inference run on CPU.
Linux installations use CPU-only PyTorch wheels, while the native macOS dependency source
is unchanged. If Linux bind mounts reject writes, add
`--user "$(id -u):$(id -g)" -e HOME=/tmp` to `docker run`.

`Dockerfile` installs locked runtime dependencies and defines the startup and health check.
`.dockerignore` excludes credentials, Git history, virtual environments, data, model artifacts,
and caches. Neither MovieLens data nor model weights are baked into or downloaded during builds.

## Using the app

1. Choose **Individual** or **Group**, with two or three viewers.
2. Click **Try example ratings** for an editable demo, or search for watched movies.
3. Each viewer rates at least three movies from 0.5 to 5. Include likes and dislikes.
4. Optionally filter by genre, release year, and minimum historical rating count.
5. Adjust **Protect the least satisfied viewer** and **Discovery**.
6. Click the recommendation button. Results display estimated ratings for every viewer.

The shortlist excludes anything **any** member rated. Changing controls requires clicking
the button again; stale results are hidden. Genre filters match any selected genre.
Year filtering excludes titles without a parsed release year. Preferences remain in memory
and are not written to disk. There are no user accounts or external tracking calls.

**Catalog limits:** the downloaded MovieLens snapshot contains historical ratings through
September 2018. It is not a current-release or streaming-availability service. The raw catalog
has 9,742 movies; our model supports the 8,255 movies present in its training split.
There are no live posters, streaming links, synopses, or invented explanations.

## What happens from start to finish

1. **Download:** fetch the official MovieLens Small archive. Copy explicitly named CSVs and
   the upstream README without extracting arbitrary ZIP paths. Record the archive SHA-256.
2. **Split:** sort each user's ratings by timestamp, breaking ties by movie ID. Hold out
   approximately the latest 10% for test and the previous 10% for validation.
3. **Prepare:** create contiguous user/movie IDs and compute movie counts and mean ratings
   from training data only. Exclude holdout movies unknown to training and record counts.
4. **Train:** learn 32-dimensional user and movie vectors, with user/movie biases.
   Train a matrix-factorization scorer and a neural interaction variant.
5. **Select:** save each model's lowest-validation-RMSE epoch. Choose between the two learned
   models using validation RMSE, without inspecting test results for selection.
6. **Evaluate:** measure rating prediction error and full-catalog top-ten ranking separately.
7. **Adapt a viewer:** freeze shared model weights and fit a temporary user vector and bias
   using the ratings entered in the dashboard.
8. **Build a shortlist:** combine member scores, apply filters, optionally favor less-rated
   movies, and display the highest-ranked eligible titles.

## Models and algorithms

### Baseline: popularity / movie quality

Use a shrinkage-adjusted movie mean rating:

```text
baseline(movie) = (count × movie_mean + 20 × global_mean) / (count + 20)
```

All statistics come from training ratings. This baseline favors well-rated movies while
shrinking low-count estimates toward the global mean. It is not simply a sort by rating count.
Select **Popularity baseline** in the app to compare an unpersonalized shortlist.

### Biased matrix factorization

```text
prediction(user, movie) = global_mean + user_bias + movie_bias
                          + dot(user_embedding, movie_embedding)
```

User/movie embeddings have 32 learned values. This is a latent-factor model, not itself
a deep neural network. It is a useful interpretable baseline for the neural experiment.

### Neural interaction model

Add an MLP residual to the matrix-factorization prediction:

```text
features = concatenate(user_vector, movie_vector, user_vector × movie_vector)
MLP      = Linear(96, 64) → ReLU → Linear(64, 1)
prediction = matrix_factorization_prediction + MLP(features)
```

Both models use MSE loss, Adam (`lr=0.005`, `weight_decay=0.0001`), batches of 1,024,
seed 42, and a maximum of 30 epochs. Predictions are clipped to [0.5, 5] for evaluation
and display; raw scores are used for training loss. There is no negative sampling because
training uses explicit ratings rather than implicit clicks.

### New viewers and groups

For a new viewer, initialize from the mean learned user vector and optimize only that
temporary vector and bias for 120 Adam steps (`lr=0.05`). A penalty discourages large
departures from the prior and large bias offsets. Movie vectors and the neural head stay frozen.
Three ratings are enough to demonstrate the flow, not enough to guarantee reliable taste estimates.

For each movie, combine clipped member predictions using:

```text
group_rating = (1 - compromise) × mean(member_predictions)
               + compromise × min(member_predictions)
```

`compromise=0` favors average enjoyment; `compromise=1` favors the lowest-scoring member.
This is an explicit product rule, not a trained group-satisfaction model.

The discovery control mixes normalized group rating with training-count rarity:

```text
rarity = 1 - log(1 + movie_count) / log(1 + maximum_movie_count)
weight = 0.3 × discovery
rank_score = (1 - weight) × (group_rating - 0.5) / 4.5 + weight × rarity
```

It contributes at most 30% of ranking score. Less-rated does not necessarily mean better,
more diverse, or unfamiliar to a viewer. The minimum-count control limits sparse candidates.

## Evaluation and initial results

The downloaded snapshot contains 100,836 ratings from 610 users. The split has:

| Partition | Original rows | Known-movie rows evaluated/trained | Unknown-movie rows excluded |
|---|---:|---:|---:|
| Training | 81,200 | 81,200 | 0 |
| Validation | 9,818 | 9,083 | 735 |
| Test | 9,818 | 8,886 | 932 |

The split is chronological **per user**, not a single global time cutoff. Other users'
training histories can therefore extend beyond a particular user's holdout date.

Initial run, seed 42, CPU:

| Method | Test RMSE ↓ | Test MAE ↓ | Recall@10 ↑ | NDCG@10 ↑ |
|---|---:|---:|---:|---:|
| Global mean | 1.0625 | 0.8362 | — | — |
| Popularity / movie mean | 0.9994 | 0.7798 | 0.0417 | 0.0402 |
| Matrix factorization | 0.8824 | 0.6772 | 0.0568 | 0.0526 |
| Neural interaction | 0.8823 | 0.6741 | 0.0247 | 0.0248 |

The neural model was selected using validation RMSE (0.8621, epoch 2), narrowly ahead
of MF (0.8642, epoch 4). **MF gave stronger top-ten rankings, while the selected neural
model's ranking was weaker than the popularity baseline.** Predicting star ratings and
ranking relevant movies are different objectives. We report that tradeoff rather than
claiming the neural model wins every task. No retraining or tuning was performed based on test results.

Ranking uses 557 users with at least one supported test rating >=4. Every training-catalog
movie is a candidate except that user's training/validation movies. There are no sampled negatives.
Recall@10 measures the fraction of relevant held-out items retrieved; binary NDCG@10 rewards
placing them earlier. Metrics are averaged equally across eligible users. A missing interaction
is not proof of dislike. Clipped-score ties preserve catalog order.

These offline results concern existing users with training histories. They do **not** validate
new-viewer adaptation, actual group enjoyment, or discovery-slider quality. A future experiment
could evaluate new users with user-disjoint splits and choose a model by validation ranking quality.
See [docs/RESULTS.md](docs/RESULTS.md) for checkpoint identities and provenance.

## Files and generated artifacts

| File | Purpose |
|---|---|
| `app.py` | Streamlit preference entry, demo, filters, recommendations, and metrics |
| `src/watch_together/__init__.py` | Package overview |
| `src/watch_together/data.py` | Safe download, chronological splits, and training-only catalog |
| `src/watch_together/model.py` | MF and neural rating scorer |
| `src/watch_together/recommend.py` | Frozen model loading, new-viewer fitting, and group ranking |
| `src/watch_together/evaluation.py` | Rating error and full-catalog ranking metrics |
| `src/watch_together/pipeline.py` | CLI, model training, checkpoint selection, evaluation |
| `tests/test_core.py` | Learning, isolation, adaptation, and ranking-rule checks |
| `docs/CODE_GUIDE.md` | Function/class walkthrough |
| `docs/RESULTS.md` | Initial experiment record |
| `pyproject.toml`, `uv.lock`, `.python-version` | Environment and CLI configuration |
| `.streamlit/config.toml`, `.gitignore` | Theme and generated-file exclusions |

Generated files stay local and are ignored by Git:

- `data/raw/`: original ZIP, selected CSVs, upstream README, source hash.
- `data/processed/`: training, validation, and supported-test CSVs.
- `artifacts/catalog.csv`: movie metadata and training-only statistics.
- `artifacts/data.json`: split sizes and data/catalog identity.
- `artifacts/mf.pt`, `neural.pt`, `selected.pt`: weights and training metadata.
- `artifacts/training.json`: epoch histories and model selection.
- `artifacts/evaluation.json`: complete metrics and protocol.

Checkpoint/catalog hashes reject mismatched artifacts. Weights are loaded with
`weights_only=True`. Seeded runs can still differ across software versions and hardware.

## Development

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

Unit tests use synthetic ratings and require no downloads. All Python modules, named
functions/methods, and classes have docstrings. After training, the dashboard can also
be exercised with Streamlit's AppTest.

## Dataset credit and terms

Uses [MovieLens Small by GroupLens](https://grouplens.org/datasets/movielens/latest/).
Its original README is retained under `data/raw/README.txt`. The dataset permits research
use subject to its conditions; commercial or revenue-bearing use requires permission.
Redistribution must retain the dataset's conditions. No GroupLens endorsement is implied.
This repository does not bundle the dataset or derived catalog.

Dataset citation: F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets:
History and Context*. ACM TiiS 5(4), Article 19. [DOI](https://doi.org/10.1145/2827872).

MovieLens Small is a development dataset that may change. Results here describe the downloaded
snapshot identified in the local provenance record, not a claim on a permanent research benchmark.

Course inspiration: [Andrew Ng's Machine Learning Specialization](https://www.deeplearning.ai/specializations/machine-learning).
The project is an independent implementation, not an official course assignment or endorsed app.
