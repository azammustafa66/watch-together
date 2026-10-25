"""Download MovieLens Small and make per-user chronological rating partitions.

All movie statistics and model mappings are computed from training ratings only.
The downloaded archive and its upstream license notices remain local.
"""

import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DATA_URL = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'


def write_json(path: Path, value: dict) -> None:
    """Create parent directories and write an indented UTF-8 JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def digest(path: Path) -> str:
    """Return a SHA-256 identity for a local artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(root: Path) -> None:
    """Download the official ZIP and copy only explicitly named CSV and notice files.

    Args:
        root: Project directory under which data/raw is created.

    Existing archives are reused. ZIP paths are never extracted to the filesystem.
    The archive hash is recorded for provenance; it is not an upstream signature.
    """
    raw = root / 'data/raw'
    raw.mkdir(parents=True, exist_ok=True)
    archive_path = raw / 'ml-latest-small.zip'
    if not archive_path.exists():
        request = urllib.request.Request(DATA_URL, headers={'User-Agent': 'WatchTogether/0.1'})
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = response.read(20_000_001)
        if len(payload) > 20_000_000:
            raise ValueError('Dataset archive exceeds the expected size limit.')
        # Check the ZIP before publishing a reusable local copy.
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            archive.getinfo('ml-latest-small/ratings.csv')
            archive.getinfo('ml-latest-small/movies.csv')
        temporary = archive_path.with_suffix('.tmp')
        temporary.write_bytes(payload)
        temporary.replace(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        for name in ('ratings.csv', 'movies.csv', 'README.txt', 'links.csv'):
            member = archive.getinfo(f'ml-latest-small/{name}')
            if member.file_size > 30_000_000:
                raise ValueError('Unexpected dataset member size.')
            (raw / name).write_bytes(archive.read(member))
    write_json(raw / 'provenance.json', {'url': DATA_URL, 'sha256': digest(archive_path)})
    print('MovieLens downloaded. Read data/raw/README.txt for dataset usage terms.', flush=True)


def chronological_split(ratings: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """Split each user's time-ordered ratings into approximately 80/10/10 partitions.

    Users need at least five ratings. Equal timestamps are ordered by movie ID.
    The validation and test sets each get at least one rating per retained user.
    Returns training, validation, and test frames with no shared user/movie pairs.
    """
    parts: list[list[pd.DataFrame]] = [[], [], []]
    if ratings.duplicated(['userId', 'movieId']).any():
        raise ValueError('Expected at most one rating per user/movie pair.')
    for _, group in ratings.groupby('userId', sort=True):
        group = group.sort_values(['timestamp', 'movieId'], kind='stable')
        if len(group) < 5:
            continue
        holdout = max(1, int(len(group) * 0.1))
        parts[0].append(group.iloc[: -2 * holdout])
        parts[1].append(group.iloc[-2 * holdout : -holdout])
        parts[2].append(group.iloc[-holdout:])
    if not parts[0]:
        raise ValueError('No users have enough ratings to split.')
    return tuple(pd.concat(part, ignore_index=True) for part in parts)


def prepare(root: Path) -> dict:
    """Create split CSVs and a training-only movie catalog with popularity statistics.

    Writes data/processed/{train,validation,test}.csv, artifacts/catalog.csv,
    and artifacts/data.json. Holdout movies absent from training are omitted
    from model evaluation and their counts are reported explicitly.
    """
    raw = root / 'data/raw'
    ratings = pd.read_csv(raw / 'ratings.csv')
    movies = pd.read_csv(raw / 'movies.csv')
    train, validation, test = chronological_split(ratings)
    user_ids = sorted(train.userId.unique().tolist())
    movie_ids = sorted(train.movieId.unique().tolist())
    user_map = {value: index for index, value in enumerate(user_ids)}
    movie_map = {value: index for index, value in enumerate(movie_ids)}
    stats = train.groupby('movieId').rating.agg(['count', 'mean']).reset_index()
    catalog = movies.merge(stats, on='movieId', how='inner')
    catalog['item_idx'] = catalog.movieId.map(movie_map)
    catalog = catalog.sort_values('item_idx').reset_index(drop=True)
    catalog['year'] = pd.to_numeric(catalog.title.str.extract(r'\((\d{4})\)\s*$', expand=False))
    global_mean = float(train.rating.mean())
    catalog['popular_score'] = (catalog['count'] * catalog['mean'] + 20 * global_mean) / (
        catalog['count'] + 20
    )
    catalog['novelty'] = 1 - np.log1p(catalog['count']) / np.log1p(catalog['count'].max())
    output = root / 'artifacts'
    output.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(output / 'catalog.csv', index=False)
    info = {
        'users': len(user_ids),
        'movies': len(movie_ids),
        'global_mean': global_mean,
        'raw_ratings': len(ratings),
        'source_sha256': digest(raw / 'ratings.csv'),
        'catalog_sha256': digest(output / 'catalog.csv'),
        'splits': {},
    }
    processed = root / 'data/processed'
    processed.mkdir(parents=True, exist_ok=True)
    for name, frame in [('train', train), ('validation', validation), ('test', test)]:
        supported = frame.movieId.isin(movie_ids) & frame.userId.isin(user_ids)
        mapped = frame.loc[supported].copy()
        mapped['user_idx'] = mapped.userId.map(user_map)
        mapped['item_idx'] = mapped.movieId.map(movie_map)
        mapped.to_csv(processed / f'{name}.csv', index=False)
        info['splits'][name] = {
            'total': len(frame),
            'supported': len(mapped),
            'unseen_movie_rows_excluded': int((~supported).sum()),
        }
    write_json(output / 'data.json', info)
    print(json.dumps(info, indent=2), flush=True)
    return info
