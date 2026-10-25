"""CLI for preparing MovieLens, training two small scorers, and evaluating them.

Model and epoch selection use validation RMSE only. Test data is read for final
reporting after selection, and never used for early stopping or model selection.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from watch_together.data import digest, download, prepare, write_json
from watch_together.evaluation import evaluate_rankings, predict_pairs, rating_error
from watch_together.model import RatingModel

ROOT = Path(__file__).resolve().parents[2]


def load_splits(root: Path) -> tuple[pd.DataFrame, ...]:
    """Read the training, validation, and supported-test processed CSV files."""
    return tuple(
        pd.read_csv(root / f'data/processed/{name}.csv') for name in ('train', 'validation', 'test')
    )


def load_checkpoint(path: Path) -> tuple[RatingModel, dict]:
    """Restore CPU tensor weights and metadata without loading arbitrary pickle objects."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = RatingModel(**checkpoint['config'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint


def train(root: Path, epochs: int = 30, seed: int = 42) -> dict:
    """Train MF and neural variants and save the best validation epoch for each.

    Args:
        root: Project directory containing prepared data and catalog artifacts.
        epochs: Maximum training passes; early stopping has patience five.
        seed: Seed for initialization and shuffled minibatches.

    Writes mf.pt, neural.pt, selected.pt, and training.json. Selection is between
    learned scorers by validation RMSE. The baseline is reported separately.
    All training uses CPU, Adam, batch size 1024, and MSE loss.
    """
    info = json.loads((root / 'artifacts/data.json').read_text())
    if info['catalog_sha256'] != digest(root / 'artifacts/catalog.csv'):
        raise ValueError('Prepared catalog changed. Run prepare again.')
    training = pd.read_csv(root / 'data/processed/train.csv')
    validation = pd.read_csv(root / 'data/processed/validation.csv')
    dataset = TensorDataset(
        torch.tensor(training.user_idx.to_numpy(), dtype=torch.long),
        torch.tensor(training.item_idx.to_numpy(), dtype=torch.long),
        torch.tensor(training.rating.to_numpy(), dtype=torch.float32),
    )
    reports = {}
    (root / 'artifacts/evaluation.json').unlink(missing_ok=True)
    for kind in ('mf', 'neural'):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        config = dict(
            users=info['users'],
            items=info['movies'],
            mean=info['global_mean'],
            kind=kind,
            dimensions=32,
        )
        model = RatingModel(**config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0001)
        loader = DataLoader(dataset, batch_size=1024, shuffle=True)
        history, best, stale = [], float('inf'), 0
        for epoch in range(1, epochs + 1):
            model.train()
            loss_sum = 0.0
            for users, items, ratings in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = ((model(users, items) - ratings) ** 2).mean()
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(ratings)
            error = rating_error(validation.rating.to_numpy(), predict_pairs(model, validation))
            row = dict(
                epoch=epoch, train_mse=loss_sum / len(dataset), validation_rmse=error['rmse']
            )
            history.append(row)
            print(json.dumps({'model': kind, **row}), flush=True)
            if error['rmse'] < best - 0.0001:
                best, stale = error['rmse'], 0
                torch.save(
                    dict(
                        state_dict=model.state_dict(),
                        config=config,
                        seed=seed,
                        epoch=epoch,
                        validation_rmse=best,
                        catalog_sha256=info['catalog_sha256'],
                        source_sha256=info['source_sha256'],
                    ),
                    root / f'artifacts/{kind}.pt',
                )
            else:
                stale += 1
                if stale >= 5:
                    break
        _, saved = load_checkpoint(root / f'artifacts/{kind}.pt')
        reports[kind] = {'best_epoch': saved['epoch'], 'validation_rmse': best, 'history': history}
    selected = min(reports, key=lambda key: reports[key]['validation_rmse'])
    (root / 'artifacts/selected.pt').write_bytes((root / f'artifacts/{selected}.pt').read_bytes())
    result = {
        'seed': seed,
        'epochs_requested': epochs,
        'device': 'cpu',
        'models': reports,
        'selected': selected,
        'selected_sha256': digest(root / 'artifacts/selected.pt'),
    }
    write_json(root / 'artifacts/training.json', result)
    print(f'Selected {selected} by validation RMSE.', flush=True)
    return result


def evaluate(root: Path) -> dict:
    """Evaluate the fixed models against movie-mean and global-mean baselines.

    Reports RMSE/MAE and full-catalog Recall@10/NDCG@10. Scores measure existing
    users with training histories; they do not validate new-viewer adaptation,
    group enjoyment, or the discovery slider.
    """
    train_data, validation, test = load_splits(root)
    catalog = pd.read_csv(root / 'artifacts/catalog.csv').sort_values('item_idx')
    info = json.loads((root / 'artifacts/data.json').read_text())
    training = json.loads((root / 'artifacts/training.json').read_text())
    if digest(root / 'artifacts/selected.pt') != training['selected_sha256']:
        raise ValueError('Selected model and training report disagree.')
    models, rating_reports = {}, {}
    for kind in ('mf', 'neural'):
        model, checkpoint = load_checkpoint(root / f'artifacts/{kind}.pt')
        if checkpoint['catalog_sha256'] != digest(root / 'artifacts/catalog.csv'):
            raise ValueError('Model and catalog disagree. Train again.')
        models[kind] = model
        rating_reports[kind] = rating_error(test.rating.to_numpy(), predict_pairs(model, test))
    rating_reports['global_mean'] = rating_error(
        test.rating.to_numpy(), np.full(len(test), info['global_mean'])
    )
    rating_reports['popularity'] = rating_error(
        test.rating.to_numpy(),
        catalog.popular_score.to_numpy()[test.item_idx.to_numpy()],
    )
    report = {
        'selected_model': training['selected'],
        'model_sha256': training['selected_sha256'],
        'data': info,
        'rating_metrics': rating_reports,
        'ranking_metrics': evaluate_rankings(models, catalog, train_data, validation, test),
        'protocol': 'Per-user chronological 80/10/10; relevant test ratings >=4; '
        'full training catalog, excluding train/validation seen movies; '
        'unseen movies excluded. No group or cold-start validation.',
    }
    write_json(root / 'artifacts/evaluation.json', report)
    print(
        json.dumps(
            {key: report[key] for key in ('selected_model', 'rating_metrics', 'ranking_metrics')},
            indent=2,
        ),
        flush=True,
    )
    return report


def main() -> None:
    """Parse validated CLI options and dispatch download, prepare, train, evaluate, or all."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['download', 'prepare', 'train', 'evaluate', 'all'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive.')
    torch.set_num_threads(4)
    if args.command in ('download', 'all'):
        download(args.root)
    if args.command in ('prepare', 'all'):
        prepare(args.root)
    if args.command in ('train', 'all'):
        train(args.root, args.epochs, args.seed)
    if args.command in ('evaluate', 'all'):
        evaluate(args.root)
