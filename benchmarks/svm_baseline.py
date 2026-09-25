"""Classical baselines on exactly the pixels and preprocessing GHOST 0.2.0 uses.

    python benchmarks/svm_baseline.py                  # SVM + random forest, ratio + fixed splits, 5 seeds

The SVM's C and gamma are chosen on the validation pixels, the same pixels the network early-stops on.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from ghost.data import indian_pines_path  # noqa: E402
from ghost.v0_2.metrics import METRIC_KEYS, compute_metrics, majority_share  # noqa: E402
from ghost.v0_2.preprocessing import build_settings, preprocess  # noqa: E402
from ghost.v0_2.scene import load_scene  # noqa: E402
from ghost.v0_2.splits import make_split  # noqa: E402
from ghost.v0_2.train import RESULT_COLUMNS  # noqa: E402

SPLIT_PARAMS = {'train_ratio': 0.2, 'val_ratio': 0.1, 'samples_per_class': 50, 'minority_samples': 15}
C_GRID = (1, 10, 100, 1000)
GAMMA_GRID = ('scale', 0.01, 0.1, 1.0)


def fit_svm(X_train, y_train, X_val, y_val, seed):
    from sklearn.svm import SVC
    best = None
    for C, gamma in itertools.product(C_GRID, GAMMA_GRID):
        model = SVC(C=C, gamma=gamma, kernel='rbf', random_state=seed).fit(X_train, y_train)
        score = float((model.predict(X_val) == y_val).mean())
        if best is None or score > best[0]:
            best = (score, model, {'C': C, 'gamma': gamma})
    return best[1], best[2]


def fit_rf(X_train, y_train, seed):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1).fit(X_train, y_train), {}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', default=None, help='Defaults to the bundled Indian Pines')
    p.add_argument('--gt', default=None)
    p.add_argument('--splits', nargs='+', default=['ratio', 'fixed'], choices=['ratio', 'fixed', 'disjoint'])
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--cr', default='auto')
    p.add_argument('--models', nargs='+', default=['svm', 'rf'], choices=['svm', 'rf'])
    p.add_argument('--out', default=os.path.join('runs', 'bench'))
    args = p.parse_args()

    from sklearn.preprocessing import StandardScaler
    data, gt_path = (args.data, args.gt) if args.data else indian_pines_path()
    scene = load_scene(data, gt_path)
    settings = build_settings(args.cr, scene['cube'], scene['valid'], scene['wavelengths'])
    feats = preprocess(scene['cube'], scene['valid'], settings)
    gt, y = scene['gt'], scene['gt'].reshape(-1)

    for split, seed, name in itertools.product(args.splits, args.seeds, args.models):
        out = os.path.join(args.out, name, split, f"{settings['cr_mode']}", f"seed{seed}")
        if os.path.exists(os.path.join(out, 'test_results.csv')):
            print(f"{name} {split} seed={seed}: done, skipping")
            continue
        train_idx, val_idx, test_idx = make_split(split, gt, seed=seed, **SPLIT_PARAMS)
        scaler = StandardScaler().fit(feats[train_idx])
        X_train, X_val, X_test = (scaler.transform(feats[i]) for i in (train_idx, val_idx, test_idx))
        t0 = time.time()
        if name == 'svm':
            model, chosen = fit_svm(X_train, y[train_idx], X_val, y[val_idx], seed)
        else:
            model, chosen = fit_rf(X_train, y[train_idx], seed)
        m = compute_metrics(model.predict(X_test), y[test_idx])
        print(f"{name} {split} seed={seed}: OA {m['OA']:.4f}  {chosen}  ({time.time() - t0:.0f}s)")

        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, 'test_results.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['best_epoch'] + RESULT_COLUMNS)
            w.writerow([''] + [f"{m[k]:.4f}" for k in METRIC_KEYS])
        with open(os.path.join(out, 'bench_meta.json'), 'w') as f:
            json.dump({'label': f'{name} baseline', 'arch': name, 'split': split, 'cr': settings['cr_mode'],
                       'pool': '-', 'loss': '-', 'seed': seed, 'chosen': chosen,
                       'majority_baseline': majority_share(y[test_idx])}, f, indent=2)


if __name__ == '__main__':
    main()
