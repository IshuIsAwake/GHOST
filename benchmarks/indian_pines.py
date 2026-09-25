"""Indian Pines protocol for GHOST 0.2.0. Runs one training at a time and skips runs that already finished.

    python benchmarks/indian_pines.py --control          # v0.2: ratio + fixed splits × 5 seeds
    python benchmarks/indian_pines.py --v01              # v0.1 flat U-Net on the same ratio-split pixels
    python benchmarks/indian_pines.py --shuffle-control  # v0.2 on permuted labels: should sit at chance
    python benchmarks/indian_pines.py --ablation         # fixed split × CR mode × pooling × 5 seeds
    python benchmarks/summarize.py                       # the table

Add --dry-run to print the commands without running them.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import scipy.io

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from ghost.data import indian_pines_path  # noqa: E402
from ghost.datasets.loader import load_labels  # noqa: E402
from ghost.v0_2.metrics import majority_share  # noqa: E402
from ghost.v0_2.splits import make_split  # noqa: E402

SPLIT_PARAMS = {'train_ratio': 0.2, 'val_ratio': 0.1, 'samples_per_class': 50, 'minority_samples': 15}
LOSS = 'dice'  # v0.1's recipe (0.5·CE + 0.5·Dice), so both architectures train on the same objective


def baseline_for(gt: np.ndarray, split: str, seed: int) -> float:
    test_idx = make_split(split, gt, seed=seed, **SPLIT_PARAMS)[2]
    return majority_share(gt.reshape(-1)[test_idx])


def plan_runs(args, data: str, gt_path: str, gt: np.ndarray) -> list:
    runs = []

    def add(label, arch, split, cr, pool, seed, gt_file=None, labels=None):
        gt_file = gt_file or gt_path
        labels = gt if labels is None else labels
        out = os.path.join(args.out, arch, split, f"{cr}-{pool}-{LOSS}" + ('-shuffled' if 'shuffle' in label else ''),
                           f"seed{seed}")
        cmd = [sys.executable, '-m', 'ghost.cli', 'train', '--arch', arch, '--data', data, '--gt', gt_file,
               '--loss', LOSS, '--seed', str(seed), '--out-dir', out]
        if arch == '0.2.0':
            cmd += ['--split', split, '--cr', cr, '--pool', pool, '--device', args.device]
            if args.epochs:
                cmd += ['--epochs', str(args.epochs)]
        meta = {'label': label, 'arch': arch, 'split': split, 'cr': cr, 'pool': pool, 'loss': LOSS, 'seed': seed,
                'majority_baseline': baseline_for(labels, split, seed)}
        runs.append((out, cmd, meta))

    for seed in args.seeds:
        if args.control:
            for split in ('ratio', 'fixed'):
                add('v0.2 control', '0.2.0', split, 'auto', 'avg', seed)
        if args.v01:
            add('v0.1 flat reference', '0.1.7', 'ratio', 'v0.1', 'unet', seed)
        if args.shuffle_control:
            shuffled = gt.copy()
            labelled = gt > 0
            shuffled[labelled] = np.random.default_rng(1000 + seed).permutation(gt[labelled])
            path = os.path.join(args.out, '_inputs', f'ip_gt_shuffled_seed{seed}.mat')
            if not args.dry_run:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                scipy.io.savemat(path, {'gt': shuffled})
            add('v0.2 shuffled labels', '0.2.0', 'fixed', 'auto', 'avg', seed, gt_file=path, labels=shuffled)
        if args.ablation:
            for cr in ('none', 'off', 'simple', 'full'):
                for pool in ('avg', 'flatten'):
                    add('v0.2 ablation', '0.2.0', 'fixed', cr, pool, seed)
    return runs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--control', action='store_true')
    p.add_argument('--v01', action='store_true')
    p.add_argument('--shuffle-control', dest='shuffle_control', action='store_true')
    p.add_argument('--ablation', action='store_true')
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--epochs', type=int, default=None, help='v0.2 only; default is ghost train\'s 300')
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--out', default=os.path.join('runs', 'bench'))
    p.add_argument('--dry-run', dest='dry_run', action='store_true')
    args = p.parse_args()
    if not (args.control or args.v01 or args.shuffle_control or args.ablation):
        p.error('choose at least one of --control, --v01, --shuffle-control, --ablation')

    data, gt_path = indian_pines_path()
    gt = np.squeeze(load_labels(gt_path)[0]).astype(np.int64)
    runs = plan_runs(args, data, gt_path, gt)
    env = dict(os.environ, PYTHONPATH=REPO + os.pathsep + os.environ.get('PYTHONPATH', ''))

    for i, (out, cmd, meta) in enumerate(runs, 1):
        done = os.path.exists(os.path.join(out, 'test_results.csv'))
        print(f"\n[{i}/{len(runs)}] {meta['label']} | {meta['arch']} {meta['split']} cr={meta['cr']} "
              f"pool={meta['pool']} seed={meta['seed']}" + ('  (done, skipping)' if done else ''))
        if args.dry_run:
            print('  ' + ' '.join(cmd))
            continue
        if done:
            continue
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, 'bench_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            sys.exit(f"Run failed ({result.returncode}): {' '.join(cmd)}")
    if not args.dry_run:
        print("\nAll runs finished. Next: python benchmarks/summarize.py")


if __name__ == '__main__':
    main()
