"""Summarise runs/bench into one table: mean ± std over seeds, beside the majority-class baseline.

    python benchmarks/summarize.py [--out runs/bench]

Readings were fixed before any run (fixed split = 50 per class, 15 for small classes):
  - v0.2 on the fixed split: about 40% or less means broken; 75–85% is expected; 95% or more means look
    for leakage.
  - v0.2 below the SVM on the same split means the ResNet is not earning its place.
  - Shuffled labels should score at chance: kappa near 0. (Chance OA is not the majority baseline here;
    training is class-balanced, so guesses spread across all classes.)
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np


def load_runs(root: str) -> dict:
    groups = defaultdict(list)
    for meta_path in glob.glob(os.path.join(root, '**', 'bench_meta.json'), recursive=True):
        results = os.path.join(os.path.dirname(meta_path), 'test_results.csv')
        if not os.path.exists(results):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        with open(results, newline='') as f:
            row = dict(zip(*csv.reader(f)))
        if 'test_oa' not in row:  # train_spt's layout: routing, OA, mIoU, Dice, Precision, Recall, AA, kappa
            row = {'test_oa': row['OA'], 'test_aa': row['AA'], 'test_kappa': row['kappa'], 'test_miou': row['mIoU']}
        key = (meta['label'], meta['arch'], meta['split'], meta['cr'], meta['pool'], meta['loss'])
        groups[key].append({'oa': float(row['test_oa']), 'aa': float(row['test_aa']),
                            'kappa': float(row['test_kappa']), 'miou': float(row['test_miou']),
                            'baseline': meta['majority_baseline'], 'seed': meta['seed']})
    return groups


def reading(key, oa_mean, kappa_mean, svm_by_split) -> str:
    label, arch, split = key[0], key[1], key[2]
    if 'shuffled' in label:
        return 'at chance' if abs(kappa_mean) <= 0.05 else 'above chance: look for leakage'
    notes = []
    if arch == '0.2.0' and split == 'fixed':
        if oa_mean <= 0.45:
            notes.append('broken?')
        elif oa_mean >= 0.95:
            notes.append('look for leakage')
        elif 0.75 <= oa_mean <= 0.85:
            notes.append('expected band')
        else:
            notes.append('outside 75–85%')
    svm = svm_by_split.get((split, key[3]))
    if arch == '0.2.0' and svm and oa_mean < svm[0] - svm[1]:
        notes.append('below SVM')
    return ', '.join(notes)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default=os.path.join('runs', 'bench'))
    p.add_argument('--csv', default=None, help='Also write the table to this CSV')
    args = p.parse_args()

    groups = load_runs(args.out)
    if not groups:
        raise SystemExit(f"No finished runs under {args.out}")
    stats = {}
    for key, runs in groups.items():
        arr = {m: np.array([r[m] for r in runs]) for m in ('oa', 'aa', 'kappa', 'miou', 'baseline')}
        stats[key] = {m: (float(v.mean()), float(v.std())) for m, v in arr.items()} | {'n': len(runs)}
    svm_by_split = {(k[2], k[3]): s['oa'] for k, s in stats.items() if k[1] == 'svm'}

    header = ['label', 'arch', 'split', 'cr', 'pool', 'loss', 'n', 'OA', 'AA', 'kappa', 'mIoU', 'majority', 'reading']
    rows = []
    for key in sorted(stats, key=lambda k: (k[2], k[0], k[3], k[4])):
        s = stats[key]
        fmt = lambda m: f"{s[m][0] * 100:.2f} ± {s[m][1] * 100:.2f}"
        rows.append(list(key) + [s['n'], fmt('oa'), fmt('aa'), fmt('kappa'), fmt('miou'),
                                 f"{s['baseline'][0] * 100:.1f}", reading(key, s['oa'][0], s['kappa'][0], svm_by_split)])

    print('| ' + ' | '.join(header) + ' |')
    print('|' + '---|' * len(header))
    for row in rows:
        print('| ' + ' | '.join(str(c) for c in row) + ' |')
    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            csv.writer(f).writerows([header] + rows)


if __name__ == '__main__':
    main()
