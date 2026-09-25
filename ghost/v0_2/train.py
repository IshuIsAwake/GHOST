"""ghost train (architecture 0.2.0): per-pixel training on one labelled scene."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch

from ghost.utils.display import (BOLD, CYAN, GRAY, GREEN, RED, RESET, epoch_bar, print_config_box,
                                 print_per_class_iou, print_results_box, print_training_done,
                                 print_training_start)
from ghost.v0_2 import ARCH_VERSION
from ghost.v0_2.checkpoint import build_payload, save_checkpoint
from ghost.v0_2.inference import predict_logits, scene_predictions
from ghost.v0_2.losses import LOSSES, build_criterion
from ghost.v0_2.metrics import METRIC_KEYS, compute_metrics, majority_share
from ghost.v0_2.model import POOLS, build_model, default_config
from ghost.v0_2.preprocessing import CR_MODES, build_settings, preprocess
from ghost.v0_2.scene import class_ids_of, load_scene, remap_labels
from ghost.v0_2.splits import SPLIT_MODES, make_split

RESULT_COLUMNS = ['test_oa', 'test_miou', 'test_dice', 'test_precision', 'test_recall', 'test_aa', 'test_kappa']
LOG_COLUMNS = ['epoch', 'train_loss', 'val_loss', 'val_oa', 'val_miou', 'val_dice', 'val_precision',
               'val_recall', 'val_aa', 'val_kappa', 'lr']


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='ghost train', description='GHOST 0.2.0 — per-pixel spectral training')
    p.add_argument('--data', required=True, help='Hyperspectral cube (.mat, .hdr, .tif, .h5)')
    p.add_argument('--gt', required=True, help='Ground-truth labels (.mat, .tif, .hdr, .h5, image)')

    p.add_argument('--cr', default='auto', choices=CR_MODES,
                   help='Continuum removal: auto (full ≥64 bands, simple 3–63, off <3), full, simple, off, '
                        'or none (scene z-score, no continuum removal)')
    p.add_argument('--split', default='ratio', choices=SPLIT_MODES,
                   help='ratio: v0.1\'s per-class split; fixed: N pixels per class; disjoint: spatial blocks')
    p.add_argument('--train_ratio', type=float, default=0.2, help='ratio/disjoint: training share (default 0.2)')
    p.add_argument('--val_ratio', type=float, default=0.1,
                   help='ratio/disjoint: validation share; fixed: share of the non-training pixels (default 0.1)')
    p.add_argument('--samples_per_class', type=int, default=50, help='fixed: training pixels per class')
    p.add_argument('--minority_samples', type=int, default=15,
                   help='fixed: training pixels for classes smaller than --samples_per_class')
    p.add_argument('--block_size', type=int, default=None, help='disjoint: block side in pixels (default H/10)')

    p.add_argument('--channels', type=int, default=64, help='Encoder width, constant through the stack')
    p.add_argument('--embed_dim', type=int, default=128)
    p.add_argument('--depth', type=int, default=5, help='Maximum residual blocks before band-count pruning')
    p.add_argument('--kernel_size', type=int, default=7)
    p.add_argument('--head_hidden', type=int, default=128)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--pool', default='avg', choices=POOLS,
                   help='avg: average over bands; flatten: keep band positions')

    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=50, help='Stop after this many epochs without a better val mIoU')
    p.add_argument('--min_epochs', type=int, default=40)
    p.add_argument('--loss', default='ce', choices=LOSSES, help="'dice' is 0.5·CE + 0.5·Dice, as in v0.1")
    p.add_argument('--focal_gamma', type=float, default=2.0)
    p.add_argument('--seed', type=int, default=42, help='Seeds the split and the initialisation')
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])

    p.add_argument('--out-dir', dest='out_dir', default='.')
    p.add_argument('--save', default='ghost_model.pt')
    p.add_argument('--log', default='training_log.csv')
    return p


def resolve_device(name: str) -> torch.device:
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError("--device cuda was requested but CUDA is not available")
    return torch.device(name)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _batches(n: int, batch_size: int, generator: torch.Generator) -> list:
    """Shuffled batches; a trailing batch of one joins the previous one, since BatchNorm needs two samples."""
    order = torch.randperm(n, generator=generator)
    batches = [order[i:i + batch_size] for i in range(0, n, batch_size)]
    if len(batches) > 1 and len(batches[-1]) == 1:
        batches[-2] = torch.cat([batches[-2], batches.pop()])
    return batches


def _git_commit() -> str | None:
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=root,
                               capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            return None
        return out.stdout.strip() + ('-dirty' if dirty.stdout.strip() else '')
    except (OSError, subprocess.SubprocessError):
        return None


def _scalar_metrics(m: dict | None) -> dict | None:
    return None if m is None else {k: m[k] for k in METRIC_KEYS}


def write_results(path: str, metrics: dict, leading: dict):
    """v0.1's test_results.csv layout: leading columns, then the seven metrics to four decimals."""
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(list(leading) + RESULT_COLUMNS)
        w.writerow(list(leading.values()) + [f"{metrics[k]:.4f}" for k in METRIC_KEYS])


def write_class_report(path: str, metrics: dict):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['class_id', 'total_test_pixels', 'correct_pixels', 'IoU', 'Precision', 'Recall'])
        for c, v in sorted(metrics['per_class'].items()):
            w.writerow([c, v['total'], v['correct'], f"{v['iou']:.6f}", f"{v['precision']:.6f}", f"{v['recall']:.6f}"])


def run(args) -> dict:
    seed_everything(args.seed)
    device = resolve_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    print_training_start()

    scene = load_scene(args.data, args.gt)
    cube, valid, gt = scene['cube'], scene['valid'], scene['gt']
    H, W, B = cube.shape
    class_ids = class_ids_of(gt)
    if len(class_ids) < 2:
        raise ValueError(f"Training needs at least two labelled classes; {args.gt} has {len(class_ids)}")
    if scene['invalid_labelled']:
        print(f"  {GRAY}{scene['invalid_labelled']} labelled pixels have no usable spectrum and are ignored{RESET}")

    t0 = time.time()
    settings = build_settings(args.cr, cube, valid, scene['wavelengths'])
    print(f"  Continuum removal ({settings['cr_mode']}) on {int(valid.sum()):,} pixels ...", end='', flush=True)
    feats = preprocess(cube, valid, settings)
    preprocessing_seconds = time.time() - t0
    print(f" {preprocessing_seconds:.1f}s")

    split_params = {'train_ratio': args.train_ratio, 'val_ratio': args.val_ratio,
                    'samples_per_class': args.samples_per_class, 'minority_samples': args.minority_samples,
                    'block_size': args.block_size}
    train_idx, val_idx, test_idx = make_split(args.split, gt, seed=args.seed, **split_params)
    if len(train_idx) < 2 or len(test_idx) == 0:
        raise ValueError(f"The {args.split} split left {len(train_idx)} training and {len(test_idx)} test pixels")

    gt_flat = gt.reshape(-1)
    targets = torch.from_numpy(remap_labels(gt, class_ids).reshape(-1))
    X = torch.from_numpy(feats)
    X_train, y_train = X[train_idx].to(device), targets[train_idx].to(device)
    X_val, y_val = X[val_idx].to(device), targets[val_idx].to(device)
    ids = np.asarray(class_ids, dtype=np.int64)

    config = default_config(B, len(class_ids), channels=args.channels, embed_dim=args.embed_dim,
                            depth=args.depth, kernel_size=args.kernel_size, head_hidden=args.head_hidden,
                            dropout=args.dropout, pool=args.pool)
    model = build_model(config).to(device)
    criterion = build_criterion(args.loss, args.focal_gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    generator = torch.Generator().manual_seed(args.seed)

    enc = model.encoder
    print_config_box(f"GHOST {ARCH_VERSION} Training", [
        ("Device", str(device)),
        ("Data", f"{H}×{W}, {B} bands, {len(class_ids)} classes"),
        ("CR mode", f"{settings['cr_mode']} (x-axis: {settings['x_axis']})"),
        ("Split", f"{args.split}: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test"),
        ("Encoder", f"kernel {enc.kernel_size}, dilations {enc.dilations}, sees {enc.receptive_field} bands, "
                    f"pool {args.pool}"),
        ("Loss", args.loss),
        ("Params", f"{sum(p.numel() for p in model.parameters()):,}"),
        ("Seed", str(args.seed)),
    ])

    log_path = os.path.join(args.out_dir, args.log)
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(LOG_COLUMNS)

    best = {'miou': -1.0, 'epoch': 0, 'state': None, 'metrics': None}
    t_train = time.time()
    epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for idx in _batches(len(X_train), args.batch_size, generator):
            idx = idx.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(X_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
            count += len(idx)
        train_loss = total / count

        if len(val_idx):
            logits = predict_logits(model, X_val)
            val_loss = criterion(logits, y_val).item()
            val = compute_metrics(ids[logits.argmax(1).cpu().numpy()], gt_flat[val_idx], class_ids)
            scheduler.step(val_loss)
            improved = val['mIoU'] > best['miou']
        else:
            val_loss, val = float('nan'), None
            scheduler.step(train_loss)
            improved = True
        if improved:
            best = {'miou': val['mIoU'] if val else 0.0, 'epoch': epoch, 'metrics': val,
                    'state': {k: v.detach().clone() for k, v in model.state_dict().items()}}

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}"]
                                   + [f"{val[k]:.6f}" if val else '' for k in METRIC_KEYS]
                                   + [f"{optimizer.param_groups[0]['lr']:.2e}"])
        epoch_bar(epoch, args.epochs, train_loss, val_loss=val_loss if val else None,
                  oa=val['OA'] if val else None, miou=val['mIoU'] if val else None,
                  aa=val['AA'] if val else None, kappa=val['kappa'] if val else None, interval=10)

        if val and epoch >= args.min_epochs and epoch - best['epoch'] >= args.patience:
            print(f"\n  Early stop at epoch {epoch}: no better val mIoU since epoch {best['epoch']}")
            break
    training_seconds = time.time() - t_train

    model.load_state_dict(best['state'])
    pred_flat = scene_predictions(model, feats, valid, class_ids, device)
    test = compute_metrics(pred_flat[test_idx], gt_flat[test_idx], class_ids)
    baseline = majority_share(gt_flat[test_idx])

    print_training_done()
    print_results_box(_scalar_metrics(test))
    print_per_class_iou(test['per_class_iou'], pixel_counts={c: v['total'] for c, v in test['per_class'].items()})
    print(f"  Majority-class baseline on these {len(test_idx):,} test pixels: {baseline:.4f}")

    write_results(os.path.join(args.out_dir, 'test_results.csv'), test, {'best_epoch': best['epoch']})
    write_class_report(os.path.join(args.out_dir, 'class_report.csv'), test)

    split_sizes = {'train': int(len(train_idx)), 'val': int(len(val_idx)), 'test': int(len(test_idx))}
    training = {'args': vars(args), 'best_epoch': best['epoch'], 'epochs_run': epoch,
                'val_metrics': _scalar_metrics(best['metrics']), 'test_metrics': _scalar_metrics(test),
                'majority_baseline': baseline}
    ckpt_path = os.path.join(args.out_dir, args.save)
    save_checkpoint(ckpt_path, build_payload(
        model=model, class_ids=class_ids, preprocessing=settings,
        scene={'shape': [H, W, B], 'fingerprint': scene['fingerprint'],
               'label_fingerprint': scene['label_fingerprint'], 'data_path': scene['data_path']},
        split={'mode': args.split, 'seed': args.seed, 'params': split_params,
               'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx},
        training=training))

    run_config = {
        'arch': ARCH_VERSION, 'ghost_version': _ghost_version(), 'git_commit': _git_commit(),
        'torch_version': torch.__version__, 'device': str(device), 'data': os.path.abspath(args.data),
        'gt': os.path.abspath(args.gt), 'args': vars(args), 'cr_mode': settings['cr_mode'],
        'x_axis': settings['x_axis'], 'num_bands': B, 'class_ids': class_ids,
        'encoder': {'kernel_size': enc.kernel_size, 'dilations': enc.dilations,
                    'receptive_field': enc.receptive_field},
        'split_sizes': split_sizes, 'majority_baseline': baseline,
        'preprocessing_seconds': round(preprocessing_seconds, 2), 'training_seconds': round(training_seconds, 2),
        'epochs_run': epoch, 'best_epoch': best['epoch'],
        'val_metrics': _scalar_metrics(best['metrics']), 'test_metrics': _scalar_metrics(test),
    }
    with open(os.path.join(args.out_dir, 'run_config.json'), 'w') as f:
        json.dump(run_config, f, indent=2)

    print(f"\n  {BOLD}Best epoch:{RESET} {best['epoch']}  |  {GRAY}{training_seconds:.0f}s training{RESET}")
    print(f"  {GREEN}{BOLD}Saved →{RESET} {ckpt_path}")
    print(f"\n{BOLD}{CYAN}  What's next?{RESET}")
    print(f"  ghost predict   --model {ckpt_path} --data {args.data} --gt {args.gt} --out-dir {args.out_dir}")
    print(f"  ghost visualize --model {ckpt_path} --data {args.data} --gt {args.gt} --out-dir {args.out_dir}\n")

    return {'out_dir': args.out_dir, 'checkpoint': ckpt_path, 'split_sizes': split_sizes,
            'test_metrics': _scalar_metrics(test), 'val_metrics': _scalar_metrics(best['metrics']),
            'best_epoch': best['epoch'], 'epochs_run': epoch, 'cr_mode': settings['cr_mode'],
            'majority_baseline': baseline}


def _ghost_version() -> str:
    from ghost import __version__
    return __version__


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (ValueError, FileNotFoundError, ImportError, KeyError) as exc:
        print(f"\n{RED}{BOLD}  {exc}{RESET}\n", file=sys.stderr)
        sys.exit(2)


if __name__ == '__main__':
    main()
