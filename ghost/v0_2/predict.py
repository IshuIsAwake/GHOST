"""ghost predict (architecture 0.2.0): segment a scene with a trained checkpoint, with or without labels."""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np

from ghost.utils.display import (BOLD, GRAY, GREEN, RED, RESET, print_config_box, print_per_class_iou,
                                 print_predict_start, print_results_box)
from ghost.v0_2 import ARCH_VERSION
from ghost.v0_2.checkpoint import load_checkpoint, model_from_checkpoint
from ghost.v0_2.inference import scene_predictions
from ghost.v0_2.metrics import METRIC_KEYS, compute_metrics, majority_share
from ghost.v0_2.preprocessing import preprocess
from ghost.v0_2.scene import load_scene
from ghost.v0_2.train import resolve_device, write_class_report, write_results


def build_parser() -> argparse.ArgumentParser:
    from ghost.visualize import CLASS_NAMES
    p = argparse.ArgumentParser(prog='ghost predict', description='GHOST 0.2.0 — segment a scene')
    p.add_argument('--model', required=True, help='Checkpoint written by ghost train (ghost_model.pt)')
    p.add_argument('--data', required=True, help='Cube to segment (.mat, .hdr, .tif, .h5)')
    p.add_argument('--gt', default=None,
                   help='Optional labels. On the training scene only its held-out test pixels are scored; '
                        'on any other scene every labelled pixel is')
    p.add_argument('--dataset', default=None, choices=sorted(CLASS_NAMES), help='Class names for the map legend')
    p.add_argument('--out-dir', dest='out_dir', default='.')
    p.add_argument('--batch_size', type=int, default=8192)
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    return p


def check_arch(ckpt: dict, path) -> dict:
    if ckpt['arch'] != ARCH_VERSION:
        raise ValueError(f"{path} was trained with architecture {ckpt['arch']}; this is {ARCH_VERSION}")
    return ckpt


def predict_scene(ckpt: dict, scene: dict, device, batch_size: int = 8192) -> np.ndarray:
    """(H, W) int32 label map for a loaded scene, repeating training's preprocessing exactly."""
    H, W, B = scene['cube'].shape
    expected = ckpt['model_config']['num_bands']
    if B != expected:
        raise ValueError(f"The model was trained on {expected} bands but {scene['data_path']} has {B}")
    model = model_from_checkpoint(ckpt).to(device)
    feats = preprocess(scene['cube'], scene['valid'], ckpt['preprocessing'])
    pred = scene_predictions(model, feats, scene['valid'], ckpt['class_ids'], device, batch_size)
    return pred.reshape(H, W).astype(np.int32)


def evaluate(ckpt: dict, scene: dict, pred_map: np.ndarray):
    """On the training scene, score only its stored test pixels; on any other scene, every labelled pixel."""
    gt_flat, pred_flat = scene['gt'].reshape(-1), pred_map.reshape(-1)
    if scene['fingerprint'] == ckpt['scene']['fingerprint']:
        mode = 'same_scene'
        idx = ckpt['split']['test_idx'].numpy()
        if scene['label_fingerprint'] != ckpt['scene'].get('label_fingerprint'):
            warnings.warn("These labels differ from the training labels; scoring the stored test pixels with them",
                          UserWarning)
        idx = idx[gt_flat[idx] > 0]
    else:
        mode = 'new_scene'
        idx = np.flatnonzero(gt_flat > 0)
    unknown = sorted(set(np.unique(gt_flat[idx]).tolist()) - set(ckpt['class_ids']))
    if unknown:
        warnings.warn(f"Labels {unknown} were never seen in training; those pixels can only count as errors",
                      UserWarning)
    return mode, idx, compute_metrics(pred_flat[idx], gt_flat[idx], ckpt['class_ids'])


def run(args) -> dict:
    from ghost.v0_2.visualize import class_names_for, save_prediction_png

    print_predict_start()
    device = resolve_device(args.device)
    ckpt = check_arch(load_checkpoint(args.model), args.model)
    scene = load_scene(args.data, args.gt)
    H, W, B = scene['cube'].shape
    print_config_box(f"GHOST {ARCH_VERSION} Predict", [
        ("Device", str(device)),
        ("Model", args.model),
        ("Data", f"{args.data} ({H}×{W}, {B} bands)"),
        ("CR mode", ckpt['preprocessing']['cr_mode']),
        ("Labels", args.gt or "none (segmentation only)"),
    ])

    pred_map = predict_scene(ckpt, scene, device, args.batch_size)
    os.makedirs(args.out_dir, exist_ok=True)
    npy_path = os.path.join(args.out_dir, 'prediction.npy')
    np.save(npy_path, pred_map)
    save_prediction_png(pred_map, ckpt['class_ids'], os.path.join(args.out_dir, 'prediction.png'),
                        class_names_for(args.dataset))
    summary = {'mode': None, 'n_eval': 0, 'metrics': None, 'prediction': npy_path}

    if scene['gt'] is not None:
        mode, idx, metrics = evaluate(ckpt, scene, pred_map)
        where = ("the training scene's held-out test pixels" if mode == 'same_scene'
                 else "every labelled pixel (new scene)")
        print(f"\n  Scored {len(idx):,} pixels: {where}")
        print_results_box({k: metrics[k] for k in METRIC_KEYS})
        print_per_class_iou(metrics['per_class_iou'],
                            pixel_counts={c: v['total'] for c, v in metrics['per_class'].items()})
        print(f"  Majority-class baseline on these pixels: {majority_share(scene['gt'].reshape(-1)[idx]):.4f}")
        write_results(os.path.join(args.out_dir, 'predict_results.csv'), metrics,
                      {'mode': mode, 'n_eval': int(len(idx))})
        write_class_report(os.path.join(args.out_dir, 'predict_class_report.csv'), metrics)
        summary.update(mode=mode, n_eval=int(len(idx)), metrics={k: metrics[k] for k in METRIC_KEYS})

    print(f"\n  {GREEN}{BOLD}Saved →{RESET} {npy_path} {GRAY}(+ prediction.png){RESET}\n")
    return summary


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (ValueError, FileNotFoundError, ImportError, KeyError) as exc:
        print(f"\n{RED}{BOLD}  {exc}{RESET}\n", file=sys.stderr)
        sys.exit(2)


if __name__ == '__main__':
    main()
