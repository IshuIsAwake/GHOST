"""ghost visualize (architecture 0.2.0): false colour, ground truth and prediction side by side."""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

from ghost.utils.display import BOLD, GREEN, RED, RESET, print_visualize_start
from ghost.visualize import CLASS_NAMES, PALETTE

BACKGROUND = '#1a1a2e'


def class_colormap(max_id: int) -> ListedColormap:
    """v0.1's palette (index 0 black), extended when a scene has more classes than it covers."""
    n = max_id + 1
    colours = list(PALETTE[:n])
    if n > len(PALETTE):
        extra = matplotlib.colormaps['gist_ncar'](np.linspace(0.05, 0.95, n - len(PALETTE)))
        colours += [tuple(c) for c in extra]
    return ListedColormap(colours)


def false_colour(cube: np.ndarray, r_band=None, g_band=None, b_band=None) -> np.ndarray:
    """RGB from three bands (default 75/50/25% of the spectrum), each stretched on its own 2–98th percentiles."""
    B = cube.shape[-1]
    bands = [b if b is not None else int(B * f) for b, f in zip((r_band, g_band, b_band), (0.75, 0.50, 0.25))]
    rgb = np.nan_to_num(cube[:, :, bands]).astype(np.float64)
    lo, hi = np.percentile(rgb, 2, axis=(0, 1)), np.percentile(rgb, 98, axis=(0, 1))
    return np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)


def class_names_for(dataset: str | None):
    return CLASS_NAMES.get(dataset) if dataset else None


def _legend(class_ids, cmap, names):
    return [mpatches.Patch(color=cmap(c), label=names[c] if names and c < len(names) else f'Class {c}')
            for c in class_ids]


def _panel(ax, image, title, cmap=None, vmax=None):
    ax.set_facecolor(BACKGROUND)
    if cmap is None:
        ax.imshow(image, interpolation='nearest')
    else:
        ax.imshow(image, cmap=cmap, vmin=0, vmax=vmax, interpolation='nearest')
    ax.set_title(title, color='white', fontsize=12, fontweight='bold', pad=8)
    ax.axis('off')


def save_prediction_png(pred_map, class_ids, path, class_names=None, title='GHOST prediction'):
    max_id = max(class_ids)
    cmap = class_colormap(max_id)
    fig = Figure(figsize=(8, 8), facecolor=BACKGROUND)
    _panel(fig.subplots(), pred_map, title, cmap, max_id)
    fig.legend(handles=_legend(class_ids, cmap, class_names), loc='lower center', ncol=min(6, len(class_ids)),
               fontsize=7, labelcolor='white', frameon=False)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BACKGROUND)


def save_segmentation_figure(cube, gt, pred_map, class_ids, path, class_names=None,
                             title='GHOST Segmentation', bands=(None, None, None)):
    max_id = max(class_ids + ([int(gt.max())] if gt is not None else []))
    cmap = class_colormap(max_id)
    panels = [(false_colour(cube, *bands), 'False Colour Composite', None)]
    if gt is not None:
        panels.append((gt, 'Ground Truth Labels', cmap))
    panels.append((pred_map, 'GHOST Prediction (every pixel)', cmap))

    fig = Figure(figsize=(6 * len(panels), 7), facecolor=BACKGROUND)
    for ax, (image, name, cm) in zip(fig.subplots(1, len(panels)), panels):
        _panel(ax, image, name, cm, max_id)
    ids = sorted(set(class_ids) | (set(np.unique(gt[gt > 0]).tolist()) if gt is not None else set()))
    fig.legend(handles=_legend(ids, cmap, class_names), loc='lower center', ncol=min(8, len(ids)), fontsize=8,
               labelcolor='white', frameon=False)
    fig.suptitle(title, color='white', fontsize=15, fontweight='bold')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BACKGROUND)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='ghost visualize', description='GHOST 0.2.0 — segmentation figure')
    p.add_argument('--model', required=True, help='Checkpoint written by ghost train')
    p.add_argument('--data', required=True)
    p.add_argument('--gt', default=None, help='Optional labels, shown as their own panel')
    p.add_argument('--dataset', default=None, choices=sorted(CLASS_NAMES), help='Class names for the legend')
    p.add_argument('--r_band', type=int, default=None)
    p.add_argument('--g_band', type=int, default=None)
    p.add_argument('--b_band', type=int, default=None)
    p.add_argument('--title', default='GHOST Segmentation')
    p.add_argument('--out-dir', dest='out_dir', default='.')
    p.add_argument('--batch_size', type=int, default=8192)
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    return p


def run(args) -> str:
    from ghost.v0_2.checkpoint import load_checkpoint
    from ghost.v0_2.predict import check_arch, predict_scene
    from ghost.v0_2.scene import load_scene
    from ghost.v0_2.train import resolve_device

    print_visualize_start()
    ckpt = check_arch(load_checkpoint(args.model), args.model)
    scene = load_scene(args.data, args.gt)
    pred_map = predict_scene(ckpt, scene, resolve_device(args.device), args.batch_size)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, 'segmentation.png')
    save_segmentation_figure(scene['cube'], scene['gt'], pred_map, ckpt['class_ids'], path,
                             class_names_for(args.dataset), args.title, (args.r_band, args.g_band, args.b_band))
    print(f"  {GREEN}{BOLD}Saved →{RESET} {path}")
    return path


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (ValueError, FileNotFoundError, ImportError, KeyError) as exc:
        print(f"\n{RED}{BOLD}  {exc}{RESET}\n", file=sys.stderr)
        sys.exit(2)


if __name__ == '__main__':
    main()
