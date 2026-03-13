import argparse
import os
import pickle
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import torch

from ghost.datasets.hyperspectral_dataset import HyperspectralDataset
from ghost.models.spectral_ssm import SpectralSSMEncoder
from ghost.rssp.rssp_inference import run_inference


# ── Dataset-specific class names ──────────────────────────────────────────────
CLASS_NAMES = {
    'indian_pines': [
        'Background', 'Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn',
        'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
        'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
        'Stone-Steel-Towers'
    ],
    'pavia': [
        'Background', 'Asphalt', 'Meadows', 'Gravel', 'Trees',
        'Painted metal sheets', 'Bare Soil', 'Bitumen',
        'Self-Blocking Bricks', 'Shadows'
    ],
    'salinas': [
        'Background', 'Weeds-1', 'Weeds-2', 'Fallow', 'Fallow-rough-plow',
        'Fallow-smooth', 'Stubble', 'Celery', 'Grapes-untrained',
        'Soil-vinyard-develop', 'Corn-senesced-green-weeds',
        'Lettuce-romaine-4wk', 'Lettuce-romaine-5wk', 'Lettuce-romaine-6wk',
        'Lettuce-romaine-7wk', 'Vinyard-untrained', 'Vinyard-vertical-trellis'
    ],
}

# 20 visually distinct colours (index 0 = black for background)
PALETTE = [
    '#000000', '#e6194b', '#3cb44b', '#ffe119', '#4363d8',
    '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45',
    '#fabed4', '#469990', '#dcbeff', '#9a6324', '#fffac8',
    '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075',
    '#a9a9a9', '#ffffff'
]


def get_cmap(num_classes):
    colours = PALETTE[:num_classes]
    return ListedColormap(colours)


def false_colour(data_chw, r_band=None, g_band=None, b_band=None):
    """
    Build a false-colour RGB image from three bands.
    Defaults to evenly spaced bands across the spectrum.
    data_chw: (C, H, W) numpy array
    """
    C = data_chw.shape[0]
    r = r_band if r_band is not None else int(C * 0.75)
    g = g_band if g_band is not None else int(C * 0.50)
    b = b_band if b_band is not None else int(C * 0.25)

    rgb = np.stack([data_chw[r], data_chw[g], data_chw[b]], axis=-1)  # (H, W, 3)

    # Percentile stretch for visual clarity
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)
    return rgb


def build_legend(class_names, num_classes, cmap):
    patches = []
    for i in range(1, num_classes):
        name = class_names[i] if i < len(class_names) else f'Class {i}'
        patches.append(mpatches.Patch(color=cmap(i), label=name))
    return patches


def visualize(data_chw, labels_hw, pred_hw, class_names,
              title='GHOST Segmentation', save_path=None,
              r_band=None, g_band=None, b_band=None):
    """
    Three-panel figure: False Colour | Ground Truth | Prediction
    """
    num_classes = int(labels_hw.max()) + 1
    cmap        = get_cmap(max(num_classes, pred_hw.max() + 1))

    rgb = false_colour(data_chw, r_band, g_band, b_band)

    fig, axes = plt.subplots(1, 3, figsize=(20, 15))
    fig.patch.set_facecolor('#1a1a2e')

    panel_titles = ['False Colour Composite', 'Ground Truth Labels', 'GHOST Prediction']
    images       = [rgb, labels_hw, pred_hw]
    cmaps        = [None, cmap, cmap]
    vnorm        = [None, (0, num_classes - 1), (0, num_classes - 1)]

    for ax, img, cm, vn, ptitle in zip(axes, images, cmaps, vnorm, panel_titles):
        ax.set_facecolor('#1a1a2e')
        if cm is None:
            ax.imshow(img, interpolation='nearest')
        else:
            ax.imshow(img, cmap=cm, vmin=vn[0], vmax=vn[1], interpolation='nearest')
        ax.set_title(ptitle, color='white', fontsize=13, fontweight='bold', pad=10)
        ax.axis('off')

    # Legend below panels
    legend_patches = build_legend(class_names, num_classes, cmap)
    fig.legend(
        handles=legend_patches,
        loc='lower center',
        ncol=min(8, len(legend_patches)),
        fontsize=8,
        framealpha=0.15,
        labelcolor='white',
        facecolor='#1a1a2e',
        edgecolor='none',
        bbox_to_anchor=(0.5, -0.02)
    )

    fig.suptitle(title, color='white', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"Saved → {save_path}")

    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='GHOST — Visualize Segmentation Results')

    parser.add_argument('--data',        type=str, required=True)
    parser.add_argument('--gt',          type=str, required=True)
    parser.add_argument('--model',       type=str, required=True, help='Path to rssp_models.pkl')
    parser.add_argument('--ssm_load',    type=str, default=None)
    parser.add_argument('--train_ratio', type=float, default=0.2)
    parser.add_argument('--val_ratio',   type=float, default=0.1)
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--routing',     type=str,   default='forest',
                        choices=['hybrid', 'forest', 'soft'])
    parser.add_argument('--dataset',     type=str,   default=None,
                        choices=['indian_pines', 'pavia', 'salinas'],
                        help='Dataset name for class labels (optional)')
    parser.add_argument('--r_band',      type=int,   default=None)
    parser.add_argument('--g_band',      type=int,   default=None)
    parser.add_argument('--b_band',      type=int,   default=None)
    parser.add_argument('--out-dir',     type=str,   default='.')
    parser.add_argument('--title',       type=str,   default='GHOST Segmentation')

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    test_ds = HyperspectralDataset(
        args.data, args.gt, split='test',
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
    data        = test_ds.data
    labels      = test_ds.labels.numpy()
    num_classes = test_ds.num_classes

    # Load model
    with open(args.model, 'rb') as f:
        checkpoint = pickle.load(f)

    trained_models = checkpoint['trained_models']
    tree           = checkpoint['tree']
    d_model        = checkpoint.get('d_model', 64)
    d_state        = checkpoint.get('d_state', 16)

    # SSM encoder
    ssm_encoder = None
    if args.routing in ('hybrid', 'soft'):
        ssm_encoder = SpectralSSMEncoder(d_model=d_model, d_state=d_state).to(DEVICE)
        if args.ssm_load:
            state = torch.load(args.ssm_load, map_location=DEVICE, weights_only=True)
        else:
            state = {k: v.to(DEVICE) for k, v in checkpoint['ssm_state'].items()}
        ssm_encoder.load_state_dict(state)
        ssm_encoder.eval()
        for p in ssm_encoder.parameters():
            p.requires_grad_(False)

    # Run inference
    print(f"Running inference (routing={args.routing}) ...")
    pred = run_inference(
        tree, trained_models, data,
        ssm_encoder, DEVICE, num_classes,
        routing=args.routing
    )

    # Class names
    class_names = CLASS_NAMES.get(args.dataset, [f'Class {i}' for i in range(num_classes)])

    # Visualize
    data_np   = data.numpy()   # (C, H, W)
    save_path = os.path.join(args.out_dir, f'segmentation_{args.routing}.png')

    visualize(
        data_chw    = data_np,
        labels_hw   = labels,
        pred_hw     = pred,
        class_names = class_names,
        title       = args.title,
        save_path   = save_path,
        r_band      = args.r_band,
        g_band      = args.g_band,
        b_band      = args.b_band,
    )


if __name__ == '__main__':
    main()