"""
Visualization utilities for GHOST hyperspectral segmentation results.

This module provides five output types:
  1. false_colour   — pick 3 bands from the 100+ available → show as an RGB image
  2. spectral_profiles — plot the mean spectrum for each class (like a fingerprint chart)
  3. segmentation_map  — colour-coded map where each class = different colour
  4. class_accuracy    — horizontal bar chart of per-class IoU scores
  5. confidence_map    — where is the model uncertain? (darker = more uncertain)

All functions save PNG files to a given output directory.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend: works on any machine, no display needed
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# 1. False-colour composite
# ─────────────────────────────────────────────────────────────────────────────

def make_false_colour(data: np.ndarray, bands: tuple = None) -> np.ndarray:
    """
    Create a false-colour RGB image from a hyperspectral cube.

    What is false colour?
        A normal camera picks up R, G, B (3 specific wavelengths).
        A hyperspectral cube has 100+ wavelengths. We pick any 3 and
        map them to R, G, B to create a viewable image.
        Scientists often choose bands that highlight specific minerals.

    Parameters
    ----------
    data  : (C, H, W) float32 — the hyperspectral cube
    bands : (r_band, g_band, b_band) — indices into the C dimension.
            Defaults to roughly: 1/3, 1/2, 2/3 of the band range —
            a reasonable spread that usually shows good contrast.

    Returns
    -------
    rgb : (H, W, 3) uint8  — a standard RGB image ready to save/display
    """
    C, H, W = data.shape
    if bands is None:
        # Auto-select: spread across the spectral range
        bands = (C // 4, C // 2, 3 * C // 4)

    r_band, g_band, b_band = bands
    rgb = np.stack([data[r_band], data[g_band], data[b_band]], axis=-1)  # (H, W, 3)

    # Normalize each channel independently to [0, 255]
    # (using 2nd–98th percentile to avoid outlier pixels blowing out the contrast)
    out = np.zeros_like(rgb, dtype=np.float32)
    for i in range(3):
        ch    = rgb[:, :, i]
        valid = ch[ch != 0]
        if valid.size > 0:
            lo = np.percentile(valid, 2)
            hi = np.percentile(valid, 98)
            if hi > lo:
                out[:, :, i] = np.clip((ch - lo) / (hi - lo), 0, 1)
                # Keep background pixels black
                out[:, :, i][ch == 0] = 0.0
            else:
                out[:, :, i] = 0.0
        else:
            out[:, :, i] = 0.0

    return (out * 255).astype(np.uint8)


def save_false_colour(data: np.ndarray, out_path: str, bands: tuple = None,
                      title: str = "False-Colour Composite"):
    C = data.shape[0]
    if bands is None:
        bands = (C // 4, C // 2, 3 * C // 4)
    rgb = make_false_colour(data, bands)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(rgb)
    ax.set_title(f"{title}\n(bands {bands[0]}, {bands[1]}, {bands[2]} → R, G, B)")
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Spectral profile chart
# ─────────────────────────────────────────────────────────────────────────────

def save_spectral_profiles(data: np.ndarray, labels: np.ndarray,
                           out_path: str,
                           class_names: list = None,
                           wavelengths: list = None,
                           title: str = "Mean Spectral Profiles per Class"):
    """
    Plot the mean spectrum for each class.

    Think of this like showing the "average colour fingerprint" of each
    material class. Minerals absorb light differently at different wavelengths,
    so each class should have a distinct line shape.

    Parameters
    ----------
    data        : (C, H, W)
    labels      : (H, W) int — class IDs (0=background, ignored)
    wavelengths : optional list of C floats (nanometres) for the X axis
    class_names : optional list of strings, one per class ID (1-indexed)
    """
    C, H, W  = data.shape
    num_classes = int(labels.max())

    # X axis: wavelength in nm if available, else band index
    x = wavelengths if (wavelengths and len(wavelengths) == C) else list(range(C))
    x_label = "Wavelength (nm)" if wavelengths else "Band index"

    # Colour palette: matplotlib's tab20 gives 20 distinct colours
    palette = plt.cm.get_cmap('tab20', max(num_classes, 1))

    fig, ax = plt.subplots(figsize=(12, 5))

    data_hw = data.reshape(C, -1).T      # (H*W, C)
    labels_flat = labels.reshape(-1)      # (H*W,)

    for cls_id in range(1, num_classes + 1):
        mask = labels_flat == cls_id
        if mask.sum() == 0:
            continue
        mean_spectrum = data_hw[mask].mean(axis=0)   # (C,)
        label = class_names[cls_id - 1] if class_names else f"Class {cls_id}"
        colour = palette(cls_id - 1)
        ax.plot(x, mean_spectrum, label=label, color=colour, linewidth=1.5)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Normalized Reflectance")
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Segmentation map
# ─────────────────────────────────────────────────────────────────────────────

def make_seg_map(pred: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert a (H, W) integer class map → (H, W, 3) uint8 RGB colour map.

    Each class gets a distinct colour from matplotlib's tab20 palette.
    Class 0 (background/unlabeled) is always black.
    """
    H, W = pred.shape
    palette = plt.cm.get_cmap('tab20', max(num_classes, 1))
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for cls_id in range(1, num_classes):
        mask = pred == cls_id
        colour = palette(cls_id - 1)[:3]  # R, G, B (0-1)
        rgb[mask] = (np.array(colour) * 255).astype(np.uint8)
    return rgb


def save_seg_map(pred: np.ndarray, num_classes: int, out_path: str,
                 class_names: list = None,
                 background_rgb: np.ndarray = None,
                 title: str = "Segmentation Map"):
    """
    Save a colour-coded segmentation map, optionally overlaid on a false-colour image.

    Parameters
    ----------
    pred           : (H, W) int — predicted class IDs
    background_rgb : (H, W, 3) uint8 — if provided, overlays seg map on this image
    """
    seg_rgb = make_seg_map(pred, num_classes)

    if background_rgb is not None:
        # Blend: where seg_rgb is non-black (classified pixels), overlay at 60% alpha
        classified = (seg_rgb.sum(axis=2) > 0)
        composite  = background_rgb.copy().astype(float)
        composite[classified] = (0.4 * background_rgb[classified].astype(float) +
                                 0.6 * seg_rgb[classified].astype(float))
        display_img = composite.clip(0, 255).astype(np.uint8)
        figsize = (12, 7)
    else:
        display_img = seg_rgb
        figsize = (10, 6)

    # Build legend patches
    palette = plt.cm.get_cmap('tab20', max(num_classes, 1))
    patches = []
    import matplotlib.patches as mpatches
    for cls_id in range(1, num_classes):
        colour = palette(cls_id - 1)[:3]
        label  = class_names[cls_id - 1] if class_names else f"Class {cls_id}"
        patches.append(mpatches.Patch(color=colour, label=label))

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(display_img)
    ax.set_title(title)
    ax.axis('off')
    if patches:
        ax.legend(handles=patches, loc='lower right', fontsize=7,
                  ncol=min(4, len(patches)), framealpha=0.8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Per-class accuracy chart
# ─────────────────────────────────────────────────────────────────────────────

def save_class_accuracy(pred: np.ndarray, labels: np.ndarray, out_path: str,
                        class_names: list = None,
                        title: str = "Per-Class IoU"):
    """
    Horizontal bar chart of IoU (Intersection over Union) per class.

    IoU is the standard metric: a score of 1.0 = perfect, 0.0 = completely wrong.
    It tells you how well the model found each specific material class.
    """
    num_classes = int(labels.max()) + 1
    mask     = labels > 0
    pred_m   = pred[mask]
    target_m = labels[mask]

    class_ids = []
    ious      = []

    for cls_id in range(1, num_classes):
        pred_c   = pred_m == cls_id
        target_c = target_m == cls_id
        tp    = (pred_c & target_c).sum()
        fp    = (pred_c & ~target_c).sum()
        fn    = (~pred_c & target_c).sum()
        union = tp + fp + fn
        if union > 0:
            class_ids.append(cls_id)
            ious.append(tp / union)

    if not class_ids:
        print("  No labeled pixels found — skipping class accuracy chart.")
        return

    names  = [class_names[i - 1] if class_names else f"Class {i}" for i in class_ids]
    ious_a = np.array(ious)

    palette = plt.cm.get_cmap('tab20', max(num_classes, 1))
    colours = [palette(i - 1) for i in class_ids]

    fig, ax = plt.subplots(figsize=(8, max(3, len(class_ids) * 0.4)))
    bars = ax.barh(names, ious_a, color=colours, edgecolor='white', linewidth=0.5)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("IoU (Intersection over Union)")
    ax.set_title(title)
    ax.axvline(ious_a.mean(), color='red', linestyle='--', linewidth=1,
               label=f"Mean IoU: {ious_a.mean():.3f}")
    ax.legend(fontsize=9)
    # Value labels on bars
    for bar, iou in zip(bars, ious_a):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{iou:.2f}", va='center', fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Confidence / uncertainty map
# ─────────────────────────────────────────────────────────────────────────────

def save_confidence_map(weighted_probs: np.ndarray, out_path: str,
                        title: str = "Model Confidence Map"):
    """
    Visualize where the model is confident vs uncertain.

    weighted_probs : (G, H, W) — the final probability distribution per pixel.
                     This is the `weighted_probs` output from cascade_soft_inference.

    We compute entropy per pixel:
        entropy = -sum(p * log(p))
    High entropy → model is uncertain (spread probability across many classes)
    Low entropy  → model is confident (most probability on one class)

    The map uses a heatmap: bright yellow = uncertain, dark purple = confident.
    This is useful to show judges where GHOST needs more training data.
    """
    G, H, W = weighted_probs.shape

    # Normalize so probabilities sum to 1 per pixel
    prob_sum = weighted_probs.sum(axis=0, keepdims=True).clip(1e-8)
    probs    = weighted_probs / prob_sum              # (G, H, W)

    # Entropy: H(x) = -sum(p * log(p + ε))
    entropy = -(probs * np.log(probs + 1e-8)).sum(axis=0)  # (H, W)

    # Normalize entropy to [0, 1] for display
    max_entropy = np.log(G)   # maximum possible entropy (uniform distribution)
    uncertainty = entropy / (max_entropy + 1e-8)
    uncertainty = uncertainty.clip(0, 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(uncertainty, cmap='plasma', vmin=0, vmax=1)
    ax.set_title(f"{title}\n(bright = uncertain, dark = confident)")
    ax.axis('off')
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Uncertainty (normalized entropy)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")
