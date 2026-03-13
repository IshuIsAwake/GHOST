"""
GHOST Visualization CLI

Generates visual outputs from a trained GHOST model + a hyperspectral scene.
Works with any supported file format: .mat, .img/.lbl, .hdr, .fits, .h5

Usage examples:

  # Visualize CRISM Mars data (PDS3 format, no trained model needed)
  python visualize.py --data data/ato00027155_01_if126s_trr3.img --out_dir results/crism/

  # Visualize with a trained model (adds segmentation map + confidence)
  python visualize.py \
      --data   data/indian_pines/Indian_pines_corrected.mat \
      --gt     data/indian_pines/Indian_pines_gt.mat \
      --model  rssp_models.pkl \
      --out_dir results/indian_pines/

Outputs (saved in --out_dir):
  false_colour.png      — 3-band RGB composite of the scene
  spectral_profiles.png — mean spectrum per class (if labels exist)
  segmentation_map.png  — colour-coded class prediction (requires --model)
  seg_overlay.png       — segmentation overlaid on false-colour (requires --model)
  class_accuracy.png    — per-class IoU bars (requires --model + --gt)
  confidence_map.png    — model uncertainty per pixel (requires --model)
"""

import argparse
import sys
import pickle
import torch
import numpy as np
from pathlib import Path

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='GHOST — Visualization')
parser.add_argument('--data',      type=str, required=True,
                    help='Hyperspectral data file (.mat / .img / .hdr / .fits / .h5)')
parser.add_argument('--gt',        type=str, default=None,
                    help='Ground truth labels file (optional)')
parser.add_argument('--model',     type=str, default=None,
                    help='Trained model file (rssp_models.pkl) — optional')
parser.add_argument('--ssm_load',  type=str, default=None,
                    help='SSM encoder weights (.pt) — used with --model for hybrid routing')
parser.add_argument('--routing',   type=str, default='hybrid',
                    choices=['hybrid', 'forest', 'soft'])
parser.add_argument('--out_dir',   type=str, default='results/',
                    help='Directory to save visualizations (default: results/)')
parser.add_argument('--bands',     type=int, nargs=3, default=None, metavar=('R', 'G', 'B'),
                    help='Band indices for false-colour RGB (default: auto)')
parser.add_argument('--seed',      type=int, default=42)
parser.add_argument('--train_ratio', type=float, default=0.2)
parser.add_argument('--val_ratio',   type=float, default=0.1)
args = parser.parse_args()

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

print(f"\n=== GHOST Visualizer ===")
print(f"Data:    {args.data}")
print(f"Output:  {out_dir.resolve()}\n")

# ── Load data ─────────────────────────────────────────────────────────────────
from datasets.hyperspectral_dataset import load_hyperspectral
from visualize.utils import (save_false_colour, save_spectral_profiles,
                              save_seg_map, save_class_accuracy,
                              save_confidence_map, make_false_colour)

print("Loading hyperspectral data ...")
data_np, labels_np, meta = load_hyperspectral(
    args.data, args.gt
)

# data_np is (C, H, W), already normalized
data_tensor = torch.tensor(data_np).float()   # (C, H, W)
bands_arg   = tuple(args.bands) if args.bands else None
wavelengths = meta.get('wavelengths', [])

# ── 1. False-colour ───────────────────────────────────────────────────────────
print("\n[1/5] False-colour composite ...")
save_false_colour(
    data_np,
    str(out_dir / 'false_colour.png'),
    bands=bands_arg,
    title=f"False-Colour | {Path(args.data).name}"
)

# ── 2. Spectral profiles (only if labels exist) ───────────────────────────────
num_classes = int(labels_np.max()) + 1
if num_classes > 1:
    print(f"[2/5] Spectral profiles ({num_classes - 1} classes) ...")
    save_spectral_profiles(
        data_np, labels_np,
        str(out_dir / 'spectral_profiles.png'),
        wavelengths=wavelengths if wavelengths else None,
        title=f"Mean Spectral Profiles | {Path(args.data).name}"
    )
else:
    print("[2/5] No labeled pixels — skipping spectral profiles.")

# ── 3–5. Model inference (optional) ──────────────────────────────────────────
if args.model is None:
    print("\nNo --model provided. Skipping segmentation, accuracy, and confidence maps.")
    print(f"\nDone! Outputs saved to: {out_dir.resolve()}")
    sys.exit(0)

# Load model
print(f"\n[3/5] Loading model: {args.model}")
with open(args.model, 'rb') as f:
    checkpoint = pickle.load(f)

trained_models = checkpoint['trained_models']
tree           = checkpoint['tree']
d_model        = checkpoint.get('d_model', 64)
d_state        = checkpoint.get('d_state', 16)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {DEVICE}")

# SSM encoder
ssm_encoder = None
if args.routing in ('hybrid', 'soft'):
    from models.spectral_ssm import SpectralSSMEncoder
    ssm_encoder = SpectralSSMEncoder(d_model=d_model, d_state=d_state).to(DEVICE)
    if args.ssm_load:
        state = torch.load(args.ssm_load, map_location=DEVICE, weights_only=True)
        ssm_encoder.load_state_dict(state)
    elif 'ssm_state' in checkpoint:
        ssm_encoder.load_state_dict(
            {k: v.to(DEVICE) for k, v in checkpoint['ssm_state'].items()})
    ssm_encoder.eval()

# Run inference
print(f"  Running cascade inference (routing={args.routing}) ...")
from rssp.rssp_inference import cascade_soft_inference

with torch.no_grad():
    weighted_probs = cascade_soft_inference(
        tree, trained_models,
        data_tensor.unsqueeze(0),          # (1, C, H, W)
        ssm_encoder, DEVICE, num_classes,
        routing=args.routing
    )                                      # (1, G, H, W)

weighted_probs_np = weighted_probs.squeeze(0).numpy()   # (G, H, W)
pred = weighted_probs_np.argmax(axis=0)                 # (H, W)

# Segmentation map
print("[3/5] Segmentation map ...")
false_colour_rgb = make_false_colour(data_np, bands_arg)
save_seg_map(
    pred, num_classes,
    str(out_dir / 'segmentation_map.png'),
    title=f"Segmentation Map | {Path(args.data).name}"
)
save_seg_map(
    pred, num_classes,
    str(out_dir / 'seg_overlay.png'),
    background_rgb=false_colour_rgb,
    title=f"Segmentation Overlay | {Path(args.data).name}"
)

# Per-class accuracy (requires ground truth labels)
print("[4/5] Per-class accuracy ...")
if num_classes > 1:
    save_class_accuracy(
        pred, labels_np,
        str(out_dir / 'class_accuracy.png'),
        title="Per-Class IoU"
    )
else:
    print("  No labels — skipping class accuracy.")

# Confidence map
print("[5/5] Confidence map ...")
save_confidence_map(
    weighted_probs_np,
    str(out_dir / 'confidence_map.png'),
    title="Model Confidence Map"
)

print(f"\nDone! All outputs saved to: {out_dir.resolve()}")
