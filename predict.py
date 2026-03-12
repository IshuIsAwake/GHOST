import argparse
import torch
import numpy as np
import pickle

from datasets.hyperspectral_dataset import HyperspectralDataset
from models.spectral_ssm import SpectralSSMEncoder
from rssp.rssp_inference import run_inference, compute_rssp_metrics

parser = argparse.ArgumentParser(description='GHOST — Standalone Inference')

parser.add_argument('--data',        type=str,   required=True)
parser.add_argument('--gt',          type=str,   required=True)
parser.add_argument('--model',       type=str,   required=True, help='Path to rssp_models.pkl')
parser.add_argument('--ssm_load',    type=str,   required=True, help='Path to ssm_pretrained.pt')
parser.add_argument('--train_ratio', type=float, default=0.2)
parser.add_argument('--val_ratio',   type=float, default=0.1)
parser.add_argument('--seed',        type=int,   default=42)
parser.add_argument('--temperature', type=float, default=10.0,
                    help='Routing sharpness: 10=near-hard (weak SSM), 1=fully soft (strong SSM). Default: 10.0')

args = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
print(f"Routing temperature: {args.temperature}")

# ── Load dataset ──────────────────────────────────────────────────────────────
test_ds = HyperspectralDataset(
    args.data, args.gt, split='test',
    train_ratio=args.train_ratio,
    val_ratio=args.val_ratio,
    seed=args.seed
)

data        = test_ds.data
labels      = test_ds.labels
num_classes = test_ds.num_classes

# ── Load trained models ───────────────────────────────────────────────────────
print(f"\nLoading models from {args.model} ...")
with open(args.model, 'rb') as f:
    checkpoint = pickle.load(f)

trained_models = checkpoint['trained_models']
tree           = checkpoint['tree']
d_model        = checkpoint.get('d_model', 64)
d_state        = checkpoint.get('d_state', 16)
print("Models loaded.")

# ── Load SSM encoder ──────────────────────────────────────────────────────────
print(f"Loading SSM encoder from {args.ssm_load} ...")
ssm_encoder = SpectralSSMEncoder(d_model=d_model, d_state=d_state).to(DEVICE)
state = torch.load(args.ssm_load, map_location=DEVICE, weights_only=True)
ssm_encoder.load_state_dict(state)
ssm_encoder.eval()
for p in ssm_encoder.parameters():
    p.requires_grad_(False)
print("SSM encoder loaded.")

# ── Run inference ─────────────────────────────────────────────────────────────
print("\n=== Running Soft Cascade Inference ===")
final_pred = run_inference(
    tree, trained_models,
    data,
    ssm_encoder, DEVICE, num_classes,
    temperature=args.temperature
)

# ── Evaluate on test pixels only ──────────────────────────────────────────────
test_mask   = test_ds.split_mask.numpy()
labels_np   = labels.numpy()

eval_labels = np.zeros_like(labels_np)
eval_labels[test_mask > 0] = labels_np[test_mask > 0]

oa, miou, dice, precision, recall = compute_rssp_metrics(
    final_pred, eval_labels, num_classes
)

print(f"\n{'='*40}")
print(f"Test OA:        {oa:.4f}")
print(f"Test mIoU:      {miou:.4f}")
print(f"Test Dice:      {dice:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall:    {recall:.4f}")
print(f"{'='*40}")