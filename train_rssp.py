import argparse
import torch
import numpy as np
import scipy.io
import pickle

from datasets.hyperspectral_dataset import HyperspectralDataset
from rssp.sam_clustering import build_rssp_tree, print_tree
from rssp.rssp_trainer import train_tree
from rssp.rssp_inference import cascade_inference, compute_rssp_metrics

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='GHOST RSSP Training')

parser.add_argument('--data',         type=str,   required=True)
parser.add_argument('--gt',           type=str,   required=True)
parser.add_argument('--train_ratio',  type=float, default=0.2)
parser.add_argument('--val_ratio',    type=float, default=0.1)
parser.add_argument('--depth',        type=str,   default='auto',
                    help='auto | full | integer')
parser.add_argument('--voting',       type=str,   default='weighted',
                    help='hard | weighted | threshold')
parser.add_argument('--threshold',    type=float, default=0.7)
parser.add_argument('--forests',      type=int,   default=5)
parser.add_argument('--base_filters', type=int,   default=32)
parser.add_argument('--num_filters',  type=int,   default=8)
parser.add_argument('--num_blocks',   type=int,   default=3)
parser.add_argument('--epochs',       type=int,   default=300)
parser.add_argument('--lr',           type=float, default=1e-4)
parser.add_argument('--seed',         type=int,   default=42)
parser.add_argument('--save',         type=str,   default='rssp_models.pkl')

args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Load datasets ─────────────────────────────────────────────────────────────
train_ds = HyperspectralDataset(args.data, args.gt, split='train',
                                train_ratio=args.train_ratio,
                                val_ratio=args.val_ratio, seed=args.seed)
val_ds   = HyperspectralDataset(args.data, args.gt, split='val',
                                train_ratio=args.train_ratio,
                                val_ratio=args.val_ratio, seed=args.seed)
test_ds  = HyperspectralDataset(args.data, args.gt, split='test',
                                train_ratio=args.train_ratio,
                                val_ratio=args.val_ratio, seed=args.seed)

data        = train_ds.data    # (C, H, W) tensor
labels      = train_ds.labels  # (H, W) tensor
num_classes = train_ds.num_classes

# ── Build tree ────────────────────────────────────────────────────────────────
depth_mode = args.depth
if depth_mode.isdigit():
    depth_mode = int(depth_mode)

print("\n=== Building RSSP Tree ===")
tree, sam_matrix, means = build_rssp_tree(
    data.numpy(), labels.numpy(),
    num_classes=num_classes,
    depth_mode=depth_mode
)
print_tree(tree)

# ── Train ─────────────────────────────────────────────────────────────────────
print("\n=== Training RSSP Forest ===")
trained_models = train_tree(
    tree, data, labels,
    total_classes=num_classes - 1,
    train_coords=train_ds.train_coords,
    val_coords=val_ds.val_coords,
    base_epochs=args.epochs,
    num_forests=args.forests,
    base_filters=args.base_filters,
    num_filters=args.num_filters,
    num_blocks=args.num_blocks,
    lr=args.lr,
    device=str(DEVICE)
)

# Save trained models
with open(args.save, 'wb') as f:
    pickle.dump({'trained_models': trained_models, 'tree': tree}, f)
print(f"\nModels saved to {args.save}")

# ── Inference on test set ─────────────────────────────────────────────────────
print("\n=== Running Cascade Inference ===")
final_pred = cascade_inference(
    tree, trained_models,
    data.unsqueeze(0).to(DEVICE),
    device=DEVICE,
    voting=args.voting,
    threshold=args.threshold
)

# Evaluate on test pixels only
test_mask   = test_ds.split_mask.numpy()
labels_np   = labels.numpy()

# Only score test pixels
eval_labels        = np.zeros_like(labels_np)
eval_labels[test_mask > 0] = labels_np[test_mask > 0]

oa, miou = compute_rssp_metrics(final_pred, eval_labels, num_classes)

print(f"\nTest OA:   {oa:.4f}")
print(f"Test mIoU: {miou:.4f}")