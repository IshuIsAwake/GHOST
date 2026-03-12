import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.hyperspectral_dataset import HyperspectralDataset
from models.hyperspectral_net import HyperspectralNet
import csv
import numpy as np

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='GHOST — Hyperspectral Segmentation Training')

# Data
parser.add_argument('--data',        type=str,   required=True,  help='Path to hyperspectral data .mat file')
parser.add_argument('--gt',          type=str,   required=True,  help='Path to ground truth labels .mat file')
parser.add_argument('--train_ratio', type=float, default=0.2,    help='Fraction of labeled pixels per class for training (default: 0.2)')
parser.add_argument('--val_ratio',   type=float, default=0.1,    help='Fraction of labeled pixels per class for validation (default: 0.1)')

# Model
parser.add_argument('--base_filters', type=int, default=32,  help='Base filter count for encoder/decoder (default: 32)')
parser.add_argument('--num_filters',  type=int, default=8,   help='Filters in Spectral3DStack (default: 8)')
parser.add_argument('--num_blocks',   type=int, default=3,   help='Number of 3D conv blocks (default: 3)')

# Training
parser.add_argument('--epochs', type=int,   default=300,  help='Number of training epochs (default: 300)')
parser.add_argument('--lr',     type=float, default=1e-4, help='Learning rate (default: 1e-4)')
parser.add_argument('--seed',   type=int,   default=42,   help='Random seed (default: 42)')

# Precision
parser.add_argument('--fp16', action='store_true', help='Use mixed precision (float16) training — faster, lower VRAM, minor performance tradeoff')

# Output
parser.add_argument('--save',    type=str, default='best_model.pth', help='Path to save best model weights (default: best_model.pth)')
parser.add_argument('--log',     type=str, default='training_log.csv', help='Path to training log CSV (default: training_log.csv)')

args = parser.parse_args()

# ── Seeds ─────────────────────────────────────────────────────────────────────
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Datasets ──────────────────────────────────────────────────────────────────
train_ds = HyperspectralDataset(args.data, args.gt, split='train', train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed, use_fp16=args.fp16)
val_ds   = HyperspectralDataset(args.data, args.gt, split='val',   train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed, use_fp16=args.fp16)
test_ds  = HyperspectralDataset(args.data, args.gt, split='test',  train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed, use_fp16=args.fp16)

train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False)

# ── Model ─────────────────────────────────────────────────────────────────────
model = HyperspectralNet(
    num_bands=train_ds.num_bands,
    num_classes=train_ds.num_classes,
    num_filters=args.num_filters,
    num_blocks=args.num_blocks,
    base_filters=args.base_filters,
    use_fp16=args.fp16
).to(DEVICE)

print(f"Training on {DEVICE} | {'fp16 (mixed precision)' if args.fp16 else 'fp32 (full precision)'}")
print(f"Bands: {train_ds.num_bands} | Classes: {train_ds.num_classes} | Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

# ── Loss, optimizer, scheduler ────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
scaler    = torch.amp.GradScaler('cuda', enabled=args.fp16)

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(pred, target, num_classes):
    pred = pred.argmax(dim=1)
    mask = target != 0
    pred_m   = pred[mask]
    target_m = target[mask]

    oa = (pred_m == target_m).sum().item() / target_m.numel()

    ious, dices, precisions, recalls = [], [], [], []
    for c in range(1, num_classes):
        pred_c   = pred_m == c
        target_c = target_m == c
        tp = (pred_c & target_c).sum().item()
        fp = (pred_c & ~target_c).sum().item()
        fn = (~pred_c & target_c).sum().item()
        union = tp + fp + fn
        if union > 0:
            ious.append(tp / union)
            dices.append((2 * tp) / (2 * tp + fp + fn + 1e-8))
            precisions.append(tp / (tp + fp + 1e-8))
            recalls.append(tp / (tp + fn + 1e-8))

    miou      = sum(ious)       / len(ious)       if ious       else 0
    dice      = sum(dices)      / len(dices)      if dices      else 0
    precision = sum(precisions) / len(precisions) if precisions else 0
    recall    = sum(recalls)    / len(recalls)    if recalls    else 0

    return oa, miou, dice, precision, recall

# ── Training loop ─────────────────────────────────────────────────────────────
with open(args.log, 'w', newline='') as f:
    csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'val_oa', 'val_miou', 'val_dice', 'val_precision', 'val_recall'])

best_val_miou = 0
best_epoch    = 0

for epoch in range(1, args.epochs + 1):
    model.train()
    for data, labels in train_loader:
        data, labels = data.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        with torch.autocast('cuda', enabled=args.fp16):
            output = model(data)
            loss   = criterion(output, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    scheduler.step(loss)

    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(DEVICE), labels.to(DEVICE)
                with torch.autocast('cuda', enabled=args.fp16):
                    output   = model(data)
                    val_loss = criterion(output, labels)
                val_oa, val_miou, val_dice, val_precision, val_recall = compute_metrics(output, labels, train_ds.num_classes)

        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Val Loss: {val_loss:.4f} | Val OA: {val_oa:.4f} | Val mIoU: {val_miou:.4f} | Dice: {val_dice:.4f}")

        with open(args.log, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, f"{loss:.4f}", f"{val_loss:.4f}", f"{val_oa:.4f}",
                                    f"{val_miou:.4f}", f"{val_dice:.4f}", f"{val_precision:.4f}", f"{val_recall:.4f}"])

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            best_epoch    = epoch
            torch.save(model.state_dict(), args.save)

# ── Test ──────────────────────────────────────────────────────────────────────
model.load_state_dict(torch.load(args.save, weights_only=True))
model.eval()
with torch.no_grad():
    for data, labels in test_loader:
        data, labels = data.to(DEVICE), labels.to(DEVICE)
        with torch.autocast('cuda', enabled=args.fp16):
            output = model(data)
        test_oa, test_miou, test_dice, test_precision, test_recall = compute_metrics(output, labels, test_ds.num_classes)

print(f"\nTest OA:        {test_oa:.4f}")
print(f"Test mIoU:      {test_miou:.4f}")
print(f"Test Dice:      {test_dice:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall:    {test_recall:.4f}")

with open('test_results.csv', 'w', newline='') as f:
    csv.writer(f).writerow(['best_epoch', 'test_oa', 'test_miou', 'test_dice', 'test_precision', 'test_recall'])
    csv.writer(f).writerow([best_epoch, f"{test_oa:.4f}", f"{test_miou:.4f}", f"{test_dice:.4f}",
                            f"{test_precision:.4f}", f"{test_recall:.4f}"])

print(f"\nLogs: {args.log} | Test results: test_results.csv | Best epoch: {best_epoch}")