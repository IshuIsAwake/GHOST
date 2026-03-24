import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ghost.datasets.hyperspectral_dataset import HyperspectralDataset
from ghost.models.hyperspectral_net import HyperspectralNet
from ghost.models.losses import build_criterion
from ghost.utils.display import (
    print_training_start, print_training_done,
    print_results_box, epoch_bar
)
import csv
import numpy as np

# ── Args ──────────────────────────────────────────────────────────────────────
def main():
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

    # Loss
    parser.add_argument('--loss',        type=str,   default='ce',
                        choices=['ce', 'squared_ce', 'focal', 'dice'],
                        help='Loss function (default: ce). '
                             'squared_ce: CE squared, amplifies hard-example penalty. '
                             'focal: focal loss with --focal_gamma. '
                             'dice: combined CrossEntropy + Dice loss.')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma — higher values focus more on hard examples (default: 2.0)')

    # Precision
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision (float16) training')

    # Output
    parser.add_argument('--out-dir', type=str, default='.', help='Output directory')
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

    _dl_kwargs = dict(batch_size=1, shuffle=False, num_workers=2,
                      pin_memory=torch.cuda.is_available(), persistent_workers=True)
    train_loader = DataLoader(train_ds, **_dl_kwargs)
    val_loader   = DataLoader(val_ds,   **_dl_kwargs)
    test_loader  = DataLoader(test_ds,  **_dl_kwargs)

    # ── Model ─────────────────────────────────────────────────────────────────────
    model = HyperspectralNet(
        num_bands=train_ds.num_bands,
        num_classes=train_ds.num_classes,
        num_filters=args.num_filters,
        num_blocks=args.num_blocks,
        base_filters=args.base_filters,
        use_fp16=args.fp16
    ).to(DEVICE)

    # channels_last_3d lets cuDNN pick faster Conv3D algorithms
    if DEVICE != 'cpu':
        model.spectral_3d.stack = model.spectral_3d.stack.to(
            memory_format=torch.channels_last_3d)

    print(f"Training on {DEVICE} | {'fp16 (mixed precision)' if args.fp16 else 'fp32 (full precision)'}")
    print(f"Bands: {train_ds.num_bands} | Classes: {train_ds.num_classes} | Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Loss: {args.loss}" + (f" | gamma={args.focal_gamma}" if 'focal' in args.loss else "") + "\n")

    # ── Loss ──────────────────────────────────────────────────────────────────────
    criterion = build_criterion(
        loss_type   = args.loss,
        focal_gamma = args.focal_gamma,
        ignore_index= 0
    )

    # ── Optimizer, scheduler ──────────────────────────────────────────────────────
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

        ious, dices, precisions, recalls, per_class_acc = [], [], [], [], []
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
            if target_c.sum() > 0:
                per_class_acc.append(tp / target_c.sum().item())

        miou      = sum(ious)           / len(ious)           if ious           else 0
        dice      = sum(dices)          / len(dices)          if dices          else 0
        precision = sum(precisions)     / len(precisions)     if precisions     else 0
        recall    = sum(recalls)        / len(recalls)        if recalls        else 0
        aa        = sum(per_class_acc)  / len(per_class_acc)  if per_class_acc  else 0

        # Cohen's kappa
        n        = target_m.numel()
        po       = oa
        pe_sum   = 0.0
        for c in range(1, num_classes):
            p_pred   = (pred_m == c).sum().item() / n
            p_target = (target_m == c).sum().item() / n
            pe_sum  += p_pred * p_target
        kappa = (po - pe_sum) / (1 - pe_sum + 1e-10) if (1 - pe_sum) > 1e-10 else 0.0

        return oa, miou, dice, precision, recall, aa, kappa

    # ── Training loop ─────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, args.log), 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'val_oa', 'val_miou',
                                 'val_dice', 'val_precision', 'val_recall', 'val_aa', 'val_kappa'])

    print_training_start()
    best_val_miou = 0
    best_epoch    = 0
    val_loss = val_oa = val_miou = val_dice = val_precision = val_recall = val_aa = val_kappa = 0.0

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

        scheduler.step(loss.item())

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                for data, labels in val_loader:
                    data, labels = data.to(DEVICE), labels.to(DEVICE)
                    with torch.autocast('cuda', enabled=args.fp16):
                        output   = model(data)
                        val_loss = criterion(output, labels)
                    val_oa, val_miou, val_dice, val_precision, val_recall, val_aa, val_kappa = \
                        compute_metrics(output, labels, train_ds.num_classes)

            epoch_bar(epoch, args.epochs, loss.item(),
                      val_loss=val_loss.item(),
                      oa=val_oa, miou=val_miou,
                      aa=val_aa, kappa=val_kappa,
                      interval=10)

            # Log metrics to CSV on validation epochs
            with open(os.path.join(args.out_dir, args.log), 'a', newline='') as f:
                csv.writer(f).writerow([epoch, f"{loss.item():.4f}", f"{val_loss.item():.4f}",
                                        f"{val_oa:.4f}", f"{val_miou:.4f}", f"{val_dice:.4f}",
                                        f"{val_precision:.4f}", f"{val_recall:.4f}",
                                        f"{val_aa:.4f}", f"{val_kappa:.4f}"])

            # Save best model checkpoint
            if val_miou > best_val_miou:
                best_val_miou = val_miou
                best_epoch    = epoch
                torch.save(model.state_dict(), os.path.join(args.out_dir, args.save))
        else:
            epoch_bar(epoch, args.epochs, loss.item(), interval=10)

    # ── Test ──────────────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(os.path.join(args.out_dir, args.save), weights_only=True))
    model.eval()
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            with torch.autocast('cuda', enabled=args.fp16):
                output = model(data)
            test_oa, test_miou, test_dice, test_precision, test_recall, test_aa, test_kappa = \
                compute_metrics(output, labels, test_ds.num_classes)

    print_training_done()

    print_results_box({
        'OA':        test_oa,
        'mIoU':      test_miou,
        'Dice':      test_dice,
        'Precision': test_precision,
        'Recall':    test_recall,
        'AA':        test_aa,
        'kappa':     test_kappa,
    })

    with open(os.path.join(args.out_dir, 'test_results.csv'), 'w', newline='') as f:
        csv.writer(f).writerow(['best_epoch', 'test_oa', 'test_miou', 'test_dice',
                                 'test_precision', 'test_recall', 'test_aa', 'test_kappa'])
        csv.writer(f).writerow([best_epoch,
                                 f"{test_oa:.4f}", f"{test_miou:.4f}", f"{test_dice:.4f}",
                                 f"{test_precision:.4f}", f"{test_recall:.4f}",
                                 f"{test_aa:.4f}", f"{test_kappa:.4f}"])

    from ghost.utils.display import print_save_and_next, GREEN, BOLD, RESET
    print(f"  {BOLD}Best epoch:{RESET} {best_epoch}  |  "
          f"Logs: {args.log}  |  Test results: test_results.csv")
    print_save_and_next(
        out_dir     = args.out_dir,
        save_file   = args.save,
        data_path   = args.data,
        gt_path     = args.gt,
        train_ratio = args.train_ratio,
        val_ratio   = args.val_ratio,
    )

if __name__ == '__main__':
    main()