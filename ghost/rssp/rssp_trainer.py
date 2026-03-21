import time
import csv
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from ghost.models.hyperspectral_net import HyperspectralNet
from ghost.models.losses import build_criterion
from ghost.rssp.sssr_router import SSSRRouter, train_router
from ghost.utils.display import (
    _c, epoch_bar, forest_banner, node_banner, forest_done_line,
    BOLD, RESET, CYAN, GRAY, GREEN, YELLOW, RED, MAGENTA
)


def _elapsed_str(start: float) -> str:
    """Return elapsed time as Xh Ym Zs string."""
    secs  = int(time.time() - start)
    h, r  = divmod(secs, 3600)
    m, s  = divmod(r, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def _colorize_metric(name: str, value: float) -> str:
    """Color a 0–1 metric value: green ≥ 0.9, yellow ≥ 0.7, red < 0.7."""
    if value >= 0.9:
        color = GREEN
    elif value >= 0.7:
        color = YELLOW
    else:
        color = RED
    return f"{color}{name} {value:.4f}{RESET}"


class NodeDataset(Dataset):
    """
    Dataset for a single RSSP tree node.
    Masks to only pixels belonging to node's classes.
    Relabels classes locally: e.g. [3, 7, 12] → [1, 2, 3]
    """
    def __init__(self, data, labels, node_classes, split_coords):
        self.data = data
        H, W = labels.shape

        self.global_to_local = {c: i+1 for i, c in enumerate(sorted(node_classes))}
        self.local_to_global = {v: k for k, v in self.global_to_local.items()}
        self.num_classes     = len(node_classes) + 1

        self.split_mask = torch.zeros(H, W, dtype=torch.long)
        for r, c in split_coords:
            global_label = labels[r, c].item() if hasattr(labels[r, c], 'item') else int(labels[r, c])
            if global_label in self.global_to_local:
                self.split_mask[r, c] = self.global_to_local[global_label]

        self.num_pixels = (self.split_mask > 0).sum().item()

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.data, self.split_mask


def compute_node_metrics(pred, target, num_classes):
    """Compute OA, mIoU, Dice, Precision, Recall, AA, kappa for a single node prediction."""
    pred     = pred.argmax(dim=1)
    mask     = target != 0
    pred_m   = pred[mask]
    target_m = target[mask]

    if target_m.numel() == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    n  = target_m.numel()
    oa = (pred_m == target_m).sum().item() / n

    ious, dices, precisions, recalls, per_class_acc = [], [], [], [], []

    for c in range(1, num_classes):
        pred_c   = pred_m == c
        target_c = target_m == c
        tp    = (pred_c & target_c).sum().item()
        fp    = (pred_c & ~target_c).sum().item()
        fn    = (~pred_c & target_c).sum().item()
        union = tp + fp + fn
        if union > 0:
            ious.append(tp / union)
            dices.append((2 * tp) / (2 * tp + fp + fn + 1e-8))
            precisions.append(tp / (tp + fp + 1e-8))
            recalls.append(tp / (tp + fn + 1e-8))
        if target_c.sum().item() > 0:
            per_class_acc.append(tp / target_c.sum().item())

    miou      = sum(ious)          / len(ious)          if ious          else 0.0
    dice      = sum(dices)         / len(dices)         if dices         else 0.0
    precision = sum(precisions)    / len(precisions)    if precisions    else 0.0
    recall    = sum(recalls)       / len(recalls)       if recalls       else 0.0
    aa        = sum(per_class_acc) / len(per_class_acc) if per_class_acc else 0.0

    # Cohen's kappa
    po     = oa
    pe_sum = 0.0
    for c in range(1, num_classes):
        p_pred   = (pred_m == c).sum().item() / n
        p_target = (target_m == c).sum().item() / n
        pe_sum  += p_pred * p_target
    kappa = (po - pe_sum) / (1 - pe_sum + 1e-10) if (1 - pe_sum) > 1e-10 else 0.0

    return oa, miou, dice, precision, recall, aa, kappa


def get_node_epochs(base_epochs, node_classes, total_classes):
    if len(node_classes) == total_classes:
        return base_epochs
    return max(base_epochs // 2, int(base_epochs * len(node_classes) / total_classes))


def train_node(node, data, labels, total_classes, train_coords, val_coords,
               fp_map,
               ssm_d_model: int = 64,
               base_epochs=300, num_forests=5, base_filters=32,
               num_filters=8, num_blocks=3, lr=1e-4, device='cuda',
               node_id='root',
               loss_type: str = 'ce', focal_gamma: float = 2.0,
               patience: int = 50, min_epochs: int = 40,
               leaf_forests: int = 3,
               warmup_epochs: int = 0,
               val_interval: int = 20,
               out_dir: str = None,
               global_start: float = None):
    """
    Train forest ensemble + SSSR routing head for one RSSP node.

    loss_type:      'ce' | 'squared_ce' | 'focal' | 'dice'
    focal_gamma:    gamma for focal loss
    patience:       early stop if val mIoU hasn't improved for this many epochs
    min_epochs:     never early-stop before this epoch
    leaf_forests:   number of forests for leaf nodes (≤ 2 classes)
    warmup_epochs:  linear LR warmup from lr/10 → lr over this many epochs
    val_interval:   validate every N epochs (default 20)
    out_dir:        output directory for CSVs (None = no CSV)
    global_start:   time.time() from the very start of train_tree
    """
    node_classes  = node['classes']
    num_classes   = len(node_classes) + 1
    num_bands     = data.shape[0]
    epochs        = get_node_epochs(base_epochs, node_classes, total_classes)
    node_start    = time.time()
    if global_start is None:
        global_start = node_start

    # Leaf nodes use fewer forests (they converge to identical answers)
    actual_forests = num_forests if len(node_classes) > 2 else min(num_forests, leaf_forests)

    train_ds = NodeDataset(data, labels, node_classes, train_coords)
    val_ds   = NodeDataset(data, labels, node_classes, val_coords)

    print(node_banner(node_id, node_classes, num_classes, epochs,
                      actual_forests, loss_type, focal_gamma,
                      train_ds.num_pixels, val_ds.num_pixels))

    _dl_kwargs = dict(batch_size=1, shuffle=False, num_workers=2,
                      pin_memory=(device != 'cpu'), persistent_workers=True)
    train_loader = DataLoader(train_ds, **_dl_kwargs)
    val_loader   = DataLoader(val_ds,   **_dl_kwargs)

    # Build criterion once per node
    criterion = build_criterion(
        loss_type   = loss_type,
        focal_gamma = focal_gamma,
        ignore_index= 0
    )

    # ── CSV epoch history ────────────────────────────────────────────────────
    csv_path = None
    csv_header_written = False
    if out_dir is not None:
        csv_path = os.path.join(out_dir, 'training_history.csv')

    forest_models  = []
    forest_mious   = []   # for node summary
    early_stop_count = 0  # for node summary

    for forest_idx in range(actual_forests):
        seed = 42 + forest_idx
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        print(forest_banner(forest_idx, actual_forests, seed, node_id))

        model = HyperspectralNet(
            num_bands    = num_bands,
            num_classes  = num_classes,
            num_filters  = num_filters,
            num_blocks   = num_blocks,
            base_filters = base_filters
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5)

        best_val_miou = 0.0
        best_state    = None
        best_epoch    = 0
        best_oa = best_aa = best_kappa = 0.0
        # Initialise val metrics so they're always bound even before first checkpoint
        v_loss = float('inf')
        val_oa = val_miou = val_dice = val_prec = val_rec = val_aa = val_kappa = 0.0
        epochs_without_improvement = 0
        stopped_early = False

        for epoch in range(1, epochs + 1):
            # LR warmup: linear ramp from lr/10 → lr over warmup_epochs
            if warmup_epochs > 0 and epoch <= warmup_epochs:
                warmup_lr = lr * (epoch / warmup_epochs)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr

            model.train()
            for d, l in train_loader:
                d, l = d.to(device), l.to(device)
                optimizer.zero_grad()
                out  = model(d)
                loss = criterion(out, l)
                loss.backward()
                optimizer.step()

            # Only step scheduler after warmup
            if epoch > warmup_epochs:
                scheduler.step(loss.item())

            if epoch % val_interval == 0 or epoch == epochs:
                model.eval()
                with torch.no_grad():
                    for d, l in val_loader:
                        d, l   = d.to(device), l.to(device)
                        out    = model(d)
                        v_loss = criterion(out, l).item()
                        val_oa, val_miou, val_dice, val_prec, val_rec, val_aa, val_kappa = \
                            compute_node_metrics(out.cpu(), l.cpu(), num_classes)

                # Update best checkpoint
                if val_miou > best_val_miou:
                    best_val_miou = val_miou
                    best_epoch    = epoch
                    best_oa       = val_oa
                    best_aa       = val_aa
                    best_kappa    = val_kappa
                    best_state    = {k: v.cpu().clone()
                                     for k, v in model.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += val_interval

                epoch_bar(epoch, epochs, loss.item(),
                          val_loss=v_loss,
                          oa=val_oa, miou=val_miou,
                          aa=val_aa, kappa=val_kappa,
                          interval=val_interval)

                # Write CSV row
                if csv_path is not None:
                    file_exists = os.path.exists(csv_path)
                    with open(csv_path, 'a', newline='') as cf:
                        writer = csv.writer(cf)
                        if not file_exists and not csv_header_written:
                            writer.writerow([
                                'node_id', 'forest_idx', 'epoch',
                                'loss', 'val_loss',
                                'OA', 'mIoU', 'Dice', 'Precision', 'Recall',
                                'AA', 'kappa'
                            ])
                            csv_header_written = True
                        writer.writerow([
                            node_id, forest_idx + 1, epoch,
                            f'{loss.item():.6f}', f'{v_loss:.6f}',
                            f'{val_oa:.6f}', f'{val_miou:.6f}',
                            f'{val_dice:.6f}', f'{val_prec:.6f}',
                            f'{val_rec:.6f}', f'{val_aa:.6f}',
                            f'{val_kappa:.6f}'
                        ])

                # Early stopping: perfect score
                if val_miou >= 0.99 and epoch >= min_epochs:
                    print(f"  ⚡ Early stop: mIoU ≥ 0.99 at epoch {epoch}")
                    stopped_early = True
                    break

                # Early stopping: patience exceeded
                if epochs_without_improvement >= patience and epoch >= min_epochs:
                    print(f"  ⚡ Early stop: no improvement for {patience} epochs at epoch {epoch}")
                    stopped_early = True
                    break

            else:
                epoch_bar(epoch, epochs, loss.item(), interval=val_interval)

        if stopped_early:
            early_stop_count += 1

        if best_state is None:
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}

        print(forest_done_line(forest_idx, actual_forests, best_val_miou,
                               best_epoch, best_oa, best_aa, best_kappa,
                               _elapsed_str(global_start)))

        forest_models.append({
            'state_dict':      best_state,
            'global_to_local': train_ds.global_to_local,
            'local_to_global': train_ds.local_to_global,
            'num_classes':     num_classes,
            'num_bands':       num_bands,
            'num_filters':     num_filters,
            'num_blocks':      num_blocks,
            'base_filters':    base_filters,
        })
        forest_mious.append(best_val_miou)

    # ── Node Summary ──────────────────────────────────────────────────────────
    if forest_mious:
        mean_miou = sum(forest_mious) / len(forest_mious)
        std_miou  = float(np.std(forest_mious)) if len(forest_mious) > 1 else 0.0
        best_idx  = int(np.argmax(forest_mious))
        worst_idx = int(np.argmin(forest_mious))
        total_time = _elapsed_str(node_start)

        mious_str = "  ".join(f"{_c(f'{v:.4f}', GREEN if v >= 0.65 else YELLOW if v >= 0.45 else RED)}"
                              for v in forest_mious)
        print(f"\n  {BOLD}Node '{node_id}'{RESET} — {actual_forests} forests trained in {total_time}")
        print(f"  mIoU  {mious_str}   {GRAY}mean {mean_miou:.4f} ± {std_miou:.4f}{RESET}")
        print(f"  {GREEN}Best{RESET}: Forest {best_idx+1} ({forest_mious[best_idx]:.4f})"
              f"  {RED}Worst{RESET}: Forest {worst_idx+1} ({forest_mious[worst_idx]:.4f})"
              f"  {GRAY}Early stops: {early_stop_count}/{actual_forests}{RESET}\n")

    # ── Train SSSR routing head (internal nodes only) ─────────────────────────
    router_state = None
    is_internal  = (node['left'] is not None and node['right'] is not None)

    if is_internal and fp_map is not None:
        print(f"\n  Training SSSR router for node '{node_id}' ...")
        router = train_router(
            node        = node,
            labels      = labels,
            train_coords= train_coords,
            val_coords  = val_coords,
            fp_map      = fp_map,
            d_model     = ssm_d_model,
            epochs      = 200,
            lr          = 1e-3,
            device      = device
        )
        if router is not None:
            router_state = {k: v.clone() for k, v in router.state_dict().items()}
            print(f"  Router trained.")
        else:
            print(f"  Router skipped (too few pixels).")

    return {
        'forests':      forest_models,
        'router_state': router_state,
        'node_classes': node_classes,
        'num_classes':  num_classes,
        'num_bands':    num_bands,
        'num_filters':  num_filters,
        'num_blocks':   num_blocks,
        'base_filters': base_filters,
        'd_model':      ssm_d_model,
    }


def train_tree(tree, data, labels, total_classes, train_coords, val_coords,
               fp_map,
               ssm_d_model: int = 64,
               base_epochs=300, num_forests=5, base_filters=32,
               num_filters=8, num_blocks=3, lr=1e-4, device='cuda',
               node_id='root',
               loss_type: str = 'ce', focal_gamma: float = 2.0,
               patience: int = 50, min_epochs: int = 40,
               leaf_forests: int = 3,
               warmup_epochs: int = 0,
               val_interval: int = 20,
               out_dir: str = None,
               global_start: float = None):
    """
    Recursively train all RSSP tree nodes.
    global_start: set once at root, passed to every node for total elapsed display.
    """
    if global_start is None:
        global_start = time.time()

    trained = {}

    trained[node_id] = train_node(
        tree, data, labels, total_classes,
        train_coords, val_coords,
        fp_map, ssm_d_model,
        base_epochs, num_forests, base_filters,
        num_filters, num_blocks, lr, device, node_id,
        loss_type=loss_type, focal_gamma=focal_gamma,
        patience=patience, min_epochs=min_epochs,
        leaf_forests=leaf_forests,
        warmup_epochs=warmup_epochs,
        val_interval=val_interval,
        out_dir=out_dir,
        global_start=global_start
    )

    if tree['left'] and len(tree['left']['classes']) > 1:
        trained.update(train_tree(
            tree['left'], data, labels, total_classes,
            train_coords, val_coords,
            fp_map, ssm_d_model,
            base_epochs, num_forests, base_filters,
            num_filters, num_blocks, lr, device,
            node_id=node_id + '_L',
            loss_type=loss_type, focal_gamma=focal_gamma,
            patience=patience, min_epochs=min_epochs,
            leaf_forests=leaf_forests,
            warmup_epochs=warmup_epochs,
            val_interval=val_interval,
            out_dir=out_dir,
            global_start=global_start
        ))

    if tree['right'] and len(tree['right']['classes']) > 1:
        trained.update(train_tree(
            tree['right'], data, labels, total_classes,
            train_coords, val_coords,
            fp_map, ssm_d_model,
            base_epochs, num_forests, base_filters,
            num_filters, num_blocks, lr, device,
            node_id=node_id + '_R',
            loss_type=loss_type, focal_gamma=focal_gamma,
            patience=patience, min_epochs=min_epochs,
            leaf_forests=leaf_forests,
            warmup_epochs=warmup_epochs,
            val_interval=val_interval,
            out_dir=out_dir,
            global_start=global_start
        ))

    return trained
