import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from ghost.models.hyperspectral_net import HyperspectralNet
from ghost.models.losses import build_criterion
from ghost.rssp.sssr_router import SSSRRouter, train_router
from ghost.utils.display import (
    epoch_bar, forest_banner, node_banner, forest_done_line,
    BOLD, RESET, CYAN, GRAY, GREEN
)


def _vram_str(device: str) -> str:
    """Return peak VRAM usage since last reset. Meaningful even after model deletion."""
    if torch.cuda.is_available() and device != 'cpu':
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        return f"{peak:.2f} GB peak"
    return "N/A (CPU)"


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
               global_start: float = None):
    """
    Train forest ensemble + SSSR routing head for one RSSP node.

    loss_type:    'ce' | 'squared_ce' | 'focal'
    focal_gamma:  gamma for focal loss
    global_start: time.time() from the very start of train_tree — used to show
                  total elapsed wall-clock time across all nodes.
    """
    node_classes  = node['classes']
    num_classes   = len(node_classes) + 1
    num_bands     = data.shape[0]
    epochs        = get_node_epochs(base_epochs, node_classes, total_classes)
    node_start    = time.time()
    if global_start is None:
        global_start = node_start

    if torch.cuda.is_available() and device != 'cpu':
        torch.cuda.reset_peak_memory_stats()  # fresh peak counter per node

    train_ds = NodeDataset(data, labels, node_classes, train_coords)
    val_ds   = NodeDataset(data, labels, node_classes, val_coords)

    print(node_banner(node_id, node_classes, num_classes, epochs,
                      num_forests, loss_type, focal_gamma,
                      train_ds.num_pixels, val_ds.num_pixels))

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False)

    # Build criterion once per node
    criterion = build_criterion(
        loss_type   = loss_type,
        focal_gamma = focal_gamma,
        ignore_index= 0
    )

    forest_models = []

    for forest_idx in range(num_forests):
        seed = 42 + forest_idx
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        print(forest_banner(forest_idx, num_forests, seed, node_id))

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
        # Initialise val metrics so they're always bound even before first checkpoint
        v_loss = float('inf')
        val_oa = val_miou = val_dice = val_prec = val_rec = val_aa = val_kappa = 0.0

        for epoch in range(1, epochs + 1):
            model.train()
            for d, l in train_loader:
                d, l = d.to(device), l.to(device)
                optimizer.zero_grad()
                out  = model(d)
                loss = criterion(out, l)
                loss.backward()
                optimizer.step()
            scheduler.step(loss)

            if epoch % 20 == 0:
                model.eval()
                with torch.no_grad():
                    for d, l in val_loader:
                        d, l   = d.to(device), l.to(device)
                        out    = model(d)
                        v_loss = criterion(out, l).item()
                        val_oa, val_miou, val_dice, val_prec, val_rec, val_aa, val_kappa = \
                            compute_node_metrics(out.cpu(), l.cpu(), num_classes)

                epoch_bar(epoch, epochs, loss.item(),
                          val_loss=v_loss,
                          oa=val_oa, miou=val_miou,
                          aa=val_aa, kappa=val_kappa,
                          interval=20)
            else:
                epoch_bar(epoch, epochs, loss.item(), interval=20)

                if val_miou > best_val_miou:
                    best_val_miou = val_miou
                    best_state    = {k: v.cpu().clone()
                                     for k, v in model.state_dict().items()}

        if best_state is None:
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}

        print(forest_done_line(forest_idx, num_forests, best_val_miou,
                               _elapsed_str(node_start),
                               _elapsed_str(global_start),
                               _vram_str(device)))

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
            global_start=global_start
        ))

    return trained