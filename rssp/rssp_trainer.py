import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models.hyperspectral_net import HyperspectralNet
from rssp.sssr_router import SSSRRouter, train_router


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
    """Compute OA, mIoU, Dice, Precision, Recall for a single node prediction."""
    pred     = pred.argmax(dim=1)
    mask     = target != 0
    pred_m   = pred[mask]
    target_m = target[mask]

    if target_m.numel() == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    oa = (pred_m == target_m).sum().item() / target_m.numel()

    ious, dices, precisions, recalls = [], [], [], []

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

    miou      = sum(ious)       / len(ious)       if ious       else 0.0
    dice      = sum(dices)      / len(dices)      if dices      else 0.0
    precision = sum(precisions) / len(precisions) if precisions else 0.0
    recall    = sum(recalls)    / len(recalls)    if recalls    else 0.0

    return oa, miou, dice, precision, recall


def get_node_epochs(base_epochs, node_classes, total_classes):
    if len(node_classes) == total_classes:
        return base_epochs
    return max(base_epochs // 2, int(base_epochs * len(node_classes) / total_classes))


def train_node(node, data, labels, total_classes, train_coords, val_coords,
               fp_map,                  # (H, W, d_model) CPU tensor — pre-computed
               ssm_d_model: int = 64,   # must match encoder's d_model
               base_epochs=300, num_forests=5, base_filters=32,
               num_filters=8, num_blocks=3, lr=1e-4, device='cuda',
               node_id='root'):
    """
    Train forest ensemble + SSSR routing head for one RSSP node.

    Returns a structured dict:
    {
        'forests':      list of model state dicts (one per forest member),
        'router_state': SSSRRouter state dict or None (None for leaf nodes),
        'node_classes', 'num_classes', 'num_bands',
        'num_filters', 'num_blocks', 'base_filters', 'd_model'
    }
    """
    node_classes = node['classes']
    num_classes  = len(node_classes) + 1
    num_bands    = data.shape[0]
    epochs       = get_node_epochs(base_epochs, node_classes, total_classes)

    print(f"\n{'='*60}")
    print(f"Node '{node_id}'")
    print(f"  Classes:       {node_classes}")
    print(f"  Local classes: {num_classes - 1}")
    print(f"  Epochs:        {epochs}")
    print(f"  Forests:       {num_forests}")

    train_ds = NodeDataset(data, labels, node_classes, train_coords)
    val_ds   = NodeDataset(data, labels, node_classes, val_coords)

    print(f"  Train pixels:  {train_ds.num_pixels}")
    print(f"  Val pixels:    {val_ds.num_pixels}")

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False)

    forest_models = []

    for forest_idx in range(num_forests):
        seed = 42 + forest_idx
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        print(f"\n  --- Forest {forest_idx+1}/{num_forests} (seed={seed}) ---")

        model = HyperspectralNet(
            num_bands    = num_bands,
            num_classes  = num_classes,
            num_filters  = num_filters,
            num_blocks   = num_blocks,
            base_filters = base_filters
        ).to(device)

        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5)

        best_val_miou = 0.0
        best_state    = None

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
                        val_oa, val_miou, val_dice, val_prec, val_rec = compute_node_metrics(
                            out.cpu(), l.cpu(), num_classes)

                print(f"  Epoch {epoch:3d} | Loss: {loss:.4f} | Val Loss: {v_loss:.4f} | "
                      f"OA: {val_oa:.4f} | mIoU: {val_miou:.4f} | "
                      f"Dice: {val_dice:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f}")

                if val_miou > best_val_miou:
                    best_val_miou = val_miou
                    best_state    = {k: v.cpu().clone()
                                     for k, v in model.state_dict().items()}

        if best_state is None:
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}

        print(f"  Forest {forest_idx+1} done | Best Val mIoU: {best_val_miou:.4f}")

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
            fp_map      = fp_map,
            d_model     = ssm_d_model,
            epochs      = 50,
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
               node_id='root'):
    """
    Recursively train all RSSP tree nodes.
    Returns: dict mapping node_id → node training result dict.
    """
    trained = {}

    trained[node_id] = train_node(
        tree, data, labels, total_classes,
        train_coords, val_coords,
        fp_map, ssm_d_model,
        base_epochs, num_forests, base_filters,
        num_filters, num_blocks, lr, device, node_id
    )

    if tree['left'] and len(tree['left']['classes']) > 1:
        trained.update(train_tree(
            tree['left'], data, labels, total_classes,
            train_coords, val_coords,
            fp_map, ssm_d_model,
            base_epochs, num_forests, base_filters,
            num_filters, num_blocks, lr, device,
            node_id=node_id + '_L'
        ))

    if tree['right'] and len(tree['right']['classes']) > 1:
        trained.update(train_tree(
            tree['right'], data, labels, total_classes,
            train_coords, val_coords,
            fp_map, ssm_d_model,
            base_epochs, num_forests, base_filters,
            num_filters, num_blocks, lr, device,
            node_id=node_id + '_R'
        ))

    return trained