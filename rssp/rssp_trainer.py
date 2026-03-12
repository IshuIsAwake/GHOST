import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models.hyperspectral_net import HyperspectralNet


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
            global_label = labels[r, c].item() if hasattr(labels[r,c], 'item') else int(labels[r, c])
            if global_label in self.global_to_local:
                self.split_mask[r, c] = self.global_to_local[global_label]

        # Count how many pixels this node actually has
        self.num_pixels = (self.split_mask > 0).sum().item()

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.data, self.split_mask


def compute_node_metrics(pred, target, num_classes):
    """Compute OA and mIoU for a single node prediction."""
    pred     = pred.argmax(dim=1)
    mask     = target != 0
    pred_m   = pred[mask]
    target_m = target[mask]

    if target_m.numel() == 0:
        return 0.0, 0.0

    oa   = (pred_m == target_m).sum().item() / target_m.numel()
    ious = []

    for c in range(1, num_classes):
        pred_c   = pred_m == c
        target_c = target_m == c
        tp    = (pred_c & target_c).sum().item()
        fp    = (pred_c & ~target_c).sum().item()
        fn    = (~pred_c & target_c).sum().item()
        union = tp + fp + fn
        if union > 0:
            ious.append(tp / union)

    miou = sum(ious) / len(ious) if ious else 0.0
    return oa, miou


def get_node_epochs(base_epochs, node_classes, total_classes):
    """
    Root always gets full base_epochs.
    Other nodes scale by class fraction, minimum base_epochs // 2.
    """
    if len(node_classes) == total_classes:
        return base_epochs
    return max(base_epochs // 2, int(base_epochs * len(node_classes) / total_classes))


def train_node(node, data, labels, total_classes, train_coords, val_coords,
               base_epochs=300, num_forests=5, base_filters=32,
               num_filters=8, num_blocks=3, lr=1e-4, device='cuda',
               node_id='root'):

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

    trained_models = []

    for forest_idx in range(num_forests):
        seed = 42 + forest_idx
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        print(f"\n  --- Forest {forest_idx+1}/{num_forests} (seed={seed}) ---")

        model = HyperspectralNet(
            num_bands=num_bands,
            num_classes=num_classes,
            num_filters=num_filters,
            num_blocks=num_blocks,
            base_filters=base_filters
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
                        val_oa, val_miou = compute_node_metrics(
                            out.cpu(), l.cpu(), num_classes)

                print(f"  Epoch {epoch:3d} | Loss: {loss:.4f} | "
                      f"Val Loss: {v_loss:.4f} | "
                      f"Val OA: {val_oa:.4f} | "
                      f"Val mIoU: {val_miou:.4f}")

                if val_miou > best_val_miou:
                    best_val_miou = val_miou
                    best_state    = {k: v.cpu().clone()
                                     for k, v in model.state_dict().items()}

        if best_state is None:
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}

        print(f"  Forest {forest_idx+1} done | Best Val mIoU: {best_val_miou:.4f}")

        trained_models.append({
        'state_dict':      best_state,
        'node_classes':    node_classes,
        'global_to_local': train_ds.global_to_local,
        'local_to_global': train_ds.local_to_global,
        'num_classes':     num_classes,
        'num_bands':       num_bands,
        'num_filters':     num_filters,   # ← add these
        'num_blocks':      num_blocks,    # ← add these
        'base_filters':    base_filters   # ← add these
    })

    return trained_models


def train_tree(tree, data, labels, total_classes, train_coords, val_coords,
               base_epochs=300, num_forests=5, base_filters=32,
               num_filters=8, num_blocks=3, lr=1e-4, device='cuda',
               node_id='root'):

    trained = {}

    trained[node_id] = train_node(
        tree, data, labels, total_classes,
        train_coords, val_coords,
        base_epochs, num_forests, base_filters,
        num_filters, num_blocks, lr, device, node_id
    )

    if tree['left'] and len(tree['left']['classes']) > 1:
        trained.update(train_tree(
            tree['left'], data, labels, total_classes,
            train_coords, val_coords,
            base_epochs, num_forests, base_filters,
            num_filters, num_blocks, lr, device,
            node_id=node_id + '_L'
        ))

    if tree['right'] and len(tree['right']['classes']) > 1:
        trained.update(train_tree(
            tree['right'], data, labels, total_classes,
            train_coords, val_coords,
            base_epochs, num_forests, base_filters,
            num_filters, num_blocks, lr, device,
            node_id=node_id + '_R'
        ))

    return trained