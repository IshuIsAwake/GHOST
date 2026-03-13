import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class SSSRRouter(nn.Module):
    """
    Selective Spectral State Router.

    Per-node routing head: SSM fingerprint → p_left ∈ (0, 1).

    The SSM encoder is shared and frozen. Only this tiny head is
    node-specific — so the full routing system costs ~(d_model × 32 + 32)
    parameters per node. Negligible.

    Input:  (B, d_model, H, W) — spatial fingerprint map
            or (N, d_model)    — pixel batch
    Output: (B, H, W)          — per-pixel routing probability toward left child
            or (N,)
    """

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )

    def forward(self, fingerprints: torch.Tensor) -> torch.Tensor:
        if fingerprints.dim() == 4:
            B, D, H, W = fingerprints.shape
            fp = fingerprints.permute(0, 2, 3, 1).reshape(-1, D)   # (B·H·W, D)
            return torch.sigmoid(self.head(fp)).reshape(B, H, W)
        return torch.sigmoid(self.head(fingerprints)).squeeze(-1)   # (N,)


def train_router(node: dict,
                 labels: torch.Tensor,
                 train_coords: np.ndarray,
                 val_coords: np.ndarray,
                 fp_map: torch.Tensor,
                 d_model: int = 64,
                 epochs: int = 200,
                 lr: float = 1e-3,
                 device: str = 'cuda') -> 'SSSRRouter | None':
    """
    Train routing head for a single internal RSSP node.

    Ground truth: pixel belongs to left_classes → target=1.0
                  pixel belongs to right_classes → target=0.0

    Training uses class-balanced BCE loss and best-checkpoint selection
    based on validation accuracy.

    node:         RSSP tree node dict
    labels:       (H, W) tensor — global labels
    train_coords: (N, 2) numpy — all training pixel coords
    val_coords:   (M, 2) numpy — all validation pixel coords
    fp_map:       (H, W, d_model) tensor on CPU — pre-computed fingerprints
    """
    left_classes  = set(node['left']['classes'])
    right_classes = set(node['right']['classes'])
    node_classes  = left_classes | right_classes

    # ── Prepare train split ───────────────────────────────────────────────────
    def prepare_split(coords):
        rows = coords[:, 0]
        cols = coords[:, 1]
        global_labels = labels[rows, cols].numpy()
        mask = np.isin(global_labels, list(node_classes))
        rows, cols = rows[mask], cols[mask]
        node_labels = global_labels[mask]

        if len(rows) < 2:
            return None, None, None

        targets = torch.tensor(
            [1.0 if l in left_classes else 0.0 for l in node_labels],
            dtype=torch.float32
        )
        fp = fp_map[rows, cols]  # (N, d_model)
        return fp, targets, node_labels

    train_fp, train_tgt, train_labels = prepare_split(train_coords)
    val_fp, val_tgt, val_labels = prepare_split(val_coords)

    if train_fp is None or len(train_fp) < 4:
        return None

    # ── Class-balanced BCE weights ────────────────────────────────────────────
    n_left  = (train_tgt == 1.0).sum().item()
    n_right = (train_tgt == 0.0).sum().item()
    total   = n_left + n_right

    if n_left > 0 and n_right > 0:
        # Weight each sample inversely proportional to its class frequency
        w_left  = total / (2.0 * n_left)
        w_right = total / (2.0 * n_right)
        sample_weights = torch.where(
            train_tgt == 1.0,
            torch.tensor(w_left),
            torch.tensor(w_right)
        ).to(device)
    else:
        sample_weights = torch.ones_like(train_tgt).to(device)

    train_fp_tensor  = train_fp.to(device)
    train_tgt_tensor = train_tgt.to(device)

    has_val = val_fp is not None and len(val_fp) >= 2
    if has_val:
        val_fp_tensor  = val_fp.to(device)
        val_tgt_tensor = val_tgt.to(device)

    # ── Train ─────────────────────────────────────────────────────────────────
    router    = SSSRRouter(d_model=d_model).to(device)
    optimizer = optim.AdamW(router.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss(reduction='none')

    best_val_acc  = 0.0
    best_state    = None

    for epoch in range(1, epochs + 1):
        router.train()
        pred = router(train_fp_tensor)
        per_sample_loss = criterion(pred, train_tgt_tensor)
        loss = (per_sample_loss * sample_weights).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == epochs or (has_val and epoch % 50 == 0):
            router.eval()
            with torch.no_grad():
                train_acc = ((router(train_fp_tensor) > 0.5).float() == train_tgt_tensor).float().mean().item()

                if has_val:
                    val_pred = router(val_fp_tensor)
                    val_acc  = ((val_pred > 0.5).float() == val_tgt_tensor).float().mean().item()

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_state   = {k: v.cpu().clone()
                                        for k, v in router.state_dict().items()}

    # ── Final report ──────────────────────────────────────────────────────────
    if has_val:
        print(f"    Router: train_acc={train_acc:.4f} | val_acc={best_val_acc:.4f} "
              f"| left={n_left} right={n_right}")
    else:
        print(f"    Router: train_acc={train_acc:.4f} | no val | left={n_left} right={n_right}")
        best_state = {k: v.cpu().clone() for k, v in router.state_dict().items()}

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in router.state_dict().items()}

    router.cpu()
    router.load_state_dict(best_state)
    router.eval()
    return router