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
                 fp_map: torch.Tensor,
                 d_model: int = 64,
                 epochs: int = 50,
                 lr: float = 1e-3,
                 device: str = 'cuda') -> 'SSSRRouter | None':
    """
    Train routing head for a single internal RSSP node.

    Ground truth: pixel belongs to left_classes → target=1.0
                  pixel belongs to right_classes → target=0.0

    node:         RSSP tree node dict
    labels:       (H, W) tensor — global labels
    train_coords: (N, 2) numpy — all training pixel coords
    fp_map:       (H, W, d_model) tensor on CPU — pre-computed fingerprints
    """
    left_classes  = set(node['left']['classes'])
    right_classes = set(node['right']['classes'])
    node_classes  = left_classes | right_classes

    rows = train_coords[:, 0]
    cols = train_coords[:, 1]

    global_labels = labels[rows, cols].numpy()

    # Keep only pixels belonging to this node's classes
    mask        = np.isin(global_labels, list(node_classes))
    rows, cols  = rows[mask], cols[mask]
    node_labels = global_labels[mask]

    if len(rows) < 4:       # not enough pixels to train a router
        return None

    routing_targets = torch.tensor(
        [1.0 if l in left_classes else 0.0 for l in node_labels],
        dtype=torch.float32
    )

    fp          = fp_map[rows, cols]                # (N, d_model)
    fp_tensor   = fp.to(device)
    tgt_tensor  = routing_targets.to(device)

    router    = SSSRRouter(d_model=d_model).to(device)
    optimizer = optim.AdamW(router.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()

    for _ in range(epochs):
        router.train()
        pred = router(fp_tensor)
        loss = criterion(pred, tgt_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    router.eval()
    return router.cpu()