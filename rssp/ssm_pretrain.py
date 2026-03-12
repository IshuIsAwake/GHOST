import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.spectral_ssm import SpectralSSMEncoder


def pretrain_ssm(data, labels, train_coords, val_coords,
                 d_model: int = 64, d_state: int = 16,
                 num_classes: int = None,
                 epochs: int = 100, lr: float = 1e-3,
                 device: str = 'cuda',
                 save_path: str = 'ssm_pretrained.pt') -> SpectralSSMEncoder:
    """
    Pre-train the SSM encoder with a pixel-level classification head.

    Completely data-agnostic: operates on raw pixel spectra, no class structure
    baked into the SSM itself. The classification head is discarded after
    pre-training — only the encoder weights are saved and reused.

    data:         (C, H, W) tensor
    labels:       (H, W)    tensor  (0 = background, ignored)
    train_coords: (N, 2)    numpy array of (row, col)
    val_coords:   (M, 2)    numpy array of (row, col)

    Returns: SpectralSSMEncoder with best weights loaded (frozen externally).
    """
    if num_classes is None:
        num_classes = int(labels.max().item()) + 1

    encoder = SpectralSSMEncoder(d_model=d_model, d_state=d_state).to(device)
    head    = nn.Linear(d_model, num_classes).to(device)

    optimizer = optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()),
        lr=lr, weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()

    def extract(coords: np.ndarray):
        """Extract pixel spectra and labels at given coordinates."""
        r, c    = coords[:, 0], coords[:, 1]
        spectra = data[:, r, c].T.float()                   # (N, C)
        targets = labels[r, c].long()                        # (N,)
        # Fake spatial dims so SpectralSSMEncoder accepts it: (N, C, 1, 1)
        return spectra.unsqueeze(-1).unsqueeze(-1).to(device), targets.to(device)

    def encode(x_4d: torch.Tensor) -> torch.Tensor:
        """(N, C, 1, 1) → (N, d_model)"""
        return encoder(x_4d).squeeze(-1).squeeze(-1)

    train_x, train_y = extract(train_coords)
    val_x,   val_y   = extract(val_coords)

    best_val_acc = 0.0
    best_state   = None

    print(f"\n=== Pre-training SSM Encoder ===")
    print(f"    d_model={d_model}  d_state={d_state}  epochs={epochs}")
    print(f"    Train pixels: {len(train_coords)}  |  Val pixels: {len(val_coords)}")

    for epoch in range(1, epochs + 1):
        encoder.train()
        head.train()

        logits = head(encode(train_x))
        loss   = criterion(logits, train_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            encoder.eval()
            head.eval()
            with torch.no_grad():
                val_logits = head(encode(val_x))
                val_acc    = (val_logits.argmax(1) == val_y).float().mean().item()

            print(f"  Epoch {epoch:4d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = {k: v.cpu().clone()
                                for k, v in encoder.state_dict().items()}

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}

    encoder.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    torch.save(best_state, save_path)
    print(f"\nSSM pre-trained | Best Val Acc: {best_val_acc:.4f} | Saved → {save_path}")

    return encoder