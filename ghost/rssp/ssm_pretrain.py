import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ghost.models.spectral_ssm import SpectralSSMEncoder


def pretrain_ssm(data, labels, train_coords, val_coords,
                 d_model: int = 64, d_state: int = 16,
                 num_classes: int = None,
                 epochs: int = 300, lr: float = 1e-3,
                 batch_size: int = 512,
                 device: str = 'cuda',
                 save_path: str = 'ssm_pretrained.pt') -> SpectralSSMEncoder:
    """
    Pre-train the SSM encoder with a pixel-level classification head.

    Completely data-agnostic: operates on raw pixel spectra, no class structure
    baked into the SSM itself. The classification head is discarded after
    pre-training — only the encoder weights are saved and reused.

    Uses mini-batch training for more stable gradients and ReduceLROnPlateau
    scheduler for better convergence.

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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=20, factor=0.5, min_lr=1e-5
    )
    criterion = nn.CrossEntropyLoss()

    def extract(coords: np.ndarray):
        """Extract pixel spectra and labels at given coordinates."""
        r, c    = coords[:, 0], coords[:, 1]
        spectra = data[:, r, c].T.float()                   # (N, C)
        targets = labels[r, c].long()                        # (N,)
        return spectra, targets

    def encode_batch(spectra_batch: torch.Tensor) -> torch.Tensor:
        """(N, C) → (N, d_model) via fake spatial dims."""
        x_4d = spectra_batch.unsqueeze(-1).unsqueeze(-1).to(device)  # (N, C, 1, 1)
        return encoder(x_4d).squeeze(-1).squeeze(-1)                 # (N, d_model)

    train_spectra, train_y = extract(train_coords)
    val_spectra,   val_y   = extract(val_coords)
    val_y = val_y.to(device)

    n_train = len(train_spectra)
    best_val_acc = 0.0
    best_state   = None

    print(f"\n=== Pre-training SSM Encoder ===")
    print(f"    d_model={d_model}  d_state={d_state}  epochs={epochs}  batch_size={batch_size}")
    print(f"    Train pixels: {len(train_coords)}  |  Val pixels: {len(val_coords)}")

    for epoch in range(1, epochs + 1):
        encoder.train()
        head.train()

        # Mini-batch training — shuffle each epoch
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            batch_spectra = train_spectra[idx]
            batch_targets = train_y[idx].to(device)

            features = encode_batch(batch_spectra)
            logits   = head(features)
            loss     = criterion(logits, batch_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / n_batches

        if epoch % 10 == 0:
            encoder.eval()
            head.eval()
            with torch.no_grad():
                # Evaluate in batches to avoid OOM
                val_correct = 0
                val_total   = 0
                for v_start in range(0, len(val_spectra), batch_size):
                    v_batch = val_spectra[v_start:v_start + batch_size]
                    v_tgt   = val_y[v_start:v_start + batch_size]
                    v_feat  = encode_batch(v_batch)
                    v_logits = head(v_feat)
                    val_correct += (v_logits.argmax(1) == v_tgt).sum().item()
                    val_total   += len(v_tgt)

                val_acc = val_correct / val_total
                scheduler.step(val_acc)

            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:4d} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.1e}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = {k: v.cpu().clone()
                                for k, v in encoder.state_dict().items()}

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}

    encoder.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    torch.save(best_state, save_path)
    print(f"\nSSM pre-trained | Best Val Acc: {best_val_acc:.4f} | Saved -> {save_path}")

    return encoder