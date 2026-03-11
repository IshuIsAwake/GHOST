import torch
import traceback
import gc

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Free VRAM at start: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")

PASS = "[PASS]"
FAIL = "[FAIL]"


def section(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────
# 1. Continuum Removal
# ─────────────────────────────────────────────────
section("1. ContinuumRemoval")
try:
    from preprocessing.continuum_removal import ContinuumRemoval
    layer = ContinuumRemoval().to(DEVICE)
    x = torch.randn(1, 200, 16, 16).to(DEVICE)
    out = layer(x)
    assert out.shape == x.shape
    print(f"{PASS} Input: {tuple(x.shape)} -> Output: {tuple(out.shape)}")
    del layer, x, out
except Exception:
    print(f"{FAIL}"); traceback.print_exc()
finally:
    cleanup()


# ─────────────────────────────────────────────────
# 2. Spectral3DStack
# ─────────────────────────────────────────────────
section("2. Spectral3DStack")
try:
    from models.spectral_3d_block import Spectral3DStack
    stack = Spectral3DStack(num_filters=8, num_blocks=3).to(DEVICE)
    x = torch.randn(1, 200, 16, 16).to(DEVICE)
    out = stack(x)
    assert out.shape == (1, 8, 200, 16, 16)
    print(f"{PASS} Input: {tuple(x.shape)} -> Output: {tuple(out.shape)}")
    print(f"      out_channels: {stack.out_channels}")
    del stack, x, out
except Exception:
    print(f"{FAIL}"); traceback.print_exc()
finally:
    cleanup()


# ─────────────────────────────────────────────────
# 3. SpectralTransformer
# ─────────────────────────────────────────────────
section("3. SpectralTransformer (200 bands, B=1, 16x16)")
try:
    from models.real_spectral_transformer import SpectralTransformer
    transformer = SpectralTransformer(embed_dim=8, num_heads=4, num_layers=2).to(DEVICE)
    x = torch.randn(1, 8, 200, 16, 16).to(DEVICE)

    torch.cuda.reset_peak_memory_stats()
    out = transformer(x)
    peak_mb = torch.cuda.max_memory_allocated() / 1e6

    assert out.shape == (1, 8, 16, 16)
    print(f"{PASS} Input: {tuple(x.shape)} -> Output: {tuple(out.shape)}")
    print(f"      Peak VRAM this test: {peak_mb:.1f} MB")
    del transformer, x, out
except Exception:
    print(f"{FAIL}"); traceback.print_exc()
finally:
    cleanup()


# ─────────────────────────────────────────────────
# 4. SpectralTransformer — band agnosticism
# ─────────────────────────────────────────────────
section("4. SpectralTransformer — band agnosticism (103 bands, B=1)")
try:
    from models.real_spectral_transformer import SpectralTransformer
    transformer = SpectralTransformer(embed_dim=8, num_heads=4, num_layers=2).to(DEVICE)
    x = torch.randn(1, 8, 103, 16, 16).to(DEVICE)
    out = transformer(x)
    assert out.shape == (1, 8, 16, 16)
    print(f"{PASS} Input: {tuple(x.shape)} -> Output: {tuple(out.shape)}")
    print(f"      Positional embeddings interpolated 500 -> 103 bands.")
    del transformer, x, out
except Exception:
    print(f"{FAIL}"); traceback.print_exc()
finally:
    cleanup()


# ─────────────────────────────────────────────────
# 5. Encoder2D
# ─────────────────────────────────────────────────
section("5. Encoder2D")
try:
    from models.encoder_2d import Encoder2D
    enc = Encoder2D(in_channels=8, base_filters=32).to(DEVICE)
    x = torch.randn(1, 8, 16, 16).to(DEVICE)
    b, skips = enc(x)
    print(f"{PASS} Input: {tuple(x.shape)}")
    print(f"      Bottleneck: {tuple(b.shape)}")
    for i, s in enumerate(skips):
        print(f"      Skip {i+1}: {tuple(s.shape)}")
    del enc, x, b, skips
except Exception:
    print(f"{FAIL}"); traceback.print_exc()
finally:
    cleanup()


# ─────────────────────────────────────────────────
# 6. Decoder2D
# ─────────────────────────────────────────────────
section("6. Decoder2D")
try:
    from models.decoder_2d import Decoder2D
    f = 32
    dec = Decoder2D(num_classes=17, base_filters=f).to(DEVICE)
    b = torch.randn(1, f * 16, 1, 1).to(DEVICE)
    skips = [
        torch.randn(1, f,      16, 16).to(DEVICE),
        torch.randn(1, f * 2,   8,  8).to(DEVICE),
        torch.randn(1, f * 4,   4,  4).to(DEVICE),
        torch.randn(1, f * 8,   2,  2).to(DEVICE),
    ]
    out = dec(b, skips)
    assert out.shape == (1, 17, 16, 16)
    print(f"{PASS} Output: {tuple(out.shape)}")
    del dec, b, skips, out
except Exception:
    print(f"{FAIL}"); traceback.print_exc()
finally:
    cleanup()


# ─────────────────────────────────────────────────
# 7. Full pipeline — Indian Pines (200 bands, B=1)
# ─────────────────────────────────────────────────
section("7. Full Pipeline — 200 bands, B=1, 16x16")
try:
    from models.hyperspectral_net import HyperspectralNet
    model = HyperspectralNet(num_bands=200, num_classes=17, base_filters=32).to(DEVICE)
    x = torch.randn(1, 200, 16, 16).to(DEVICE)

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out = model(x)
    peak_mb = torch.cuda.max_memory_allocated() / 1e6

    assert out.shape == (1, 17, 16, 16)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{PASS} Input: {tuple(x.shape)} -> Output: {tuple(out.shape)}")
    print(f"      Parameters: {total_params:,}")
    print(f"      Peak VRAM this test: {peak_mb:.1f} MB")
    del model, x, out
except Exception:
    print(f"{FAIL}"); traceback.print_exc()
finally:
    cleanup()


# ─────────────────────────────────────────────────
# 8. Full pipeline — Pavia zero-shot (103 bands, B=1)
# ─────────────────────────────────────────────────
section("8. Full Pipeline — zero-shot 103 bands (Pavia)")
try:
    from models.hyperspectral_net import HyperspectralNet
    model = HyperspectralNet(num_bands=200, num_classes=17, base_filters=32).to(DEVICE)
    x = torch.randn(1, 103, 16, 16).to(DEVICE)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 17, 16, 16)
    print(f"{PASS} Input: {tuple(x.shape)} -> Output: {tuple(out.shape)}")
    print(f"      Model accepts 103 bands despite never seeing them.")
    del model, x, out
except Exception:
    print(f"{FAIL}"); traceback.print_exc()
finally:
    cleanup()


# ─────────────────────────────────────────────────
# 9. Backward pass — this is the real training memory cost
# ─────────────────────────────────────────────────
section("9. Backward Pass — B=1, 16x16")
try:
    from models.hyperspectral_net import HyperspectralNet
    model = HyperspectralNet(num_bands=200, num_classes=17, base_filters=32).to(DEVICE)
    x      = torch.randn(1, 200, 16, 16).to(DEVICE)
    labels = torch.randint(0, 17, (1, 16, 16)).to(DEVICE)

    torch.cuda.reset_peak_memory_stats()
    out  = model(x)
    loss = torch.nn.CrossEntropyLoss(ignore_index=0)(out, labels)
    loss.backward()
    peak_mb = torch.cuda.max_memory_allocated() / 1e6

    dead = [n for n, p in model.named_parameters() if p.grad is None]
    if dead:
        print(f"{FAIL} Params with no gradient: {dead}")
    else:
        print(f"{PASS} Loss: {loss.item():.4f}. All parameters received gradients.")
        print(f"      Peak VRAM (forward + backward): {peak_mb:.1f} MB")
        print(f"      This is what training will cost per step.")
    del model, x, labels, out, loss
except Exception:
    print(f"{FAIL}"); traceback.print_exc()
finally:
    cleanup()


print(f"\n{'='*50}")
print("  Done. Fix any FAILs before running train.py.")
print(f"{'='*50}\n")


section("10. VRAM check — B=4, 32x32 (proposed training config)")
model = HyperspectralNet(num_bands=200, num_classes=17, base_filters=32).to(DEVICE)
x = torch.randn(4, 200, 32, 32).to(DEVICE)
labels = torch.randint(0, 17, (4, 32, 32)).to(DEVICE)
torch.cuda.reset_peak_memory_stats()
out = model(x)
loss = criterion(out, labels)
loss.backward()
peak_mb = torch.cuda.max_memory_allocated() / 1e6
print(f"Peak VRAM: {peak_mb:.1f} MB")