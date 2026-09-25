"""Per-pixel encoder: constant length and width, depth pruned by band count, no spatial operators."""
import pytest
import torch
import torch.nn as nn

from ghost.v0_2.model import DilatedResNet1D, SpectralNet, build_model, default_config


def test_output_shapes():
    enc = DilatedResNet1D(200)
    assert enc(torch.randn(8, 1, 200)).shape == (8, 128)
    assert enc(torch.randn(8, 200)).shape == (8, 128)
    net = SpectralNet(default_config(200, 16))
    assert net(torch.randn(8, 200)).shape == (8, 16)


@pytest.mark.parametrize("bands,blocks", [(1, 0), (3, 0), (32, 1), (61, 2), (103, 3), (200, 4), (400, 5)])
def test_depth_follows_band_count(bands, blocks):
    enc = DilatedResNet1D(bands)
    assert len(enc.blocks) == blocks
    assert enc.dilations == [2 ** i for i in range(blocks)]
    if blocks:
        assert enc.receptive_field <= bands
    enc.eval()
    assert enc(torch.randn(2, bands)).shape == (2, 128)


def test_receptive_field_at_indian_pines():
    assert DilatedResNet1D(200).receptive_field == 187


def test_length_and_width_are_preserved_in_every_block():
    enc = DilatedResNet1D(200)
    shapes = []
    hooks = [b.register_forward_hook(lambda m, i, o: shapes.append((tuple(i[0].shape), tuple(o.shape))))
             for b in enc.blocks]
    enc(torch.randn(4, 1, 200))
    for h in hooks:
        h.remove()
    assert len(shapes) == 4
    assert all(i == o == (4, 64, 200) for i, o in shapes)


def test_flatten_projects_from_channels_times_bands():
    enc = DilatedResNet1D(200, pool="flatten")
    assert enc.proj.in_features == 64 * 200
    assert enc(torch.randn(3, 200)).shape == (3, 128)


@pytest.mark.parametrize("pool", ["avg", "flatten"])
@pytest.mark.parametrize("bands", [3, 61])
def test_gradients_reach_every_parameter(pool, bands):
    net = SpectralNet(default_config(bands, 5, pool=pool))
    net.train()
    net(torch.randn(16, bands)).pow(2).mean().backward()
    missing = [n for n, p in net.named_parameters() if p.grad is None]
    assert not missing
    assert net.encoder.stem[0].weight.grad.abs().sum() > 0


def test_eval_is_deterministic_and_pixels_are_independent():
    net = SpectralNet(default_config(64, 4))
    net.eval()
    x = torch.randn(10, 64)
    with torch.no_grad():
        a, b, alone = net(x), net(x), net(x[3:4])
    assert torch.equal(a, b)
    torch.testing.assert_close(alone, a[3:4])


def test_no_spatial_convolutions():
    net = SpectralNet(default_config(200, 16))
    assert not [m for m in net.modules() if isinstance(m, (nn.Conv2d, nn.Conv3d))]


def test_band_mismatch_raises():
    with pytest.raises(ValueError, match="200"):
        SpectralNet(default_config(200, 4))(torch.randn(2, 150))


def test_build_model_keeps_its_config():
    cfg = default_config(50, 3, channels=16, pool="flatten")
    assert build_model(cfg).config == cfg
