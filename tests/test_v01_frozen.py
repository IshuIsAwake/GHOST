"""v0.1.7 stays importable and runnable next to v0.2."""
import importlib

import pytest
import torch

V01_MODULES = [
    "ghost.train", "ghost.train_rssp", "ghost.predict", "ghost.visualize", "ghost.losses",
    "ghost.models.hyperspectral_net", "ghost.models.losses", "ghost.models.spectral_ssm",
    "ghost.rssp.rssp_trainer", "ghost.rssp.rssp_inference", "ghost.rssp.sam_clustering",
    "ghost.rssp.sssr_router", "ghost.rssp.ssm_pretrain", "ghost.datasets.hyperspectral_dataset",
    "ghost.preprocessing.continuum_removal", "ghost.convert", "ghost.cli",
]


@pytest.mark.parametrize("name", V01_MODULES)
def test_v01_module_imports(name):
    importlib.import_module(name)


def test_hyperspectral_net_forward_shape():
    from ghost.models.hyperspectral_net import HyperspectralNet
    net = HyperspectralNet(num_bands=16, num_classes=4)
    net.eval()
    with torch.no_grad():
        assert net(torch.rand(1, 16, 32, 32)).shape == (1, 4, 32, 32)
