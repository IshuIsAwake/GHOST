"""Splits: the ratio split reproduces v0.1's pixels exactly; every split partitions the labelled pixels."""
import numpy as np
import pytest

from conftest import make_scene, write_mat
from ghost.v0_2.splits import SPLIT_MODES, disjoint_split, fixed_split, make_split, ratio_split


@pytest.mark.parametrize("seed", [0, 42])
def test_ratio_split_reproduces_v01_pixels(tmp_path, seed):
    from ghost.datasets.hyperspectral_dataset import HyperspectralDataset
    cube, gt = make_scene(H=16, W=12, B=10, n_classes=4, seed=3)
    data_path, gt_path = write_mat(tmp_path / "s.mat", cube, gt)
    v01 = HyperspectralDataset(data_path, gt_path, split="train", train_ratio=0.2, val_ratio=0.1, seed=seed)
    ours = ratio_split(gt, train_ratio=0.2, val_ratio=0.1, seed=seed)
    W = gt.shape[1]
    for coords, flat in zip((v01.train_coords, v01.val_coords, v01.test_coords), ours):
        np.testing.assert_array_equal(coords[:, 0] * W + coords[:, 1], flat)


def test_indian_pines_split_sizes():
    from ghost.data import indian_pines_path
    from ghost.datasets.loader import load_labels
    gt = np.squeeze(load_labels(indian_pines_path()[1])[0]).astype(np.int64)
    assert [len(s) for s in ratio_split(gt, 0.2, 0.1, seed=42)] == [2045, 1018, 7186]
    assert [len(s) for s in fixed_split(gt, 50, 15, 0.1, seed=42)] == [695, 950, 8604]


def test_fixed_split_minority_rule():
    labels = np.zeros(200, dtype=np.int64)
    labels[:120], labels[120:160], labels[160:170] = 1, 2, 3
    with pytest.warns(UserWarning, match="class 3"):
        train, val, test = fixed_split(labels.reshape(10, 20), 50, 15, 0.1, seed=0)
    flat = labels
    assert [int((flat[train] == c).sum()) for c in (1, 2, 3)] == [50, 15, 10]
    assert [int((flat[val] == c).sum()) for c in (1, 2, 3)] == [7, 2, 0]
    assert [int((flat[test] == c).sum()) for c in (1, 2, 3)] == [63, 23, 0]


def test_disjoint_blocks_never_straddle_splits():
    _, gt = make_scene(H=40, W=40, B=8, n_classes=4, seed=0)
    block = 5
    splits = disjoint_split(gt, 0.3, 0.2, seed=1, block_size=block)
    W = gt.shape[1]
    block_sets = [set(((s // W) // block * 1000 + (s % W) // block).tolist()) for s in splits]
    assert not (block_sets[0] & block_sets[1] or block_sets[0] & block_sets[2] or block_sets[1] & block_sets[2])


@pytest.mark.parametrize("seed", range(5))
def test_disjoint_split_trains_and_tests_every_class_that_spans_two_blocks(seed):
    from ghost.data import indian_pines_path
    from ghost.datasets.loader import load_labels
    gt = np.squeeze(load_labels(indian_pines_path()[1])[0]).astype(np.int64)
    with pytest.warns(UserWarning, match="test split has no pixels of classes \\[7\\]"):  # one 14×14 block
        train, _, test = disjoint_split(gt, 0.2, 0.1, seed=seed)
    flat = gt.reshape(-1)
    trained, tested = set(flat[train].tolist()), set(flat[test].tolist())
    assert trained == set(range(1, 17))
    assert tested == set(range(1, 17)) - {7}
    share = np.bincount(flat[train], minlength=17)[1:] / np.bincount(flat[flat > 0], minlength=17)[1:]
    assert share.min() >= 0.05


@pytest.mark.parametrize("mode", SPLIT_MODES)
def test_split_partitions_the_labelled_pixels(mode):
    _, gt = make_scene(H=30, W=24, B=8, n_classes=3, seed=2)
    train, val, test = make_split(mode, gt, seed=7, samples_per_class=20, minority_samples=5)
    sets = [set(train.tolist()), set(val.tolist()), set(test.tolist())]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert sets[0] | sets[1] | sets[2] == set(np.flatnonzero(gt.reshape(-1) > 0).tolist())
    assert len(train) > 0 and len(test) > 0


@pytest.mark.parametrize("mode", SPLIT_MODES)
def test_split_is_deterministic_per_seed(mode):
    _, gt = make_scene(H=30, W=24, B=8, n_classes=3, seed=2)
    a = make_split(mode, gt, seed=1, samples_per_class=20, minority_samples=5)
    b = make_split(mode, gt, seed=1, samples_per_class=20, minority_samples=5)
    c = make_split(mode, gt, seed=2, samples_per_class=20, minority_samples=5)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    assert any(not np.array_equal(x, z) for x, z in zip(a, c))


def test_unknown_split_mode_raises():
    with pytest.raises(ValueError):
        make_split("kfold", np.ones((4, 4), dtype=np.int64), seed=0)
