"""Losses train every class including index 0; metrics match v0.1's definitions exactly."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ghost.v0_2.losses import LOSSES, build_criterion
from ghost.v0_2.metrics import compute_metrics, majority_share


@pytest.mark.parametrize("name", LOSSES)
def test_every_loss_is_finite(name):
    logits = torch.randn(32, 5, requires_grad=True)
    loss = build_criterion(name)(logits, torch.randint(0, 5, (32,)))
    assert loss.dim() == 0 and torch.isfinite(loss)
    loss.backward()


@pytest.mark.parametrize("name", LOSSES)
def test_class_zero_is_trained_not_ignored(name):
    logits = torch.zeros(16, 4, requires_grad=True)
    build_criterion(name)(logits, torch.zeros(16, dtype=torch.long)).backward()
    assert logits.grad[:, 0].abs().sum() > 0


def test_dice_loss_is_half_ce_half_dice():
    torch.manual_seed(0)
    logits, target = torch.randn(40, 3), torch.randint(0, 3, (40,))
    probs, onehot = F.softmax(logits, 1), F.one_hot(target, 3).float()
    dice = (2 * (probs * onehot).sum(0) + 1) / (probs.sum(0) + onehot.sum(0) + 1)
    expected = 0.5 * F.cross_entropy(logits, target) + 0.5 * (1 - dice.mean())
    torch.testing.assert_close(build_criterion("dice")(logits, target), expected)


def test_unknown_loss_raises():
    with pytest.raises(ValueError):
        build_criterion("hinge")


def test_perfect_prediction_scores_one():
    y = np.array([1, 2, 2, 3, 3, 3])
    m = compute_metrics(y, y)
    for key in ("OA", "mIoU", "AA", "kappa", "Dice", "Precision", "Recall"):
        assert m[key] == pytest.approx(1.0, abs=1e-6)


def test_hand_built_confusion():
    target = np.array([1, 1, 1, 2, 2, 3])
    pred = np.array([1, 1, 2, 2, 2, 3])
    m = compute_metrics(pred, target)
    assert m["OA"] == pytest.approx(5 / 6)
    assert m["mIoU"] == pytest.approx(7 / 9)
    assert m["AA"] == pytest.approx(8 / 9)
    assert m["kappa"] == pytest.approx(17 / 23)
    assert m["per_class_iou"] == pytest.approx({1: 2 / 3, 2: 2 / 3, 3: 1.0})
    assert m["per_class"][1]["total"] == 3 and m["per_class"][1]["correct"] == 2


@pytest.mark.parametrize("seed", range(20))
def test_metrics_match_v01(seed):
    from ghost.rssp.rssp_inference import compute_rssp_metrics
    rng = np.random.default_rng(seed)
    k = 6
    target = rng.integers(1, k - 1 if seed % 3 == 0 else k + 1, size=300)  # some runs leave classes out
    pred = np.where(rng.random(300) < 0.6, target, rng.integers(1, k + 1, size=300))
    oa, miou, dice, prec, rec, aa, kappa, per_class = compute_rssp_metrics(pred, target, k + 1)
    m = compute_metrics(pred, target)
    for ours, theirs in ((m["OA"], oa), (m["mIoU"], miou), (m["Dice"], dice), (m["Precision"], prec),
                         (m["Recall"], rec), (m["AA"], aa), (m["kappa"], kappa)):
        assert ours == pytest.approx(float(theirs), rel=1e-9, abs=1e-12)
    assert m["per_class_iou"] == pytest.approx(per_class)


def test_majority_share():
    assert majority_share(np.array([1, 1, 2, 3])) == pytest.approx(0.5)
