"""Segmentation metrics in original label space, defined exactly as v0.1's so the numbers compare."""
from __future__ import annotations

import numpy as np

METRIC_KEYS = ('OA', 'mIoU', 'Dice', 'Precision', 'Recall', 'AA', 'kappa')


def _mean(values: list) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def compute_metrics(pred, target, class_ids=None) -> dict:
    """Scores pixels whose target is > 0. IoU, Dice, precision and recall average over classes present in
    prediction or target; AA over classes present in the target (v0.1's inclusion rules)."""
    pred, target = np.asarray(pred).ravel(), np.asarray(target).ravel()
    mask = target > 0
    pred_m, target_m = pred[mask], target[mask]
    n = int(target_m.size)
    if n == 0:
        raise ValueError("No labelled pixels to evaluate")

    classes = set(np.unique(target_m).tolist()) | set(np.unique(pred_m[pred_m > 0]).tolist())
    classes = sorted(classes | set(class_ids or []))

    oa = float((pred_m == target_m).sum() / n)
    ious, dices, precisions, recalls, accs, per_class = [], [], [], [], [], {}
    for c in classes:
        pc, tc = pred_m == c, target_m == c
        tp, fp, fn = int((pc & tc).sum()), int((pc & ~tc).sum()), int((~pc & tc).sum())
        union = tp + fp + fn
        iou = tp / union if union > 0 else 0.0
        precision, recall = tp / (tp + fp + 1e-8), tp / (tp + fn + 1e-8)
        if union > 0:
            ious.append(iou)
            dices.append((2 * tp) / (2 * tp + fp + fn + 1e-8))
            precisions.append(precision)
            recalls.append(recall)
        total = int(tc.sum())
        if total > 0:
            accs.append(tp / total)
            per_class[int(c)] = {'total': total, 'correct': tp, 'iou': float(iou),
                                 'precision': float(precision), 'recall': float(recall)}

    pe = 0.0
    for c in classes:
        pe += float((pred_m == c).sum()) / n * float((target_m == c).sum()) / n
    kappa = (oa - pe) / (1 - pe + 1e-10) if (1 - pe) > 1e-10 else 0.0

    return {'OA': oa, 'mIoU': _mean(ious), 'Dice': _mean(dices), 'Precision': _mean(precisions),
            'Recall': _mean(recalls), 'AA': _mean(accs), 'kappa': float(kappa), 'n': n,
            'per_class_iou': {c: v['iou'] for c, v in per_class.items()}, 'per_class': per_class}


def majority_share(target) -> float:
    """Share of the most common class among labelled pixels: the score of always guessing it."""
    target = np.asarray(target).ravel()
    target = target[target > 0]
    return float(np.bincount(target).max() / target.size) if target.size else 0.0
