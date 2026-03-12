import torch
import torch.nn.functional as F
import numpy as np
from models.hyperspectral_net import HyperspectralNet
from rssp.sssr_router import SSSRRouter


# ─────────────────────────────────────────────────────────────────────────────
# Forest ensemble → global probability map
# ─────────────────────────────────────────────────────────────────────────────

def get_global_soft_probs(node_info: dict, data: torch.Tensor,
                          device, num_global_classes: int) -> torch.Tensor:
    """
    Run the forest ensemble for one node.
    Averages softmax probabilities across all forest members.
    Maps local class IDs → global class IDs.

    Returns: (B, num_global_classes, H, W) on CPU
    """
    forests         = node_info['forests']
    local_to_global = forests[0]['local_to_global']
    B, _, H, W      = data.shape

    prob_sum = torch.zeros(B, forests[0]['num_classes'], H, W)

    for info in forests:
        model = HyperspectralNet(
            num_bands    = info['num_bands'],
            num_classes  = info['num_classes'],
            num_filters  = info['num_filters'],
            num_blocks   = info['num_blocks'],
            base_filters = info['base_filters']
        ).to(device)
        model.load_state_dict({k: v.to(device) for k, v in info['state_dict'].items()})
        model.eval()

        with torch.no_grad():
            logits   = model(data.to(device)).cpu()
            prob_sum += F.softmax(logits, dim=1)

        del model
        if device != 'cpu':
            torch.cuda.empty_cache()

    prob_avg = prob_sum / len(forests)                      # (B, local_classes, H, W)

    # Remap to global class space
    global_probs = torch.zeros(B, num_global_classes, H, W)
    for local_id, global_id in local_to_global.items():
        if global_id < num_global_classes:
            global_probs[:, global_id] = prob_avg[:, local_id]

    return global_probs                                      # CPU


# ─────────────────────────────────────────────────────────────────────────────
# Temperature-sharpened routing
# ─────────────────────────────────────────────────────────────────────────────

def sharpen(p: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Sharpen routing probability p toward 0 or 1.

    temperature = 1.0  → no change (full soft routing)
    temperature = 10.0 → near-hard routing (recommended when SSM is weak)
    temperature → ∞    → hard routing

    Formula: p^t / (p^t + (1-p)^t)
    """
    if temperature <= 1.0:
        return p
    p_c     = p.clamp(1e-6, 1 - 1e-6)
    p_t     = p_c ** temperature
    one_p_t = (1.0 - p_c) ** temperature
    return p_t / (p_t + one_p_t)


# ─────────────────────────────────────────────────────────────────────────────
# Soft cascade inference
# ─────────────────────────────────────────────────────────────────────────────

def cascade_soft_inference(tree: dict,
                           trained_models: dict,
                           data: torch.Tensor,
                           ssm_encoder,
                           device,
                           num_global_classes: int,
                           node_id: str = 'root',
                           path_weight: torch.Tensor = None,
                           fingerprints: torch.Tensor = None,
                           temperature: float = 10.0) -> torch.Tensor:
    """
    Temperature-controlled soft-routed RSSP cascade inference.

    temperature dial:
      1.0  = fully soft (only use with strong SSM — spreads prob everywhere)
      10.0 = near-hard (default — uses SSSR signal only when confident)
      50+  = effectively hard routing

    path_weight:  (B, 1, H, W) cumulative routing weight along this branch.
    fingerprints: (B, d_model, H, W) pre-computed once at root, passed down.

    Returns: (B, num_global_classes, H, W) weighted probability tensor on CPU.
    """
    B, _, H, W = data.shape

    # ── Compute fingerprints once at root ────────────────────────────────────
    if fingerprints is None:
        ssm_encoder.eval()
        with torch.no_grad():
            fingerprints = ssm_encoder(data.to(device)).cpu()  # (B, d_model, H, W)

    if path_weight is None:
        path_weight = torch.ones(B, 1, H, W)

    # ── Single-class leaf: no model trained, trivial prediction ──────────────
    if node_id not in trained_models:
        classes      = tree['classes']
        global_probs = torch.zeros(B, num_global_classes, H, W)
        if len(classes) == 1 and classes[0] < num_global_classes:
            global_probs[:, classes[0]] = 1.0
        return global_probs * path_weight

    node_info  = trained_models[node_id]
    node_probs = get_global_soft_probs(
        node_info, data, device, num_global_classes)           # (B, G, H, W) CPU

    # ── Structural leaf: no children in tree ─────────────────────────────────
    if tree['left'] is None and tree['right'] is None:
        return node_probs * path_weight

    # ── Load router, compute and sharpen routing probabilities ───────────────
    router = SSSRRouter(d_model=node_info['d_model'])
    if node_info['router_state'] is not None:
        router.load_state_dict(node_info['router_state'])
    router.eval()

    with torch.no_grad():
        p_left_raw = router(fingerprints)                       # (B, H, W) CPU

    p_left  = sharpen(p_left_raw, temperature).unsqueeze(1)    # (B, 1, H, W)
    p_right = 1.0 - p_left

    result = torch.zeros(B, num_global_classes, H, W)

    # ── Left subtree ─────────────────────────────────────────────────────────
    if tree['left'] is not None:
        left_id = node_id + '_L'
        result += cascade_soft_inference(
            tree['left'], trained_models, data, ssm_encoder, device,
            num_global_classes, left_id,
            path_weight * p_left, fingerprints, temperature
        )

    # ── Right subtree ────────────────────────────────────────────────────────
    if tree['right'] is not None:
        right_id = node_id + '_R'
        result += cascade_soft_inference(
            tree['right'], trained_models, data, ssm_encoder, device,
            num_global_classes, right_id,
            path_weight * p_right, fingerprints, temperature
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Final prediction and metrics
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(tree, trained_models, data, ssm_encoder, device,
                  num_global_classes, temperature: float = 10.0) -> np.ndarray:
    """
    Runs cascade_soft_inference and returns argmax prediction.

    temperature: 10.0 = near-hard routing (recommended with weak SSM)
                 1.0  = fully soft routing (use only with strong SSM)

    Returns: (H, W) numpy array of predicted global class IDs.
    """
    with torch.no_grad():
        weighted_probs = cascade_soft_inference(
            tree, trained_models,
            data.unsqueeze(0) if data.dim() == 3 else data,
            ssm_encoder, device, num_global_classes,
            temperature=temperature
        )                                                       # (1, G, H, W)

    pred = weighted_probs.squeeze(0).argmax(dim=0).numpy()     # (H, W)
    return pred


def compute_rssp_metrics(pred: np.ndarray,
                         labels_np: np.ndarray,
                         num_classes: int):
    """OA, mIoU, Dice, Precision, Recall over labeled pixels only."""
    mask     = labels_np > 0
    pred_m   = pred[mask]
    target_m = labels_np[mask]

    oa = (pred_m == target_m).sum() / target_m.size

    ious, dices, precisions, recalls = [], [], [], []

    for c in range(1, num_classes):
        pred_c   = pred_m == c
        target_c = target_m == c
        tp    = (pred_c & target_c).sum()
        fp    = (pred_c & ~target_c).sum()
        fn    = (~pred_c & target_c).sum()
        union = tp + fp + fn
        if union > 0:
            ious.append(tp / union)
            dices.append((2 * tp) / (2 * tp + fp + fn + 1e-8))
            precisions.append(tp / (tp + fp + 1e-8))
            recalls.append(tp / (tp + fn + 1e-8))

    miou      = sum(ious)       / len(ious)       if ious       else 0.0
    dice      = sum(dices)      / len(dices)      if dices      else 0.0
    precision = sum(precisions) / len(precisions) if precisions else 0.0
    recall    = sum(recalls)    / len(recalls)    if recalls    else 0.0

    return float(oa), float(miou), float(dice), float(precision), float(recall)