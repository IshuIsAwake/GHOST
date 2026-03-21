import torch
import torch.nn.functional as F
import numpy as np
from ghost.models.hyperspectral_net import HyperspectralNet
from ghost.rssp.sssr_router import SSSRRouter


# ─────────────────────────────────────────────────────────────────────────────
# Forest ensemble → global probability map
# ─────────────────────────────────────────────────────────────────────────────

def get_global_soft_probs(node_info: dict, data: torch.Tensor,
                          device, num_global_classes: int) -> torch.Tensor:
    forests         = node_info['forests']
    local_to_global = forests[0]['local_to_global']
    B, _, H, W      = data.shape

    prob_sum = torch.zeros(B, forests[0]['num_classes'], H, W, device=device)

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
            logits    = model(data.to(device))
            prob_sum += F.softmax(logits, dim=1)

        del model
        if device != 'cpu':
            torch.cuda.empty_cache()

    prob_avg = (prob_sum / len(forests)).cpu()

    global_probs = torch.zeros(B, num_global_classes, H, W)
    for local_id, global_id in local_to_global.items():
        if global_id < num_global_classes:
            global_probs[:, global_id] = prob_avg[:, local_id]

    return global_probs


# ─────────────────────────────────────────────────────────────────────────────
# Forest-derived routing probability
# ─────────────────────────────────────────────────────────────────────────────

def forest_routing_prob(node_probs: torch.Tensor,
                        left_classes: list,
                        num_global_classes: int) -> torch.Tensor:
    B, G, H, W = node_probs.shape
    p_left = torch.zeros(B, 1, H, W)
    for c in left_classes:
        if c < G:
            p_left += node_probs[:, c:c+1, :, :]
    return p_left.clamp(1e-6, 1 - 1e-6)


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
                           routing: str = 'hybrid') -> torch.Tensor:
    B, _, H, W = data.shape

    if fingerprints is None and routing in ('hybrid', 'soft'):
        ssm_encoder.eval()
        with torch.no_grad():
            fp_gpu = ssm_encoder(data.to(device))
            fingerprints = fp_gpu.cpu()
            del fp_gpu

    if path_weight is None:
        path_weight = torch.ones(B, 1, H, W)

    if node_id not in trained_models:
        classes      = tree['classes']
        global_probs = torch.zeros(B, num_global_classes, H, W)
        if len(classes) == 1 and classes[0] < num_global_classes:
            global_probs[:, classes[0]] = 1.0
        return global_probs * path_weight

    node_info  = trained_models[node_id]
    node_probs = get_global_soft_probs(
        node_info, data, device, num_global_classes)

    if tree['left'] is None and tree['right'] is None:
        return node_probs * path_weight

    left_classes = tree['left']['classes']

    if routing == 'forest':
        p_left = forest_routing_prob(node_probs, left_classes, num_global_classes)

    elif routing == 'soft':
        router = SSSRRouter(d_model=node_info['d_model'])
        if node_info['router_state'] is not None:
            router.load_state_dict(node_info['router_state'])
        router.eval()
        with torch.no_grad():
            p_left = router(fingerprints).unsqueeze(1)

    elif routing == 'hybrid':
        f_p_left = forest_routing_prob(
            node_probs, left_classes, num_global_classes)

        router = SSSRRouter(d_model=node_info['d_model'])
        if node_info['router_state'] is not None:
            router.load_state_dict(node_info['router_state'])
        router.eval()

        with torch.no_grad():
            ssm_p_left = router(fingerprints).unsqueeze(1)

        ssm_confidence = (2.0 * (ssm_p_left - 0.5).abs()).clamp(0, 1)
        p_left = f_p_left + ssm_confidence * (ssm_p_left - f_p_left)
        p_left = p_left.clamp(1e-6, 1 - 1e-6)

    else:
        raise ValueError(f"Unknown routing mode: '{routing}'. Use 'hybrid', 'forest', or 'soft'.")

    p_right = 1.0 - p_left
    result  = torch.zeros(B, num_global_classes, H, W)

    if tree['left'] is not None:
        left_id = node_id + '_L'
        result += cascade_soft_inference(
            tree['left'], trained_models, data, ssm_encoder, device,
            num_global_classes, left_id,
            path_weight * p_left, fingerprints, routing
        )

    if tree['right'] is not None:
        right_id = node_id + '_R'
        result += cascade_soft_inference(
            tree['right'], trained_models, data, ssm_encoder, device,
            num_global_classes, right_id,
            path_weight * p_right, fingerprints, routing
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Final prediction
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(tree, trained_models, data, ssm_encoder, device,
                  num_global_classes, routing: str = 'hybrid') -> np.ndarray:
    with torch.no_grad():
        weighted_probs = cascade_soft_inference(
            tree, trained_models,
            data.unsqueeze(0) if data.dim() == 3 else data,
            ssm_encoder, device, num_global_classes,
            routing=routing
        )

    pred = weighted_probs.squeeze(0).argmax(dim=0).numpy()
    return pred


def per_class_iou(pred: np.ndarray,
                  labels_np: np.ndarray,
                  num_classes: int) -> dict:
    """
    Per-class IoU over labeled pixels only.
    Returns dict: {class_id: iou_float}  (classes with zero ground-truth pixels are omitted)
    """
    mask     = labels_np > 0
    pred_m   = pred[mask]
    target_m = labels_np[mask]
    result   = {}
    for c in range(1, num_classes):
        pred_c   = pred_m == c
        target_c = target_m == c
        tp    = (pred_c & target_c).sum()
        fp    = (pred_c & ~target_c).sum()
        fn    = (~pred_c & target_c).sum()
        union = tp + fp + fn
        if target_c.sum() > 0:
            result[c] = float(tp / union) if union > 0 else 0.0
    return result


def compute_rssp_metrics(pred: np.ndarray,
                         labels_np: np.ndarray,
                         num_classes: int):
    """
    OA, mIoU, Dice, Precision, Recall, AA, kappa over labeled pixels only.
    Returns: (oa, miou, dice, precision, recall, aa, kappa, per_class_ious)
    where per_class_ious is a dict {class_id: iou_float}.
    """
    mask     = labels_np > 0
    pred_m   = pred[mask]
    target_m = labels_np[mask]
    n        = target_m.size

    oa = (pred_m == target_m).sum() / n

    ious, dices, precisions, recalls, per_class_acc = [], [], [], [], []
    per_class_ious = {}

    for c in range(1, num_classes):
        pred_c   = pred_m == c
        target_c = target_m == c
        tp    = (pred_c & target_c).sum()
        fp    = (pred_c & ~target_c).sum()
        fn    = (~pred_c & target_c).sum()
        union = tp + fp + fn
        if union > 0:
            iou = float(tp / union)
            ious.append(iou)
            dices.append((2 * tp) / (2 * tp + fp + fn + 1e-8))
            precisions.append(tp / (tp + fp + 1e-8))
            recalls.append(tp / (tp + fn + 1e-8))
        else:
            iou = 0.0
        if target_c.sum() > 0:
            per_class_acc.append(tp / target_c.sum())
            per_class_ious[c] = iou

    miou      = sum(ious)          / len(ious)          if ious          else 0.0
    dice      = sum(dices)         / len(dices)         if dices         else 0.0
    precision = sum(precisions)    / len(precisions)    if precisions    else 0.0
    recall    = sum(recalls)       / len(recalls)       if recalls       else 0.0
    aa        = sum(per_class_acc) / len(per_class_acc) if per_class_acc else 0.0

    # Cohen's kappa
    po     = float(oa)
    pe_sum = 0.0
    for c in range(1, num_classes):
        p_pred   = (pred_m == c).sum() / n
        p_target = (target_m == c).sum() / n
        pe_sum  += float(p_pred) * float(p_target)
    kappa = (po - pe_sum) / (1 - pe_sum + 1e-10) if (1 - pe_sum) > 1e-10 else 0.0

    return (float(oa), float(miou), float(dice), float(precision), float(recall),
            float(aa), float(kappa), per_class_ious)