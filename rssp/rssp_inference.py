import torch
import torch.nn.functional as F
import numpy as np
from models.hyperspectral_net import HyperspectralNet


def predict_node(models_info, data, device, voting='weighted'):
    num_classes     = models_info[0]['num_classes']
    local_to_global = models_info[0]['local_to_global']
    H, W            = data.shape[2], data.shape[3]
    num_bands       = data.shape[1]
    prob_sum        = torch.zeros(1, num_classes, H, W)

    for info in models_info:
        model = HyperspectralNet(
            num_bands=info['num_bands'],
            num_classes=info['num_classes'],
            num_filters=info['num_filters'],
            num_blocks=info['num_blocks'],
            base_filters=info['base_filters']
        ).to(device)
        model.load_state_dict({k: v.to(device) for k, v in info['state_dict'].items()})
        model.eval()

        with torch.no_grad():
            logits = model(data.to(device)).cpu()
            probs  = F.softmax(logits, dim=1)

        if voting == 'hard':
            prob_sum += (probs == probs.max(dim=1, keepdim=True).values).float()
        else:
            prob_sum += probs

    prob_avg              = prob_sum / len(models_info)
    confidence, local_pred = prob_avg.max(dim=1)
    confidence  = confidence[0].numpy()
    local_pred  = local_pred[0].numpy()

    pred_global = np.zeros_like(local_pred)
    for local_id, global_id in local_to_global.items():
        pred_global[local_pred == local_id] = global_id

    return pred_global, confidence


def cascade_inference(tree, trained_models, data, device,
                      voting='weighted', threshold=0.7,
                      node_id='root', pixel_mask=None):
    """
    pixel_mask: (H, W) boolean numpy array
                Only these pixels are valid for this node.
                None means all pixels (root call).
    """
    H, W = data.shape[2], data.shape[3]

    if pixel_mask is None:
        pixel_mask = np.ones((H, W), dtype=bool)

    result = np.zeros((H, W), dtype=np.int64)

    # Get this node's predictions
    node_models              = trained_models[node_id]
    pred_global, confidence  = predict_node(node_models, data, device, voting)

    # Only trust predictions within our pixel_mask
    pred_global[~pixel_mask] = 0
    confidence[~pixel_mask]  = 0.0

    # Leaf node - return directly
    if tree['left'] is None and tree['right'] is None:
        result[pixel_mask] = pred_global[pixel_mask]
        return result

    left_classes  = set(tree['left']['classes'])  if tree['left']  else set()
    right_classes = set(tree['right']['classes']) if tree['right'] else set()

    # Route pixels to children based on this node's predictions
    left_pixel_mask  = pixel_mask & np.isin(pred_global, list(left_classes))
    right_pixel_mask = pixel_mask & np.isin(pred_global, list(right_classes))

    # Handle threshold voting
    if voting == 'threshold':
        uncertain              = pixel_mask & (confidence < threshold)
        left_pixel_mask        = left_pixel_mask  & (confidence >= threshold)
        right_pixel_mask       = right_pixel_mask & (confidence >= threshold)
        result[uncertain]      = pred_global[uncertain]  # keep parent prediction

    # Set current predictions for pixels not going deeper
    result[pixel_mask] = pred_global[pixel_mask]

    # Recurse left
    if tree['left'] and len(tree['left']['classes']) > 1:
        left_id = node_id + '_L'
        if left_id in trained_models and left_pixel_mask.sum() > 0:
            left_pred          = cascade_inference(
                tree['left'], trained_models, data,
                device, voting, threshold,
                left_id, left_pixel_mask          # ← pass mask
            )
            result[left_pixel_mask] = left_pred[left_pixel_mask]

    # Recurse right
    if tree['right'] and len(tree['right']['classes']) > 1:
        right_id = node_id + '_R'
        if right_id in trained_models and right_pixel_mask.sum() > 0:
            right_pred          = cascade_inference(
                tree['right'], trained_models, data,
                device, voting, threshold,
                right_id, right_pixel_mask        # ← pass mask
            )
            result[right_pixel_mask] = right_pred[right_pixel_mask]

    return result


def compute_rssp_metrics(pred, labels_np, num_classes):
    mask     = labels_np > 0
    pred_m   = pred[mask]
    target_m = labels_np[mask]

    oa = (pred_m == target_m).sum() / target_m.size

    ious = []
    for c in range(1, num_classes):
        pred_c   = pred_m == c
        target_c = target_m == c
        tp    = (pred_c & target_c).sum()
        fp    = (pred_c & ~target_c).sum()
        fn    = (~pred_c & target_c).sum()
        union = tp + fp + fn
        if union > 0:
            ious.append(tp / union)

    miou = sum(ious) / len(ious) if ious else 0
    return float(oa), float(miou)