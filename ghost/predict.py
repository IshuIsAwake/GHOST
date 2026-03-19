import argparse
import time
import torch
import numpy as np
import pickle
import os
import csv

from ghost.datasets.hyperspectral_dataset import HyperspectralDataset
from ghost.models.spectral_ssm import SpectralSSMEncoder
from ghost.rssp.rssp_inference import run_inference, compute_rssp_metrics
from ghost.utils.display import (
    print_predict_start, print_results_box, print_per_class_iou, print_training_done,
    print_config_box,
    BOLD, RESET, CYAN, GRAY, GREEN
)

def main():
    parser = argparse.ArgumentParser(description='GHOST — Standalone Inference')

    parser.add_argument('--data',        type=str,   required=True)
    parser.add_argument('--gt',          type=str,   required=True)
    parser.add_argument('--model',       type=str,   required=True, help='Path to rssp_models.pkl')
    parser.add_argument('--ssm_load',    type=str,   default=None)
    parser.add_argument('--train_ratio', type=float, default=0.2)
    parser.add_argument('--val_ratio',   type=float, default=0.1)
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--routing',     type=str,   default='all',
                        choices=['hybrid', 'forest', 'soft', 'all'])
    parser.add_argument('--out-dir',     type=str,   default='.')

    args = parser.parse_args()

    print_predict_start()

    os.makedirs(args.out_dir, exist_ok=True)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    device_str = str(DEVICE)
    if torch.cuda.is_available() and DEVICE.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        device_str = f"cuda ({gpu_name}, {total_mem:.0f} GB)"

    print_config_box("GHOST Predict", [
        ("Device",  device_str),
        ("Model",   args.model),
        ("Data",    args.data),
        ("Routing", args.routing),
    ])

    test_ds = HyperspectralDataset(
        args.data, args.gt, split='test',
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    data        = test_ds.data
    labels      = test_ds.labels
    num_classes = test_ds.num_classes

    print(f"\nLoading models from {args.model} ...")
    with open(args.model, 'rb') as f:
        checkpoint = pickle.load(f)

    trained_models = checkpoint['trained_models']
    tree           = checkpoint['tree']
    d_model        = checkpoint.get('d_model', 64)
    d_state        = checkpoint.get('d_state', 16)
    print("Models loaded.")

    routings = ['forest', 'hybrid', 'soft'] if args.routing == 'all' else [args.routing]

    overall_start = time.time()

    for current_routing in routings:
        print(f"\n=== Running Cascade Inference (routing={current_routing}) ===")

        ssm_encoder = None

        if current_routing in ('hybrid', 'soft'):
            if args.ssm_load is None:
                if 'ssm_state' in checkpoint:
                    print("Loading SSM encoder from checkpoint ...")
                    ssm_encoder = SpectralSSMEncoder(d_model=d_model, d_state=d_state).to(DEVICE)
                    ssm_encoder.load_state_dict(
                        {k: v.to(DEVICE) for k, v in checkpoint['ssm_state'].items()})
                else:
                    print("WARNING: No SSM weights found. Falling back to forest-only routing.")
                    current_routing = 'forest'
            else:
                print(f"Loading SSM encoder from {args.ssm_load} ...")
                ssm_encoder = SpectralSSMEncoder(d_model=d_model, d_state=d_state).to(DEVICE)
                state = torch.load(args.ssm_load, map_location=DEVICE, weights_only=True)
                ssm_encoder.load_state_dict(state)

            if ssm_encoder is not None:
                ssm_encoder.eval()
                for p in ssm_encoder.parameters():
                    p.requires_grad_(False)
        else:
            print("Forest-only routing — SSM encoder not needed.")

        t0 = time.time()

        final_pred = run_inference(
            tree, trained_models,
            data,
            ssm_encoder, DEVICE, num_classes,
            routing=current_routing
        )

        infer_secs = time.time() - t0
        print(f"  Inference complete in {infer_secs:.1f}s")

        test_mask   = test_ds.split_mask.numpy()
        labels_np   = labels.numpy()

        eval_labels = np.zeros_like(labels_np)
        eval_labels[test_mask > 0] = labels_np[test_mask > 0]

        oa, miou, dice, precision, recall, aa, kappa, per_class_ious = compute_rssp_metrics(
            final_pred, eval_labels, num_classes
        )

        print_results_box({
            'OA':        oa,
            'mIoU':      miou,
            'Dice':      dice,
            'Precision': precision,
            'Recall':    recall,
            'AA':        aa,
            'kappa':     kappa,
        }, routing=current_routing)

        # Per-class pixel counts from the test split
        pixel_counts = {c: int((eval_labels == c).sum()) for c in per_class_ious}
        print_per_class_iou(per_class_ious, pixel_counts=pixel_counts)

        # ── Summary CSV (aggregate metrics) ─────────────────────────────────
        out_csv = os.path.join(args.out_dir, f'test_results_{current_routing}.csv')
        with open(out_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['routing', 'test_oa', 'test_miou', 'test_dice',
                             'test_precision', 'test_recall', 'test_aa', 'test_kappa'])
            writer.writerow([current_routing,
                             f"{oa:.4f}", f"{miou:.4f}", f"{dice:.4f}",
                             f"{precision:.4f}", f"{recall:.4f}",
                             f"{aa:.4f}", f"{kappa:.4f}"])

        # ── Class report CSV (per-class details) ────────────────────────────
        report_csv = os.path.join(args.out_dir, f'class_report_{current_routing}.csv')
        with open(report_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['class_id', 'total_test_pixels', 'correct_pixels',
                             'IoU', 'Precision', 'Recall'])
            for c in sorted(per_class_ious.keys()):
                total_px   = int((eval_labels == c).sum())
                correct_px = int(((final_pred == c) & (eval_labels == c)).sum())
                pred_c     = (final_pred[eval_labels > 0] == c)
                target_c   = (eval_labels[eval_labels > 0] == c)
                tp = int((pred_c & target_c).sum())
                fp = int((pred_c & ~target_c).sum())
                fn = int((~pred_c & target_c).sum())
                c_prec = tp / (tp + fp + 1e-8)
                c_rec  = tp / (tp + fn + 1e-8)
                writer.writerow([c, total_px, correct_px,
                                 f"{per_class_ious[c]:.6f}",
                                 f"{c_prec:.6f}", f"{c_rec:.6f}"])

        print(f"  {GREEN}Saved →{RESET} {out_csv}")
        print(f"  {GREEN}Saved →{RESET} {report_csv}")

    total_secs = time.time() - overall_start
    print(f"\n  {GRAY}Total predict time: {total_secs:.1f}s{RESET}")
    print_training_done()

if __name__ == '__main__':
    main()