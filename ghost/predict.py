import argparse
import torch
import numpy as np
import pickle
import os
import csv

from ghost.datasets.hyperspectral_dataset import HyperspectralDataset
from ghost.models.spectral_ssm import SpectralSSMEncoder
from ghost.rssp.rssp_inference import run_inference, compute_rssp_metrics

def main():
    parser = argparse.ArgumentParser(description='GHOST — Standalone Inference')

    parser.add_argument('--data',        type=str,   required=True)
    parser.add_argument('--gt',          type=str,   required=True)
    parser.add_argument('--model',       type=str,   required=True, help='Path to rssp_models.pkl')
    parser.add_argument('--ssm_load',    type=str,   default=None,
                        help='Path to ssm_pretrained.pt (optional for --routing forest)')
    parser.add_argument('--train_ratio', type=float, default=0.2)
    parser.add_argument('--val_ratio',   type=float, default=0.1)
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--routing',     type=str,   default='all',
                        choices=['hybrid', 'forest', 'soft', 'all'],
                        help='Routing mode: all (runs all three), hybrid (forest+SSM), forest (no SSM), soft (SSM-only)')
    parser.add_argument('--out-dir',     type=str,   default='.', help='Output directory for saving results')

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {DEVICE}")

    # ── Load dataset ──────────────────────────────────────────────────────────────
    test_ds = HyperspectralDataset(
        args.data, args.gt, split='test',
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    data        = test_ds.data
    labels      = test_ds.labels
    num_classes = test_ds.num_classes

    # ── Load trained models ───────────────────────────────────────────────────────
    print(f"\nLoading models from {args.model} ...")
    with open(args.model, 'rb') as f:
        checkpoint = pickle.load(f)

    trained_models = checkpoint['trained_models']
    tree           = checkpoint['tree']
    d_model        = checkpoint.get('d_model', 64)
    d_state        = checkpoint.get('d_state', 16)
    print("Models loaded.")

    # ── Interpret Routings ───────────────────────────────────────────────────────
    routings = ['forest', 'hybrid', 'soft'] if args.routing == 'all' else [args.routing]

    for current_routing in routings:
        print(f"\n=== Running Cascade Inference (routing={current_routing}) ===")
        # ── Load SSM encoder (only needed for hybrid / soft routing) ──────────────────
        ssm_encoder = None

        if current_routing in ('hybrid', 'soft'):
            if args.ssm_load is None:
                # Try loading from checkpoint's embedded SSM state
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
                print("SSM encoder loaded.")
        else:
            print("Forest-only routing — SSM encoder not needed.")

        # ── Run inference ─────────────────────────────────────────────────────────────
        final_pred = run_inference(
            tree, trained_models,
            data,
            ssm_encoder, DEVICE, num_classes,
            routing=current_routing
        )

        # ── Evaluate on test pixels only ──────────────────────────────────────────────
        test_mask   = test_ds.split_mask.numpy()
        labels_np   = labels.numpy()

        eval_labels = np.zeros_like(labels_np)
        eval_labels[test_mask > 0] = labels_np[test_mask > 0]

        oa, miou, dice, precision, recall = compute_rssp_metrics(
            final_pred, eval_labels, num_classes
        )

        print(f"\n{'='*40}")
        print(f"Routing [{current_routing}] Results:")
        print(f"Test OA:        {oa:.4f}")
        print(f"Test mIoU:      {miou:.4f}")
        print(f"Test Dice:      {dice:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall:    {recall:.4f}")
        print(f"{'='*40}")

        out_csv = os.path.join(args.out_dir, f'test_results_{current_routing}.csv')
        with open(out_csv, 'w', newline='') as f:
            csv.writer(f).writerow(['routing', 'test_oa', 'test_miou', 'test_dice', 'test_precision', 'test_recall'])
            csv.writer(f).writerow([current_routing, f"{oa:.4f}", f"{miou:.4f}", f"{dice:.4f}",
                                    f"{precision:.4f}", f"{recall:.4f}"])

if __name__ == '__main__':
    main()