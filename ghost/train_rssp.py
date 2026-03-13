import argparse
import os
import torch
import numpy as np
import pickle

from ghost.datasets.hyperspectral_dataset import HyperspectralDataset
from ghost.rssp.sam_clustering import build_rssp_tree, print_tree
from ghost.rssp.rssp_trainer import train_tree
from ghost.rssp.rssp_inference import run_inference, compute_rssp_metrics
from ghost.rssp.ssm_pretrain import pretrain_ssm
from ghost.models.spectral_ssm import SpectralSSMEncoder

# ── Args ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='GHOST RSSP + SSSR Training')

    # Data
    parser.add_argument('--data',         type=str,   required=True)
    parser.add_argument('--gt',           type=str,   required=True)
    parser.add_argument('--train_ratio',  type=float, default=0.2)
    parser.add_argument('--val_ratio',    type=float, default=0.1)

    # RSSP tree
    parser.add_argument('--depth',        type=str,   default='auto',
                        help='auto | full | integer')

    # Forest
    parser.add_argument('--forests',      type=int,   default=5)
    parser.add_argument('--base_filters', type=int,   default=32)
    parser.add_argument('--num_filters',  type=int,   default=8)
    parser.add_argument('--num_blocks',   type=int,   default=3)
    parser.add_argument('--epochs',       type=int,   default=300)
    parser.add_argument('--lr',           type=float, default=1e-4)

    # SSM / SSSR
    parser.add_argument('--d_model',      type=int,   default=64,
                        help='SSM fingerprint dimension (default: 64)')
    parser.add_argument('--d_state',      type=int,   default=16,
                        help='SSM state dimension (default: 16)')
    parser.add_argument('--ssm_epochs',   type=int,   default=300,
                        help='SSM pre-training epochs (default: 300)')
    parser.add_argument('--ssm_lr',       type=float, default=1e-3,
                        help='SSM pre-training learning rate (default: 1e-3)')
    parser.add_argument('--ssm_save',     type=str,   default='ssm_pretrained.pt',
                        help='Path to save/load SSM weights')
    parser.add_argument('--ssm_load',     type=str,   default=None,
                        help='Load pre-trained SSM weights (skip pre-training)')
    parser.add_argument('--routing',      type=str,   default='hybrid',
                        choices=['hybrid', 'forest', 'soft'],
                        help='Routing mode: hybrid (forest+SSM), forest (no SSM), soft (SSM-only)')

    # General
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--save',         type=str,   default='rssp_models.pkl')
    parser.add_argument('--out-dir',      type=str,   default='.',
                        help='Output directory to save weights and logs')

    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {DEVICE}")

    # ── Load datasets ─────────────────────────────────────────────────────────────
    train_ds = HyperspectralDataset(args.data, args.gt, split='train',
                                    train_ratio=args.train_ratio,
                                    val_ratio=args.val_ratio, seed=args.seed)
    val_ds   = HyperspectralDataset(args.data, args.gt, split='val',
                                    train_ratio=args.train_ratio,
                                    val_ratio=args.val_ratio, seed=args.seed)
    test_ds  = HyperspectralDataset(args.data, args.gt, split='test',
                                    train_ratio=args.train_ratio,
                                    val_ratio=args.val_ratio, seed=args.seed)

    data        = train_ds.data       # (C, H, W)
    labels      = train_ds.labels     # (H, W)
    num_classes = train_ds.num_classes

    # ── SSM pre-training (or load) ────────────────────────────────────────────────
    print("\n=== SSM Encoder ===")

    ssm_encoder = SpectralSSMEncoder(
        d_model=args.d_model,
        d_state=args.d_state
    ).to(DEVICE)

    if args.ssm_load is not None:
        print(f"Loading SSM weights from {args.ssm_load}")
        state = torch.load(args.ssm_load, map_location=DEVICE, weights_only=True)
        ssm_encoder.load_state_dict(state)
    else:
        print(f"SSM will be saved to: {os.path.join(args.out_dir, args.ssm_save)}")
        ssm_encoder = pretrain_ssm(
            data         = data,
            labels       = labels,
            train_coords = train_ds.train_coords,
            val_coords   = val_ds.val_coords,
            d_model      = args.d_model,
            d_state      = args.d_state,
            num_classes  = num_classes,
            epochs       = args.ssm_epochs,
            lr           = args.ssm_lr,
            device       = str(DEVICE),
            save_path    = os.path.join(args.out_dir, args.ssm_save)
        )

    # Freeze SSM — it is shared across all nodes and never updated again
    for p in ssm_encoder.parameters():
        p.requires_grad_(False)
    ssm_encoder.eval()

    # ── Pre-compute fingerprint map (once for the full image) ─────────────────────
    print("\nPre-computing fingerprint map ...")
    C, H, W = data.shape
    fp_map = torch.zeros(H, W, args.d_model)
    chunk_size = 32  # process 32 rows at a time

    with torch.no_grad():
        for row_start in range(0, H, chunk_size):
            row_end = min(row_start + chunk_size, H)
            chunk = data[:, row_start:row_end, :].unsqueeze(0).to(DEVICE)
            fp_chunk = ssm_encoder(chunk).squeeze(0).permute(1, 2, 0).cpu()
            fp_map[row_start:row_end, :, :] = fp_chunk
            del chunk, fp_chunk
            torch.cuda.empty_cache()

    print(f"Fingerprint map shape: {tuple(fp_map.shape)}")

    # ── Build RSSP tree ───────────────────────────────────────────────────────────
    depth_mode = args.depth
    if depth_mode.isdigit():
        depth_mode = int(depth_mode)

    print("\n=== Building RSSP Tree ===")
    tree, sam_matrix, means = build_rssp_tree(
        data.numpy(), labels.numpy(),
        num_classes=num_classes,
        depth_mode=depth_mode
    )
    print_tree(tree)

    # ── Train ─────────────────────────────────────────────────────────────────────
    print("\n=== Training RSSP Forest + SSSR Routers ===")
    trained_models = train_tree(
        tree, data, labels,
        total_classes = num_classes - 1,
        train_coords  = train_ds.train_coords,
        val_coords    = val_ds.val_coords,
        fp_map        = fp_map,
        ssm_d_model   = args.d_model,
        base_epochs   = args.epochs,
        num_forests   = args.forests,
        base_filters  = args.base_filters,
        num_filters   = args.num_filters,
        num_blocks    = args.num_blocks,
        lr            = args.lr,
        device        = str(DEVICE)
    )

    # Save everything needed for inference
    with open(os.path.join(args.out_dir, args.save), 'wb') as f:
        pickle.dump({
            'trained_models': trained_models,
            'tree':           tree,
            'ssm_state':      {k: v.cpu() for k, v in ssm_encoder.state_dict().items()},
            'd_model':        args.d_model,
            'd_state':        args.d_state,
        }, f)
    print(f"\nSaved → {os.path.join(args.out_dir, args.save)}")

    # ── Test inference ────────────────────────────────────────────────────────────
    print(f"\n=== Running Cascade Inference (routing={args.routing}) ===")

    final_pred = run_inference(
        tree, trained_models,
        data,
        ssm_encoder, DEVICE, num_classes,
        routing=args.routing
    )

    # Evaluate on test pixels only
    test_mask  = test_ds.split_mask.numpy()
    labels_np  = labels.numpy()

    eval_labels = np.zeros_like(labels_np)
    eval_labels[test_mask > 0] = labels_np[test_mask > 0]

    oa, miou, dice, precision, recall = compute_rssp_metrics(final_pred, eval_labels, num_classes)

    print(f"\n{'='*40}")
    print(f"Test OA:        {oa:.4f}")
    print(f"Test mIoU:      {miou:.4f}")
    print(f"Test Dice:      {dice:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall:    {recall:.4f}")
    print(f"{'='*40}")
if __name__ == '__main__':
    main()
