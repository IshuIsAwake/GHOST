# GHOST — TODO

## Immediate (Hackathon)
- [ ] Ryugu NIRS3 data parsing and reprojection onto shape model
- [ ] Ryugu config and dataset loader
- [ ] Interactive 3D visualization of Ryugu segmentation map (Three.js / Unity)
- [ ] Segmentation map visualization for Indian Pines (color coded output)

## Architecture Upgrades
- [ ] Spectral Transformer — replace SE block with multi-head attention
      across the spectral dimension (200 bands as 200 tokens)
      Motivation: captures co-occurring absorption features across wavelengths
      physically more meaningful than static channel weighting
- [ ] Residual connections inside Spectral3DStack
- [ ] Deep supervision — auxiliary losses at intermediate decoder levels
- [ ] Mixed precision training (torch.cuda.amp) for faster training

## Framework / QoL (nnU-Net style)
- [ ] Dataset fingerprint analyzer
      Reads any hyperspectral cube and outputs:
      - suggested patch size
      - suggested spectral kernel size
      - class distribution
      - recommended augmentation strategy
- [ ] Automatic spectral band grouping / clustering
- [ ] Band interpolation and alignment across sensors
      (AVIRIS 200 bands ↔ Hyperion 242 bands ↔ EnMAP 244 bands)
- [ ] CLI interface — train and infer with one command
- [ ] Config-driven inference script

## Data Loaders
- [ ] ENVI format (.hdr / .img) — AVIRIS, Hyperion
- [ ] NetCDF (.nc) — satellite missions  
- [ ] HDF5 (.h5) — EnMAP, EMIT
- [ ] NIRS3 binary parser (Hayabusa-2) — Ryugu
- [ ] Generic CSV loader for custom instruments

## Evaluation
- [ ] Per-class IoU breakdown table
- [ ] Confusion matrix
- [ ] Segmentation map visualization with ground truth overlay
- [ ] Comparison table against published Indian Pines baselines

## Future Datasets
- [ ] Pavia University (urban remote sensing)
- [ ] Houston 2013 (urban, LiDAR fusion potential)
- [ ] Medical hyperspectral (tissue segmentation)
- [ ] Geology field scan dataset