# GHOST — TODO

## Current Status
- [x] Phase 1: Band-count agnosticism (SpectralTransformer with interpolated positional embeddings)
- [x] Phase 2: Normalization stability (GroupNorm everywhere, BatchNorm fully removed)
- [ ] Phase 3: Multi-scale spatial training
- [ ] Phase 4: Multi-dataset pretraining
- [ ] Phase 5: pip package + release

---

## Immediate (Hackathon)

- [ ] Finish Indian Pines training run (300 epochs, real SpectralTransformer)
- [ ] Run test_pavia_zeroshot.py after training completes
- [ ] Fine-tune on Pavia University (9 classes)
- [ ] Segmentation map visualization — color-coded output for Indian Pines and Pavia
- [ ] Ryugu NIRS3 data parsing and reprojection onto shape model
- [ ] Ryugu config and dataset loader
- [ ] Interactive 3D visualization of Ryugu segmentation map

---

## Phase 3: Multi-Scale Spatial Training

- [ ] During training, randomly sample patch size from {16, 32, 48} per batch
- [ ] Update dataset loader to accept dynamic patch size
- [ ] Update train.py sampler logic accordingly
- [ ] Verify U-Net handles variable spatial input natively (it does — fully convolutional)

---

## Phase 4: Multi-Dataset Pretraining

- [ ] Gather datasets: Indian Pines, Pavia, Salinas, Botswana, Houston 2013
- [ ] Decide on label strategy: shared head vs. per-dataset output heads
- [ ] Multi-task training loop — sample batches from each dataset, combined loss
- [ ] Save pretrained weights as `ghost_foundation.pth`
- [ ] Fine-tune on unseen dataset (e.g. medical hyperspectral) — verify faster convergence

---

## Phase 5: Package & Release

- [ ] `pip install GHOST`
- [ ] Clean public API: `ghost.train()`, `ghost.infer()`, `ghost.finetune()`
- [ ] Pretrained weight hosting (HuggingFace Hub or GitHub Releases)
- [ ] Example notebooks: Indian Pines, Pavia, zero-shot transfer demo
- [ ] Name pretrained weights something fun (peek-a-boo, scary, etc.)

---

## Architecture Upgrades

- [ ] Gradient checkpointing in SpectralTransformer — reduces memory for larger patches
- [ ] Mixed precision training (torch.cuda.amp) — would allow larger batches on 6GB GPU
- [ ] Residual connections inside Spectral3DStack
- [ ] Deep supervision — auxiliary losses at intermediate decoder levels
- [ ] Full-image GPU inference via sliding window — currently runs on CPU due to transformer memory

---

## Framework / QoL

- [ ] Dataset fingerprint analyzer
      Reads any hyperspectral cube, outputs suggested patch size, spectral kernel size,
      class distribution, recommended augmentation strategy
- [ ] Automatic spectral band alignment across sensors
      (AVIRIS 200 bands ↔ Hyperion 242 bands ↔ EnMAP 244 bands)
- [ ] CLI interface — train and infer with one command
- [ ] Config-driven inference script

---

## Data Loaders

- [ ] ENVI format (.hdr / .img) — AVIRIS, Hyperion
- [ ] NetCDF (.nc) — satellite missions
- [ ] HDF5 (.h5) — EnMAP, EMIT
- [ ] NIRS3 binary parser (Hayabusa-2) — Ryugu
- [ ] Generic CSV loader

---

## Evaluation

- [ ] Per-class IoU breakdown table
- [ ] Confusion matrix
- [ ] Segmentation map visualization with ground truth overlay
- [ ] Comparison table against published Indian Pines baselines

---

## Future Datasets

- [ ] Houston 2013 (urban, LiDAR fusion potential)
- [ ] Salinas Valley
- [ ] Botswana
- [ ] Medical hyperspectral (tissue segmentation)
- [ ] Geology field scan