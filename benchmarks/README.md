# Benchmarks

Indian Pines protocol for architecture 0.2.0. Every script runs one process at a time and skips finished
runs, so it can be stopped and resumed. Results land in `runs/bench/`.

```bash
python benchmarks/indian_pines.py --control             # 0.2.0 on the ratio and fixed splits, 5 seeds
python benchmarks/indian_pines.py --v01                 # v0.1 flat U-Net on the ratio split, 5 seeds
python benchmarks/indian_pines.py --v01-fixed           # v0.1 flat U-Net on the fixed split, 5 seeds
python benchmarks/indian_pines.py --disjoint            # 0.2.0 and v0.1 flat on the block split, 5 seeds
python benchmarks/indian_pines.py --shuffle-control     # permuted labels: should score at chance
python benchmarks/indian_pines.py --ablation            # continuum removal × pooling, 5 seeds
python benchmarks/indian_pines.py --v01-spt             # v0.1 SPT 32/8 on ratio and fixed, seed 0, as shipped
python benchmarks/indian_pines.py --v01-spt-train-tree  # the same, tree built from training pixels only
python benchmarks/svm_baseline.py --splits ratio fixed disjoint  # SVM and random forest, same pixels
python benchmarks/summarize.py                          # the table
```

The v0.1 modes other than `--v01` run v0.1.7's own trainers through `v01_split.py`, which swaps in v0.2's
split for the same seed so both architectures score identical pixels (for the ratio split it changes
nothing). Shipped v0.1.7 builds SPT's tree from every labelled pixel, test pixels included;
`--tree-from-train` builds it from training pixels only. An SPT run takes about 1h20m on a laptop RTX 3050.

Three splits, each with a validation set for early stopping:

| Split | Train / val / test | Always guessing the biggest class |
|-------|--------------------|-----------------------------------|
| `ratio` (v0.1's, same pixels per seed) | 2,045 / 1,018 / 7,186 | 23.9% |
| `fixed` (50 per class, 15 for classes under 50) | 695 / 950 / 8,604 | 25.2% |
| `disjoint` (14×14 blocks, each whole in one split) | 3,029–3,192 / 1,802–2,053 / 5,039–5,418 | 29.5–32.1% |

The block split gives every class a training block, rarest class first, and a test block where the class
spans more than one. It then hands out blocks until each class has about 20% of its pixels in training and
10% in validation; whole blocks overshoot that, to about 30% and 19% here. Grass-pasture-mowed sits inside
one block, so it trains but is never tested: 15 of the 16 classes are scored.

Readings fixed before any run:

- `fixed` split: about 40% or less means broken; 75–85% is expected; 95% or more means look for leakage.
- 0.2.0 below the SVM on the same pixels means the ResNet is not earning its place.
- Shuffled labels should score at chance: kappa near 0. (Chance OA is not the majority baseline here;
  training is class-balanced, so guesses spread across all 16 classes. This line was corrected after the
  run; the original said "near the majority baseline".)
- `ratio` split: compare with the v0.1 flat runs, which see identical pixels and seeds. README's 97.20% is
  v0.1 with SPT, so it is context, not a like-for-like comparison.

## Results (25 Sept 2026)

Five seeds; every method in a row group sees the same pixels for a given seed.

| Split | Method | OA | AA | mIoU |
|-------|--------|----|----|------|
| fixed | 0.2.0 (`--cr auto` → full, avg pool) | 73.9 ± 1.4 | 82.8 ± 2.5 | 61.2 ± 2.2 |
| fixed | SVM | 66.4 ± 1.2 | 78.2 ± 0.7 | 53.4 ± 1.4 |
| fixed | Random forest | 66.7 ± 1.6 | 78.4 ± 1.4 | 53.5 ± 2.2 |
| ratio | 0.2.0 | 87.2 ± 0.5 | 79.5 ± 1.5 | 72.0 ± 1.2 |
| ratio | v0.1 flat U-Net | 87.9 ± 2.6 | 58.1 ± 4.9 | 51.3 ± 5.4 |
| ratio | SVM | 80.9 ± 0.5 | 78.0 ± 0.8 | 67.8 ± 0.6 |
| ratio | Random forest | 80.0 ± 0.3 | 72.6 ± 2.0 | 63.5 ± 1.3 |

- **Controls:**
  - 0.2.0 beats the SVM and random forest in every seed on both splits.
  - Against v0.1's flat U-Net it ties on OA and leads by 21 points of AA and mIoU in every seed.
  - Shuffled labels: kappa −0.004, i.e. chance.
- **Weak spot:** the six corn and soybean classes (65% of fixed-split test pixels) reach IoU 0.41–0.51.
  Grass, hay, wheat and woods reach 0.83–0.95.
- **Ablation, fixed split:**
  - Skipping continuum removal (`none`, 75.7 ± 1.4) beat full by 1.8 OA in all 5 seeds.
  - Full versus plain max-normalisation (`off`) made no consistent difference.
  - Averaging over bands matched or beat flattening in all four modes.
  - Within one scene, brightness is a useful cue that continuum removal throws away. Whether
    continuum removal pays off across scenes is still untested.
- **Not tested yet:** all of this is one scene with random pixel splits. A random split rewards spatial
  context, which only v0.1 uses. The claim v0.2 exists for, transfer to other scenes, needs a spatially
  disjoint split or a second scene.
