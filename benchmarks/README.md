# Benchmarks

Indian Pines protocol for architecture 0.2.0. Every script runs one process at a time and skips finished
runs, so it can be stopped and resumed. Results land in `runs/bench/`.

```bash
python benchmarks/indian_pines.py --control          # 0.2.0 on both splits, 5 seeds
python benchmarks/indian_pines.py --v01              # v0.1 flat U-Net on the same pixels, 5 seeds
python benchmarks/indian_pines.py --shuffle-control  # permuted labels: should score at chance
python benchmarks/svm_baseline.py                    # SVM and random forest on the same pixels
python benchmarks/indian_pines.py --ablation         # continuum removal × pooling, 5 seeds
python benchmarks/summarize.py                       # the table
```

Two splits, both with a validation set for early stopping:

| Split | Train / val / test | Always guessing the biggest class |
|-------|--------------------|-----------------------------------|
| `ratio` (v0.1's, same pixels per seed) | 2,045 / 1,018 / 7,186 | 23.9% |
| `fixed` (50 per class, 15 for classes under 50) | 695 / 950 / 8,604 | 25.2% |

Readings fixed before any run:

- `fixed` split: about 40% or less means broken; 75–85% is expected; 95% or more means look for leakage.
- 0.2.0 below the SVM on the same pixels means the ResNet is not earning its place.
- Shuffled labels should land near the majority-class baseline.
- `ratio` split: compare with the v0.1 flat runs, which see identical pixels and seeds. README's 97.20% is
  v0.1 with SPT, so it is context, not a like-for-like comparison.
