"""
Quick smoke test — verifies the PDS3 reader loads the real CRISM file correctly.
Run from the GHOST root: python tmp/test_pds3.py
"""
import sys
sys.path.insert(0, '.')

from datasets.pds3_reader import read
import numpy as np

img = 'data/ato00027155_01_if126s_trr3'  # Windows may have dropped extension
import os
from pathlib import Path

# Find the actual .img file (Windows may show it without extension in Explorer)
candidates = list(Path('data').glob('ato00027155*'))
print("Files found in data/:")
for c in candidates:
    print(f"  {c.name}  ({c.stat().st_size:,} bytes)")

# Try with detected paths
img_file = None
lbl_file = None
for c in candidates:
    if c.suffix.lower() in ('.img', '') and c.stat().st_size > 1_000_000:
        img_file = str(c)
    if c.suffix.lower() == '.lbl':
        lbl_file = str(c)

if img_file is None:
    print("ERROR: Could not find .img file > 1MB in data/")
    sys.exit(1)

print(f"\nReading: {img_file}")
print(f"Label:   {lbl_file}")

cube, labels, meta = read(img_file, lbl_file)

print(f"\n✓ cube shape:    {cube.shape}")
print(f"✓ labels shape:  {labels.shape}")
print(f"✓ value range:   [{cube.min():.3f}, {cube.max():.3f}]")
print(f"✓ mean / std:    {cube.mean():.4f} / {cube.std():.4f}")
print(f"✓ NaN count:     {np.isnan(cube).sum()}")
print(f"\nMeta: {meta}")
print("\nPDS3 reader smoke test PASSED ✓")
