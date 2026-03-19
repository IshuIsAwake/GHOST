#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# build_wheel.sh — Build a compiled (Cython) manylinux wheel for ghost-hsi
#
# Usage:  ./build_wheel.sh
# Output: dist/ghost_hsi-*.manylinux*.whl  (PyPI-uploadable)
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

echo "Cleaning previous builds..."
rm -rf build/ dist/ wheelhouse/ ghost_hsi.egg-info/ ghost_net.egg-info/
find ghost/ -name '*.c' -delete
find ghost/ -name '*.so' -delete
find ghost/ -name '*.pyd' -delete

echo "Compiling with Cython & building wheel..."
python setup.py bdist_wheel

# ── Strip .c and .py source files from the wheel ────────────────────
echo "Stripping source files from wheel..."
WHEEL=$(ls dist/ghost_hsi-*.whl)
TMPDIR=$(mktemp -d)
unzip -q "$WHEEL" -d "$TMPDIR"

# Remove Cython-generated .c files and any leftover .py (except __init__.py)
find "$TMPDIR/ghost" -name '*.c' -delete
find "$TMPDIR/ghost" -name '*.py' ! -name '__init__.py' -delete

# Remove empty directory entries (they confuse auditwheel)
find "$TMPDIR" -empty -type d -delete 2>/dev/null || true

# Rebuild the wheel (without directory entries)
rm "$WHEEL"
(cd "$TMPDIR" && zip -qr "$OLDPWD/$WHEEL" . -x '*/')

# Fix RECORD (regenerate hashes)
python -c "
import zipfile, hashlib, base64, os, io, csv, tempfile, shutil
whl_path = '$WHEEL'
with zipfile.ZipFile(whl_path, 'r') as zin:
    names = [n for n in zin.namelist() if not n.endswith('/')]
    record_name = [n for n in names if n.endswith('RECORD')][0]

buf = io.StringIO()
w = csv.writer(buf)
with zipfile.ZipFile(whl_path, 'r') as zin:
    for name in names:
        if name == record_name:
            w.writerow([name, '', ''])
            continue
        data = zin.read(name)
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
        w.writerow([name, f'sha256={digest}', str(len(data))])

tmp = tempfile.mktemp(suffix='.whl')
with zipfile.ZipFile(whl_path, 'r') as zin, zipfile.ZipFile(tmp, 'w') as zout:
    for name in names:
        if name == record_name:
            zout.writestr(name, buf.getvalue())
        else:
            zout.writestr(name, zin.read(name))
shutil.move(tmp, whl_path)
"

rm -rf "$TMPDIR"

# ── Convert to manylinux (required by PyPI) ─────────────────────────
echo "Converting to manylinux wheel..."
auditwheel repair "$WHEEL" -w dist/ --plat manylinux_2_17_x86_64 2>&1

# Remove the original linux_x86_64 wheel, keep only manylinux
rm -f dist/ghost_hsi-*linux_x86_64.whl 2>/dev/null || true

echo ""
echo "Wheel built successfully!"
echo ""
ls -lh dist/*.whl
echo ""
echo "To install locally:  pip install dist/ghost_hsi-*.whl"
echo "To upload to PyPI:   twine upload dist/ghost_hsi-*.whl"
