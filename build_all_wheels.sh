#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# build_all_wheels.sh — Build manylinux wheels for Python 3.10, 3.11, 3.12
#
# Usage:  ./build_all_wheels.sh
# Output: dist/ghost_hsi-*manylinux*.whl  (one per Python version)
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

VERSIONS=("3.10" "3.11" "3.12")
CONDA_ENVS=("ghost_build_310" "ai_env" "ghost_build_312")

echo "Cleaning previous builds..."
rm -rf build/ dist/ wheelhouse/ ghost_hsi.egg-info/ ghost_net.egg-info/
find ghost/ -name '*.c' -delete
find ghost/ -name '*.so' -delete
find ghost/ -name '*.pyd' -delete
mkdir -p dist

for i in "${!VERSIONS[@]}"; do
    PY_VER="${VERSIONS[$i]}"
    ENV_NAME="${CONDA_ENVS[$i]}"

    echo ""
    echo "=========================================="
    echo "  Building for Python ${PY_VER} (env: ${ENV_NAME})"
    echo "=========================================="

    # Get the Python path from the conda env
    PYTHON="$(conda run -n "$ENV_NAME" which python)"

    # Ensure build deps are installed
    conda run -n "$ENV_NAME" pip install cython setuptools wheel auditwheel patchelf 2>&1 | tail -1

    # Clean build artifacts from previous version
    rm -rf build/
    find ghost/ -name '*.c' -delete
    find ghost/ -name '*.so' -delete

    # Build wheel
    conda run -n "$ENV_NAME" python setup.py bdist_wheel 2>&1 | tail -3

    # Find the wheel we just built
    WHEEL=$(ls dist/ghost_hsi-*-cp${PY_VER//./}*.whl 2>/dev/null | grep linux_x86_64 | head -1)
    if [ -z "$WHEEL" ]; then
        echo "ERROR: No wheel found for Python ${PY_VER}"
        continue
    fi

    # Strip source files
    echo "  Stripping source from ${WHEEL}..."
    TMPDIR=$(mktemp -d)
    unzip -q "$WHEEL" -d "$TMPDIR"
    find "$TMPDIR/ghost" -name '*.c' -delete
    find "$TMPDIR/ghost" -name '*.py' ! -name '__init__.py' -delete
    find "$TMPDIR" -empty -type d -delete 2>/dev/null || true
    rm "$WHEEL"
    (cd "$TMPDIR" && zip -qr "$OLDPWD/$WHEEL" . -x '*/')

    # Fix RECORD
    conda run -n "$ENV_NAME" python -c "
import zipfile, hashlib, base64, io, csv, tempfile, shutil
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

    # Convert to manylinux
    echo "  Converting to manylinux..."
    conda run -n "$ENV_NAME" auditwheel repair "$WHEEL" -w dist/ --plat manylinux_2_17_x86_64 2>&1 | tail -2
    rm -f "$WHEEL"  # remove the linux_x86_64 version

    echo "  Done: Python ${PY_VER}"
done

# Clean any remaining linux_x86_64 wheels
rm -f dist/ghost_hsi-*linux_x86_64.whl 2>/dev/null || true

echo ""
echo "=========================================="
echo "  All wheels built!"
echo "=========================================="
echo ""
ls -lh dist/*.whl
echo ""
echo "To upload all:  twine upload dist/ghost_hsi-*.whl"
