"""
setup.py — Cython-compiled build for ghost-hsi.

Compiles all .py → .so via Cython. The .py source files are stripped
from the final wheel so only compiled binaries ship to users.
"""
import os
import shutil
from pathlib import Path
from setuptools import setup, find_packages, Extension
from setuptools.command.build_py import build_py as _build_py
from Cython.Build import cythonize


# ── Collect every .py under ghost/ (except __init__.py) ──────────────────
def collect_extensions(root="ghost"):
    extensions = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.endswith(".py") or fn == "__init__.py":
                continue
            filepath = os.path.join(dirpath, fn)
            modname = filepath.replace(os.sep, ".").removesuffix(".py")
            extensions.append(Extension(modname, [filepath]))
    return extensions


class StripSourceBuildPy(_build_py):
    """Custom build_py that removes .py and .c files from the wheel.

    Only __init__.py files and compiled .so extensions survive.
    """
    def build_packages(self):
        super().build_packages()
        compiled = {
            ext.name.replace(".", os.sep) + ".py"
            for ext in collect_extensions()
        }
        build_lib = Path(self.build_lib)
        # Remove .py source files that have compiled .so counterparts
        for py_rel in compiled:
            py_path = build_lib / py_rel
            if py_path.exists():
                py_path.unlink()
        # Remove ALL .c files — these are Cython intermediates and
        # can be reverse-read to understand the original Python logic
        for c_file in build_lib.rglob("*.c"):
            c_file.unlink()


ext_modules = cythonize(
    collect_extensions(),
    compiler_directives={"language_level": "3"},
    nthreads=os.cpu_count() or 4,
)

setup(
    ext_modules=ext_modules,
    packages=find_packages(include=["ghost*"]),
    package_data={"ghost": ["configs/*.yaml", "data/indian_pines/*.mat"]},
    cmdclass={"build_py": StripSourceBuildPy},
)
