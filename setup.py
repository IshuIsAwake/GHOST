"""
setup.py — Cython-compiled build for ghost-hsi.

Compiles all .py → .so/.pyd via Cython. The .py source files are stripped
from the final wheel so only compiled binaries ship to users.
"""
import os
import sysconfig
from pathlib import Path
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as _build_ext
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


class StripAfterCompile(_build_ext):
    """Strip .py source and .c intermediates ONLY after .so/.pyd compilation succeeds.

    Runs after build_ext (which runs after build_py), so both .py and .so
    exist in build_lib at this point. Only removes .py if its compiled
    counterpart was actually produced — prevents shipping an empty package
    if Cython compilation fails.
    """
    def run(self):
        super().run()
        build_lib  = Path(self.build_lib)
        ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')  # e.g. .cpython-312-x86_64-linux-gnu.so

        stripped = 0
        for ext in self.extensions:
            mod_path = ext.name.replace('.', os.sep)
            so_path  = build_lib / (mod_path + ext_suffix)
            py_path  = build_lib / (mod_path + '.py')

            if so_path.exists() and py_path.exists():
                py_path.unlink()
                stripped += 1

        # Remove .c Cython intermediates (can be reverse-read)
        for c_file in build_lib.rglob("*.c"):
            c_file.unlink()

        print(f"Stripped {stripped} .py source files (compiled .so exist)")


ext_modules = cythonize(
    collect_extensions(),
    compiler_directives={"language_level": "3"},
    nthreads=os.cpu_count() or 4,
)

setup(
    ext_modules=ext_modules,
    packages=find_packages(include=["ghost*"]),
    package_data={"ghost": ["configs/*.yaml", "data/indian_pines/*.mat"]},
    cmdclass={"build_ext": StripAfterCompile},
)
