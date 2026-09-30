"""Build the Rust engine's PyO3 bridge and (re)install it into this Python.

Run after any change under engine/RustEngine/:  python scripts/build_rust_engine.py
Needs a Rust toolchain and maturin (pip install maturin).
"""

import glob
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PYBIND = os.path.join(ROOT, 'engine', 'RustEngine', 'crates', 'pybind')
WHEELS = os.path.join(ROOT, 'engine', 'RustEngine', 'target', 'wheels')


def main():
    subprocess.run(['maturin', 'build', '--release', '--quiet'], cwd=PYBIND, check=True)
    wheels = sorted(glob.glob(os.path.join(WHEELS, 'rustengine-*.whl')), key=os.path.getmtime)
    if not wheels:
        sys.exit(f"no rustengine wheel found in {WHEELS}")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--force-reinstall', '--no-deps', '--quiet', wheels[-1]],
                   check=True)
    print(f"installed {os.path.basename(wheels[-1])}")


if __name__ == '__main__':
    main()
