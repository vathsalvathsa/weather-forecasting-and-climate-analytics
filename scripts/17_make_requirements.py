"""Write a LEAN, pinned requirements.txt for deployment (Streamlit Community Cloud).

The deployed app only needs these packages. Pinning the exact versions used for training matters:
the saved models are pickles and must be loaded with the same scikit-learn version.
The old full list is kept as requirements-dev.txt.

    python scripts/17_make_requirements.py
"""
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ["streamlit", "pandas", "numpy", "scipy", "scikit-learn", "plotly", "requests", "joblib", "pyarrow"]


def main():
    pins = []
    for pkg in RUNTIME:
        try:
            pins.append(f"{pkg}=={version(pkg)}")
        except PackageNotFoundError:
            sys.exit(f"{pkg} is not installed in this environment. Activate .venv and run: pip install -r requirements.txt")
    req, dev = ROOT / "requirements.txt", ROOT / "requirements-dev.txt"
    if req.exists() and not dev.exists():
        dev.write_text(req.read_text())
        print("Kept the old full list as requirements-dev.txt")
    req.write_text("\n".join(pins) + "\n")
    print("Wrote requirements.txt:")
    print("\n".join("  " + p for p in pins))


if __name__ == "__main__":
    main()
