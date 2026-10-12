"""Compatibility script for the explicit read-only data commands."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from load_signature_analysis.cli import main
if __name__ == "__main__":
    raise SystemExit(main(["data", *sys.argv[1:]]))
