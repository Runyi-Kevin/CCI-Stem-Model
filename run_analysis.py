"""Command-line entry point:  python run_analysis.py --input data/...xlsx --output outputs"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from stem_model.pipeline import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
