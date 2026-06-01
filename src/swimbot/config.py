from pathlib import Path

from swimbot_data.references import DATA_DIR, TMP_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[2]

__all__ = ["PROJECT_ROOT", "DATA_DIR", "TMP_DIR"]
