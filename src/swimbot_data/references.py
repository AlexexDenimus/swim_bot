from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
TMP_DIR = DATA_DIR / "tmp"

REFERENCE_VIDEOS: dict[str, Path] = {
    "bras": DATA_DIR / "Брасс сверху.mp4",
    "crawl": DATA_DIR / "Кроль сверху.mp4",
}
