from pathlib import Path

# Find project root automatically (where pyproject.toml lives)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FINAL_DIR = DATA_DIR / "final"


def get_data_path(subfolder: str, filename: str) -> Path:
    """
    Build a portable path to a file inside the data folder.

    Args:
        subfolder (str): One of "raw", "cleaned", "final".
        filename (str): Name of the file (e.g., "precipitation.csv").

    Returns:
        Path: Full path to the requested file.
    """
    subfolder_map = {
        "raw": RAW_DIR,
        "processed": PROCESSED_DIR,
        "final": FINAL_DIR,
    }

    if subfolder not in subfolder_map:
        raise ValueError(
            f"Invalid subfolder '{subfolder}'. "
            "Use 'raw', 'processed', or 'final'."
        )

    return subfolder_map[subfolder] / filename
