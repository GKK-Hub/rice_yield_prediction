from pathlib import Path

# Find project root automatically (where pyproject.toml lives)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLEANED_DIR = DATA_DIR / "cleaned"
FINAL_DIR = DATA_DIR / "final"

subfolder_map = {
        "raw": RAW_DIR,
        "cleaned": CLEANED_DIR,
        "final": FINAL_DIR,
    }


def get_data_dir(subfolder: str) -> Path:
    """
    Build a portable path to a data subfolder.

    Args:
        subfolder (str): One of "raw", "cleaned", "final".

    Returns:
        Path: Full path to the requested directory.
    """
    valid = {"raw", "cleaned", "final"}
    if subfolder not in valid:
        raise ValueError(f"Invalid subfolder {subfolder}."
                         f"Must be one of {valid}.")
    return subfolder_map[subfolder]


def get_data_file(subfolder_path: Path, filename: str) -> Path:
    """
    Build a portable path to a file inside the data folder.

    Args:
        subfolder_path (Path): One of RAW_DIR, CLEANED_DIR, FINAL_DIR
        filename (str): Name of the file (e.g., "precipitation.csv").

    Returns:
        Path: Full path to the requested file.
    """
    return subfolder_path / filename
