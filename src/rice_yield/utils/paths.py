from pathlib import Path

# Find project root automatically (where pyproject.toml lives)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLEANED_DIR = DATA_DIR / "cleaned"
FINAL_DIR = DATA_DIR / "final"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
VALID_DIR = OUTPUT_DIR / "validation"

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
        raise ValueError(f"Invalid subfolder {subfolder}.\
                         Must be one of {valid}.")

    folder = subfolder_map[subfolder]

    # Create 'cleaned' and 'final' folders if missing
    if subfolder in {"cleaned", "final"}:
        folder.mkdir(parents=True, exist_ok=True)

    return folder


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


def get_validation_dir() -> Path:
    """
    Return the base path where validation curve plots are stored.

    Returns:
        Path: Full path to the validation curves directory.
    """

    return VALID_DIR


def create_model_dir(model_name: str) -> Path:
    """
    Build a portable path to the validation curve folder for a given model.

    Args:
        model_name (str): Name of the model (e.g., "Random Forest").

    Returns:
        Path: Full path to the model's validation curve directory.
    """
    safe_name = model_name.replace(" ", "_")
    folder = VALID_DIR / safe_name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def get_output_dir() -> Path:
    return OUTPUT_DIR
