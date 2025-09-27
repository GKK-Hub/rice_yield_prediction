"""
notebook_utils.py
-----------------
Helper functions used in notebooks for data exploration and visualization.
"""

import json
import joblib
from pathlib import Path

import pandas as pd
from IPython.display import display, HTML
from shiny.express import ui
from sklearn.pipeline import Pipeline


def render_print(text: str) -> ui.HTML:
    """Render text in a Shiny UI."""
    return ui.markdown(text)


def render_table(df: pd.DataFrame, max_height: str = "600px") -> None:
    """
    Renders the pandas dataframe and highlights the rows on mouse hover.
    The regular dataframe object when printed doesn't have a scroll option
    and occupies space

    Args:
        `df` (`pd.DataFrame`)
        `max_height` (`str`) - Maximum height of the window
    Returns:
        `None`
    """
    html = df.to_html(index=True, table_id="scrollable-table")
    css = """
    <style>
    #scrollable-table {
        border-collapse: collapse;
        width: auto;
        margin-left: auto;
        margin-right: auto;
    }
    #scrollable-table th {
        position: sticky;
        top: 0;
        background-color: white;
        z-index: 10;
        border-bottom: 2px solid #ddd;
        padding: 8px;
        text-align: left;
        font-weight: bold;
        text-transform: none;
    }
    #scrollable-table td {
        padding: 8px;
        border-bottom: 1px solid #ddd;
    }
    #scrollable-table tr:hover {
        background-color: #87CEEB !important;  /* Sky blue on hover */
        cursor: pointer;
    }
    /* Ensure header doesn't get hover effect */
    #scrollable-table thead tr:hover {
        background-color: white !important;
        cursor: default;
    }
    </style>
    """
    scrollable = f"""
    {css}
    <div style='max-height: {max_height}; overflow-y: auto; \
        border: 1px solid #ddd;'>
        {html}
    </div>
    """
    display(HTML(scrollable))


def get_model_names(base_path: Path) -> list[str]:
    """
    List all available model names by scanning the base directory.

    Args:
        base_path (Path): Path to validation curves base folder.

    Returns:
        List of model names (folder names).
    """
    if not base_path.exists():
        return []
    return sorted([f.name for f in base_path.iterdir() if f.is_dir()])


def get_param_names(model_folder: Path) -> list[str]:
    """
    List all available parameter plots (without extension) in a model's folder.

    Args:
        model_folder (Path): Path to the model folder.

    Returns:
        List of parameter names.
    """
    if not model_folder.exists():
        return []
    return sorted([f.stem for f in model_folder.glob("*.png")])


def get_plot_path(model_folder: Path, param_name: str) -> Path:
    """
    Return the full path to the plot image file for a given parameter.

    Args:
        model_folder (Path): Path to the model’s folder.
        param_name (str): Parameter name.

    Returns:
        Path to the plot file.
    """
    return model_folder / f"{param_name}.png"


def save_model(model: Pipeline, model_name: str, save_dir: Path) -> Path:
    """
    Save the trained model pipeline to disk.

    Args:
        model (Pipeline): Trained scikit-learn pipeline.
        model_name (str): Name of the model.
        save_dir (Path): Directory where the model will be saved.

    Returns:
        Path: Full path to the saved model file.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / f"{model_name}.joblib"
    joblib.dump(model, file_path)
    return file_path


def load_model(file_path: Path) -> Pipeline:
    """
    Load a model from a file.

    Args:
        file_path (Path): Path to the saved model file.

    Returns:
        Pipeline: The loaded scikit-learn pipeline.
    """
    return joblib.load(file_path)


def save_metrics(metrics: dict, save_dir: Path, model_name: str) -> Path:
    """
    Save metrics as a JSON file.

    Args:
        metrics (dict): Dictionary of metrics.
        save_dir (Path): Directory to save the metrics.
        model_name (str): Name of the model.

    Returns:
        Path: Path to the saved JSON file.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / f"{model_name}_metrics.json"
    with open(file_path, "w") as f:
        json.dump(metrics, f, indent=4)
    return file_path


if __name__ == "__main__":
    print(__doc__)
