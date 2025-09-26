"""
notebook_utils.py
-----------------
Helper functions used in notebooks for data exploration and visualization.
"""

import pandas as pd
from pathlib import Path
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import seaborn as sns
from shiny.express import ui, render, input
from htmltools import TagList, tags, Tag

from sklearn.pipeline import Pipeline
import joblib
import json


def render_print(text: str) -> ui.HTML:
    """Render text in a Shiny UI."""
    return ui.markdown(text)


def render_table(df: pd.DataFrame, max_height: str = "600px") -> None:
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


def correlation_with_target(df: pd.DataFrame) -> tuple[Tag,
                                                       render.plot,
                                                       render.ui]:
    corr_variables = [
                    'x',
                    'act_etranspiration',
                    'pot_etranspiration',
                    'area',
                    'production',
                    'irrigated_area',
                    'max_temperature',
                    'min_temperature',
                    'precipitation',
                    'water_deficit',
                    'rainfall'
    ]
    corr_remarks = {"dist_name": (
        "Most of the districts have 26 years' data, except Nagapattinam."),
        "act_etranspiration": (
        "Almost normally distributed with no outliers or skewness."),
        "area": (
        "The data is approximately normal with right skew."),
        "production": (
        "The data is approximately normal with right skew."),
        "yield": (
        "The `distribution is approximately normal with slight left skew."),
        "irrigated_area": (
        "The data is approximately normal with right skew."),
        "max_temperature": (
            "Distribution is mostly symmetric with left skew due to outliers."
            ),
        "min_temperature": "Almost same as max_temperature",
        "precipitation": (
        "The distribution is approximately normal with slight right skew."),
        "water_deficit": (
        "Almost normally distributed with no outliers or skewness."),
        "rainfall": "Almost normally distributed with outliers.",
        "pot_etranspiration": (
            "Distribution is approximately normal with slight left skew."
            )}
    user_input = ui.input_select("corr_var",
                                 "Choose Variable",
                                 choices=corr_variables)

    @render.plot(width=900, height=400)  # type: ignore
    def corr_plot() -> None:
        col = input.corr_var()

        corr = df['yield'].corr(df[col])
        # print(f'Correlation: {corr:.2f}')

        # 2. Scatter plot with regression line
        plt.style.use('fivethirtyeight')

        fig, ax = plt.subplots(1)
        sns.regplot(x=col, y='yield', data=df, ax=ax)
        plt.title(f'Scatter Plot (Correlation: {corr:.2f})',
                  fontdict={'fontsize': 12,
                            'fontweight': 'bold',
                            'family': 'Arial'})
        ax.set_xlabel(ax.get_xlabel(), fontsize=11)
        ax.set_ylabel(ax.get_ylabel(), fontsize=11)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)

    @render.ui  # type: ignore
    def corr_remarks_ui() -> TagList:
        col = input.corr_var()
        return TagList(tags.p("Remarks for ", tags.code(col),
                              ": ",
                              tags.strong(corr_remarks.get(col))))
    return user_input, corr_plot, corr_remarks_ui


# def get_base_path() -> Path:
#     """
#     Return the base path where validation curve plots are stored.
#     Adjust this if needed to read from environment or config.
#     """
#     return Path("outputs/validation_curves")
# *** Insead of this, we'll call get_validation_curve_dir from paths.py ***


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


# def get_model_folder(base_path: Path, model_name: str) -> Path:
#     """
#     Return the folder path for the selected model.

#     Args:
#         base_path (Path): Base directory.
#         model_name (str): Model name folder.

#     Returns:
#         Path object to the model’s folder.
#     """
#     return base_path / model_name


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
    with open(file_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    return file_path
