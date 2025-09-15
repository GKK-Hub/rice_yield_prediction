# """
# plot_utils.py
# -------------
# Reusable plotting functions for data exploration and analysis.
# """

import pandas as pd
import warnings
# import numpy as np
# from IPython.display import display, HTML
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from shiny.express import ui, render, input
from htmltools import Tag
from sklearn.model_selection import ValidationCurveDisplay, BaseCrossValidator
from sklearn.pipeline import Pipeline
from .paths import get_validation_dir
from .notebook_utils import get_model_names, get_param_names, get_plot_path

warnings.filterwarnings('ignore')
# import matplotlib.colors as mcolors


def show_splits(df: pd.DataFrame) -> tuple[Tag, render.plot]:
    plot_columns = [
                "max_temperature",
                "min_temperature",
                "precipitation",
                "act_etranspiration",
                "pot_etranspiration",
                "yield",
                "production",
                "water_deficit",
                "rainfall",
                "yield",
                "area"
            ]
    user_input = ui.input_select('feature',
                                 'Choose a feature',
                                 choices=plot_columns)

    @render.plot(width=900, height=400)  # type: ignore
    def train_test_plot() -> None:
        plt.style.use('fivethirtyeight')

        fig, ax = plt.subplots(1)
        sns.scatterplot(data=df,
                        x='year',
                        y=input.feature(),
                        hue='split',
                        ax=ax)
        plt.title('Train test split', fontdict={'fontsize': 13,
                                                'fontweight': 'bold',
                                                'family': 'Arial'})
        plt.xlabel('Year')
        plt.legend(
                bbox_to_anchor=(1, 1),
                loc="upper left",
                ncol=2,
                fontsize=12)
        ax.set_xlabel(ax.get_xlabel(), fontsize=11)
        ax.set_ylabel(ax.get_ylabel(), fontsize=11)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
    return user_input, train_test_plot


def validation_curve_display(estimator: Pipeline,
                             X: pd.DataFrame,
                             y: pd.Series,
                             param_name: str,
                             param_range: list,
                             cv: BaseCrossValidator,
                             scoring: str,
                             negate_score: bool = True):
    """
    Create and style a validation curve plot using
    ValidationCurveDisplay.from_estimator.

    Args:
        estimator: The estimator object implementing 'fit'.
        X: Training data features.
        y: Target values.
        param_name (str): Name of the parameter to vary (e.g., 'svr__C').
        param_range (array-like): Values of the parameter to evaluate.
        cv: Cross-validation strategy.
        scoring (str): Scoring metric.
        negate_score (bool): Whether to negate the scoring metric.

    Returns:
        fig, ax: The matplotlib figure and axis objects for further
        customization or saving.
    """
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots(figsize=(18, 8))

    # Create the validation curve plot
    _ = ValidationCurveDisplay.from_estimator(
        estimator=estimator,
        X=X,
        y=y,
        param_name=param_name,
        param_range=param_range,
        cv=cv,
        scoring=scoring,
        negate_score=negate_score,
        ax=ax
    )

    # Customize the plot appearance
    ax.set_title(f"Validation Curve for {param_name.split('__')[-1]}",
                 fontdict={
                'fontsize': 12,
                'fontweight': 'bold',
                'family': 'Arial'
            })
    ax.set_xlabel(param_name.split('__')[-1], fontsize=11)
    ax.set_ylabel(ax.get_ylabel(), fontsize=11)

    # Adjust tick label fonts and disable gridlines
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
    # ax.xaxis.grid(False)
    # ax.yaxis.grid(False)

    return fig, ax


def show_validation_curves():
    base_path = get_validation_dir()
    model_names = get_model_names(base_path)
    default_model = model_names[0] if model_names else None

    @render.ui
    def dropdown_model():
        return ui.input_select(id='model',
                               label='Select Model',
                               choices=model_names,
                               selected=default_model)

    def model_folder():
        return base_path / input.model()

    def available_params():
        model = model_folder()
        return get_param_names(model)

    @render.ui
    def dropdown_param():
        params = available_params()
        default_param = params[0] if params else None
        return ui.input_select(id='param',
                               label='Select Hyperparameter',
                               choices=params,
                               selected=default_param)

    @render.plot(width=800, height=500)  # type: ignore
    def show_plot():
        # req(input.model(), cancel_output=True)
        # req(input.param(), cancel_output=True)
        # req(input.param())
        plot_path = get_plot_path(model_folder(), input.param())
        if not plot_path.exists():
            return None
        fig, ax = plt.subplots(figsize=(10, 20))
        plot = Image.open(plot_path)
        ax.imshow(plot)
        ax.axis('off')  # Hide axes for image display
        ax.set_title(f"{input.model()} — {input.param()}")
        plot.close()
    return dropdown_model, dropdown_param, show_plot
