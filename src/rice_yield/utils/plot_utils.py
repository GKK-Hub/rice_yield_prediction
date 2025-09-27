# """
# plot_utils.py
# -------------
# Reusable plotting functions for data exploration and analysis.
# """

# Standard library imports
import warnings
from typing import Tuple

# Third-party library imports
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from htmltools import Tag, TagList, tags
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from PIL import Image
from shiny.express import input, render, ui
from sklearn.metrics import PredictionErrorDisplay
from sklearn.model_selection import BaseCrossValidator, ValidationCurveDisplay
from sklearn.pipeline import Pipeline

# Local/relative imports
from .notebook_utils import get_model_names, get_param_names, get_plot_path
from .paths import get_validation_dir

warnings.filterwarnings('ignore')


def show_correlation_heatmap(corr_df: pd.DataFrame) -> render.plot:
    @render.plot(width=900, height=700)  # type: ignore
    def heatmap_plot() -> None:

        plt.style.use('fivethirtyeight')

        fig, ax = plt.subplots(1)
        # sky_blue_cmap = sns.light_palette('deepskyblue',
        # as_cmap=True, reverse=True)
        colors = colors = ['deepskyblue', 'white', 'deepskyblue']
        cmap = mcolors.LinearSegmentedColormap.from_list('custom_deepskyblue',
                                                         colors,
                                                         N=100)
        norm = mcolors.CenteredNorm()
        # Plot the heatmap
        sns.heatmap(corr_df,
                    annot=True,
                    cmap=cmap,
                    vmin=-1,
                    vmax=1,
                    fmt='.1f',
                    mask=np.triu(np.ones_like(corr_df, dtype=bool)),
                    norm=norm
                    )
        plt.title('Heatmap of Correlation Coefficient Matrix',
                  fontdict={'fontsize': 14,
                            'fontweight': 'bold',
                            'family': 'Arial'})
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=11)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=11)
        ax.xaxis.grid(False)
        ax.yaxis.grid(False)
    return heatmap_plot


def show_distribution(df: pd.DataFrame) -> tuple[Tag, render.plot, render.ui]:
    # Define variables and remarks
    dist_variables = [
                    'dist_name',
                    'act_etranspiration',
                    'pot_etranspiration',
                    'area',
                    'production',
                    'yield',
                    'irrigated_area',
                    'max_temperature',
                    'min_temperature',
                    'precipitation',
                    'water_deficit',
                    'rainfall'
    ]
    dist_remarks = {"dist_name": (
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
    # Create UI elements
    user_input = ui.input_select("dist_var",
                                 "Choose Variable",
                                 choices=dist_variables
                                 )

    # Plot renderer
    @render.plot(width=900, height=400)  # type: ignore
    def dist_plot() -> None:
        col = input.dist_var()
        if col != 'dist_name':
            fig, ax = plt.subplots(1, 2)
            plt.style.use('fivethirtyeight')
            # Axes 1: original distribution
            sns.distplot(df[col], ax=ax[0])
            ax[0].set_xlabel(col, fontsize=11)
            ax[0].set_ylabel("Density", fontsize=11)
            # Axes 2: box plot of the original distribution
            sns.boxplot(df[col], ax=ax[1])
            ax[1].set_ylabel(col, fontsize=11)
            fig.suptitle(f"Distribution of {col}",
                         fontsize=14,
                         fontweight='bold',
                         family='Arial')
        else:
            plt.style.use('fivethirtyeight')
            fig, ax = plt.subplots(1)
            df[col].value_counts().plot(kind='bar')
            ax.set_ylim(0, 27)
            ax.set_yticks(range(1, 27, 5))
            ax.set_xlabel('')  # Hide the x-axis label
            ax.set_ylabel('Year Count', fontsize=11)
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=11)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=11)
            ax.xaxis.grid(False)
            fig.suptitle(f"Distribution of {col}",
                         fontsize=14,
                         fontweight='bold',
                         family='Arial')
        return

    # Remarks renderer
    @render.ui  # type: ignore
    def dist_remarks_ui() -> TagList:
        col = input.dist_var()
        return TagList(
            tags.p("Remarks for ",
                   tags.code(col),
                   ": ",
                   tags.strong(dist_remarks.get(col)))
        )
    return user_input, dist_plot, dist_remarks_ui


def show_splits(df: pd.DataFrame) -> tuple[Tag, render.plot]:
    plot_columns = [
                "temperature",
                "precipitation",
                "act_etranspiration",
                "pot_etranspiration",
                "yield",
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
        plt.title('Train-Test split', fontdict={'fontsize': 14,
                                                'fontweight': 'bold',
                                                'family': 'Arial'})
        plt.xlabel('year')
        plt.legend(
                bbox_to_anchor=(1, 1),
                loc="upper left",
                ncol=1,
                fontsize=12)
        ax.set_xlabel(ax.get_xlabel(), fontsize=11)
        ax.set_ylabel(ax.get_ylabel(), fontsize=11)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=11)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=11)
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
    fig, ax = plt.subplots(figsize=(15, 10))

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
                'fontsize': 14,
                'fontweight': 'bold',
                'family': 'Arial'
            })
    ax.set_xlabel(param_name.split('__')[-1], fontsize=11)
    ax.set_ylabel(ax.get_ylabel(), fontsize=11)

    # Adjust tick label fonts and disable gridlines
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=11)
    # ax.xaxis.grid(False)
    # ax.yaxis.grid(False)
    ax.legend(
                bbox_to_anchor=(1, 1),
                loc="upper left",
                ncol=1,
                fontsize=11)

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
        plot_path = get_plot_path(model_folder(), input.param())
        if not plot_path.exists():
            return None
        fig, ax = plt.subplots(figsize=(10, 20))
        plot = Image.open(plot_path)
        ax.imshow(plot)
        ax.axis('off')  # Hide axes for image display
        ax.set_title(f"{input.model()} — {input.param()}",
                     fontdict={'fontsize': 16,
                               'fontweight': 'bold',
                               'family': 'Arial'})
        plot.close()
    return dropdown_model, dropdown_param, show_plot


def plot_prediction_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    figsize: Tuple[int, int] = (10, 6)
) -> Tuple[Figure, Axes]:
    """
    Wrapper for sklearn's PredictionErrorDisplay with 'fivethirtyeight' style.

    Args:
        y_true (np.ndarray): Actual values.
        y_pred (np.ndarray): Predicted values.
        model_name (str): Name of the model for the title.
        figsize (tuple): Size of the figure.

    Returns:
        Tuple[plt.Figure, plt.Axes]: The matplotlib figure and axis objects.
    """
    plt.style.use('fivethirtyeight')
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    # Create the PredictionErrorDisplay
    _ = PredictionErrorDisplay.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        ax=axs[0],
        kind='actual_vs_predicted'
    )
    axs[0].set_title("Actual vs Predicted",
                     fontsize=14,
                     fontweight='bold',
                     fontfamily='Arial')
    axs[0].set_xlabel(axs[0].get_xlabel(), fontsize=11)
    axs[0].set_ylabel(axs[0].get_ylabel(), fontsize=11)
    axs[0].set_xticklabels(axs[0].get_xticklabels(), fontsize=11)
    axs[0].set_yticklabels(axs[0].get_yticklabels(), fontsize=11)

    _ = PredictionErrorDisplay.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        ax=axs[1],
        kind='residual_vs_predicted'
    )
    axs[1].set_title("Residual vs Predicted",
                     fontsize=14,
                     fontweight='bold',
                     fontfamily='Arial')
    axs[1].set_xlabel(axs[1].get_xlabel(), fontsize=11)
    axs[1].set_ylabel(axs[1].get_ylabel(), fontsize=11)
    axs[1].set_xticklabels(axs[1].get_xticklabels(), fontsize=11)
    axs[1].set_yticklabels(axs[1].get_yticklabels(), fontsize=11)
    return fig, axs


def plot_train_test_scores(
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    model_name: str,
    scoring_name: str = "RMSE",
    figsize: Tuple[int, int] = (8, 6)
) -> Tuple[Figure, Axes]:
    """
    Plot train vs test scores with standard deviation as error bars.

    Args:
        train_scores (np.ndarray): Array of train scores from cross_validate.
        test_scores (np.ndarray): Array of test scores from cross_validate.
        model_name (str): Name of the model for the title.
        scoring_name (str): Name of the metric for the y-axis.
        figsize (tuple): Size of the figure.

    Returns:
        Tuple[plt.Figure, plt.Axes]: Matplotlib figure and axis objects.
    """
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots(figsize=figsize)
    # Prepare data
    means = [np.mean(train_scores), np.mean(test_scores)]
    stds = [np.std(train_scores), np.std(test_scores)]
    labels = ['Train', 'Test']
    # Bar positions
    x_pos = np.arange(len(labels))
    # Plot bars with error bars
    ax.bar(x_pos, means, yerr=stds, capsize=5, color=['#1f77b4',
                                                      '#ff69b4'])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(scoring_name, fontsize=11)
    ax.set_title(f"{model_name}: Train vs Test {scoring_name}",
                 fontdict={'fontsize': 14,
                           'fontweight': 'bold',
                           'family': 'Arial'})
    return fig, ax


def train_test_plot(df: pd.DataFrame, feature: str) -> Tuple:
    plt.style.use('fivethirtyeight')

    fig, ax = plt.subplots(1)
    sns.scatterplot(data=df,
                    x='year',
                    y=feature,
                    hue='split',
                    ax=ax)
    ax.set_title('Train-Test split',
                 fontdict={
                     'fontsize': 14,
                     'fontweight': 'bold',
                     'family': 'Arial'})
    ax.set_xlabel('year')
    ax.legend(
            bbox_to_anchor=(1, 1),
            loc="upper left",
            ncol=1,
            fontsize=12)
    ax.set_xlabel(ax.get_xlabel(), fontsize=11)
    ax.set_ylabel(ax.get_ylabel(), fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=11)
    return fig, ax
