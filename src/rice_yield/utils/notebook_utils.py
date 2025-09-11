"""
notebook_utils.py
-----------------
Helper functions used in notebooks for data exploration and visualization.
"""

import pandas as pd
import numpy as np
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import seaborn as sns
from shiny.express import ui, render, input
from htmltools import TagList, tags, Tag
import matplotlib.colors as mcolors


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns in a DataFrame for brevity or clarity.

    Args:
        df (pd.DataFrame): Input DataFrame.
        rename_dict (dict): Dictionary mapping old column names to new ones.

    Returns:
        pd.DataFrame: DataFrame with renamed columns.
    """
    rename_cols = {
        'rice_area_1000_ha': 'area',
        'rice_production_1000_tons': 'production',
        'rice_yield_kg_per_ha': 'yield',
        'average_actual_evapotranspiration': 'act_etranspiration',
        'average_potential_evapotranspiration': 'pot_etranspiration',
        'rice_irrigated_area_1000_ha': 'irrigated_area',
        'average_precipitation': 'precipitation',
        'average_water_deficit': 'water_deficit',
        'average_maximum_temperature': 'max_temperature',
        'average_minimum_temperature': 'min_temperature',
        'average_rainfall': 'rainfall'
    }
    return df.rename(columns=rename_cols)


def show_output(df: pd.DataFrame, max_height: str = "600px") -> None:
    html = df.to_html(index=True, table_id="scrollable-table")
    css = """
    <style>
    #scrollable-table {
        border-collapse: collapse;
        width: 100%;
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
        "Most of the districts have 26 years' data, except a few."),
        "act_etranspiration": (
        "Almost normally distributed with no outliers or skewness."),
        "area": (
        "The data is approximately normal with right skew."),
        "production": (
        "The data is approximately normal with slight right skew."),
        "yield": (
        "The `rice_yield` variable is approximately normal with left skew."),
        "irrigated_area": (
        "The data is approximately normal with right skew."),
        "max_temperature": "Distribution is bi-modal",
        "min_temperature": "Distribution is bi-modal.",
        "precipitation": (
        "The data is approximately normal with slight right skew."),
        "water_deficit": (
        "Almost normally distributed with no outliers or skewness."),
        "rainfall": "Almost normally distributed",
        "pot_etranspiration": "Almost normally distributed"}
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
        else:
            plt.style.use('fivethirtyeight')
            fig, ax = plt.subplots(1)
            df[col].value_counts().plot(kind='bar')
            ax.set_ylim(0, 27)
            ax.set_yticks(range(1, 27, 5))
            ax.set_xlabel('')  # Hide the x-axis label
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=11)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=11)
            ax.xaxis.grid(False)
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
                  fontdict={'fontsize': 12,
                            'fontweight': 'bold',
                            'family': 'Arial'})
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
        ax.xaxis.grid(False)
        ax.yaxis.grid(False)
    return heatmap_plot


corr_output_type = tuple[Tag, render.plot, render.ui]


def show_correlation_with_target(df: pd.DataFrame) -> corr_output_type:
    # Define variables and remarks
    corr_variables = [
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
    corr_remarks = {
        "act_etranspiration": "No correlation with rice_yield.",
        "pot_etranspiration": "No correlation with rice_yield.",
        "area": "No correlation with rice_yield.",
        "production": "Strong positive correlation with rice_yield.",
        "irrigated_area": "Moderate positive correlation with rice_yield.",
        "max_temperature": "Weak negative correlation with rice_yield.",
        "min_temperature": "Weak negative correlation with rice_yield.",
        "precipitation": "Moderate positive correlation with rice_yield.",
        "water_deficit": "Strong negative correlation with rice_yield.",
        "rainfall": " Moderate positive correlation with rice_yield."
    }
    # Create UI elements
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
        return TagList(tags.p("Remarks for ",
                              tags.code(col), ": ",
                              tags.strong(corr_remarks.get(col))))
    return user_input, corr_plot, corr_remarks_ui
