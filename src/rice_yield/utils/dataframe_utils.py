"""
`dataframe_utils.py`

Shared utility functions for the rice yield prediction project.

This module contains general-purpose cleaning functions that are reused across
multiple data processing scripts, such as:
- Reading and saving CSV files
- Dropping unneeded location identifier columns
- Standardizing district names to present-day naming conventions
- Normalizing column names to lowercase, snake_case

These functions ensure consistency and reduce duplication in the preprocessing
pipeline.
"""

import pandas as pd
from pathlib import Path

district_names = {'Chengalpattu MGR Kancheepuram': 'Kancheepuram',
                  'Tiruchirapalli Trichy': 'Tiruchirappalli',
                  'Periyar(Erode)': 'Erode',
                  'Dindigul Anna': 'Dindigul',
                  'Virudhunagar Kamarajar': 'Virudhunagar',
                  'Sivagangai Pasumpon': 'Sivagangai',
                  'Chidambanar Toothukudi': 'Thoothukudi',
                  'North Arcot Vellore': 'Vellore',
                  'South Arcot Cuddalore': 'Cuddalore',
                  'Tiruvarur': 'Thiruvarur',
                  'Thiruppur': 'Tiruppur',
                  'Thiruvannamalai': 'Tiruvannamalai'
                  }


def read_csv(path: Path) -> pd.DataFrame:
    """
        Returns a `DataFrame` object

        Args:
            `path` (`Path`) - Path of the file to be read
        Returns:
            `pd.DataFrame`
    """
    return pd.read_csv(path)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """
        Creates a `csv` file on the `path` by converting a `DataFrame` object

        Args:
            `df` (`pd.DataFrame`)
            `path` (`Path`)
        Returns:
            `None`
    """
    df.to_csv(path, index=False)


def drop_location_id(df: pd.DataFrame) -> pd.DataFrame:
    """
        Drops the location identifiers and returns the `DataFrame` object

        Args:
            `df` (`pd.DataFrame`)
        Returns:
            `pd.DataFrame`
    """
    return df.drop(
                   columns=['Dist Code', 'State Code', 'State Name'],
                   inplace=False
                  )


def standardize_district_names(df: pd.DataFrame) -> pd.DataFrame:
    """
        Standardizes the old district names to present-day names

        Args:
            `df` (`pd.DataFrame`)
        Returns:
            `pd.DataFrame`

    """
    return df.replace(district_names, inplace=False)


def normalize_column_names(column: str) -> str:
    """
        Removes the spaces in column names

        Args:
            `column`: (`str`)
        Returns:
            `str`
    """
    return column.replace(' ', '_').replace('(', '').replace(')', '').lower()


def map_columns(df: pd.DataFrame) -> pd.Index:
    """
        Maps the column names with the function `rename_column`

        Args:
            `df` (`pd.DataFrame`)
        Returns:
            `pd.Index`

    """
    return df.columns.map(lambda col: normalize_column_names(col))


def annual_average(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
        Averages values over a period of 12 months and store the average
        in `column_name` For example, if a dataframe has columns like this:
        'January Precipitation, ..., December Precipitation, this function will
        take average over all the months and produce a single value which is
        stored under `column_name`.

        Args:
            `df` (`pd.DataFrame`)
            `column_name` (`str`)
        Returns:
            `pd.DataFrame`
    """
    monthly_cols = df.columns.difference(['year', 'dist_name'])
    df[column_name] = df[monthly_cols].mean(axis=1).round(2)
    return df[['year', 'dist_name', column_name]]


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


if __name__ == "__main__":
    print(__doc__)
