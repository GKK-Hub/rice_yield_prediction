"""
`combine.py`
Combine all cleaned CSV files into a single dataset.

Workflow:
    - Reads all CSVs from data/cleaned/
    - Merges them on ["year", "dist_name"] using inner join
    - Writes the combined dataset to `data/final/rice_yield.csv`
"""

import pandas as pd
from pathlib import Path

from rice_yield.utils.dataframe_utils import read_csv, write_csv
from rice_yield.utils.paths import get_data_dir, get_data_file


def combine_cleaned_files(input_dir: Path, output_file: Path) -> None:
    """
    Combines all cleaned CSVs into one final dataset.

    Args:
        `input_dir` (`Path`): Directory containing cleaned CSVs.
        `output_file` (`Path`): Path to save the combined dataset.
    Returns:
        `None`
    """
    files = sorted(input_dir.glob("*.csv"))
    if not files:
        print("No CSV files found to combine.")
        return

    combined_df = None

    for file in files:
        df = read_csv(file)

        if combined_df is None:
            combined_df = df
        else:
            combined_df = pd.merge(
                combined_df,
                df,
                how="inner",
                on=["year", "dist_name"],
            )

    if combined_df is not None:
        write_csv(combined_df, output_file)
        print(f"Combined dataset written to {output_file}")
    else:
        print("No dataframes were combined; nothing to write.")


if __name__ == "__main__":
    input_dir = get_data_dir("cleaned")
    output_file = get_data_file(get_data_dir("final"), "combine.csv")
    combine_cleaned_files(input_dir, output_file)
