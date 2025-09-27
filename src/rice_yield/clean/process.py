"""
`all_process.py`

Batch processing script that reads all raw files in the `raw_files/` directory,
cleans them using functions from `common.py`, and writes the cleaned versions
to disk in the `processed_files/` directory.

- Monthly value files are averaged annually before saving.
- Other files undergo standard column cleanup and renaming.
"""


from pathlib import Path
from rice_yield.utils import dataframe_utils as du
from rice_yield.utils.paths import get_data_dir, get_data_file


def process_file(file_path: Path, output_dir: Path) -> None:
    """
    Processes a single file and save the cleaned version

    Args:
        `file_path` (`Path`) - Input file path
        `output_dir` (`Path`) - Directory for the file to be saved
    Returns:
        `None`
    """
    monthly_data_files = ["precipitation",
                          "minimum_temperature",
                          "maximum_temperature",
                          "water_deficit",
                          "actual_evapotranspiration",
                          "potential_evapotranspiration",
                          "rainfall"]
    df = du.read_csv(file_path)
    df = du.drop_location_id(df)
    df = du.standardize_district_names(df)
    df.columns = du.map_columns(df)

    if file_path.stem in monthly_data_files:
        df = du.annual_average(df, "average_" + file_path.stem)

    write_path = get_data_file(output_dir, file_path.parts[-1])

    du.write_csv(df, write_path)


def process_all_files() -> None:
    """
    Processes all raw files in `data\raw_files`

    Args:
        None
    Returns:
        `None`
    """
    raw_dir = get_data_dir("raw")
    cleaned_dir = get_data_dir("cleaned")

    for file_path in raw_dir.glob("*.csv"):
        process_file(file_path, cleaned_dir)


if __name__ == "__main__":
    process_all_files()
