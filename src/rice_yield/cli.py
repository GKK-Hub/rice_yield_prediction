"""
Command-line interface for Rice Yield project.

Usage examples:
    rice-yield process   # Clean raw CSVs → cleaned/
    rice-yield combine   # Merge cleaned/ → final/rice_yield.csv
"""

import argparse
from rice_yield.clean.process import process_all_files
from rice_yield.clean.combine import combine_cleaned_files
from rice_yield.utils.paths import get_data_dir, get_data_file


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rice-yield",
        description="Rice Yield Prediction Pipeline CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: process
    subparsers.add_parser("process",
                          help="Process raw CSVs into cleaned/")

    # Subcommand: combine
    subparsers.add_parser("combine",
                          help="Combine cleaned CSVs into final dataset")

    args = parser.parse_args()

    if args.command == "process":
        process_all_files()

    elif args.command == "combine":
        input_dir = get_data_dir("cleaned")
        output_file = get_data_file(get_data_dir("final"), "rice_yield.csv")
        combine_cleaned_files(input_dir, output_file)


if __name__ == "__main__":
    main()
