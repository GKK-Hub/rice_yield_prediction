import pytest
from pathlib import Path
from rice_yield.utils.paths import get_data_dir, get_data_file


def test_get_data_file_valid() -> None:
    path = get_data_file(get_data_dir("raw"), "example.csv")
    assert isinstance(path, Path)
    assert path.parts[-3:] == ("data", "raw", "example.csv")


def test_get_data_dir_valid() -> None:
    path = get_data_dir("cleaned")
    assert isinstance(path, Path)
    assert path.parts[-2:] == ("data", "cleaned")


def test_invalid_subfolder_path() -> None:
    with pytest.raises(ValueError):
        get_data_file(get_data_dir("invalid"), "file.csv")


def test_invalid_subfolder_dir() -> None:
    with pytest.raises(ValueError):
        get_data_dir("invalid")
