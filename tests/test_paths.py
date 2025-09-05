import pytest
from rice_yield.utils.paths import get_data_path, RAW_DIR


def test_get_data_path_raw() -> None:
    path = get_data_path("raw", "precipitation.csv")
    assert str(path).endswith("data\\raw\\precipitation.csv")
    assert path.parent == RAW_DIR


# def test_get_data_path_processed():
#     path = get_data_path("processed", "cleaned_precipitation.csv")
#     assert str(path).endswith("data/processed/cleaned_precipitaion.csv")
#     assert path.parent == PROCESSED_DIR

# def test_get_data_path_final():
#     path = get_data_path("final", "rice_yield.csv")
#     assert str(path).endswith("data/final/rice_yield.csv")
#     assert path.parent == FINAL_DIR


def test_get_data_path_invalid_subfolder() -> None:
    with pytest.raises(ValueError, match="Invalid subfolder 'invalid'"):
        get_data_path("invalid", "file.csv")
