import pandas as pd
import pytest

from nfl_two_point.validation import temporal_split
from nfl_two_point.win_probability import assert_no_leakage


def test_temporal_split_is_disjoint_and_ordered() -> None:
    states = pd.DataFrame({"season": [2019, 2020, 2023], "x": [1, 2, 3]})
    split = temporal_split(states, 2019, 2022)
    assert split["development"].season.tolist() == [2019]
    assert split["validation"].season.tolist() == [2020]
    assert split["test"].season.tolist() == [2023]


def test_leaky_feature_is_rejected() -> None:
    with pytest.raises(ValueError, match="Leaky"):
        assert_no_leakage(["score_margin", "wp"])
