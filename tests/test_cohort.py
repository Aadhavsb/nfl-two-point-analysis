import pandas as pd

from nfl_two_point.cohort import build_conversion_cohort


def _row(**overrides):
    row = {
        "season": 2024,
        "game_id": "x",
        "play_id": 1,
        "qtr": 4,
        "home_team": "H",
        "away_team": "A",
        "posteam": "H",
        "total_home_score": 21,
        "total_away_score": 14,
        "extra_point_attempt": 1,
        "two_point_attempt": 0,
        "extra_point_result": "good",
        "two_point_conv_result": None,
        "game_seconds_remaining": 100,
    }
    row.update(overrides)
    return row


def test_made_pat_reconstructs_touchdown_only_margin() -> None:
    cohort = build_conversion_cohort(pd.DataFrame([_row()]))
    assert cohort.iloc[0].margin_after_td == 6


def test_made_two_point_reconstructs_touchdown_only_margin_for_away_team() -> None:
    cohort = build_conversion_cohort(
        pd.DataFrame(
            [
                _row(
                    posteam="A",
                    total_home_score=14,
                    total_away_score=22,
                    extra_point_attempt=0,
                    two_point_attempt=1,
                    extra_point_result=None,
                    two_point_conv_result="success",
                )
            ]
        )
    )
    assert cohort.iloc[0].margin_after_td == 6
