from __future__ import annotations

import pandas as pd

REQUIRED_TRY_COLUMNS = {
    "season",
    "game_id",
    "play_id",
    "qtr",
    "home_team",
    "away_team",
    "posteam",
    "total_home_score",
    "total_away_score",
    "extra_point_attempt",
    "two_point_attempt",
    "extra_point_result",
    "two_point_conv_result",
    "game_seconds_remaining",
}


def _success(value: object) -> bool:
    return str(value).lower() in {"good", "success"}


def build_conversion_cohort(pbp: pd.DataFrame, regulation_only: bool = True) -> pd.DataFrame:
    """Make one auditable row per actual post-touchdown conversion attempt.

    nflverse score totals are post-play. A made conversion is therefore removed to
    reconstruct the decision-time score, immediately after the touchdown.
    """
    missing = REQUIRED_TRY_COLUMNS - set(pbp.columns)
    if missing:
        raise ValueError(f"PBP data is missing required columns: {sorted(missing)}")
    tries = pbp.loc[(pbp.extra_point_attempt == 1) | (pbp.two_point_attempt == 1)].copy()
    if regulation_only:
        tries = tries.loc[tries.qtr <= 4].copy()
    tries["action"] = tries.apply(
        lambda row: "two_point" if row.two_point_attempt == 1 else "kick", axis=1
    )
    tries["success"] = tries.apply(
        lambda row: (
            _success(row.two_point_conv_result)
            if row.action == "two_point"
            else _success(row.extra_point_result)
        ),
        axis=1,
    )
    # A defensive return/multi-point anomaly is deliberately excluded from v1.
    action_points = tries.apply(
        lambda row: 2 if row.action == "two_point" and row.success else 1 if row.success else 0,
        axis=1,
    )
    is_home = tries.posteam == tries.home_team
    tries["scoring_team_is_home"] = is_home
    tries["margin_after_td"] = tries.total_home_score - tries.total_away_score
    tries.loc[is_home & (action_points > 0), "margin_after_td"] -= action_points[
        is_home & (action_points > 0)
    ]
    tries.loc[~is_home & (action_points > 0), "margin_after_td"] += action_points[
        ~is_home & (action_points > 0)
    ]
    tries.loc[~is_home, "margin_after_td"] *= -1
    columns = [
        "season",
        "game_id",
        "play_id",
        "action",
        "success",
        "margin_after_td",
        "game_seconds_remaining",
        "scoring_team_is_home",
        "home_team",
        "away_team",
        "posteam",
    ]
    return tries[columns].dropna().reset_index(drop=True)
