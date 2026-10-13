from __future__ import annotations

import pandas as pd

from .win_probability import fit_win_probability, metrics


def temporal_split(
    states: pd.DataFrame, development_end: int, validation_end: int
) -> dict[str, pd.DataFrame]:
    """Return disjoint chronological datasets; never shuffle seasons."""
    return {
        "development": states.loc[states.season <= development_end].copy(),
        "validation": states.loc[
            (states.season > development_end) & (states.season <= validation_end)
        ].copy(),
        "test": states.loc[states.season > validation_end].copy(),
    }


def rolling_origin_metrics(states: pd.DataFrame, first_test_year: int) -> pd.DataFrame:
    records = []
    for year in sorted(states.season.unique()):
        if year < first_test_year:
            continue
        train, test = states.loc[states.season < year], states.loc[states.season == year]
        if train.empty or test.empty:
            continue
        records.append({"test_season": int(year), **metrics(fit_win_probability(train), test)})
    return pd.DataFrame(records)
