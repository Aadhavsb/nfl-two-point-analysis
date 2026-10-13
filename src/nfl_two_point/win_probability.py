from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

FEATURES = ["score_margin", "seconds_remaining", "is_home"]
FORBIDDEN_FEATURES = {"wp", "home_wp", "away_wp", "wpa", "final_score", "result", "spread_line"}


@dataclass
class WinProbabilityModel:
    pipeline: Pipeline

    def predict(self, states: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(states[FEATURES])[:, 1]


def assert_no_leakage(features: list[str]) -> None:
    bad = set(features) & FORBIDDEN_FEATURES
    if bad:
        raise ValueError(f"Leaky features prohibited: {sorted(bad)}")


def fit_win_probability(states: pd.DataFrame) -> WinProbabilityModel:
    """Fit an interpretable state-only model of eventual team win probability."""
    assert_no_leakage(FEATURES)
    required = set(FEATURES + ["won"])
    if missing := required - set(states.columns):
        raise ValueError(f"Missing training fields: {sorted(missing)}")
    preprocessor = ColumnTransformer(
        [
            (
                "score",
                Pipeline(
                    [
                        ("spline", SplineTransformer(n_knots=7, degree=3)),
                        ("scale", StandardScaler()),
                    ]
                ),
                ["score_margin"],
            ),
            (
                "time",
                Pipeline(
                    [
                        ("spline", SplineTransformer(n_knots=7, degree=3)),
                        ("scale", StandardScaler()),
                    ]
                ),
                ["seconds_remaining"],
            ),
            ("home", "passthrough", ["is_home"]),
        ]
    )
    pipeline = Pipeline(
        [("features", preprocessor), ("model", LogisticRegression(C=1.0, max_iter=1000))]
    )
    pipeline.fit(states[FEATURES], states.won.astype(int))
    return WinProbabilityModel(pipeline)


def metrics(model: WinProbabilityModel, states: pd.DataFrame) -> dict[str, float]:
    probabilities = model.predict(states)
    return {
        "brier": float(brier_score_loss(states.won, probabilities)),
        "log_loss": float(log_loss(states.won, probabilities, labels=[0, 1])),
    }


def game_states(pbp: pd.DataFrame) -> pd.DataFrame:
    """Build non-leaky pre-play states using nflverse's pre-play score differential."""
    needed = {
        "season",
        "game_id",
        "posteam",
        "posteam_type",
        "score_differential",
        "game_seconds_remaining",
        "home_team",
        "away_team",
        "total_home_score",
        "total_away_score",
    }
    if missing := needed - set(pbp.columns):
        raise ValueError(f"PBP data is missing required columns: {sorted(missing)}")
    rows = pbp.loc[(pbp.qtr <= 4) & pbp.posteam.notna() & pbp.score_differential.notna()].copy()
    finals = rows.groupby("game_id")[["total_home_score", "total_away_score"]].max()
    rows = rows.join(finals, on="game_id", rsuffix="_final")
    rows["is_home"] = (rows.posteam_type == "home").astype(int)
    rows["score_margin"] = rows.score_differential.astype(float)
    rows["seconds_remaining"] = rows.game_seconds_remaining.astype(float)
    rows["won"] = np.where(
        rows.is_home == 1,
        rows.total_home_score_final > rows.total_away_score_final,
        rows.total_away_score_final > rows.total_home_score_final,
    )
    return rows[["season", "game_id", *FEATURES, "won"]].dropna()
