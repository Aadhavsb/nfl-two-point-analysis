from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class AnalysisConfig:
    start_season: int
    end_season: int
    include_playoffs: bool
    regulation_only: bool
    pat_era_cutover: int
    random_seed: int
    bootstrap_draws: int
    recommendation_margin_pp: float
    development_end: int
    validation_end: int
    test_end: int


def load_config(path: str | Path) -> AnalysisConfig:
    with Path(path).open() as handle:
        data = yaml.safe_load(handle)
    splits = data["splits"]
    return AnalysisConfig(
        **{key: data[key] for key in AnalysisConfig.__dataclass_fields__ if key in data},
        development_end=splits["development_end"],
        validation_end=splits["validation_end"],
        test_end=splits["test_end"],
    )
