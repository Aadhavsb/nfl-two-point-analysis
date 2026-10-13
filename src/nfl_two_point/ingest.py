from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import nflreadpy as nfl
import polars as pl

from .config import AnalysisConfig


def load_play_by_play(config: AnalysisConfig) -> pl.DataFrame:
    """Fetch completed-season nflverse play-by-play data as a Polars frame."""
    seasons = list(range(config.start_season, config.end_season + 1))
    frame = nfl.load_pbp(seasons)
    if not config.include_playoffs:
        frame = frame.filter(pl.col("season_type") == "REG")
    return frame


def write_manifest(frame: pl.DataFrame, config: AnalysisConfig, destination: str | Path) -> None:
    """Record provenance without committing downloaded play-by-play files."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    counts = frame.group_by("season").len().sort("season").to_dicts()
    schema = [(name, str(dtype)) for name, dtype in frame.schema.items()]
    payload = {
        "source": "nflverse play-by-play via nflreadpy",
        "downloaded_at": datetime.now(UTC).isoformat(),
        "seasons": list(range(config.start_season, config.end_season + 1)),
        "row_counts": counts,
        "schema_sha256": hashlib.sha256(repr(schema).encode()).hexdigest(),
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n")
