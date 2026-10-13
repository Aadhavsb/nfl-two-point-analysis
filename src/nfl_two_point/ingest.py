from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import nflreadpy as nfl
import polars as pl

from .config import AnalysisConfig


def load_play_by_play(config: AnalysisConfig, attempts: int = 3) -> pl.DataFrame:
    """Fetch seasons individually so transient upstream failures can be retried."""
    frames: list[pl.DataFrame] = []
    for season in range(config.start_season, config.end_season + 1):
        for attempt in range(1, attempts + 1):
            try:
                print(f"Loading {season} (attempt {attempt}/{attempts})...", flush=True)
                frames.append(nfl.load_pbp(season))
                break
            except ConnectionError:
                if attempt == attempts:
                    raise
                time.sleep(2**attempt)
    # nflverse's parquet schema evolves across seasons; safely promote compatible types.
    frame = pl.concat(frames, how="diagonal_relaxed")
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
