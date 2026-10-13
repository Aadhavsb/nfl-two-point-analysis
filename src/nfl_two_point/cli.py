from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .cohort import build_conversion_cohort
from .config import load_config
from .conversion_rates import estimate_rates
from .ingest import load_play_by_play, write_manifest
from .validation import rolling_origin_metrics, temporal_split
from .win_probability import fit_win_probability, game_states, metrics


def build(config_path: str) -> None:
    config = load_config(config_path)
    pbp = load_play_by_play(config)
    write_manifest(pbp, config, "data/manifests/pbp.json")
    pandas_pbp = pbp.to_pandas()
    cohort = build_conversion_cohort(pandas_pbp, config.regulation_only)
    states = game_states(pandas_pbp)
    splits = temporal_split(states, config.development_end, config.validation_end)
    model = fit_win_probability(splits["development"])
    output = Path("data/processed")
    output.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(output / "conversion_cohort.csv", index=False)
    pd.DataFrame(
        [
            {"split": name, **metrics(model, frame)}
            for name, frame in splits.items()
            if not frame.empty
        ]
    ).to_csv(output / "wp_metrics.csv", index=False)
    rolling_origin_metrics(states, config.validation_end + 1).to_csv(
        output / "rolling_metrics.csv", index=False
    )
    rates = estimate_rates(
        cohort.loc[cohort.season <= config.validation_end], config.pat_era_cutover
    )
    pd.DataFrame(
        [
            {
                "era": era,
                "action": action,
                "attempts": rate.attempts,
                "successes": rate.successes,
                "posterior_mean": rate.mean,
            }
            for (era, action), rate in rates.items()
        ]
    ).to_csv(output / "conversion_rates.csv", index=False)
    print(f"Built point-in-time model artifacts in {output}")


def live_recommendation(args: argparse.Namespace) -> None:
    # This command uses fitted artifacts only after `build`; the notebook provides the full workflow.
    raise SystemExit(
        "Run `nfl-two-point build` first, then use the notebook for interactive state queries."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="NFL point-in-time two-point decision analysis")
    parser.add_argument("command", choices=["build"])
    parser.add_argument("--config", default="config/analysis.yaml")
    args = parser.parse_args()
    if args.command == "build":
        build(args.config)


if __name__ == "__main__":
    main()
