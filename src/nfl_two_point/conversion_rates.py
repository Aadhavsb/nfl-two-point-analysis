from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BetaRate:
    successes: int
    attempts: int
    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def sample(self, draws: int, rng: np.random.Generator) -> np.ndarray:
        return rng.beta(self.alpha, self.beta, size=draws)


def estimate_rates(cohort: pd.DataFrame, cutover: int) -> dict[tuple[str, str], BetaRate]:
    """Estimate weakly smoothed action success rates by PAT-rule era."""
    rates: dict[tuple[str, str], BetaRate] = {}
    for era, subset in cohort.assign(era=np.where(cohort.season < cutover, "old", "new")).groupby(
        "era"
    ):
        for action in ("two_point", "kick"):
            tries = subset.loc[subset.action == action, "success"]
            successes, attempts = int(tries.sum()), len(tries)
            rates[(era, action)] = BetaRate(
                successes, attempts, 1 + successes, 1 + attempts - successes
            )
    return rates
