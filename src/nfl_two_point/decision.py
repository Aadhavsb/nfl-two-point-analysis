from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DecisionResult:
    recommendation: str
    go_ewp: float
    kick_ewp: float
    delta_ewp_pp: float
    break_even_two_point_rate: float
    interval_pp: tuple[float, float] | None = None


def recommend(
    win_probability: Callable[[int], float],
    margin_after_touchdown: int,
    p_two: float,
    p_pat: float,
    margin_threshold_pp: float = 0.10,
    p_two_draws: np.ndarray | None = None,
    p_pat_draws: np.ndarray | None = None,
) -> DecisionResult:
    """Value the two actions now, varying only post-try score margin."""
    wp0, wp1, wp2 = (win_probability(margin_after_touchdown + points) for points in (0, 1, 2))
    go = p_two * wp2 + (1 - p_two) * wp0
    kick = p_pat * wp1 + (1 - p_pat) * wp0
    delta = go - kick
    denominator = wp2 - wp0
    break_even = float("nan") if denominator <= 0 else (kick - wp0) / denominator
    interval = None
    if p_two_draws is not None and p_pat_draws is not None:
        p_two_draws, p_pat_draws = np.asarray(p_two_draws), np.asarray(p_pat_draws)
        deltas = (p_two_draws * wp2 + (1 - p_two_draws) * wp0) - (
            p_pat_draws * wp1 + (1 - p_pat_draws) * wp0
        )
        interval = tuple(float(value * 100) for value in np.quantile(deltas, [0.05, 0.95]))
    if (
        interval is not None
        and interval[0] <= 0 <= interval[1]
        or abs(delta * 100) < margin_threshold_pp
    ):
        label = "indeterminate"
    else:
        label = "go_for_two" if delta > 0 else "kick"
    return DecisionResult(label, go, kick, delta * 100, break_even, interval)
