# NFL Two-Point Decision Assistant

A transparent, **in-the-moment** model for the question asked immediately after an NFL touchdown:

> Should we kick, go for two, or wait because the evidence is too close to call?

It values actions by expected **win probability**, not expected points and not whether a historical coach’s decision happened to work out. Historical play-by-play is used to estimate state value and execution rates, then to validate forecast calibration on later seasons.

## First-version scope

- 2010–2025 completed seasons; regulation only.
- State: team-relative score margin after the touchdown, time remaining, and home/away orientation.
- Action probabilities: smoothed, league-level two-point/PAT success estimates separated by the 2015 PAT-rule change.
- Output: expected WP for both choices, the break-even two-point rate, uncertainty, and `go_for_two` / `kick` / `indeterminate`.
- Validation: chronological held-out seasons, Brier/log-loss metrics, and rolling-origin tests.

This is **not** causal proof that a past coach would have won by picking the other option. Coaches select attempts non-randomly, and the v1 execution probabilities are intentionally conservative league-level estimates.

## Setup

```bash
cd ~/repos/nfl-two-point-analysis
uv sync --extra dev --extra notebook
uv run pytest
uv run nfl-two-point build --config config/analysis.yaml
```

The build downloads nflverse data with `nflreadpy`, writes provenance to `data/manifests/pbp.json`, and stores ignored derived tables in `data/processed/`.

## Model calculation

For a state with touchdown-only margin `m`, the model evaluates predicted WP at `m`, `m+1`, and `m+2`:

```text
EWP(go)   = p(two) × WP(m + 2) + [1 - p(two)] × WP(m)
EWP(kick) = p(PAT) × WP(m + 1) + [1 - p(PAT)] × WP(m)
```

The report/notebook present a sensitivity range for `p(two)` and return `indeterminate` when uncertainty crosses zero. That makes the recommendation auditable instead of falsely precise.

## Limitations and next steps

V1 does not yet model team/QB/defense-specific conversion skill, remaining timeouts, future possession, weather, or overtime. Those belong in a later version only after the score/time baseline is well calibrated. A later causal-policy study should add propensity/overlap diagnostics and doubly robust evaluation before making claims about policy improvement.
