import pytest

from nfl_two_point.decision import recommend


def test_expected_win_probability_and_break_even_rate() -> None:
    # A deliberately simple monotonic score-to-WP fixture.
    result = recommend(lambda margin: {0: 0.50, 1: 0.55, 2: 0.60}[margin], 0, 0.50, 0.95)
    assert result.go_ewp == pytest.approx(0.55)
    assert result.kick_ewp == pytest.approx(0.5475)
    assert result.delta_ewp_pp == pytest.approx(0.25)
    assert result.break_even_two_point_rate == pytest.approx(0.475)
    assert result.recommendation == "go_for_two"


def test_interval_that_crosses_zero_is_indeterminate() -> None:
    result = recommend(
        lambda margin: {0: 0.50, 1: 0.55, 2: 0.60}[margin],
        0,
        0.48,
        0.95,
        p_two_draws=[0.40, 0.60],
        p_pat_draws=[0.95, 0.95],
    )
    assert result.recommendation == "indeterminate"
