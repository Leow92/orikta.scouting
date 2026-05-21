"""
Unit tests for utils/percentile_engine.py — pure functions only, no API calls.
"""
from __future__ import annotations
import math
import pytest
import pandas as pd

from utils.percentile_engine import (
    _safe,
    _is_cup_by_name,
    stats_entry_to_per90,
    compute_percentiles,
    build_season_comparison,
)


# Helpers — minimal API-football player objects
def _player(stats: list) -> dict:
    return {"player": {"id": 1, "name": "X", "position": "Midfielder"}, "statistics": stats}


def _stat(league_id: int, minutes: int, **extra) -> dict:
    base = {"league": {"id": league_id, "name": f"League {league_id}"}, "games": {"minutes": minutes}}
    base.update(extra)
    return base


# ===========================================================================
# _safe
# ===========================================================================
class TestSafe:
    def test_int_to_float(self):
        assert _safe(5) == 5.0

    def test_float_unchanged(self):
        assert _safe(3.14) == pytest.approx(3.14)

    def test_numeric_string(self):
        assert _safe("42") == 42.0

    def test_none_returns_none(self):
        assert _safe(None) is None

    def test_nan_returns_none(self):
        assert _safe(math.nan) is None

    def test_invalid_string_returns_none(self):
        assert _safe("abc") is None

    def test_zero(self):
        assert _safe(0) == 0.0

    def test_negative(self):
        assert _safe(-1.5) == pytest.approx(-1.5)


# ===========================================================================
# _is_cup_by_name
# ===========================================================================
class TestIsCupByName:
    def test_fa_cup(self):
        assert _is_cup_by_name("FA Cup") is True

    def test_coppa_italia(self):
        assert _is_cup_by_name("Coppa Italia") is True

    def test_dfb_pokal(self):
        assert _is_cup_by_name("DFB-Pokal") is True

    def test_coupe_de_france(self):
        assert _is_cup_by_name("Coupe de France") is True

    def test_copa_del_rey(self):
        assert _is_cup_by_name("Copa del Rey") is True

    def test_taca_de_portugal(self):
        assert _is_cup_by_name("Taça de Portugal") is True

    def test_knvb_beker(self):
        assert _is_cup_by_name("KNVB Beker") is True

    def test_premier_league_is_not_cup(self):
        assert _is_cup_by_name("Premier League") is False

    def test_bundesliga_is_not_cup(self):
        assert _is_cup_by_name("Bundesliga") is False

    def test_serie_a_is_not_cup(self):
        assert _is_cup_by_name("Serie A") is False

    def test_case_insensitive(self):
        assert _is_cup_by_name("fa cup") is True
        assert _is_cup_by_name("FA CUP") is True


# ===========================================================================
# stats_entry_to_per90
# ===========================================================================
def _entry(minutes: int, **sections) -> dict:
    """Build a minimal stats entry with only the given sections."""
    defaults = {
        "games": {"minutes": minutes},
        "goals": {"total": None, "assists": None, "saves": None, "conceded": None},
        "shots": {}, "passes": {}, "tackles": {}, "duels": {}, "dribbles": {}, "fouls": {},
    }
    for k, v in sections.items():
        defaults[k] = v
    return defaults


class TestStatsEntryToPer90:
    def test_zero_minutes_returns_empty(self):
        assert stats_entry_to_per90(_entry(0)) == {}

    def test_none_minutes_returns_empty(self):
        assert stats_entry_to_per90({"games": {"minutes": None}}) == {}

    def test_minutes_key_present(self):
        result = stats_entry_to_per90(_entry(1800, goals={"total": 5, "assists": 3, "saves": None, "conceded": None}))
        assert result["Minutes"] == 1800.0

    def test_goals_per90_correct(self):
        # 900 min = 10 nineties; 10 goals → 1.0 per 90
        result = stats_entry_to_per90(_entry(900, goals={"total": 10, "assists": None, "saves": None, "conceded": None}))
        assert result["Goals per 90"] == pytest.approx(1.0)

    def test_ga_per90_correct(self):
        # 1800 min = 20 nineties; 10 G + 10 A → 1.0 G+A/90
        result = stats_entry_to_per90(_entry(1800, goals={"total": 10, "assists": 10, "saves": None, "conceded": None}))
        assert result["G+A per 90"] == pytest.approx(1.0)

    def test_shot_accuracy_pct(self):
        result = stats_entry_to_per90(_entry(900, shots={"total": 20, "on": 10}))
        assert result["Shot Accuracy %"] == pytest.approx(50.0)

    def test_zero_shots_no_shot_accuracy(self):
        result = stats_entry_to_per90(_entry(900, shots={"total": 0, "on": 0}))
        assert "Shot Accuracy %" not in result

    def test_dribble_success_pct(self):
        result = stats_entry_to_per90(_entry(900, dribbles={"attempts": 20, "success": 15}))
        assert result["Dribble Success %"] == pytest.approx(75.0)

    def test_zero_dribble_attempts_no_success_pct(self):
        result = stats_entry_to_per90(_entry(900, dribbles={"attempts": 0, "success": 0}))
        assert "Dribble Success %" not in result

    def test_duels_won_pct(self):
        result = stats_entry_to_per90(_entry(900, duels={"total": 100, "won": 60}))
        assert result["Duels Won %"] == pytest.approx(60.0)

    def test_zero_duels_no_pct(self):
        result = stats_entry_to_per90(_entry(900, duels={"total": 0, "won": 0}))
        assert "Duels Won %" not in result

    def test_gk_save_pct(self):
        # 80 saves + 20 conceded → 80%
        result = stats_entry_to_per90(_entry(3060, goals={"total": None, "assists": None, "saves": 80, "conceded": 20}))
        assert result["Save %"] == pytest.approx(80.0)

    def test_gk_no_saves_no_save_pct(self):
        result = stats_entry_to_per90(_entry(3060, goals={"total": None, "assists": None, "saves": None, "conceded": 30}))
        assert "Save %" not in result

    def test_pass_completion_is_passed_through(self):
        # pass accuracy is already a percentage — should not be divided by nineties
        result = stats_entry_to_per90(_entry(900, passes={"key": None, "accuracy": 85.0}))
        assert result["Pass Completion %"] == pytest.approx(85.0)

    def test_tackles_per90(self):
        # 900 min = 10 nineties; 18 tackles → 1.8 per 90
        result = stats_entry_to_per90(_entry(900, tackles={"total": 18, "blocks": None, "interceptions": None}))
        assert result["Tackles per 90"] == pytest.approx(1.8)

    def test_fouls_drawn_per90(self):
        # 900 min = 10 nineties; 9 fouls drawn → 0.9 per 90
        result = stats_entry_to_per90(_entry(900, fouls={"committed": None, "drawn": 9}))
        assert result["Fouls Drawn per 90"] == pytest.approx(0.9)


# ===========================================================================
# compute_percentiles
# ===========================================================================
class TestComputePercentiles:
    def _pool(self, values: list[float], minutes: int = 900) -> list[dict]:
        return [{"metric": v, "Minutes": minutes} for v in values]

    def test_top_value_returns_100(self):
        result = compute_percentiles({"metric": 100.0}, self._pool(list(range(1, 10))))
        assert result["metric"] == 100.0

    def test_bottom_value_returns_0(self):
        result = compute_percentiles({"metric": 0.0}, self._pool(list(range(1, 10))))
        assert result["metric"] == 0.0

    def test_small_pool_fallback_50(self):
        """Fewer than 5 qualifying pool players → fallback to 50.0."""
        result = compute_percentiles({"metric": 99.0}, self._pool([1.0, 2.0, 3.0]))
        assert result["metric"] == 50.0

    def test_pool_filtered_by_minutes(self):
        """Players below MIN_POOL_MINUTES (450) are excluded; triggers fallback."""
        low_min_pool = self._pool(list(range(1, 20)), minutes=100)
        result = compute_percentiles({"metric": 10.0}, low_min_pool)
        assert result["metric"] == 50.0

    def test_missing_metric_in_pool_fallback_50(self):
        """Metric absent from pool → fallback to 50.0."""
        result = compute_percentiles({"rare_metric": 5.0}, self._pool(list(range(1, 10))))
        assert result["rare_metric"] == 50.0

    def test_percentile_is_between_0_and_100(self):
        pool = self._pool([float(i) for i in range(1, 100)])
        result = compute_percentiles({"metric": 50.0}, pool)
        assert 0.0 <= result["metric"] <= 100.0


# ===========================================================================
# build_season_comparison
# ===========================================================================
class TestBuildSeasonComparison:
    def _make_player(self, league_id: int, minutes: int, goals: int) -> dict:
        return {
            "player": {"id": 1, "name": "X"},
            "statistics": [{
                "league": {"id": league_id},
                "games": {"minutes": minutes, "position": "Midfielder"},
                "goals": {"total": goals, "assists": 0, "saves": None, "conceded": None},
                "shots": {}, "passes": {}, "tackles": {}, "duels": {}, "dribbles": {}, "fouls": {},
            }],
        }

    def test_returns_two_dicts(self):
        curr = self._make_player(39, 2700, 15)
        prev = self._make_player(39, 2700, 10)
        curr_m, prev_m = build_season_comparison(curr, prev)
        assert isinstance(curr_m, dict) and isinstance(prev_m, dict)

    def test_current_goals_correct(self):
        # 2700 min = 30 nineties; 15 goals → 0.5 per 90
        curr = self._make_player(39, 2700, 15)
        curr_m, _ = build_season_comparison(curr, None)
        assert curr_m["Goals per 90"] == pytest.approx(0.5)

    def test_no_prev_returns_empty_prev(self):
        curr = self._make_player(39, 2700, 15)
        _, prev_m = build_season_comparison(curr, None)
        assert prev_m == {}

    def test_minutes_key_excluded(self):
        curr = self._make_player(39, 2700, 15)
        curr_m, _ = build_season_comparison(curr, None)
        assert "Minutes" not in curr_m
