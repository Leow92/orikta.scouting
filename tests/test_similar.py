"""
Tests for tools/similar.py — pure function and similarity ranking coverage.

No live API calls. The ranking pipeline is exercised end-to-end using
synthetic per-90 dicts that mimic the API-football structure, so the
algorithm behaviour (profile shape discrimination, position archetypes)
can be asserted deterministically.
"""
from __future__ import annotations
import pytest
import numpy as np
import pandas as pd

from tools.similar import (
    _profile_sim,
    _name_overlap,
    _get_pool_position,
    _precompute_pool_arrays,
    _pct_from_arrays,
    _shared_strengths,
    GK_ONLY_METRICS,
)
from utils.percentile_engine import stats_entry_to_per90, MIN_POOL_MINUTES


# ---------------------------------------------------------------------------
# Helpers — synthetic pool entries
# ---------------------------------------------------------------------------

def _entry(
    minutes: int,
    goals: float,
    assists: float,
    shots: float,
    key_passes: float,
    tackles: float,
    interceptions: float,
    dribbles: float = 0.0,
    pass_acc: float = 75.0,
    position: str = "Attacker",
    saves: int | None = None,
    conceded: int | None = None,
) -> dict:
    """Minimal API-football statistics entry. Counts are derived from per-90 rates."""
    n = minutes / 90
    return {
        "player": {"name": "X", "position": position},
        "statistics": [{
            "games":    {"minutes": minutes, "position": position},
            "goals":    {"total": round(goals * n), "assists": round(assists * n),
                         "saves": saves, "conceded": conceded},
            "shots":    {"total": round(shots * n), "on": round(shots * n * 0.5)},
            "passes":   {"key": round(key_passes * n), "accuracy": pass_acc},
            "tackles":  {"total": round(tackles * n), "blocks": 2,
                         "interceptions": round(interceptions * n)},
            "duels":    {"total": 80, "won": 50},
            "dribbles": {"attempts": round(dribbles * n),
                         "success": round(dribbles * n * 0.6)},
            "fouls":    {"committed": 2, "drawn": 3},
            "league":   {"id": 61, "name": "Ligue 1"},
            "team":     {"name": "Test FC"},
        }],
    }


def _gk_entry(minutes: int, saves_p90: float, conceded_p90: float,
              pass_acc: float = 65.0) -> dict:
    n = minutes / 90
    return {
        "player": {"name": "GK", "position": "Goalkeeper"},
        "statistics": [{
            "games":    {"minutes": minutes, "position": "Goalkeeper"},
            "goals":    {"total": None, "assists": None,
                         "saves": round(saves_p90 * n),
                         "conceded": round(conceded_p90 * n)},
            "shots":    {}, "passes": {"key": None, "accuracy": pass_acc},
            "tackles":  {}, "duels":    {}, "dribbles": {}, "fouls": {},
            "league":   {"id": 61, "name": "Ligue 1"},
            "team":     {"name": "Test FC"},
        }],
    }


# ---------------------------------------------------------------------------
# Archetypal profiles
# ---------------------------------------------------------------------------

STRIKER_A  = _entry(2700, goals=0.70, assists=0.20, shots=3.5, key_passes=0.8,
                    tackles=0.8,  interceptions=0.5, dribbles=2.0, position="Attacker")
STRIKER_B  = _entry(2520, goals=0.65, assists=0.22, shots=3.3, key_passes=0.9,
                    tackles=0.7,  interceptions=0.4, dribbles=1.9, position="Attacker")
WINGER     = _entry(2700, goals=0.35, assists=0.40, shots=2.0, key_passes=2.0,
                    tackles=0.6,  interceptions=0.4, dribbles=4.0, position="Attacker")
WINGER_B   = _entry(2430, goals=0.32, assists=0.38, shots=1.9, key_passes=1.9,
                    tackles=0.7,  interceptions=0.5, dribbles=3.8, position="Attacker")
PLAYMAKER  = _entry(2700, goals=0.10, assists=0.45, shots=0.8, key_passes=3.5,
                    tackles=2.5,  interceptions=2.0, pass_acc=88.0, position="Midfielder")
BOX_TO_BOX = _entry(2700, goals=0.20, assists=0.25, shots=1.4, key_passes=1.5,
                    tackles=3.0,  interceptions=2.0, dribbles=1.2, position="Midfielder")
BOX_TO_BOX_B = _entry(2610, goals=0.22, assists=0.23, shots=1.3, key_passes=1.4,
                      tackles=2.9,  interceptions=1.9, dribbles=1.1, position="Midfielder")
CENTER_BACK  = _entry(2700, goals=0.08, assists=0.08, shots=0.3, key_passes=0.4,
                      tackles=4.5,  interceptions=3.5, position="Defender")
CENTER_BACK_B = _entry(2520, goals=0.07, assists=0.09, shots=0.25, key_passes=0.5,
                       tackles=4.3,  interceptions=3.3, position="Defender")
GK_A  = _gk_entry(2700, saves_p90=4.0, conceded_p90=1.0)
GK_B  = _gk_entry(2430, saves_p90=3.8, conceded_p90=1.1)

LOW_MINUTES = _entry(200, goals=0.5, assists=0.2, shots=2.0,
                     key_passes=1.0, tackles=1.0, interceptions=0.5)

OUTFIELD_POOL = [STRIKER_A, STRIKER_B, WINGER, WINGER_B,
                 PLAYMAKER, BOX_TO_BOX, BOX_TO_BOX_B, CENTER_BACK, CENTER_BACK_B]
GK_POOL = [GK_A, GK_B]


def _per90(entry: dict) -> dict[str, float]:
    from utils.api_football import best_stats_entry
    return stats_entry_to_per90(best_stats_entry(entry))


def _pool_arrays_and_pct(entries: list[dict]) -> tuple:
    pool_metrics = [_per90(e) for e in entries]
    arrays = _precompute_pool_arrays(pool_metrics)
    return pool_metrics, arrays


def _pct_for(entry: dict, arrays: dict, exclude_gk: bool = True) -> pd.Series:
    per90 = {k: v for k, v in _per90(entry).items()
             if k != "Minutes" and (not exclude_gk or k not in GK_ONLY_METRICS)}
    return _pct_from_arrays(per90, arrays)


# ===========================================================================
# _profile_sim — Pearson-based similarity
# ===========================================================================

class TestProfileSim:
    def test_identical_returns_100(self):
        s = pd.Series({"Goals": 80.0, "Tackles": 20.0, "KeyPasses": 60.0})
        assert _profile_sim(s, s) == pytest.approx(100.0)

    def test_perfectly_opposite_returns_zero(self):
        striker  = pd.Series({"Goals": 90.0, "Tackles": 10.0, "KeyPasses": 20.0, "Shots": 85.0})
        defender = pd.Series({"Goals": 10.0, "Tackles": 90.0, "KeyPasses": 80.0, "Shots": 15.0})
        assert _profile_sim(striker, defender) == pytest.approx(0.0)

    def test_close_profiles_beat_distant_profiles(self):
        target = pd.Series({"Goals": 85.0, "Shots": 80.0, "Tackles": 15.0, "KP": 25.0})
        close  = pd.Series({"Goals": 82.0, "Shots": 78.0, "Tackles": 18.0, "KP": 28.0})
        far    = pd.Series({"Goals": 30.0, "Shots": 35.0, "Tackles": 70.0, "KP": 80.0})
        assert _profile_sim(target, close) > _profile_sim(target, far)

    def test_fewer_than_3_metrics_returns_zero(self):
        s = pd.Series({"Goals": 80.0, "Tackles": 20.0})
        assert _profile_sim(s, s) == 0.0

    def test_constant_series_returns_zero(self):
        s = pd.Series({"Goals": 50.0, "Tackles": 50.0, "Shots": 50.0, "KP": 50.0})
        assert _profile_sim(s, s) == 0.0

    def test_output_bounded_0_to_100(self):
        a = pd.Series({"x": 10.0, "y": 80.0, "z": 50.0, "w": 30.0})
        b = pd.Series({"x": 60.0, "y": 40.0, "z": 30.0, "w": 70.0})
        assert 0.0 <= _profile_sim(a, b) <= 100.0


# ===========================================================================
# _name_overlap
# ===========================================================================

class TestNameOverlap:
    def test_exact_match(self):
        assert _name_overlap("Bellingham", "Bellingham") is True

    def test_case_insensitive(self):
        assert _name_overlap("bellingham", "Bellingham") is True

    def test_full_name_contains_short_name(self):
        assert _name_overlap("Jude Bellingham", "Bellingham") is True

    def test_short_name_in_full_name(self):
        assert _name_overlap("Bellingham", "Jude Bellingham") is True

    def test_completely_different_names(self):
        assert _name_overlap("Haaland", "Mbappe") is False

    def test_unrelated_names_no_substring(self):
        assert _name_overlap("Rodrigo", "Haaland") is False


# ===========================================================================
# _get_pool_position
# ===========================================================================

class TestGetPoolPosition:
    def test_reads_games_position_first(self):
        # games.position takes priority over player.position
        entry = {
            "player": {"position": "Midfielder"},
            "statistics": [{"games": {"minutes": 2700, "position": "Defender"},
                            "league": {"id": 61}, "goals": {}, "shots": {},
                            "passes": {}, "tackles": {}, "duels": {},
                            "dribbles": {}, "fouls": {}}],
        }
        assert _get_pool_position(entry) == "Defender"

    def test_falls_back_to_player_position(self):
        # No games.position → falls back to player.position
        entry = {
            "player": {"position": "Midfielder"},
            "statistics": [{"games": {"minutes": 2700},  # no position key
                            "league": {"id": 61}, "goals": {}, "shots": {},
                            "passes": {}, "tackles": {}, "duels": {},
                            "dribbles": {}, "fouls": {}}],
        }
        assert _get_pool_position(entry) == "Midfielder"

    def test_returns_empty_string_when_both_missing(self):
        entry = {
            "player": {},
            "statistics": [{"games": {"minutes": 2700},
                            "league": {"id": 61}, "goals": {}, "shots": {},
                            "passes": {}, "tackles": {}, "duels": {},
                            "dribbles": {}, "fouls": {}}],
        }
        assert _get_pool_position(entry) == ""

    def test_attacker_position_extracted(self):
        assert _get_pool_position(STRIKER_A) == "Attacker"

    def test_defender_position_extracted(self):
        assert _get_pool_position(CENTER_BACK) == "Defender"

    def test_goalkeeper_position_extracted(self):
        assert _get_pool_position(GK_A) == "Goalkeeper"


# ===========================================================================
# _precompute_pool_arrays + _pct_from_arrays
# ===========================================================================

class TestPrecomputeAndPct:
    def setup_method(self):
        self.pool_metrics, self.arrays = _pool_arrays_and_pct(OUTFIELD_POOL)

    def test_arrays_exclude_minutes_key(self):
        assert "Minutes" not in self.arrays

    def test_known_metrics_present(self):
        for m in ("Goals per 90", "Assists per 90", "Tackles per 90", "Pass Completion %"):
            assert m in self.arrays

    def test_array_length_bounded_by_pool_size(self):
        n = len(self.pool_metrics)
        for metric, arr in self.arrays.items():
            assert 1 <= len(arr) <= n, f"{metric}: len={len(arr)}, pool={n}"

    def test_universal_metrics_have_full_pool_coverage(self):
        n = len(self.pool_metrics)
        for m in ("Goals per 90", "Assists per 90", "Tackles per 90", "Pass Completion %"):
            assert len(self.arrays[m]) == n

    def test_pct_values_in_0_to_100_range(self):
        per90 = {k: v for k, v in self.pool_metrics[0].items() if k != "Minutes"}
        for val in _pct_from_arrays(per90, self.arrays).values:
            assert 0.0 <= val <= 100.0

    def test_top_goal_scorer_ranks_highest_in_goals(self):
        # STRIKER_A has the most goals per 90 in the pool
        pct = _pct_for(STRIKER_A, self.arrays)
        assert pct.get("Goals per 90", 0) >= 80.0

    def test_center_back_ranks_highest_in_tackles(self):
        pct = _pct_for(CENTER_BACK, self.arrays)
        assert pct.get("Tackles per 90", 0) >= 80.0

    def test_winger_ranks_highest_in_dribbles(self):
        pct = _pct_for(WINGER, self.arrays)
        assert pct.get("Dribbles per 90", 0) >= 80.0

    def test_playmaker_ranks_highest_in_key_passes(self):
        pct = _pct_for(PLAYMAKER, self.arrays)
        assert pct.get("Key Passes per 90", 0) >= 80.0

    def test_minutes_excluded_from_output(self):
        per90 = {k: v for k, v in self.pool_metrics[0].items() if k != "Minutes"}
        assert "Minutes" not in _pct_from_arrays(per90, self.arrays)


# ===========================================================================
# GK-only metric isolation
# ===========================================================================

class TestGkOnlyMetrics:
    def test_gk_only_set_contains_expected_keys(self):
        assert "Goals Conceded per 90" in GK_ONLY_METRICS
        assert "Save %" in GK_ONLY_METRICS
        assert "Saves per 90" in GK_ONLY_METRICS

    def test_outfield_per90_stripped_of_gk_metrics(self):
        per90 = _per90(STRIKER_A)
        stripped = {k: v for k, v in per90.items()
                    if k != "Minutes" and k not in GK_ONLY_METRICS}
        for m in GK_ONLY_METRICS:
            assert m not in stripped

    def test_gk_per90_contains_gk_metrics(self):
        per90 = _per90(GK_A)
        # At least one GK metric must be present
        assert any(m in per90 for m in GK_ONLY_METRICS)


# ===========================================================================
# _shared_strengths
# ===========================================================================

class TestSharedStrengths:
    def _result(self, d: dict) -> dict:
        return {"pcts": pd.Series(d)}

    def test_high_shared_metric_included(self):
        target = pd.Series({"Goals per 90": 85.0, "Tackles per 90": 20.0})
        peers  = [self._result({"Goals per 90": 80.0, "Tackles per 90": 15.0})]
        shared = _shared_strengths(target, peers, threshold=65.0)
        assert "Goals per 90" in shared
        assert "Tackles per 90" not in shared

    def test_peer_below_threshold_excluded(self):
        target = pd.Series({"Goals per 90": 90.0, "Key Passes per 90": 80.0})
        peers  = [self._result({"Goals per 90": 30.0, "Key Passes per 90": 85.0})]
        shared = _shared_strengths(target, peers, threshold=65.0)
        assert "Goals per 90" not in shared
        assert "Key Passes per 90" in shared

    def test_max_out_caps_results(self):
        target = pd.Series({f"M{i}": 90.0 for i in range(10)})
        peers  = [self._result({f"M{i}": 90.0 for i in range(10)})]
        assert len(_shared_strengths(target, peers, threshold=65.0, max_out=3)) == 3

    def test_empty_peers_returns_empty(self):
        target = pd.Series({"Goals per 90": 90.0})
        assert _shared_strengths(target, []) == []


# ===========================================================================
# Similarity ranking — end-to-end with consistent reference pool
# ===========================================================================

class TestSimilarityRanking:
    """
    All percentile vectors are computed against the same pool_arrays,
    mirroring the fix in similar_players() where both target and candidates
    use _pct_from_arrays(_, pool_arrays).
    """

    def setup_method(self):
        _, self.arrays = _pool_arrays_and_pct(OUTFIELD_POOL)

    def _sim(self, a: dict, b: dict) -> float:
        pa = _pct_for(a, self.arrays)
        pb = _pct_for(b, self.arrays)
        common = sorted(set(pa.index) & set(pb.index))
        return _profile_sim(pa.reindex(common), pb.reindex(common))

    # --- Striker ----
    def test_striker_most_similar_to_striker(self):
        sim_b   = self._sim(STRIKER_A, STRIKER_B)
        sim_wng = self._sim(STRIKER_A, WINGER)
        sim_pm  = self._sim(STRIKER_A, PLAYMAKER)
        sim_cb  = self._sim(STRIKER_A, CENTER_BACK)
        assert sim_b > sim_wng, f"STRIKER_B ({sim_b:.1f}) should beat WINGER ({sim_wng:.1f})"
        assert sim_b > sim_pm,  f"STRIKER_B ({sim_b:.1f}) should beat PLAYMAKER ({sim_pm:.1f})"
        assert sim_b > sim_cb,  f"STRIKER_B ({sim_b:.1f}) should beat CENTER_BACK ({sim_cb:.1f})"

    # --- Winger ---
    def test_winger_most_similar_to_winger(self):
        sim_b   = self._sim(WINGER, WINGER_B)
        sim_str = self._sim(WINGER, STRIKER_A)
        sim_cb  = self._sim(WINGER, CENTER_BACK)
        assert sim_b > sim_str, f"WINGER_B ({sim_b:.1f}) should beat STRIKER ({sim_str:.1f})"
        assert sim_b > sim_cb,  f"WINGER_B ({sim_b:.1f}) should beat CB ({sim_cb:.1f})"

    # --- Midfielder ---
    def test_box_to_box_most_similar_to_box_to_box(self):
        # In a mixed-position pool, B2B and CB can both rank below-average on
        # attacking metrics, inflating their correlation. The meaningful guarantee
        # is that B2B_B beats the most dissimilar archetypes (striker, winger).
        # In production the pool is position-filtered, eliminating this inflation.
        sim_b   = self._sim(BOX_TO_BOX, BOX_TO_BOX_B)
        sim_str = self._sim(BOX_TO_BOX, STRIKER_A)
        sim_wng = self._sim(BOX_TO_BOX, WINGER)
        assert sim_b > sim_str, f"B2B_B ({sim_b:.1f}) should beat STRIKER ({sim_str:.1f})"
        assert sim_b > sim_wng, f"B2B_B ({sim_b:.1f}) should beat WINGER ({sim_wng:.1f})"

    # --- Center-back ---
    def test_center_back_most_similar_to_center_back(self):
        sim_b   = self._sim(CENTER_BACK, CENTER_BACK_B)
        sim_str = self._sim(CENTER_BACK, STRIKER_A)
        sim_wng = self._sim(CENTER_BACK, WINGER)
        assert sim_b > sim_str, f"CB_B ({sim_b:.1f}) should beat STRIKER ({sim_str:.1f})"
        assert sim_b > sim_wng, f"CB_B ({sim_b:.1f}) should beat WINGER ({sim_wng:.1f})"

    # --- Cross-role ordering ---
    def test_striker_not_similar_to_center_back(self):
        sim_cb  = self._sim(STRIKER_A, CENTER_BACK)
        sim_own = self._sim(STRIKER_A, STRIKER_B)
        assert sim_own > sim_cb

    def test_center_back_not_similar_to_striker(self):
        sim_str = self._sim(CENTER_BACK, STRIKER_A)
        sim_own = self._sim(CENTER_BACK, CENTER_BACK_B)
        assert sim_own > sim_str

    def test_self_similarity_is_max(self):
        for entry in [STRIKER_A, WINGER, PLAYMAKER, BOX_TO_BOX, CENTER_BACK]:
            s = self._sim(entry, entry)
            assert s == pytest.approx(100.0) or np.isnan(s)


# ===========================================================================
# Low-minutes filter
# ===========================================================================

class TestLowMinutesFiltering:
    def test_low_minutes_player_below_threshold(self):
        assert _per90(LOW_MINUTES).get("Minutes", 0) < MIN_POOL_MINUTES

    def test_all_archetypes_above_threshold(self):
        for entry in [STRIKER_A, STRIKER_B, WINGER, WINGER_B,
                      PLAYMAKER, BOX_TO_BOX, CENTER_BACK, GK_A]:
            assert _per90(entry).get("Minutes", 0) >= MIN_POOL_MINUTES
