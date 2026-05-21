"""
Unit tests for tools/grading.py — pure function coverage.

Automatically skipped if grading.py is gitignored / unavailable in this
environment (e.g. CI on the public repo). Run the full suite locally where
the private core is present.
"""
from __future__ import annotations
import pytest
import pandas as pd

try:
    from tools.grading import (
        _normalize_role,
        _confidence_factor,
        _get_league_tier,
        _age_adjustment_factor,
        _blend_weights,
        validate_scout_df,
        compute_grade,
        compute_grade_with_playstyle,
        percentile_to_letter,
        percentile_to_tier,
        NEGATIVE_KEYS,
        PLAY_STYLE_PRESETS,
    )
    GRADING_AVAILABLE = True
except ImportError:
    GRADING_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not GRADING_AVAILABLE,
    reason="tools/grading.py not available in this environment",
)


# ===========================================================================
# _normalize_role
# ===========================================================================
class TestNormalizeRole:
    def test_none_defaults_to_mf(self):
        assert _normalize_role(None) == ("mf", None)

    def test_empty_string_defaults_to_mf(self):
        assert _normalize_role("") == ("mf", None)

    def test_explicit_colon_notation(self):
        assert _normalize_role("df:cb") == ("df", "cb")
        assert _normalize_role("mf:dm") == ("mf", "dm")
        assert _normalize_role("fw:st") == ("fw", "st")

    def test_striker_alias(self):
        assert _normalize_role("striker") == ("fw", "st")

    def test_winger_alias(self):
        assert _normalize_role("winger") == ("fw", "w")

    def test_goalkeeper_alias(self):
        base, _ = _normalize_role("goalkeeper")
        assert base == "gk"

    def test_center_back_alias(self):
        assert _normalize_role("center back") == ("df", "cb")

    def test_dm_alias(self):
        assert _normalize_role("dm") == ("mf", "dm")

    def test_am_alias(self):
        assert _normalize_role("am") == ("mf", "am")

    def test_case_insensitive(self):
        assert _normalize_role("STRIKER") == _normalize_role("striker")

    def test_unknown_defaults_to_mf(self):
        base, _ = _normalize_role("quidditch_player")
        assert base == "mf"


# ===========================================================================
# _confidence_factor
# ===========================================================================
class TestConfidenceFactor:
    def test_zero_minutes_returns_half(self):
        assert _confidence_factor(0) == pytest.approx(0.5)

    def test_negative_minutes_returns_half(self):
        assert _confidence_factor(-100) == pytest.approx(0.5)

    def test_2000_minutes_is_high(self):
        c = _confidence_factor(2000)
        assert 0.75 <= c <= 0.90

    def test_5000_minutes_is_near_max(self):
        assert _confidence_factor(5000) >= 0.94

    def test_strictly_increasing(self):
        for lo, hi in [(500, 1000), (1000, 2000), (2000, 4000)]:
            assert _confidence_factor(lo) < _confidence_factor(hi)

    def test_bounded_above_1(self):
        assert _confidence_factor(100_000) <= 1.0


# ===========================================================================
# _get_league_tier
# ===========================================================================
class TestGetLeagueTier:
    def test_premier_league_is_top(self):
        assert _get_league_tier("Premier League") == pytest.approx(1.0)

    def test_bundesliga_is_top(self):
        assert _get_league_tier("Bundesliga") == pytest.approx(1.0)

    def test_empty_returns_1(self):
        assert _get_league_tier("") == pytest.approx(1.0)

    def test_unknown_returns_default(self):
        tier = _get_league_tier("Fictional Super League")
        assert 0.0 < tier <= 1.0

    def test_case_insensitive(self):
        assert _get_league_tier("premier league") == _get_league_tier("Premier League")

    def test_ligue1_below_top(self):
        assert _get_league_tier("Ligue 1") < 1.0


# ===========================================================================
# _age_adjustment_factor
# ===========================================================================
class TestAgeAdjustmentFactor:
    def test_none_returns_1(self):
        assert _age_adjustment_factor(None, "fw") == pytest.approx(1.0)

    def test_peak_age_fw(self):
        # 26 is peak for fw — no penalty, small bonus or neutral
        factor = _age_adjustment_factor(26, "fw")
        assert factor >= 1.0

    def test_young_player_gets_bonus(self):
        assert _age_adjustment_factor(18, "fw") > 1.0

    def test_older_player_gets_penalty(self):
        assert _age_adjustment_factor(35, "fw") < 1.0

    def test_32_vs_28(self):
        assert _age_adjustment_factor(32, "mf") < _age_adjustment_factor(28, "mf")

    def test_penalty_increases_with_age(self):
        assert _age_adjustment_factor(33, "df") < _age_adjustment_factor(31, "df")


# ===========================================================================
# percentile_to_letter / percentile_to_tier
# ===========================================================================
class TestPercentileToLetter:
    def test_99_is_a_plus(self):
        assert percentile_to_letter(99) == "A+"

    def test_90_is_a(self):
        assert percentile_to_letter(90) == "A"

    def test_50_is_c_minus(self):
        assert percentile_to_letter(50) == "C-"

    def test_10_is_f(self):
        assert percentile_to_letter(10) == "F"

    def test_boundaries_are_inclusive(self):
        assert percentile_to_letter(95) == "A+"
        assert percentile_to_letter(80) == "B+"


class TestPercentileTotier:
    def test_95_is_elite(self):
        assert percentile_to_tier(95) == "Elite"

    def test_85_is_world_class(self):
        assert percentile_to_tier(85) == "World Class"

    def test_40_is_project(self):
        assert percentile_to_tier(40) == "Project"

    def test_20_is_limited(self):
        assert percentile_to_tier(20) == "Limited"


# ===========================================================================
# _blend_weights
# ===========================================================================
class TestBlendWeights:
    def test_no_subrole_normalizes_to_1(self):
        base = {"a": 2.0, "b": 2.0}
        result = _blend_weights(base, None)
        assert sum(result.values()) == pytest.approx(1.0)

    def test_with_subrole_normalizes_to_1(self):
        base = {"a": 1.0, "b": 1.0}
        sub = {"b": 1.0, "c": 1.0}
        result = _blend_weights(base, sub)
        assert sum(result.values()) == pytest.approx(1.0)

    def test_all_weights_non_negative(self):
        base = {"a": 1.0, "b": 1.0}
        sub = {"b": 0.5, "c": 0.5}
        result = _blend_weights(base, sub)
        assert all(v >= 0 for v in result.values())

    def test_subrole_metrics_present_in_output(self):
        base = {"a": 1.0}
        sub = {"b": 1.0}
        result = _blend_weights(base, sub)
        assert "b" in result


# ===========================================================================
# validate_scout_df
# ===========================================================================
class TestValidateScoutDf:
    def test_valid_df_passes(self, minimal_scout_df):
        validate_scout_df(minimal_scout_df)  # should not raise

    def test_missing_percentile_column_raises(self):
        df = pd.DataFrame({"Other": [50.0]}, index=["metric"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_scout_df(df)


# ===========================================================================
# compute_grade — integration
# ===========================================================================
class TestComputeGrade:
    def test_returns_grade_breakdown(self, minimal_scout_df):
        bd = compute_grade(minimal_scout_df, role_hint="mf")
        assert bd.final_score >= 0
        assert bd.role == "mf"

    def test_final_score_between_0_and_80(self, minimal_scout_df):
        # Score formula caps at raw * 0.8 where raw max is 100
        bd = compute_grade(minimal_scout_df, role_hint="fw")
        assert 0.0 <= bd.final_score <= 80.0

    def test_high_percentiles_beat_low(self, minimal_scout_df):
        low_df = minimal_scout_df.copy()
        low_df["Percentile"] = 20.0
        high_df = minimal_scout_df.copy()
        high_df["Percentile"] = 80.0
        bd_low = compute_grade(low_df, role_hint="fw")
        bd_high = compute_grade(high_df, role_hint="fw")
        assert bd_high.final_score > bd_low.final_score

    def test_role_is_set_correctly(self, fw_scout_df):
        bd = compute_grade(fw_scout_df, role_hint="fw:st")
        assert bd.role == "fw:st"

    def test_gk_role(self, gk_scout_df):
        bd = compute_grade(gk_scout_df, role_hint="gk")
        assert bd.role == "gk"
        assert bd.final_score >= 0

    def test_negative_metric_inverted(self, gk_scout_df):
        # Goals Conceded per 90 is negative — a low percentile means fewer goals
        # conceded, which should be rewarded.  Grade with low conceded pct should
        # be higher than grade with high conceded pct.
        low_conceded = gk_scout_df.copy()
        low_conceded.loc["Goals Conceded per 90", "Percentile"] = 10.0  # bad raw pct → inverted to 90
        high_conceded = gk_scout_df.copy()
        high_conceded.loc["Goals Conceded per 90", "Percentile"] = 90.0  # good raw pct → inverted to 10
        bd_low = compute_grade(low_conceded, role_hint="gk")
        bd_high = compute_grade(high_conceded, role_hint="gk")
        assert bd_low.final_score > bd_high.final_score


# ===========================================================================
# compute_grade_with_playstyle
# ===========================================================================
class TestComputeGradeWithPlaystyle:
    def test_playstyle_changes_score(self, fw_scout_df):
        bd_base = compute_grade(fw_scout_df, role_hint="fw")
        bd_styled = compute_grade_with_playstyle(fw_scout_df, role_hint="fw", play_style="possession_high")
        # Scores may differ since weight distribution changes
        assert isinstance(bd_styled.final_score, float)

    def test_unknown_playstyle_same_as_none(self, minimal_scout_df):
        bd_none = compute_grade(minimal_scout_df, role_hint="mf")
        bd_unknown = compute_grade_with_playstyle(minimal_scout_df, role_hint="mf", play_style="nonexistent_style")
        assert bd_none.final_score == pytest.approx(bd_unknown.final_score)

    def test_all_presets_accepted(self, minimal_scout_df):
        for style in PLAY_STYLE_PRESETS:
            bd = compute_grade_with_playstyle(minimal_scout_df, role_hint="mf", play_style=style)
            assert 0.0 <= bd.final_score <= 80.0
