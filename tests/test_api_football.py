"""
Unit tests for utils/api_football.py — pure functions only, no live API calls.

All I/O is either deterministic (current_season) or mocked
(_search_in_league patched via unittest.mock).
"""
from __future__ import annotations
from datetime import date
from unittest.mock import patch
import pytest

from utils.api_football import (
    _normalize_name,
    _name_score,
    _search_variants,
    pick_best_player,
    best_stats_entry,
    current_season,
    search_player,
    MAJOR_LEAGUE_IDS,
    SEARCH_LEAGUES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _player(pid: int, name: str, firstname: str, lastname: str, stats: list) -> dict:
    return {
        "player": {"id": pid, "name": name, "firstname": firstname, "lastname": lastname},
        "statistics": stats,
    }


def _stat(league_id: int, minutes: int) -> dict:
    return {"league": {"id": league_id}, "games": {"minutes": minutes}}


# ===========================================================================
# _normalize_name
# ===========================================================================
class TestNormalizeName:
    def test_lowercase(self):
        assert _normalize_name("Bellingham") == "bellingham"

    def test_accent_e(self):
        assert _normalize_name("Kanté") == "kante"

    def test_accent_u(self):
        assert _normalize_name("Müller") == "muller"

    def test_accent_circumflex(self):
        assert _normalize_name("Mbappé") == "mbappe"

    def test_cedilla(self):
        assert _normalize_name("Çalhanoğlu") == "calhanoglu"

    def test_straight_apostrophe(self):
        assert _normalize_name("N'Golo") == "ngolo"

    def test_curly_apostrophe(self):
        assert _normalize_name("N’Golo") == "ngolo"

    def test_empty_string(self):
        assert _normalize_name("") == ""

    def test_whitespace_stripped(self):
        assert _normalize_name("  Messi  ") == "messi"

    def test_idempotent(self):
        n = "Çalhanoğlu"
        assert _normalize_name(_normalize_name(n)) == _normalize_name(n)


# ===========================================================================
# _search_variants
# ===========================================================================
class TestSearchVariants:
    def test_full_name_is_first(self):
        assert _search_variants("Rayan Cherki")[0] == "Rayan Cherki"

    def test_last_name_present(self):
        assert "Cherki" in _search_variants("Rayan Cherki")

    def test_first_name_present(self):
        assert "Rayan" in _search_variants("Rayan Cherki")

    def test_accent_stripped_variant(self):
        assert "Mbappe" in _search_variants("Kylian Mbappé")

    def test_no_duplicates_for_full_name(self):
        v = _search_variants("Alphonso Davies")
        assert len(v) == len(set(v))

    def test_no_duplicates_for_single_name(self):
        v = _search_variants("Messi")
        assert len(v) == len(set(v))

    def test_no_empty_variants(self):
        for name in ["Bellingham", "N'Golo Kanté", "Pedri"]:
            assert all(v.strip() for v in _search_variants(name))

    def test_first_name_before_last_name(self):
        v = _search_variants("Alphonso Davies")
        fi = next(i for i, x in enumerate(v) if x == "Alphonso")
        li = next(i for i, x in enumerate(v) if x == "Davies")
        assert fi < li


# ===========================================================================
# current_season
# ===========================================================================
class TestCurrentSeason:
    def _mock_date(self, d: date):
        return patch("utils.api_football.date", **{"today.return_value": d})

    def test_july_starts_new_season(self):
        with self._mock_date(date(2025, 7, 1)):
            assert current_season() == 2025

    def test_august_is_current_year(self):
        with self._mock_date(date(2025, 8, 15)):
            assert current_season() == 2025

    def test_january_is_previous_year(self):
        with self._mock_date(date(2026, 1, 15)):
            assert current_season() == 2025

    def test_june_is_previous_year(self):
        with self._mock_date(date(2025, 6, 30)):
            assert current_season() == 2024

    def test_boundary_june_vs_july(self):
        with self._mock_date(date(2024, 6, 30)):
            jun = current_season()
        with self._mock_date(date(2024, 7, 1)):
            jul = current_season()
        assert jul == jun + 1


# ===========================================================================
# best_stats_entry
# ===========================================================================
class TestBestStatsEntry:
    def test_empty_returns_none(self):
        assert best_stats_entry({"player": {}, "statistics": []}) is None

    def test_major_league_beats_high_minutes_minor(self):
        obj = _player(1, "X", "X", "X", [_stat(39, 500), _stat(9999, 3000)])
        assert best_stats_entry(obj)["league"]["id"] == 39

    def test_more_minutes_wins_within_major(self):
        obj = _player(1, "X", "X", "X", [_stat(39, 1000), _stat(140, 2500)])
        assert best_stats_entry(obj)["league"]["id"] == 140

    def test_single_entry_always_returned(self):
        obj = _player(1, "X", "X", "X", [_stat(39, 1000)])
        assert best_stats_entry(obj) is not None

    def test_none_minutes_treated_as_zero(self):
        obj = _player(1, "X", "X", "X", [
            {"league": {"id": 39}, "games": {"minutes": None}},
            _stat(140, 500),
        ])
        assert best_stats_entry(obj)["league"]["id"] == 140

    def test_saudi_is_major(self):
        obj = _player(1, "X", "X", "X", [_stat(307, 2000), _stat(9999, 9000)])
        assert best_stats_entry(obj)["league"]["id"] == 307


# ===========================================================================
# pick_best_player
# ===========================================================================
class TestPickBestPlayer:
    def test_empty_returns_none(self):
        assert pick_best_player([], "Messi") is None

    def test_exact_combined_name_wins(self):
        results = [
            _player(1, "L. Messi", "Lionel", "Messi", [_stat(253, 2800)]),
            _player(2, "T. Messi", "Thiago", "Messi", [_stat(128, 200)]),
        ]
        assert pick_best_player(results, "Lionel Messi")["player"]["id"] == 1

    def test_accent_normalization(self):
        results = [_player(1, "K. Mbappé", "Kylian", "Mbappé", [_stat(307, 2400)])]
        assert pick_best_player(results, "Mbappe")["player"]["id"] == 1

    def test_apostrophe_normalization(self):
        results = [
            _player(1, "N. Kanté", "N'Golo", "Kanté", [_stat(39, 2500)]),
            _player(2, "B. Kanté", "Boubacar", "Kanté", [_stat(61, 900)]),
        ]
        assert pick_best_player(results, "Ngolo Kante")["player"]["id"] == 1

    def test_disambiguation_by_first_name(self):
        results = [
            _player(1, "J. Bellingham", "Jude", "Bellingham", [_stat(140, 2700)]),
            _player(2, "J. Bellingham", "Jobe", "Bellingham", [_stat(39, 3000)]),
        ]
        assert pick_best_player(results, "Jude Bellingham")["player"]["id"] == 1

    def test_minutes_tiebreak(self):
        results = [
            _player(1, "Silva", "Thiago", "Silva", [_stat(39, 1800)]),
            _player(2, "Silva", "David", "Silva", [_stat(140, 3000)]),
        ]
        assert pick_best_player(results, "Silva")["player"]["id"] == 2


# ===========================================================================
# search_player — quality-gating (mocked)
# ===========================================================================
class TestSearchPlayerGating:
    def _run(self, name: str, league_results: dict, season: int = 2025) -> list:
        def _mock(search_name, league_id, _season):
            return league_results.get((search_name, league_id), [])

        with patch("utils.api_football._search_in_league", side_effect=_mock):
            return search_player(name, season=season)

    def test_commits_on_exact_match(self):
        lamine = _player(1, "L. Yamal", "Lamine", "Yamal", [_stat(140, 2500)])
        results = self._run("Lamine Yamal", {("Lamine Yamal", 140): [lamine]})
        assert pick_best_player(results, "Lamine Yamal")["player"]["id"] == 1

    def test_skips_weak_match_finds_correct(self):
        ben = _player(2, "B. Davies", "Ben", "Davies", [_stat(39, 2800)])
        alphonso = _player(1, "A. Davies", "Alphonso", "Boyle Davies", [_stat(78, 2200)])
        results = self._run(
            "Alphonso Davies",
            {("Alphonso Davies", 39): [ben], ("Alphonso Davies", 78): [alphonso]},
        )
        assert pick_best_player(results, "Alphonso Davies")["player"]["id"] == 1

    def test_empty_api_returns_empty_list(self):
        assert self._run("Nobody AtAll", {}) == []

    def test_fallback_when_no_strong_match(self):
        ben = _player(2, "B. Davies", "Ben", "Davies", [_stat(39, 2800)])
        results = self._run("Alphonso Davies", {("Davies", 39): [ben]})
        assert results  # fallback, not empty


# ===========================================================================
# League coverage
# ===========================================================================
class TestLeagueCoverage:
    BIG5 = [39, 140, 78, 135, 61]

    def test_big5_in_search_leagues(self):
        for lid in self.BIG5:
            assert lid in SEARCH_LEAGUES

    def test_big5_in_major_league_ids(self):
        for lid in self.BIG5:
            assert lid in MAJOR_LEAGUE_IDS

    def test_no_duplicates_in_search_leagues(self):
        assert len(SEARCH_LEAGUES) == len(set(SEARCH_LEAGUES))

    def test_big5_before_tier2(self):
        tier2 = [40, 79, 62, 136, 141, 95]
        for top in self.BIG5:
            for t2 in tier2:
                if top in SEARCH_LEAGUES and t2 in SEARCH_LEAGUES:
                    assert SEARCH_LEAGUES.index(top) < SEARCH_LEAGUES.index(t2)
