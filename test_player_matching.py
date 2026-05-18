# test_player_matching.py
"""
Unit tests for player-name matching, normalization, and search utilities.
All tests are fully offline — no API calls, no Streamlit app context required.

Run:
    pip install pytest
    pytest test_player_matching.py -v
    pytest test_player_matching.py -v --tb=short    # compact tracebacks
"""
from __future__ import annotations

import sys
import time
import types
from datetime import date
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Stub streamlit before importing the module under test.
# api_football.py applies @st.cache_data decorators at module level; the
# stub turns those into no-ops so the module loads without a running app.
# sys.modules.setdefault leaves the real streamlit in place if it was already
# imported (e.g. by pytest plugins), otherwise installs the stub.
# ---------------------------------------------------------------------------
if "streamlit" not in sys.modules:
    _st_stub = types.ModuleType("streamlit")
    _st_stub.cache_data = lambda **kw: (lambda f: f)
    sys.modules["streamlit"] = _st_stub

from utils.api_football import (  # noqa: E402
    _normalize_name,
    _search_variants,
    pick_best_player,
    best_stats_entry,
    current_season,
    MAJOR_LEAGUE_IDS,
    SEARCH_LEAGUES,
)


# ---------------------------------------------------------------------------
# Helpers — build minimal mock API-football player objects
# ---------------------------------------------------------------------------
def _player(
    player_id: int,
    name: str,
    firstname: str,
    lastname: str,
    stats: list[dict],
) -> dict:
    """Minimal replica of one entry in the /players API response."""
    return {
        "player": {
            "id": player_id,
            "name": name,
            "firstname": firstname,
            "lastname": lastname,
        },
        "statistics": stats,
    }


def _stat(league_id: int, minutes: int) -> dict:
    """Minimal statistics block for a given league."""
    return {
        "league": {"id": league_id, "name": f"League {league_id}"},
        "games": {"minutes": minutes},
    }


# ===========================================================================
# _normalize_name
# ===========================================================================
class TestNormalizeName:
    """Pure accent- and apostrophe-stripping, lowercasing."""

    def test_lowercase(self):
        assert _normalize_name("Bellingham") == "bellingham"

    def test_accent_e(self):
        assert _normalize_name("Kanté") == "kante"

    def test_accent_e_umlaut(self):
        assert _normalize_name("Müller") == "muller"

    def test_accent_o_umlaut(self):
        assert _normalize_name("Özil") == "ozil"

    def test_cedilla(self):
        # ç → c, ğ → g
        assert _normalize_name("Çalhanoğlu") == "calhanoglu"

    def test_portuguese_tilde(self):
        assert _normalize_name("Ronaldo") == "ronaldo"

    def test_straight_apostrophe(self):
        assert _normalize_name("N'Golo") == "ngolo"

    def test_curly_apostrophe(self):
        # U+2019 RIGHT SINGLE QUOTATION MARK
        assert _normalize_name("N’Golo") == "ngolo"

    def test_backtick(self):
        assert _normalize_name("N`Golo") == "ngolo"

    def test_combined_apostrophe_and_accent(self):
        assert _normalize_name("N'Golo Kanté") == "ngolo kante"

    def test_dembele(self):
        assert _normalize_name("Dembélé") == "dembele"

    def test_full_dembele(self):
        assert _normalize_name("Ousmane Dembélé") == "ousmane dembele"

    def test_empty_string(self):
        assert _normalize_name("") == ""

    def test_leading_trailing_whitespace(self):
        assert _normalize_name("  Messi  ") == "messi"

    def test_already_ascii(self):
        assert _normalize_name("haaland") == "haaland"

    def test_mbappe(self):
        assert _normalize_name("Mbappé") == "mbappe"

    def test_idempotent(self):
        name = "Çalhanoğlu"
        assert _normalize_name(_normalize_name(name)) == _normalize_name(name)


# ===========================================================================
# _search_variants
# ===========================================================================
class TestSearchVariants:
    """Variant generation for the API name-search strategy."""

    def test_single_name_present(self):
        assert "Bellingham" in _search_variants("Bellingham")

    def test_no_duplicates_single(self):
        v = _search_variants("Bellingham")
        assert len(v) == len(set(v))

    def test_no_empty_variants(self):
        for name in ["Bellingham", "N'Golo Kanté", "Lamine Yamal", "Pedri"]:
            assert all(v.strip() for v in _search_variants(name))

    def test_full_name_included(self):
        assert "Rayan Cherki" in _search_variants("Rayan Cherki")

    def test_last_name_included(self):
        assert "Cherki" in _search_variants("Rayan Cherki")

    def test_first_name_included(self):
        assert "Rayan" in _search_variants("Rayan Cherki")

    def test_accent_stripped_last_name(self):
        v = _search_variants("N'Golo Kanté")
        assert "Kante" in v

    def test_accent_stripped_full(self):
        v = _search_variants("Ousmane Dembélé")
        assert "Ousmane Dembele" in v

    def test_starts_with_original(self):
        assert _search_variants("Cristiano Ronaldo")[0] == "Cristiano Ronaldo"

    def test_no_duplicates_full_name(self):
        v = _search_variants("Rayan Cherki")
        assert len(v) == len(set(v))

    def test_single_word_no_duplicates(self):
        # first == last when only one word; should not duplicate
        v = _search_variants("Messi")
        assert len(v) == len(set(v))

    def test_mbappe_stripped(self):
        v = _search_variants("Kylian Mbappé")
        assert "Mbappe" in v

    def test_calhanoglu_stripped(self):
        v = _search_variants("Hakan Çalhanoğlu")
        assert "Calhanoglu" in v


# ===========================================================================
# pick_best_player
# ===========================================================================
class TestPickBestPlayer:
    """Name-matching logic: normalization, scoring, minutes tiebreak."""

    # ── Edge cases ────────────────────────────────────────────────────────

    def test_empty_results_returns_none(self):
        assert pick_best_player([], "Messi") is None

    def test_single_result_always_returned(self):
        results = [_player(1, "A. Player", "Any", "Player", [_stat(39, 100)])]
        assert pick_best_player(results, "Nobody") is not None

    # ── Apostrophe normalization ──────────────────────────────────────────

    def test_ngolo_kante_no_apostrophe(self):
        """'Ngolo Kante' must find N'Golo Kanté over another Kanté."""
        results = [
            _player(1, "N. Kanté",  "N'Golo",   "Kanté", [_stat(39, 2500)]),
            _player(2, "B. Kanté",  "Boubacar",  "Kanté", [_stat(61,  900)]),
        ]
        assert pick_best_player(results, "Ngolo Kante")["player"]["id"] == 1

    def test_ngolo_kante_curly_apostrophe(self):
        """Curly apostrophe in query."""
        results = [_player(1, "N. Kanté", "N'Golo", "Kanté", [_stat(39, 2500)])]
        assert pick_best_player(results, "N’Golo Kanté")["player"]["id"] == 1

    def test_ngolo_kante_with_accent(self):
        """Standard 'N'Golo Kanté' query."""
        results = [
            _player(1, "N. Kanté",  "N'Golo",  "Kanté", [_stat(39, 2500)]),
            _player(2, "B. Kanté",  "Boubacar", "Kanté", [_stat(61,  400)]),
        ]
        assert pick_best_player(results, "N'Golo Kanté")["player"]["id"] == 1

    # ── Accent normalization ──────────────────────────────────────────────

    def test_dembele_no_accent(self):
        """'Dembele' without accent finds 'Dembélé' (more minutes wins tiebreak)."""
        results = [
            _player(1, "O. Dembélé", "Ousmane", "Dembélé", [_stat(140, 2000)]),
            _player(2, "M. Dembele", "Moussa",  "Dembele",  [_stat(88,   500)]),
        ]
        assert pick_best_player(results, "Ousmane Dembele")["player"]["id"] == 1

    def test_mbappe_no_accent(self):
        results = [_player(1, "K. Mbappé", "Kylian", "Mbappé", [_stat(307, 2400)])]
        assert pick_best_player(results, "Mbappe")["player"]["id"] == 1

    def test_mbappe_full_no_accent(self):
        results = [_player(1, "K. Mbappé", "Kylian", "Mbappé", [_stat(307, 2400)])]
        assert pick_best_player(results, "Kylian Mbappe")["player"]["id"] == 1

    def test_muller_umlaut(self):
        results = [_player(1, "T. Müller", "Thomas", "Müller", [_stat(78, 2000)])]
        assert pick_best_player(results, "Muller")["player"]["id"] == 1

    def test_calhanoglu(self):
        results = [_player(1, "H. Çalhanoğlu", "Hakan", "Çalhanoğlu", [_stat(135, 2800)])]
        assert pick_best_player(results, "Calhanoglu")["player"]["id"] == 1

    def test_ozil_umlaut(self):
        results = [_player(1, "M. Özil", "Mesut", "Özil", [_stat(39, 2000)])]
        assert pick_best_player(results, "Ozil")["player"]["id"] == 1

    # ── Exact display-name match (score 4 beats more minutes) ────────────

    def test_exact_display_name_beats_minutes(self):
        """Score-4 match wins even with fewer minutes than a score-1 match."""
        results = [
            _player(1, "Rayan Cherki", "Rayan", "Cherki", [_stat(61, 2000)]),
            _player(2, "E. Cherki",    "Eric",  "Cherki", [_stat(61, 3000)]),
        ]
        assert pick_best_player(results, "Rayan Cherki")["player"]["id"] == 1

    # ── Full-name queries ─────────────────────────────────────────────────

    def test_cristiano_ronaldo_full_name(self):
        """CR7 wins over another Ronaldo (more minutes in Saudi league)."""
        results = [
            _player(1, "C. Ronaldo", "Cristiano", "Ronaldo", [_stat(307, 2700)]),
            _player(2, "R. Ronaldo", "Roberto",   "Ronaldo", [_stat(253, 1500)]),
        ]
        assert pick_best_player(results, "Cristiano Ronaldo")["player"]["id"] == 1

    def test_cristiano_ronaldo_last_name_only(self):
        """'Ronaldo' alone: CR7 with most minutes wins the tiebreak."""
        results = [
            _player(1, "C. Ronaldo", "Cristiano", "Ronaldo", [_stat(307, 2700)]),
            _player(2, "R. Ronaldo", "Roberto",   "Ronaldo", [_stat(88,   800)]),
        ]
        assert pick_best_player(results, "Ronaldo")["player"]["id"] == 1

    def test_lamine_yamal(self):
        results = [
            _player(1, "L. Yamal", "Lamine", "Yamal", [_stat(140, 2500)]),
            _player(2, "Y. Yamal", "Yusuf",  "Yamal", [_stat(88,   200)]),
        ]
        assert pick_best_player(results, "Lamine Yamal")["player"]["id"] == 1

    def test_jude_bellingham(self):
        results = [
            _player(1, "J. Bellingham", "Jude", "Bellingham", [_stat(140, 2700)]),
            _player(2, "J. Bellingham", "Jobe", "Bellingham", [_stat(39,  1200)]),
        ]
        assert pick_best_player(results, "Jude Bellingham")["player"]["id"] == 1

    def test_vinicius_junior(self):
        results = [_player(1, "Vinicius Junior", "Vinicius", "Junior", [_stat(140, 2900)])]
        assert pick_best_player(results, "Vinicius Junior")["player"]["id"] == 1

    def test_rayan_cherki(self):
        results = [_player(1, "R. Cherki", "Rayan", "Cherki", [_stat(61, 2200)])]
        assert pick_best_player(results, "Rayan Cherki")["player"]["id"] == 1

    def test_pedri(self):
        results = [_player(1, "Pedri", "Pedro", "González López", [_stat(140, 2200)])]
        assert pick_best_player(results, "Pedri")["player"]["id"] == 1

    def test_messi(self):
        results = [
            _player(1, "L. Messi", "Lionel", "Messi", [_stat(253, 2800)]),
            _player(2, "T. Messi", "Thiago", "Messi", [_stat(128,  200)]),
        ]
        assert pick_best_player(results, "Messi")["player"]["id"] == 1

    def test_rodri(self):
        results = [_player(1, "Rodri", "Rodrigo", "Hernández Cascante", [_stat(39, 2500)])]
        assert pick_best_player(results, "Rodri")["player"]["id"] == 1

    def test_erling_haaland(self):
        results = [_player(1, "E. Haaland", "Erling", "Haaland", [_stat(39, 2700)])]
        assert pick_best_player(results, "Erling Haaland")["player"]["id"] == 1

    def test_salah(self):
        results = [
            _player(1, "M. Salah", "Mohamed", "Salah", [_stat(39, 2800)]),
        ]
        assert pick_best_player(results, "Mohamed Salah")["player"]["id"] == 1

    def test_de_bruyne(self):
        results = [_player(1, "K. De Bruyne", "Kevin", "De Bruyne", [_stat(39, 1800)])]
        assert pick_best_player(results, "Kevin De Bruyne")["player"]["id"] == 1

    def test_virgil_van_dijk(self):
        results = [_player(1, "V. van Dijk", "Virgil", "van Dijk", [_stat(39, 3000)])]
        assert pick_best_player(results, "Virgil Van Dijk")["player"]["id"] == 1

    # ── Minutes tiebreak ─────────────────────────────────────────────────

    def test_minutes_tiebreak_same_name_score(self):
        """Equal name score → player with more minutes wins."""
        results = [
            _player(1, "Silva", "Thiago", "Silva", [_stat(39, 1800)]),
            _player(2, "Silva", "David",  "Silva", [_stat(140, 3000)]),
        ]
        assert pick_best_player(results, "Silva")["player"]["id"] == 2

    def test_zero_minutes_loses(self):
        results = [
            _player(1, "J. Player", "John", "Player", [_stat(39, 0)]),
            _player(2, "J. Player", "Jack", "Player", [_stat(39, 1500)]),
        ]
        assert pick_best_player(results, "Player")["player"]["id"] == 2


# ===========================================================================
# best_stats_entry
# ===========================================================================
class TestBestStatsEntry:
    """Preference for major-league stats over cups and lower divisions."""

    def test_empty_returns_none(self):
        assert best_stats_entry(_player(1, "X", "X", "X", [])) is None

    def test_single_entry_returned(self):
        obj = _player(1, "X", "X", "X", [_stat(39, 1000)])
        assert best_stats_entry(obj)["league"]["id"] == 39

    def test_major_beats_more_minutes_in_minor(self):
        """Major league with 500 min beats unknown league with 3 000 min."""
        obj = _player(1, "X", "X", "X", [
            _stat(39, 500),    # Premier League — major
            _stat(9999, 3000), # Unknown league — not major
        ])
        assert best_stats_entry(obj)["league"]["id"] == 39

    def test_more_minutes_wins_within_major(self):
        obj = _player(1, "X", "X", "X", [
            _stat(39, 1000),
            _stat(140, 2500),
        ])
        assert best_stats_entry(obj)["league"]["id"] == 140

    def test_saudi_league_is_major(self):
        """Saudi Pro League (307) must be treated as a major league."""
        obj = _player(1, "X", "X", "X", [
            _stat(307, 2000),   # Saudi — major
            _stat(9999, 9000),  # Unknown — not major
        ])
        assert best_stats_entry(obj)["league"]["id"] == 307

    def test_championship_is_major(self):
        obj = _player(1, "X", "X", "X", [
            _stat(40, 3000),   # Championship — major
            _stat(9999, 9000), # Unknown
        ])
        assert best_stats_entry(obj)["league"]["id"] == 40

    def test_serie_b_is_major(self):
        obj = _player(1, "X", "X", "X", [
            _stat(136, 3000),  # Serie B — major
            _stat(9999, 9000),
        ])
        assert best_stats_entry(obj)["league"]["id"] == 136

    def test_ligue2_is_major(self):
        obj = _player(1, "X", "X", "X", [
            _stat(62, 2800),
            _stat(9999, 9000),
        ])
        assert best_stats_entry(obj)["league"]["id"] == 62

    def test_mls_is_major(self):
        obj = _player(1, "X", "X", "X", [
            _stat(253, 2500),
            _stat(9999, 9000),
        ])
        assert best_stats_entry(obj)["league"]["id"] == 253

    def test_none_minutes_treated_as_zero(self):
        obj = _player(1, "X", "X", "X", [
            {"league": {"id": 39, "name": "PL"}, "games": {"minutes": None}},
            _stat(140, 500),
        ])
        assert best_stats_entry(obj)["league"]["id"] == 140


# ===========================================================================
# current_season
# ===========================================================================
class TestCurrentSeason:
    """Season boundary: before July → previous year, July+ → current year."""

    def _mock(self, d: date):
        return patch("utils.api_football.date", **{"today.return_value": d})

    def test_july_1_is_new_season(self):
        with self._mock(date(2025, 7, 1)):
            assert current_season() == 2025

    def test_august_is_new_season(self):
        with self._mock(date(2025, 8, 15)):
            assert current_season() == 2025

    def test_december_is_same_start_year(self):
        with self._mock(date(2025, 12, 31)):
            assert current_season() == 2025

    def test_january_is_previous_start_year(self):
        with self._mock(date(2026, 1, 15)):
            assert current_season() == 2025

    def test_june_is_previous_start_year(self):
        with self._mock(date(2025, 6, 30)):
            assert current_season() == 2024

    def test_boundary_june_vs_july(self):
        with self._mock(date(2024, 6, 30)):
            june = current_season()
        with self._mock(date(2024, 7, 1)):
            july = current_season()
        assert july == june + 1


# ===========================================================================
# League coverage
# ===========================================================================
class TestLeagueCoverage:
    """Sanity-check that critical league IDs appear in the right lists."""

    BIG_5 = [39, 140, 78, 135, 61]

    def test_big5_in_search_leagues(self):
        for lid in self.BIG_5:
            assert lid in SEARCH_LEAGUES, f"Big-5 league {lid} missing from SEARCH_LEAGUES"

    def test_big5_in_major_league_ids(self):
        for lid in self.BIG_5:
            assert lid in MAJOR_LEAGUE_IDS, f"Big-5 league {lid} missing from MAJOR_LEAGUE_IDS"

    def test_saudi_in_search_leagues(self):
        assert 307 in SEARCH_LEAGUES, "Saudi Pro League (307) must be in SEARCH_LEAGUES"

    def test_saudi_in_major_league_ids(self):
        assert 307 in MAJOR_LEAGUE_IDS, "Saudi Pro League (307) must be in MAJOR_LEAGUE_IDS"

    def test_belgian_first_division_in_search_leagues(self):
        assert 144 in SEARCH_LEAGUES, "Belgian First Division A (144) missing"

    def test_belgian_first_division_in_major(self):
        assert 144 in MAJOR_LEAGUE_IDS

    def test_mls_in_search_leagues(self):
        assert 253 in SEARCH_LEAGUES

    def test_liga_mx_in_search_leagues(self):
        assert 262 in SEARCH_LEAGUES

    def test_primera_liga_argentina_in_search(self):
        assert 128 in SEARCH_LEAGUES

    def test_serie_a_brasileira_in_search(self):
        assert 71 in SEARCH_LEAGUES

    def test_k_league_in_search(self):
        assert 292 in SEARCH_LEAGUES

    def test_chinese_super_league_in_search(self):
        assert 169 in SEARCH_LEAGUES

    def test_no_duplicates_in_search_leagues(self):
        assert len(SEARCH_LEAGUES) == len(set(SEARCH_LEAGUES)), (
            "Duplicate IDs detected in SEARCH_LEAGUES"
        )

    def test_big5_appear_before_tier2_in_search_leagues(self):
        """Big-5 leagues should be checked before Championship, Serie B, etc."""
        tier2 = [40, 79, 62, 136, 141, 95]
        for top in self.BIG_5:
            for t2 in tier2:
                if t2 in SEARCH_LEAGUES and top in SEARCH_LEAGUES:
                    assert SEARCH_LEAGUES.index(top) < SEARCH_LEAGUES.index(t2), (
                        f"League {top} (big-5) should come before {t2} (tier-2)"
                    )

    def test_saudi_before_tier2_in_search_leagues(self):
        tier2 = [40, 79, 62, 136, 141, 95]
        for t2 in tier2:
            if t2 in SEARCH_LEAGUES:
                assert SEARCH_LEAGUES.index(307) < SEARCH_LEAGUES.index(t2), (
                    "Saudi Pro League should be checked before European tier-2"
                )


# ===========================================================================
# Performance
# ===========================================================================
class TestPerformance:
    """Pure-function throughput checks — no I/O, should finish in milliseconds."""

    NAMES = [
        "Bellingham", "Kylian Mbappe", "N'Golo Kanté", "Cristiano Ronaldo",
        "Lamine Yamal", "Rayan Cherki", "Ousmane Dembélé", "Pedri",
        "Vinicius Junior", "Rodri", "Erling Haaland", "Lionel Messi",
        "Mohamed Salah", "Kevin De Bruyne", "Virgil Van Dijk",
        "Bukayo Saka", "Phil Foden", "Declan Rice",
        "Çalhanoğlu", "Thomas Müller",
    ]

    def test_normalize_name_1000_calls_under_100ms(self):
        t0 = time.perf_counter()
        for _ in range(50):
            for name in self.NAMES:
                _normalize_name(name)
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.10, f"_normalize_name: {elapsed*1000:.1f} ms for 1 000 calls (limit 100 ms)"

    def test_search_variants_1000_calls_under_100ms(self):
        t0 = time.perf_counter()
        for _ in range(50):
            for name in self.NAMES:
                _search_variants(name)
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.10, f"_search_variants: {elapsed*1000:.1f} ms for 1 000 calls (limit 100 ms)"

    def test_pick_best_player_200_calls_under_50ms(self):
        """pick_best_player on a 50-candidate list, 200 repetitions."""
        results = [
            _player(i, f"Player {i}", f"First{i}", f"Last{i}", [_stat(39, i * 100)])
            for i in range(49)
        ]
        results.append(_player(99, "N. Kanté", "N'Golo", "Kanté", [_stat(39, 9999)]))

        t0 = time.perf_counter()
        for _ in range(200):
            pick_best_player(results, "Ngolo Kante")
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.05, f"pick_best_player: {elapsed*1000:.1f} ms for 200 calls (limit 50 ms)"

    def test_best_stats_entry_1000_calls_under_50ms(self):
        obj = _player(1, "X", "X", "X", [_stat(39, 1000), _stat(140, 2000), _stat(9999, 9000)])
        t0 = time.perf_counter()
        for _ in range(1000):
            best_stats_entry(obj)
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.05, f"best_stats_entry: {elapsed*1000:.1f} ms for 1 000 calls (limit 50 ms)"
