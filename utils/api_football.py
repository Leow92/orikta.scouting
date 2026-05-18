# utils/api_football.py
#
# API-football (v3.football.api-sports.io) client.
# All endpoints are cached via Streamlit to minimise daily quota usage.

from __future__ import annotations
import os
from datetime import date
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

_API_KEY = os.getenv("API_FOOTBALL_KEY")
_BASE = "https://v3.football.api-sports.io"

# League IDs for the most common European competitions.
# Used to prefer major-league stats entries when a player has multiple.
MAJOR_LEAGUE_IDS: set[int] = {
    39,   # Premier League
    140,  # La Liga
    78,   # Bundesliga
    135,  # Serie A
    61,   # Ligue 1
    94,   # Primeira Liga (Portugal)
    88,   # Eredivisie
    203,  # Süper Lig
    262,  # Liga MX
    253,  # MLS
    144,  # Jupiler Pro League
    207,  # Scottish Premiership
    65,   # Copa del Rey (exclude — cup, fewer minutes)
    119,  # Eliteserien
    197,  # Super League Greece
}

# ------------------------------------------------------------------ #
# Season helper                                                        #
# ------------------------------------------------------------------ #
def current_season() -> int:
    """Return the API-football season year for the current period.
    Seasons are identified by their start year (2024 = 2024-25).
    European seasons start in July/August, so before July we're still
    in the season that started the previous year.
    """
    today = date.today()
    return today.year - 1 if today.month < 7 else today.year

@st.cache_data(ttl=24 * 3600)
def get_player_seasons(player_id: int, seasons: list[int]) -> dict[int, list[dict]]:
    """Fetch a player's stats for multiple seasons in one call."""
    return {season: get_player_by_id(player_id, season) for season in seasons}

def get_player_comparison(player_id: int) -> dict:
    """Helper to fetch current + last season stats for a player."""
    current = current_season()
    last = current - 1
    return {
        "current": get_player_by_id(player_id, current),
        "last": get_player_by_id(player_id, last),
    }

# ------------------------------------------------------------------ #
# Internal HTTP helper                                                 #
# ------------------------------------------------------------------ #
def _headers() -> dict[str, str]:
    if not _API_KEY:
        raise RuntimeError(
            "Missing API_FOOTBALL_KEY in environment / .env — "
            "add API_FOOTBALL_KEY=<your-key> to .env"
        )
    return {"x-apisports-key": _API_KEY}


def _get(endpoint: str, params: dict, timeout: int = 20) -> dict:
    resp = requests.get(
        f"{_BASE}/{endpoint}",
        headers=_headers(),
        params=params,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


# ------------------------------------------------------------------ #
# Cached API calls                                                     #
# ------------------------------------------------------------------ #
# Leagues to scan when searching by name (in priority order)
SEARCH_LEAGUES: list[int] = [39, 140, 78, 135, 61, 94, 88, 203, 262, 253]

@st.cache_data(ttl=12 * 3600)
def _search_in_league(name: str, league_id: int, season: int) -> list[dict]:
    """Name search within a single league/season. Cached per (name, league, season)."""
    data = _get("players", {"search": name, "league": league_id, "season": season})
    return data.get("response", [])

def _search_variants(name: str) -> list[str]:
    """Generate search variants to handle API-football's abbreviated name format.

    API-football stores players as "O. Dembélé" (initial + last name), so a
    full first+last query like "ousmane dembélé" never matches.  We try:
      1. The original string (works for single-word queries like "Bellingham")
      2. The last word only — usually the last name ("dembélé")
      3. Accent-stripped last name ("dembele") — handles API inconsistencies
      4. Accent-stripped full name ("ousmane dembele")
    """
    import unicodedata

    def _strip(s: str) -> str:
        return unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode().strip()

    parts   = name.strip().split()
    last    = parts[-1] if parts else name
    last_stripped  = _strip(last)
    full_stripped  = _strip(name)

    seen, variants = set(), []
    for v in [name, last, last_stripped, full_stripped]:
        if v and v not in seen:
            seen.add(v)
            variants.append(v)
    return variants


def search_player(name: str, season: int | None = None) -> list[dict]:
    """Search for a player by name across major leagues.

    The free plan requires a league+season context for name searches.
    Tries multiple name variants (full name, last name, accent-stripped)
    before moving to the next league.  All calls are cached individually.
    """
    if season is None:
        season = current_season()

    for search_name in _search_variants(name):
        for league_id in SEARCH_LEAGUES:
            results = _search_in_league(search_name, league_id, season)
            if results:
                return results
    return []


@st.cache_data(ttl=24 * 3600)
def get_player_by_id(player_id: int, season: int) -> list[dict]:
    """Fetch a player's full stats for a specific season by player ID."""
    data = _get("players", {"id": player_id, "season": season})
    return data.get("response", [])


@st.cache_data(ttl=24 * 3600)
def get_league_players(league_id: int, season: int, max_pages: int = 5) -> list[dict]:
    """Fetch up to `max_pages` pages of players from a league.
    Each page contains ~20 players.  Results are cached for 24 h.
    """
    all_players: list[dict] = []
    for page in range(1, max_pages + 1):
        data = _get(
            "players",
            {"league": league_id, "season": season, "page": page},
        )
        players = data.get("response", [])
        if not players:
            break
        all_players.extend(players)
        total_pages = data.get("paging", {}).get("total", page)
        if page >= total_pages:
            break
    return all_players


# ------------------------------------------------------------------ #
# Selection helpers                                                    #
# ------------------------------------------------------------------ #
def best_stats_entry(player_obj: dict) -> dict | None:
    """Return the statistics entry with the most minutes, preferring
    major-league competitions over cups and lower divisions.
    """
    stats: list[dict] = player_obj.get("statistics", [])
    if not stats:
        return None

    def _sort_key(s: dict) -> tuple[int, int]:
        league_id = (s.get("league") or {}).get("id", 0)
        minutes = (s.get("games") or {}).get("minutes") or 0
        return (1 if league_id in MAJOR_LEAGUE_IDS else 0, int(minutes))

    return max(stats, key=_sort_key)


def pick_best_player(results: list[dict], query: str) -> dict | None:
    """From a list of search results pick the player whose name best
    matches `query` and who has the most minutes played.
    """
    if not results:
        return None

    q = query.lower().strip()
    scored: list[tuple[int, int, dict]] = []
    for r in results:
        p = r.get("player") or {}
        full = (p.get("name") or "").lower()
        first = (p.get("firstname") or "").lower()
        last = (p.get("lastname") or "").lower()

        if q == full:
            name_score = 4
        elif q in full or full in q:
            name_score = 3
        elif q == last or q == first:
            name_score = 2
        elif any(part in full for part in q.split() if len(part) > 2):
            name_score = 1
        else:
            name_score = 0

        entry = best_stats_entry(r)
        minutes = int((entry.get("games") or {}).get("minutes") or 0) if entry else 0
        scored.append((name_score, minutes, r))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored[0][2] if scored else None
