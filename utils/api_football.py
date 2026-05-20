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

# European and national cup competition IDs.
EUROPEAN_CUP_IDS: set[int] = {
    2,    # UEFA Champions League
    3,    # UEFA Europa League
    848,  # UEFA Europa Conference League
    531,  # UEFA Super Cup
}

NATIONAL_CUP_IDS: set[int] = {
    45,   # FA Cup (England)
    48,   # EFL Cup / Carabao Cup (England)
    66,   # Coupe de France
    81,   # DFB-Pokal (Germany)
    137,  # Coppa Italia
    143,  # Copa del Rey (Spain)
    96,   # Taça de Portugal (Portugal)
    90,   # KNVB Beker (Netherlands)
    146,  # Belgian Cup
    204,  # Turkish Cup
    209,  # Scottish FA Cup
    199,  # Greek Cup
    120,  # Norwegian Cup (NM Cupen)
    114,  # Svenska Cupen
    236,  # Russian Cup
    279,  # Ukrainian Cup
    257,  # US Open Cup
    73,   # Copa do Brasil
    308,  # King Cup (Saudi Arabia)
}

CUP_COMPETITION_IDS: set[int] = EUROPEAN_CUP_IDS | NATIONAL_CUP_IDS

# League IDs used to prefer real-competition stats over cups / lower divisions.
MAJOR_LEAGUE_IDS: set[int] = {
    # Big-5 Europe
    39,   # Premier League (England)
    140,  # La Liga (Spain)
    78,   # Bundesliga (Germany)
    135,  # Serie A (Italy)
    61,   # Ligue 1 (France)
    # Europe — tier-2 domestic
    40,   # EFL Championship (England)
    79,   # 2. Bundesliga (Germany)
    62,   # Ligue 2 (France)
    136,  # Serie B (Italy)
    141,  # Segunda División (Spain)
    95,   # Liga Portugal 2
    # Europe — other top flights
    94,   # Primeira Liga (Portugal)
    88,   # Eredivisie (Netherlands)
    203,  # Süper Lig (Turkey)
    144,  # Jupiler Pro League / Belgian First Division A
    207,  # Scottish Premiership
    197,  # Super League 1 (Greece)
    119,  # Eliteserien (Norway)
    113,  # Allsvenskan (Sweden)
    235,  # Russian Premier League
    283,  # Ukrainian Premier League
    218,  # Austrian Bundesliga
    144,  # Belgian First Division A (Jupiler)
    # Americas
    253,  # MLS (USA)
    262,  # Liga MX (Mexico)
    128,  # Argentine Primera División
    71,   # Brazilian Série A
    239,  # Colombian Primera A
    265,  # Ecuadorian Serie A
    # Middle East
    307,  # Saudi Pro League (Ronaldo, Benzema, Neymar…)
    188,  # Qatar Stars League
    274,  # UAE Pro League
    # Asia & Oceania
    169,  # Chinese Super League
    292,  # K League 1 (South Korea)
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
# Leagues scanned when searching by name, ordered by prestige / likelihood.
# All calls are cached, so extra leagues only cost API quota on cold misses.
SEARCH_LEAGUES: list[int] = [
    # Big-5 Europe (most players live here)
    39, 140, 78, 135, 61,
    # Other major European top flights
    94, 88, 203, 144, 207, 197, 113, 119, 235, 283, 218,
    # Saudi Pro League (Ronaldo, Benzema…) + other Middle East
    307, 188, 274,
    # Americas
    253, 262, 128, 71, 239, 265,
    # Asia
    169, 292,
    # European tier-2 (talented youngsters, loan players)
    40, 79, 62, 136, 141, 95,
]

@st.cache_data(ttl=12 * 3600)
def _search_in_league(name: str, league_id: int, season: int) -> list[dict]:
    """Name search within a single league/season. Cached per (name, league, season)."""
    data = _get("players", {"search": name, "league": league_id, "season": season})
    return data.get("response", [])

def _normalize_name(s: str) -> str:
    """Lowercase, strip accents and apostrophes for fuzzy comparison."""
    import unicodedata
    s = s.lower().strip()
    s = s.replace("'", "").replace("’", "").replace("`", "")
    return unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode()


def _search_variants(name: str) -> list[str]:
    """Generate search variants to handle API-football's abbreviated name format.

    API-football stores players as "O. Dembélé" (initial + last name), so a
    full first+last query like "ousmane dembélé" never matches.  We try:
      1. The original string ("Rayan Cherki", "Bellingham")
      2. Last name only ("Cherki") — most reliable single-token search
      3. First name only — useful when last name is ambiguous
      4. Accent-stripped last name ("dembele")
      5. Accent-stripped full name ("ousmane dembele")
    """
    import unicodedata

    def _strip(s: str) -> str:
        return unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode().strip()

    parts  = name.strip().split()
    last   = parts[-1] if parts else name
    first  = parts[0] if len(parts) > 1 else ""
    last_stripped  = _strip(last)
    full_stripped  = _strip(name)
    first_stripped = _strip(first) if first else ""

    seen, variants = set(), []
    # Try full name first, then first name (more distinctive than last for
    # players with common surnames like Davies, Silva, Müller), then last name.
    for v in [name, first, last, full_stripped, first_stripped, last_stripped]:
        v = v.strip()
        if v and v not in seen:
            seen.add(v)
            variants.append(v)
    return variants


def _name_score(result: dict, query: str) -> int:
    """Score (0–5) how well a search result matches the queried player name.

    5  exact match on combined first+last or display name
    4  both first name AND last name words individually appear in the query
    3  display name fully contained in query or vice-versa
    2  query equals the last or first name alone
    1  at least one query word (>2 chars) appears anywhere in the name fields
    0  no overlap
    """
    p      = result.get("player") or {}
    q_norm = _normalize_name(query)
    q_parts = [w for w in q_norm.split() if len(w) > 2]

    full     = _normalize_name(p.get("name")      or "")
    first    = _normalize_name(p.get("firstname")  or "")
    last     = _normalize_name(p.get("lastname")   or "")
    combined = f"{first} {last}".strip()

    if q_norm in (full, combined):
        return 5
    # All query words appear somewhere in the player's full name — handles
    # compound surnames like "Boyle Davies" when query is "Alphonso Davies"
    if q_parts and first and last and all(qp in combined for qp in q_parts):
        return 4
    # Guard the "X in q_norm" directions: a single-word token ("Rayan") must not
    # spuriously match as a prefix of a multi-word query ("Rayan Cherki").
    _full_multi = len(full.split()) > 1
    _comb_multi = len(combined.split()) > 1
    if (
        (q_norm in full)
        or (_full_multi and full in q_norm)
        or (q_norm in combined)
        or (_comb_multi and combined in q_norm)
    ):
        return 3
    if q_norm in (last, first):
        return 2
    if q_parts and any(w in full or w in first or w in last for w in q_parts):
        return 1
    return 0


def search_player(name: str, season: int | None = None) -> list[dict]:
    """Search for a player by name across major leagues.

    The free plan requires a league+season context for name searches.
    Tries multiple name variants (full name, last name, accent-stripped)
    before moving to the next league.  All calls are cached individually.

    For multi-word queries a result set is only committed when the best
    candidate scores ≥ 3 (both first+last name matched, or display name
    overlap), preventing common-surname false positives like returning
    "Ben Davies" when "Alphonso Davies" is requested.  A weaker fallback
    is kept in case no strong match exists anywhere.
    """
    if season is None:
        season = current_season()

    q_parts = _normalize_name(name).split()
    commit_threshold = 3 if len(q_parts) >= 2 else 1

    fallback: list[dict] = []

    for search_name in _search_variants(name):
        for league_id in SEARCH_LEAGUES:
            results = _search_in_league(search_name, league_id, season)
            if not results:
                continue
            best = pick_best_player(results, name)
            if best is None:
                continue
            score = _name_score(best, name)
            if score >= commit_threshold:
                return results
            if not fallback:
                fallback = results

    return fallback


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

    Uses _name_score() for normalised matching so apostrophes/accents
    don't block correct identification ("Ngolo Kante" → "N'Golo Kanté").
    """
    if not results:
        return None

    scored: list[tuple[int, int, dict]] = []
    for r in results:
        score = _name_score(r, query)
        entry = best_stats_entry(r)
        minutes = int((entry.get("games") or {}).get("minutes") or 0) if entry else 0
        scored.append((score, minutes, r))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored[0][2] if scored else None
