# tools/team_builder.py
#
# Formation definitions, per-slot player fetch + grade, and team scoring
# for the Team Builder feature.

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from utils.api_football import (
    search_player, get_player_by_id, get_league_players,
    best_stats_entry, pick_best_player, current_season,
)
from utils.percentile_engine import build_scout_df, get_position_str
from tools.grading import compute_grade


# ------------------------------------------------------------------ #
# Formation definitions                                               #
# ------------------------------------------------------------------ #
@dataclass
class Slot:
    id: int
    label: str    # display label shown in the UI (GK, RB, CB, …)
    role: str     # grading role hint (gk, df:cb, mf:cm, …)
    zone: str     # gk | defense | midfield | attack
    x: float      # 0=left 1=right on the pitch diagram
    y: float      # 0=GK end 1=attack end


FORMATIONS: dict[str, list[Slot]] = {
    "4-3-3": [
        Slot(0,  "GK",  "gk",    "gk",       0.50, 0.07),
        Slot(1,  "RB",  "df:fb", "defense",  0.82, 0.24),
        Slot(2,  "CB",  "df:cb", "defense",  0.61, 0.24),
        Slot(3,  "CB",  "df:cb", "defense",  0.39, 0.24),
        Slot(4,  "LB",  "df:fb", "defense",  0.18, 0.24),
        Slot(5,  "CM",  "mf:cm", "midfield", 0.67, 0.52),
        Slot(6,  "CM",  "mf:cm", "midfield", 0.50, 0.52),
        Slot(7,  "CM",  "mf:cm", "midfield", 0.33, 0.52),
        Slot(8,  "RW",  "fw:w",  "attack",   0.82, 0.82),
        Slot(9,  "ST",  "fw:st", "attack",   0.50, 0.82),
        Slot(10, "LW",  "fw:w",  "attack",   0.18, 0.82),
    ],
    "4-4-2": [
        Slot(0,  "GK",  "gk",    "gk",       0.50, 0.07),
        Slot(1,  "RB",  "df:fb", "defense",  0.82, 0.24),
        Slot(2,  "CB",  "df:cb", "defense",  0.61, 0.24),
        Slot(3,  "CB",  "df:cb", "defense",  0.39, 0.24),
        Slot(4,  "LB",  "df:fb", "defense",  0.18, 0.24),
        Slot(5,  "RM",  "mf:wm", "midfield", 0.82, 0.54),
        Slot(6,  "CM",  "mf:cm", "midfield", 0.61, 0.54),
        Slot(7,  "CM",  "mf:cm", "midfield", 0.39, 0.54),
        Slot(8,  "LM",  "mf:wm", "midfield", 0.18, 0.54),
        Slot(9,  "ST",  "fw:st", "attack",   0.62, 0.82),
        Slot(10, "ST",  "fw:st", "attack",   0.38, 0.82),
    ],
}

ZONE_ORDER = ["gk", "defense", "midfield", "attack"]


# ------------------------------------------------------------------ #
# Per-slot player fetch and grade                                     #
# ------------------------------------------------------------------ #
@dataclass
class SlotResult:
    slot: Slot
    query: str = ""
    found_name: str = ""
    grade: Optional[float] = None
    error: str = ""


def fetch_slot(slot: Slot, query: str, season: int | None = None) -> SlotResult:
    """Search, grade and return a single player for a position slot."""
    result = SlotResult(slot=slot, query=query)
    if not query.strip():
        return result

    if season is None:
        season = current_season()

    try:
        results = search_player(query, season=season)
        if not results:
            results = search_player(query, season=season - 1)
            if results:
                season = season - 1
        if not results:
            result.error = "Player not found"
            return result

        player_obj = pick_best_player(results, query)
        if not player_obj:
            result.error = "No match found"
            return result

        player_info = player_obj.get("player") or {}
        result.found_name = player_info.get("name") or query.title()
        player_id = player_info.get("id")

        # Enrich with full multi-competition stats
        if player_id:
            full_objs = get_player_by_id(player_id, season)
            if full_objs:
                player_obj = full_objs[0]

        entry = best_stats_entry(player_obj)
        if not entry:
            result.error = "No stats available"
            return result

        league_id = (entry.get("league") or {}).get("id")
        position_str = get_position_str(player_obj)

        pool: list[dict] = []
        if league_id:
            pool = get_league_players(league_id, season, max_pages=5)

        scout_df = build_scout_df(player_obj, pool, position_filter=position_str)
        grade_bd = compute_grade(scout_df, role_hint=slot.role)
        result.grade = round(grade_bd.final_score, 1)

    except Exception as e:
        result.error = str(e)

    return result


# ------------------------------------------------------------------ #
# Team score computation                                              #
# ------------------------------------------------------------------ #
def compute_team_scores(results: list[SlotResult]) -> dict:
    """Compute overall score and per-zone scores from slot results.

    Returns:
        {"overall": float|None, "zones": {"gk": float|None, ...}}
    """
    zone_grades: dict[str, list[float]] = {z: [] for z in ZONE_ORDER}
    for r in results:
        if r.grade is not None:
            zone_grades[r.slot.zone].append(r.grade)

    zone_scores: dict[str, float | None] = {}
    all_grades: list[float] = []
    for zone, grades in zone_grades.items():
        if grades:
            zone_scores[zone] = round(sum(grades) / len(grades), 1)
            all_grades.extend(grades)
        else:
            zone_scores[zone] = None

    overall = round(sum(all_grades) / len(all_grades), 1) if all_grades else None
    return {"overall": overall, "zones": zone_scores}
