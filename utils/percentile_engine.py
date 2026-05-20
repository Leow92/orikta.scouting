# utils/percentile_engine.py
#
# Converts raw API-football player objects into the scout_df format
# (index=Metric, columns=[Per90, Percentile]) consumed by the grading
# and visualisation layers.
#
# Percentiles are computed by comparing the target player against a pool
# of players from the same league / season.

from __future__ import annotations
import numpy as np
import pandas as pd
from utils.api_football import best_stats_entry

# Maps API-football position strings to our role taxonomy
POSITION_MAP: dict[str, str] = {
    "Goalkeeper": "gk",
    "Defender": "df",
    "Midfielder": "mf",
    "Attacker": "fw",
}

# Minimum minutes for a pool player to be included in percentile reference set
MIN_POOL_MINUTES = 450


# ------------------------------------------------------------------ #
# Raw stats → per-90 dict                                             #
# ------------------------------------------------------------------ #
def _safe(x) -> float | None:
    try:
        v = float(x)
        return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None


def stats_entry_to_per90(entry: dict) -> dict[str, float]:
    """Convert a single statistics entry from API-football into a dict of
    per-90 (or percentage) metrics.

    Returns an empty dict if the player has 0 minutes.
    Includes a special "Minutes" key (raw value) used for pool filtering.
    """
    minutes = _safe((entry.get("games") or {}).get("minutes")) or 0.0
    if minutes < 1:
        return {}
    nineties = minutes / 90.0

    g   = entry.get("goals")     or {}
    sh  = entry.get("shots")     or {}
    pa  = entry.get("passes")    or {}
    ta  = entry.get("tackles")   or {}
    du  = entry.get("duels")     or {}
    dr  = entry.get("dribbles")  or {}
    fo  = entry.get("fouls")     or {}

    goals      = _safe(g.get("total"))
    assists    = _safe(g.get("assists"))
    saves      = _safe(g.get("saves"))
    conceded   = _safe(g.get("conceded"))

    shots_tot  = _safe(sh.get("total"))
    shots_on   = _safe(sh.get("on"))

    pass_key   = _safe(pa.get("key"))
    pass_acc   = _safe(pa.get("accuracy"))  # already a percentage

    tackles    = _safe(ta.get("total"))
    blocks     = _safe(ta.get("blocks"))
    intercepts = _safe(ta.get("interceptions"))

    duels_tot  = _safe(du.get("total"))
    duels_won  = _safe(du.get("won"))

    drib_att   = _safe(dr.get("attempts"))
    drib_suc   = _safe(dr.get("success"))

    fouls_c    = _safe(fo.get("committed"))
    fouls_d    = _safe(fo.get("drawn"))

    m: dict[str, float] = {"Minutes": minutes}

    # ---- Outfield attacking ----
    if goals is not None:
        m["Goals per 90"] = goals / nineties
    if assists is not None:
        m["Assists per 90"] = assists / nineties
    if goals is not None and assists is not None:
        m["G+A per 90"] = (goals + assists) / nineties
    if shots_tot is not None:
        m["Shots per 90"] = shots_tot / nineties
        if shots_on is not None and shots_tot > 0:
            m["Shot Accuracy %"] = shots_on / shots_tot * 100

    # ---- Passing / creativity ----
    if pass_key is not None:
        m["Key Passes per 90"] = pass_key / nineties
    if pass_acc is not None:
        m["Pass Completion %"] = pass_acc

    # ---- Defensive ----
    if tackles is not None:
        m["Tackles per 90"] = tackles / nineties
    if intercepts is not None:
        m["Interceptions per 90"] = intercepts / nineties
    if blocks is not None:
        m["Blocks per 90"] = blocks / nineties

    # ---- Dribbling ----
    if drib_att is not None:
        m["Dribbles per 90"] = drib_att / nineties
        if drib_suc is not None and drib_att > 0:
            m["Dribble Success %"] = drib_suc / drib_att * 100

    # ---- Duels ----
    if duels_tot is not None and duels_tot > 0 and duels_won is not None:
        m["Duels Won %"] = duels_won / duels_tot * 100

    # ---- Discipline ----
    if fouls_c is not None:
        m["Fouls per 90"] = fouls_c / nineties
    if fouls_d is not None:
        m["Fouls Drawn per 90"] = fouls_d / nineties

    # ---- Goalkeeper ----
    if saves is not None:
        m["Saves per 90"] = saves / nineties
        if conceded is not None and (saves + conceded) > 0:
            m["Save %"] = saves / (saves + conceded) * 100
    if conceded is not None:
        m["Goals Conceded per 90"] = conceded / nineties

    return m


# ------------------------------------------------------------------ #
# Percentile computation                                              #
# ------------------------------------------------------------------ #
def compute_percentiles(
    target: dict[str, float],
    pool: list[dict[str, float]],
) -> dict[str, float]:
    """Compute percentile rank (0–100) for each metric in `target`
    against the provided pool of per-90 dicts.

    Falls back to 50.0 if fewer than 5 pool values are available
    for a given metric.
    """
    filtered = [p for p in pool if p.get("Minutes", 0) >= MIN_POOL_MINUTES]

    result: dict[str, float] = {}
    for metric, val in target.items():
        vals = [p[metric] for p in filtered if metric in p]
        if len(vals) < 5:
            result[metric] = 50.0
            continue
        arr = np.array(vals, dtype=float)
        pct = float(np.mean(arr <= val) * 100)
        result[metric] = round(pct, 1)
    return result


# ------------------------------------------------------------------ #
# Public builders                                                     #
# ------------------------------------------------------------------ #
def build_scout_df(
    player_obj: dict,
    pool: list[dict],
    position_filter: str | None = None,
) -> pd.DataFrame:
    """Build a scout_df (Metric / Per90 / Percentile) from an API-football
    player object and a league player pool.

    Args:
        player_obj: Single player+statistics dict from API-football.
        pool: List of player+statistics dicts for the same league/season.
        position_filter: If given (e.g. "Midfielder"), restrict pool to that
            position for more meaningful percentile comparisons.
    """
    entry = best_stats_entry(player_obj)
    if not entry:
        raise RuntimeError("No statistics entry found for player.")

    target = stats_entry_to_per90(entry)
    target.pop("Minutes", None)

    # Build per-90 dicts for pool players
    pool_metrics: list[dict[str, float]] = []
    for p in pool:
        pe = best_stats_entry(p)
        if pe is None:
            continue
        pm = stats_entry_to_per90(pe)
        if position_filter:
            pool_pos = (p.get("player") or {}).get("position", "")
            if pool_pos and pool_pos != position_filter:
                continue
        pool_metrics.append(pm)

    percentiles = compute_percentiles(target, pool_metrics)

    if not target:
        raise RuntimeError(
            "All statistics fields are null for this player entry — "
            "the API returned minutes but no counting stats."
        )

    rows = [
        {
            "Metric": metric,
            "Per90": round(val, 3),
            "Percentile": percentiles.get(metric, 50.0),
        }
        for metric, val in target.items()
    ]
    return pd.DataFrame(rows).set_index("Metric")


def get_position_str(player_obj: dict) -> str:
    """Return position string from statistics entry (more reliable than player.position)."""
    entry = best_stats_entry(player_obj)
    if entry:
        pos = (entry.get("games") or {}).get("position")
        if pos:
            return pos
    return "Midfielder"


def build_profile(player_obj: dict) -> dict:
    """Build a profile dict compatible with the existing presentation layer.

    Returns:
        {
            "name": str | None,
            "attributes": [{"label": str, "value": str}, ...],
            "paragraphs": [],
            "position_hint": "fw"|"mf"|"df"|"gk"|None,
        }
    """
    p = player_obj.get("player") or {}
    entry = best_stats_entry(player_obj)
    raw_pos = get_position_str(player_obj)

    attrs: list[dict[str, str]] = []
    if p.get("nationality"):
        attrs.append({"label": "Nationality", "value": p["nationality"]})
    if p.get("age"):
        attrs.append({"label": "Age", "value": str(p["age"])})
    h = p.get("height")
    if h:
        attrs.append({"label": "Height", "value": f"{h} cm" if str(h).isdigit() else str(h)})
    w = p.get("weight")
    if w:
        attrs.append({"label": "Weight", "value": f"{w} kg" if str(w).isdigit() else str(w)})
    if raw_pos:
        attrs.append({"label": "Position", "value": raw_pos})

    if entry:
        team    = (entry.get("team")   or {}).get("name")
        league  = (entry.get("league") or {}).get("name")
        country = (entry.get("league") or {}).get("country")
        apps    = (entry.get("games")  or {}).get("appearences")
        mins    = (entry.get("games")  or {}).get("minutes")
        rating  = (entry.get("games")  or {}).get("rating")
        if team:
            attrs.append({"label": "Club", "value": team})
        if league:
            league_str = f"{league} ({country})" if country else league
            attrs.append({"label": "League", "value": league_str})
        if apps is not None:
            attrs.append({"label": "Appearances", "value": str(apps)})
        if mins is not None:
            attrs.append({"label": "Minutes", "value": str(mins)})
        if rating:
            attrs.append({"label": "Avg Rating", "value": str(round(float(rating), 2))})

    return {
        "name": p.get("name"),
        "attributes": attrs,
        "paragraphs": [],
        "position_hint": POSITION_MAP.get(raw_pos),
    }


def build_season_comparison(
    current_obj: dict,
    prev_obj: dict | None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return (current_per90, prev_per90) metric dicts for trend analysis."""
    curr_entry = best_stats_entry(current_obj)
    curr_metrics = stats_entry_to_per90(curr_entry) if curr_entry else {}
    curr_metrics.pop("Minutes", None)

    prev_metrics: dict[str, float] = {}
    if prev_obj:
        prev_entry = best_stats_entry(prev_obj)
        if prev_entry:
            prev_metrics = stats_entry_to_per90(prev_entry)
            prev_metrics.pop("Minutes", None)

    return curr_metrics, prev_metrics
