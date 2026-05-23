# tools/similar.py
#
# Finds players with the most similar statistical profile to a target player
# within the same league pool. Similarity is measured via cosine distance on
# percentile vectors (same metric scale used everywhere else in the pipeline).

from __future__ import annotations
from typing import List, Dict

import numpy as np
import pandas as pd

from utils.api_football import (
    search_player, best_stats_entry, pick_best_player,
    get_league_players, current_season,
)
from utils.percentile_engine import (
    build_scout_df, get_position_str,
    stats_entry_to_per90, compute_percentiles, MIN_POOL_MINUTES,
)
from tools.analyze import _t, _player_card_html
from tools.grading import compute_grade, label_from_pair
import utils.pipeline_log as pipeline_log

SEPARATOR = "\n\n---\n\n"
TOP_N = 5

# Metrics that are only meaningful for goalkeepers. Including them for outfield
# players produces 100th-percentile artefacts (goals.conceded = 0 → rank-best).
GK_ONLY_METRICS = {"Goals Conceded per 90", "Save %", "Saves per 90"}


# ------------------------------------------------------------------
# Internals
# ------------------------------------------------------------------

def _profile_sim(a: pd.Series, b: pd.Series) -> float:
    """Pearson correlation mapped to 0–100.

    Cosine similarity on raw percentiles (all positive) only captures
    magnitude ratio and ignores profile shape — players at opposite extremes
    score nearly identical to truly similar players. Pearson correlation
    centers both vectors first, so above-average vs below-average per metric
    is what drives the score.
    """
    if len(a) < 3:
        return 0.0
    r = float(a.corr(b))
    if np.isnan(r):
        return 0.0
    return max(0.0, min(1.0, (r + 1) / 2.0)) * 100.0


def _name_overlap(pool_name: str, target_name: str) -> bool:
    a, b = pool_name.lower().strip(), target_name.lower().strip()
    return a == b or a in b or b in a


def _get_pool_position(p: dict) -> str:
    """Return position string for a pool entry.

    Mirrors get_position_str: tries games.position from the best stats entry
    first (more reliable), then falls back to player.position. The pool entries
    returned by get_league_players often have player.position = null, which
    is why using only player.position breaks the filter.
    """
    pe = best_stats_entry(p)
    if pe:
        pos = (pe.get("games") or {}).get("position")
        if pos:
            return pos
    return (p.get("player") or {}).get("position", "")


def _precompute_pool_arrays(
    pool_metrics: List[Dict[str, float]],
) -> Dict[str, np.ndarray]:
    """Build {metric: sorted_array} from the pool for fast percentile lookup."""
    acc: Dict[str, list] = {}
    for pm in pool_metrics:
        for k, v in pm.items():
            if k == "Minutes":
                continue
            acc.setdefault(k, []).append(v)
    return {k: np.array(v, dtype=float) for k, v in acc.items()}


def _pct_from_arrays(
    per90: Dict[str, float],
    arrays: Dict[str, np.ndarray],
) -> pd.Series:
    result: Dict[str, float] = {}
    for metric, val in per90.items():
        if metric == "Minutes":
            continue
        arr = arrays.get(metric)
        if arr is None or len(arr) < 5:
            result[metric] = 50.0
        else:
            result[metric] = round(float(np.mean(arr <= val) * 100), 1)
    return pd.Series(result)


def _shared_strengths(
    target_pct: pd.Series,
    top_results: List[dict],
    threshold: float = 65.0,
    max_out: int = 4,
) -> List[str]:
    """Metrics where both the target and the similar group rank high."""
    out = []
    for metric in target_pct.index:
        if target_pct[metric] < threshold:
            continue
        peer_vals = [
            r["pcts"].get(metric, np.nan)
            for r in top_results
            if metric in r["pcts"].index
        ]
        peer_vals_clean = [v for v in peer_vals if not np.isnan(v)]
        if peer_vals_clean and np.mean(peer_vals_clean) >= threshold:
            out.append((metric, float(target_pct[metric])))
    out.sort(key=lambda x: x[1], reverse=True)
    return [m for m, _ in out[:max_out]]


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def similar_players(
    players: list[str],
    language: str = "English",
    skip_llm: bool = False,
    user_query: str = "",
) -> str:
    """
    Find the TOP_N most statistically similar players to the target
    in the same league pool. Returns a markdown string.
    """
    try:
        assert isinstance(players, list) and len(players) >= 1
        target_raw = players[0]
        season = current_season()

        # ---- 1. Fetch target player ----
        pipeline_log.log(f"[similar] Searching for '{target_raw}' (season {season})…")
        results = search_player(target_raw, season=season)
        if not results:
            pipeline_log.log(f"[similar] Retrying season {season - 1}…", level="warning")
            results = search_player(target_raw, season=season - 1)
            if results:
                season = season - 1
        if not results:
            raise RuntimeError(f"No player found for: {target_raw}")

        player_obj = pick_best_player(results, target_raw)
        if not player_obj:
            raise RuntimeError(f"Could not identify best match for: {target_raw}")

        player_info = player_obj.get("player") or {}
        full_name   = player_info.get("name") or target_raw.title()
        photo_url   = player_info.get("photo") or ""
        position_str = get_position_str(player_obj)
        pipeline_log.log(f"[similar] Matched → {full_name}, position: {position_str}", level="success")

        entry = best_stats_entry(player_obj)
        if not entry:
            raise RuntimeError(f"No statistics for {full_name} in season {season}.")

        league_id   = (entry.get("league") or {}).get("id")
        league_name = (entry.get("league") or {}).get("name", "Unknown")

        # ---- 2. Fetch league pool ----
        pool: list[dict] = []
        if league_id:
            pipeline_log.log(f"[similar] Fetching pool: {league_name} (id={league_id})…")
            pool = get_league_players(league_id, season, max_pages=30)
            pipeline_log.log(f"[similar] Pool size: {len(pool)} players", level="success")

        is_gk = position_str == "Goalkeeper"

        # ---- 3. Build eligible pool (same position, 450+ min) ----
        def _eligible(p: dict) -> tuple[dict, dict] | None:
            pe = best_stats_entry(p)
            if pe is None:
                return None
            pm = stats_entry_to_per90(pe)
            if pm.get("Minutes", 0) < MIN_POOL_MINUTES:
                return None
            return p, pm

        def _build_pool(position_filter: str | None) -> tuple[list, list]:
            metrics, entries = [], []
            for p in pool:
                result = _eligible(p)
                if result is None:
                    continue
                p_obj, pm = result
                if position_filter:
                    pool_pos = _get_pool_position(p_obj)
                    if pool_pos and pool_pos != position_filter:
                        continue
                metrics.append(pm)
                entries.append((p_obj, pm))
            return metrics, entries

        pool_metrics, pool_entries = _build_pool(position_str)
        pipeline_log.log(f"[similar] Eligible pool players: {len(pool_entries)}")

        # Fallback: if fewer than 10 same-position players, open to all positions
        if len(pool_entries) < 10:
            pipeline_log.log("[similar] Too few position-matched players — opening to all positions", level="warning")
            pool_metrics, pool_entries = _build_pool(None)

        # ---- 4. Single reference pool for ALL percentile computations ----
        # Both the target and every candidate are ranked against the same pool,
        # so Pearson correlation compares vectors on the same scale.
        pool_arrays = _precompute_pool_arrays(pool_metrics)

        target_per90 = stats_entry_to_per90(entry)
        target_per90.pop("Minutes", None)
        if not is_gk:
            for m in GK_ONLY_METRICS:
                target_per90.pop(m, None)
        target_pct = _pct_from_arrays(target_per90, pool_arrays)
        pipeline_log.log(f"[similar] Target profile: {len(target_pct)} metrics")

        # ---- 5. Compute similarity for each pool player ----
        candidates = []
        for p_obj, p_metrics in pool_entries:
            p_info = p_obj.get("player") or {}
            p_name = p_info.get("name", "Unknown")
            if _name_overlap(p_name, full_name):
                continue

            per90 = {k: v for k, v in p_metrics.items()
                     if k != "Minutes" and (is_gk or k not in GK_ONLY_METRICS)}
            p_pct = _pct_from_arrays(per90, pool_arrays)
            common = sorted(set(target_pct.index) & set(p_pct.index))
            if len(common) < 5:
                continue

            sim = _profile_sim(target_pct.reindex(common), p_pct.reindex(common))

            p_entry = best_stats_entry(p_obj)
            p_team  = (p_entry.get("team") or {}).get("name", "—") if p_entry else "—"
            p_age   = p_info.get("age")
            p_pos   = _get_pool_position(p_obj) or "—"

            candidates.append({
                "name":       p_name,
                "team":       p_team,
                "age":        p_age,
                "position":   p_pos,
                "similarity": sim,
                "pcts":       p_pct,
            })

        candidates.sort(key=lambda x: x["similarity"], reverse=True)
        top_results = candidates[:TOP_N]
        pipeline_log.log(
            f"[similar] Top match: {top_results[0]['name']} ({top_results[0]['similarity']:.1f}/100)"
            if top_results else "[similar] No candidates found",
            level="success" if top_results else "warning",
        )

        if not top_results:
            return _t(
                f"⚠️ No similar players found for {full_name} in {league_name}.",
                f"⚠️ Aucun joueur similaire trouvé pour {full_name} dans {league_name}.",
                language,
            )

        # ---- 6. Grade & role label (build_scout_df used only for grading) ----
        scout_df = build_scout_df(player_obj, pool, position_filter=position_str)
        grade_bd = compute_grade(scout_df, role_hint=position_str)
        if ":" in grade_bd.role:
            _base, _sub = grade_bd.role.split(":", 1)
        else:
            _base, _sub = grade_bd.role, None
        role_label = label_from_pair(_base, _sub)

        # ---- 8. Shared profile strengths ----
        shared = _shared_strengths(target_pct, top_results)

        # ---- 9. Build markdown ----
        title = _t(
            f"### 🔍 Similar Players — {full_name} · {role_label}",
            f"### 🔍 Joueurs similaires — {full_name} · {role_label}",
            language,
        )
        subtitle = _t(
            f"_Top {TOP_N} most statistically similar players in the same league pool ({league_name})._",
            f"_Top {TOP_N} joueurs statistiquement les plus proches dans le même pool ({league_name})._",
            language,
        )

        header_row = (
            f"| # | {_t('Player', 'Joueur', language)} "
            f"| {_t('Club', 'Club', language)} "
            f"| {_t('Position', 'Poste', language)} "
            f"| {_t('Age', 'Âge', language)} "
            f"| {_t('Similarity', 'Similarité', language)} |"
        )
        sep_row = "|---|---|---|---|---:|---:|"
        rows = []
        for i, r in enumerate(top_results, 1):
            age_str = str(r["age"]) if r["age"] else "—"
            rows.append(
                f"| {i} | **{r['name']}** | {r['team']} "
                f"| {r['position']} | {age_str} | {r['similarity']:.1f}/100 |"
            )
        table_md = "\n".join([header_row, sep_row] + rows)

        shared_md = ""
        if shared:
            label = _t("Shared profile strengths", "Forces communes du profil", language)
            shared_md = f"\n**{label}:** {', '.join(shared)}\n"

        photo_html = (
            '<div style="margin:12px 0 20px;">'
            + _player_card_html(photo_url, full_name, "#4CAF50")
            + "</div>"
        ) if photo_url else ""

        return (
            f"{photo_html}\n\n"
            f"{title}\n\n"
            f"{subtitle}\n\n"
            f"{table_md}\n"
            f"{shared_md}"
        )

    except Exception as e:
        pipeline_log.log(f"[similar] Unhandled error: {e}", level="error")
        return f"⚠️ Similar players search failed: {e}"
