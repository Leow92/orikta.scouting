# tools/compare.py

from __future__ import annotations
from typing import Tuple, List, Dict, Optional
import math
import pandas as pd
import numpy as np

from utils.api_football import (
    search_player, get_player_by_id, get_league_players,
    best_stats_entry, pick_best_player, current_season,
)
from utils.percentile_engine import (
    build_scout_df, build_profile, get_position_str, build_season_comparison,
)
from tools.analyze import (
    build_trend_block_for_llm, _t, _nodata, _to_numeric_safely, _player_card_html,
    _merge_profile_items,
)
from tools.grading import (
    compute_grade, label_from_pair,
    DEFAULT_WEIGHTS, SUBROLE_WEIGHTS, SUBROLE_BLEND,
    NEGATIVE_KEYS, ALIASES,
    PLAY_STYLE_PRESETS, PLAY_STYLE_PRETTY,
)
from utils.llm_analysis_comparison import compare_llm_workflow
from utils.lang import _is_fr
import utils.pipeline_log as pipeline_log
from prompts.lang import glossary_block as _glossary_block_for
from ui.graph import create_spider_graph_duo
from requests.exceptions import ReadTimeout
from utils.llm_client import llm_chat

# -----------------------------
# Deterministic helpers
# -----------------------------
SEPARATOR = "\n\n---\n\n"

def _invert_if_negative(metric: str, pct: float) -> float:
    for neg in NEGATIVE_KEYS:
        if neg.lower() in metric.lower():
            return 100.0 - pct
    return pct

def _blend_role_weights(base_w: Dict[str, float], sub_w: Optional[Dict[str, float]]) -> Dict[str, float]:
    if not sub_w:
        total = sum(base_w.values()) or 1.0
        return {k: v / total for k, v in base_w.items()}
    out: Dict[str, float] = {}
    for k in set(base_w) | set(sub_w):
        out[k] = base_w.get(k, 0.0) * (1 - SUBROLE_BLEND) + sub_w.get(k, 0.0) * SUBROLE_BLEND
    tot = sum(out.values()) or 1.0
    return {k: v / tot for k, v in out.items()}

def _normalize_weight_keys(W: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in (W or {}).items():
        if isinstance(v, (int, float)):
            out[str(k).lower()] = float(v)
    return out

def _resolve_style_preset(style_key: Optional[str], base: Optional[str], sub: Optional[str]) -> Dict[str, float]:
    P = PLAY_STYLE_PRESETS.get(style_key, {}) if style_key else {}
    flat: Dict[str, float] = {}
    if not isinstance(P, dict):
        return flat
    for k, v in P.items():
        if isinstance(v, (int, float)):
            flat[str(k).lower()] = flat.get(str(k).lower(), 0.0) + float(v)
    for scope in (f"{base}", f"{base}:{sub}" if sub else None):
        if scope and scope in P and isinstance(P[scope], dict):
            for k, v in P[scope].items():
                if isinstance(v, (int, float)):
                    flat[str(k).lower()] = flat.get(str(k).lower(), 0.0) + float(v)
    return flat

def _style_reweight(
    W_role: Dict[str, float],
    style_key: Optional[str],
    style_influence: float,
    base: Optional[str] = None,
    sub: Optional[str] = None,
) -> Dict[str, float]:
    Wb = _normalize_weight_keys(W_role)
    if not style_key:
        return Wb
    delta = _resolve_style_preset(style_key, base, sub)
    delta = _normalize_weight_keys(delta)
    keys = set(Wb) | set(delta)
    out = {k: max(0.0, Wb.get(k, 0.0) + style_influence * delta.get(k, 0.0)) for k in keys}
    tot = sum(out.values()) or 1.0
    return {k: v / tot for k, v in out.items()}

def _align_metrics(dfA: pd.DataFrame, dfB: pd.DataFrame) -> Tuple[pd.Series, pd.Series, List[str]]:
    sA = pd.to_numeric(dfA["Percentile"], errors="coerce")
    sB = pd.to_numeric(dfB["Percentile"], errors="coerce")
    sA.index = dfA.index.astype(str)
    sB.index = dfB.index.astype(str)
    sA = sA.groupby(level=0).mean()
    sB = sB.groupby(level=0).mean()

    def _map_aliases(idx: List[str]) -> Dict[str, str]:
        present = {name.lower(): name for name in idx}
        out: Dict[str, str] = {}
        for canonical, aliases in ALIASES.items():
            cl = canonical.lower()
            if cl in present:
                out[cl] = present[cl]
                continue
            for al in aliases:
                al_l = al.lower()
                if al_l in present:
                    out[cl] = present[al_l]
                    break
        for name in idx:
            out.setdefault(name.lower(), present[name.lower()])
        return out

    mapA = _map_aliases(sA.index.tolist())
    mapB = _map_aliases(sB.index.tolist())
    common = sorted(set(mapA.keys()) & set(mapB.keys()))
    if not common:
        return pd.Series(dtype=float), pd.Series(dtype=float), []

    A_vals, B_vals = {}, {}
    for k in common:
        vA = sA.get(mapA[k], np.nan)
        vB = sB.get(mapB[k], np.nan)
        if isinstance(vA, pd.Series):
            vA = float(np.nanmean(vA.values))
        if isinstance(vB, pd.Series):
            vB = float(np.nanmean(vB.values))
        A_vals[k] = float(vA) if pd.notna(vA) else np.nan
        B_vals[k] = float(vB) if pd.notna(vB) else np.nan

    A = pd.Series(A_vals, name="A", dtype=float).dropna()
    B = pd.Series(B_vals, name="B", dtype=float).dropna()
    common_idx = sorted(set(A.index) & set(B.index))
    return A.reindex(common_idx), B.reindex(common_idx), common_idx

def _head_to_head_score(A: pd.Series, B: pd.Series, W: Dict[str, float]) -> Tuple[float, float, pd.DataFrame]:
    rows = []
    for canon in A.index:
        w = float(W.get(canon, 0.0))
        a_raw = A.loc[canon]
        b_raw = B.loc[canon]
        if isinstance(a_raw, pd.Series):
            a_raw = float(np.nanmean(a_raw.values))
        if isinstance(b_raw, pd.Series):
            b_raw = float(np.nanmean(b_raw.values))
        a = _invert_if_negative(canon, float(a_raw))
        b = _invert_if_negative(canon, float(b_raw))
        rows.append((canon, w, a, b, w * a, w * b, a - b))
    det = pd.DataFrame(rows, columns=["Metric", "w", "A_pct", "B_pct", "A_contrib", "B_contrib", "Δp"])
    wsum = det["w"].sum() or 1.0
    scoreA = det["A_contrib"].sum() / wsum
    scoreB = det["B_contrib"].sum() / wsum
    return float(scoreA), float(scoreB), det.sort_values("w", ascending=False)

def _cosine_similarity(A: pd.Series, B: pd.Series) -> float:
    a = A.values; b = B.values
    na = math.sqrt((a * a).sum()); nb = math.sqrt((b * b).sum())
    if na == 0 or nb == 0:
        return 0.0
    sim = float((a * b).sum() / (na * nb))
    return max(0.0, min(1.0, (sim + 1) / 2.0)) * 100.0

def _comparison_profile_md(
    name_a: str, profile_a: dict,
    name_b: str, profile_b: dict,
    language: str,
) -> str:
    title = _t("### 👤 Players at a Glance", "### 👤 Présentation des joueurs", language)
    items_a = _merge_profile_items(profile_a)
    items_b = _merge_profile_items(profile_b)

    dict_a = {it["label"].lower(): it["value"] for it in items_a}
    dict_b = {it["label"].lower(): it["value"] for it in items_b}

    # Union of all labels, A-order first then B-only additions
    seen: dict[str, str] = {}
    for it in items_a:
        seen.setdefault(it["label"].lower(), it["label"])
    for it in items_b:
        seen.setdefault(it["label"].lower(), it["label"])

    header = f"| Field | {name_a} | {name_b} |"
    sep = "|---|---|---|"
    rows = []
    for label_lower, label_display in seen.items():
        val_a = dict_a.get(label_lower, "—")
        val_b = dict_b.get(label_lower, "—")
        if val_a != "—" or val_b != "—":
            rows.append(f"| **{label_display}** | {val_a} | {val_b} |")

    if not rows:
        return f"{title}\n\n**{name_a}** vs **{name_b}**\n"
    return f"{title}\n\n" + "\n".join([header, sep] + rows)


# -----------------------------
# Fetch & normalise one player
# -----------------------------
def _fetch_player(name: str, language: str = "English") -> dict:
    """
    Returns dict with:
      full_name, profile, scout_df (Metric/Per90/Percentile),
      current_metrics, prev_metrics, trend_block_md, position_str, photo_url
    """
    season = current_season()

    pipeline_log.log(f"[compare] Searching API-football for '{name}' (season {season})…")
    results = search_player(name, season=season)
    if not results:
        pipeline_log.log(f"[compare] No results in season {season}, retrying season {season - 1}…", level="warning")
        results = search_player(name, season=season - 1)
        if results:
            season = season - 1
    if not results:
        pipeline_log.log(f"[compare] Player not found: {name}", level="error")
        raise RuntimeError(f"No player found for: {name}")

    player_obj = pick_best_player(results, name)
    if not player_obj:
        pipeline_log.log(f"[compare] Could not pick best match for: {name}", level="error")
        raise RuntimeError(f"Could not identify best match for: {name}")

    player_info  = player_obj.get("player") or {}
    full_name    = player_info.get("name") or name.title()
    player_id    = player_info.get("id")
    photo_url    = player_info.get("photo") or ""
    position_str = get_position_str(player_obj)
    pipeline_log.log(f"[compare] Matched → {full_name} (id={player_id})", level="success")

    entry = best_stats_entry(player_obj)
    if not entry:
        pipeline_log.log(f"[compare] No stats entry for {full_name} season {season}", level="error")
        raise RuntimeError(f"No statistics for {full_name} in season {season}.")

    league_id   = (entry.get("league") or {}).get("id")
    league_name = (entry.get("league") or {}).get("name", "Unknown")
    pool: list[dict] = []
    if league_id:
        pipeline_log.log(f"[compare] Fetching league pool: {league_name} (id={league_id}), season {season}…")
        pool = get_league_players(league_id, season, max_pages=5)
        if pool:
            pipeline_log.log(f"[compare] Pool fetched: {len(pool)} players in {league_name}", level="success")
        else:
            pipeline_log.log(f"[compare] No pool data for {league_name} — percentiles unavailable", level="warning")

    scout_df = build_scout_df(player_obj, pool, position_filter=position_str)
    pipeline_log.log(f"[compare] Scout DataFrame for {full_name}: {len(scout_df)} metrics")
    profile  = build_profile(player_obj)

    # Per-90 dicts for trend analysis
    prev_objs = get_player_by_id(player_id, season - 1) if player_id else []
    prev_obj  = prev_objs[0] if prev_objs else None
    current_metrics, prev_metrics = build_season_comparison(player_obj, prev_obj)

    trend_block_md, _ = build_trend_block_for_llm(current_metrics, prev_metrics, language)

    return {
        "full_name":       full_name,
        "profile":         profile,
        "scout_df":        scout_df,
        "current_metrics": current_metrics,
        "prev_metrics":    prev_metrics,
        "trend_block_md":  trend_block_md,
        "position_str":    position_str,
        "photo_url":       photo_url,
    }

# -----------------------------
# LLM streaming
# -----------------------------
def _call_twice(prompt_text: str, language: str) -> str:
    try:
        return llm_chat(prompt_text, language)
    except ReadTimeout:
        return llm_chat(prompt_text, language)

def _photos_header_html(url_a: str, name_a: str, url_b: str, name_b: str) -> str:
    return (
        '<div style="display:flex;gap:32px;align-items:flex-start;margin:12px 0 20px;">'
        + _player_card_html(url_a, name_a, "#4CAF50")
        + '<div style="font-size:2em;padding-top:28px;">⚡</div>'
        + _player_card_html(url_b, name_b, "#2196F3")
        + "</div>"
    )

# -----------------------------
# Public API
# -----------------------------
def compare_players(
    players: list[str],
    language: str = "English",
    target_role: str | None = None,
    styles: list[str] | None = None,
    style_influence: float = 0,
    skip_llm: bool = False,
) -> str:
    """
    Head-to-head report between two players (API-football edition).
    Returns markdown string; never raises (returns error text on failure).
    """
    try:
        assert isinstance(players, list) and len(players) == 2
        A_raw, B_raw = players[0], players[1]

        A = _fetch_player(A_raw, language=language)
        B = _fetch_player(B_raw, language=language)

        A_name, B_name = A["full_name"], B["full_name"]
        scoutA, scoutB = A["scout_df"], B["scout_df"]
        photos_html = _photos_header_html(
            A.get("photo_url", ""), A_name,
            B.get("photo_url", ""), B_name,
        )
        comparison_profile_md = _comparison_profile_md(
            A_name, A["profile"],
            B_name, B["profile"],
            language,
        )
        pipeline_log.log(f"[compare] Both players fetched: {A_name} vs {B_name}", level="success")

        # Align percentile series
        pipeline_log.log("[compare] Aligning common metrics…")
        A_p, B_p, common_idx = _align_metrics(scoutA, scoutB)
        if len(common_idx) == 0:
            pipeline_log.log("[compare] No common metrics found — cannot compare", level="error")
            return _t("⚠️ No common metrics to compare.", "⚠️ Aucune métrique commune à comparer.", language)
        pipeline_log.log(f"[compare] Aligned {len(common_idx)} common metrics", level="success")

        # Determine target role (infer from A if not provided)
        if not target_role:
            bdA = compute_grade(scoutA, role_hint=A.get("position_str"))
            target_role = bdA.role
        base = target_role.split(":")[0]
        sub  = target_role.split(":")[1] if ":" in target_role else None
        role_label = label_from_pair(base, sub)
        pipeline_log.log(f"[compare] Target role: {role_label} ({target_role})")

        # Duo spider chart
        pipeline_log.log("[compare] Generating duo spider chart…")
        spider_duo_fig = create_spider_graph_duo(
            playerA_data=scoutA,
            playerB_data=scoutB,
            playerA_name=A_name,
            playerB_name=B_name,
            role_hint=target_role,
            language=language,
            threshold=75.0,
            show_threshold=True,
        )
        duo_plot_html = (
            f"<!--PLOTLY_START-->"
            f"{spider_duo_fig.to_html(full_html=False, include_plotlyjs='inline')}"
            f"<!--PLOTLY_END-->"
        )
        pipeline_log.log("[compare] Duo spider chart generated", level="success")

        # Build role weights
        base_w = DEFAULT_WEIGHTS.get(base, DEFAULT_WEIGHTS["mf"])
        sub_w  = SUBROLE_WEIGHTS.get(sub) if sub else None
        W_role = _blend_role_weights(base_w, sub_w)
        W_role = _normalize_weight_keys(W_role)

        # Style head-to-head rows
        styles = styles or list(PLAY_STYLE_PRESETS.keys())
        style_rows: List[str] = []
        best_style_label = "—"
        best_abs_margin  = -1.0

        for s in styles:
            W = _style_reweight(W_role, s, style_influence, base=base, sub=sub)
            scoreA, scoreB, _ = _head_to_head_score(A_p, B_p, W)
            edge = scoreA - scoreB
            s_label = PLAY_STYLE_PRETTY.get(s, s)
            winner  = A_name if edge >= 0 else B_name
            style_rows.append(f"| {s_label} | {scoreA:.1f} | {scoreB:.1f} | **{winner}** |")
            if abs(edge) > best_abs_margin:
                best_abs_margin  = abs(edge)
                best_style_label = s_label

        # Per-metric winner table
        W_default = _style_reweight(W_role, None, 0.0)
        EPS = 0.5
        details = []
        for m in common_idx:
            w     = float(W_default.get(m, 0.0))
            a_raw = float(A_p[m]); b_raw = float(B_p[m])
            a = _invert_if_negative(m, a_raw)
            b = _invert_if_negative(m, b_raw)
            dp     = a - b
            impact = abs(w * dp)
            winner = (
                A_name if dp > EPS else
                (B_name if dp < -EPS else _t("Tie", "Égalité", language))
            )
            details.append((m, w, a_raw, b_raw, winner, impact))

        det_df = pd.DataFrame(
            details, columns=["Metric", "w", "A_pct", "B_pct", "Winner", "impact"]
        ).sort_values(["w", "impact"], ascending=[False, False])

        winners_rows = [
            f"| Metric | w | {A_name} (p) | {B_name} (p) | {_t('Winner', 'Vainqueur', language)} |",
            "|---|---:|---:|---:|---|",
        ]
        for _, r in det_df.head(12).iterrows():
            winners_rows.append(
                f"| {r['Metric']} | {r['w']:.2f} | {r['A_pct']:.0f} | {r['B_pct']:.0f} | {r['Winner']} |"
            )
        aligned_diff_md = "\n".join(winners_rows)

        # Percentiles tables for LLM
        scout_pct_A = pd.DataFrame({"Percentile": A_p}).to_markdown(tablefmt="pipe", index=True)
        scout_pct_B = pd.DataFrame({"Percentile": B_p}).to_markdown(tablefmt="pipe", index=True)

        """ # Style table
        style_header      = f"| {_t('Style','Style',language)} | {A_name} (p/100) | {B_name} (p/100) | {_t('Winner','Vainqueur',language)} |"
        style_section_title = _t("#### Style head-to-head (summary)", "#### Duel de styles (résumé)", language)
        style_sep         = "|---|---:|---:|---|"
        style_rows_md     = "\n".join([style_header, style_sep] + style_rows) """

        # Cosine similarity
        similarity = _cosine_similarity(A_p, B_p)

        # Deterministic header
        title      = f"### 🆚 Head-to-Head — {A_name} vs {B_name}"
        role_line  = _t("**Target role**", "**Poste cible**", language) + f": **{role_label}**"
        style_line = _t("**Best style context**", "**Meilleur style**", language) + f": **{best_style_label}**"
        sim_line   = _t("**Profile similarity**", "**Similarité de profil**", language) + f": {similarity:.1f}/100"

        deterministic_md = f"""
{title}
{SEPARATOR}
{photos_html}
{SEPARATOR}
{comparison_profile_md}
{SEPARATOR}
{role_line}
{style_line}
{sim_line}

{duo_plot_html}

#### {_t('Key role-critical differences (top 12 by role weight × gap)', 'Différences clés (top 12 par poids × écart)', language)}
{aligned_diff_md}
"""

        # Glossary
        present_metrics = []
        for md in (scout_pct_A + "\n" + scout_pct_B).splitlines():
            if md.startswith("|") and not md.startswith("|---"):
                parts = [p.strip() for p in md.strip("|").split("|")]
                if parts:
                    present_metrics.append(parts[0])
        glossary_block = _glossary_block_for(language, list(dict.fromkeys(present_metrics)))

        pipeline_log.log(f"[compare] Style head-to-head computed ({len(styles)} styles), best: {best_style_label}", level="success")

        if skip_llm:
            pipeline_log.log("[compare] Fast preview mode — skipping LLM narrative", level="success")
            deterministic_md += "\n\n> ⚡ **Fast preview:** LLM analysis skipped."
            return deterministic_md

        pipeline_log.log("[compare] Calling LLM for comparison narrative…")
        compare_llm_md = compare_llm_workflow(
            A_name=A_name, B_name=B_name,
            language=language,
            role_label=role_label,
            style_label=best_style_label,
            style_influence=style_influence,
            scout_md_A=scout_pct_A, scout_md_B=scout_pct_B,
            trend_md_A=A["trend_block_md"], trend_md_B=B["trend_block_md"],
            aligned_diff_md=aligned_diff_md,
            similarity_0_100=similarity,
            glossary_block=glossary_block,
            call_fn=_call_twice,
        )

        pipeline_log.log("[compare] Report generation complete", level="success")
        return deterministic_md + "\n\n---\n\n" + compare_llm_md

    except Exception as e:
        pipeline_log.log(f"[compare] Unhandled error: {e}", level="error")
        return f"⚠️ Compare failed: {e}"
