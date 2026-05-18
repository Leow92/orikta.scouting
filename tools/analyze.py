# tools/analyze.py

from __future__ import annotations
import textwrap
import pandas as pd

from utils.api_football import (
    search_player, get_player_by_id, get_league_players,
    best_stats_entry, pick_best_player, current_season,
)
from utils.percentile_engine import (
    build_scout_df, build_profile, get_position_str,
    build_season_comparison,
)
from utils.llm_analysis_player import analyze_single_player_workflow
from tools.grading import (
    compute_grade_for_positions_and_styles,
    compute_grade,
    normalize_positions_from_profile,
    compute_grade_for_positions,
    label_from_pair,
    PLAY_STYLE_PRESETS,
    PLAY_STYLE_PRETTY,
)
from utils.lang import _is_fr
from ui.graph import create_spider_graph

# ------------------------- #
# Config & small utilities  #
# ------------------------- #
VERBOSE = False

def _log(msg: str) -> None:
    if VERBOSE:
        print(msg)

def _md(s: str) -> str:
    return textwrap.dedent(s).strip()

SEPARATOR = "\n\n---\n\n"

def _t(en: str, fr: str, language: str) -> str:
    return fr if _is_fr(language) else en

def _nodata(language: str) -> str:
    return "donnée indisponible" if _is_fr(language) else "insufficient data"

# ------------------------------------------------------------------ #
# Trend analysis (API-football season-over-season)                    #
# ------------------------------------------------------------------ #
TREND_CANDIDATES = [
    "G+A per 90", "Goals per 90", "Assists per 90",
    "Key Passes per 90", "Pass Completion %",
    "Tackles per 90", "Interceptions per 90",
    "Dribble Success %", "Duels Won %",
    "Shots per 90", "Shot Accuracy %",
]

def _get_available_metrics(player_obj: dict) -> list[str]:
    """Extract all available per-90/percentage metrics from a player's stats."""
    entry = best_stats_entry(player_obj)
    if not entry:
        return []
    metrics = []
    for key in ["goals", "passes", "tackles", "dribbles", "duels", "shots"]:
        if key in entry:
            subkeys = entry[key].keys()
            for subkey in subkeys:
                if subkey in ["total", "accuracy", "success", "won"]:
                    metric_name = f"{key.capitalize()} {subkey.capitalize()}"
                    if subkey == "total":
                        metric_name += " per 90"  # Assume per-90 is derived
                    metrics.append(metric_name)
    return metrics

def _trend_threshold(col: str) -> float:
    c = col.lower()
    if "%" in c:
        return 1.5   # percentage points
    if "/90" in c or "per 90" in c:
        return 0.10  # per-90 delta
    return 0.5

def _classify_trends(
    deltas: dict[str, float], top_k: int = 3
) -> tuple[list[str], list[str], list[str]]:
    improving = [(k, v) for k, v in deltas.items() if v >  _trend_threshold(k)]
    declining = [(k, v) for k, v in deltas.items() if v < -_trend_threshold(k)]
    consistent = [k for k, v in deltas.items()
                  if -_trend_threshold(k) <= v <= _trend_threshold(k)]

    improving.sort(key=lambda kv: kv[1], reverse=True)
    declining.sort(key=lambda kv: kv[1])

    return (
        [k for k, _ in improving[:top_k]],
        sorted(consistent)[:top_k],
        [k for k, _ in declining[:top_k]],
    )

def build_trend_block_for_llm(
    current_metrics: dict[str, float],
    prev_metrics: dict[str, float],
    language: str,
    player_obj: dict | None = None,  # NEW: Accept player_obj to extract metrics
) -> tuple[str, dict]:
    """Compute number-free LLM trend block from two seasons of per-90 metric dicts."""
    # Use dynamic metrics if player_obj is provided, else fall back to TREND_CANDIDATES
    trend_candidates = _get_available_metrics(player_obj) if player_obj else TREND_CANDIDATES

    deltas: dict[str, float] = {}
    for col in trend_candidates:  # Use dynamic list
        curr = current_metrics.get(col)
        prev = prev_metrics.get(col)
        if curr is not None and prev is not None:
            deltas[col] = curr - prev

    improving, consistent, declining = _classify_trends(deltas, top_k=3)

    if _is_fr(language):
        title = "### Évolution des performances (saison en cours vs précédente)"
        block = [
            title, "",
            f"- **En hausse** : {(', '.join(improving)) or '—'}",
            f"- **Stables** : {(', '.join(consistent)) or '—'}",
            f"- **En baisse** : {(', '.join(declining)) or '—'}",
            "",
            "_Utilise ces listes uniquement ; ne cite aucun nombre._",
        ]
    else:
        title = "### Performance Evolution (current vs previous season)"
        block = [
            title, "",
            f"- **Improving Metrics**: {(', '.join(improving)) or '—'}",
            f"- **Consistent Metrics**: {(', '.join(consistent)) or '—'}",
            f"- **Declining Metrics**: {(', '.join(declining)) or '—'}",
            "",
            "_Use these lists only; do not cite any numbers._",
        ]

    return "\n".join(block).strip(), {
        "deltas": deltas,
        "improving": improving,
        "consistent": consistent,
        "declining": declining,
    }


# ------------------------------------------------------------------ #
# Season stats table (2 seasons)                                      #
# ------------------------------------------------------------------ #
def _season_stats_md(
    current_metrics: dict[str, float],
    prev_metrics: dict[str, float],
    current_season_year: int,
    language: str,
    player_obj: dict | None = None,  # NEW: Accept player_obj to extract metrics
) -> str:
    if not current_metrics:
        return _nodata(language)

    title = _t(
        f"### 📚 Season Stats ({current_season_year - 1}–{current_season_year} vs {current_season_year - 2}–{current_season_year - 1})",
        f"### 📚 Stats de saison ({current_season_year - 1}–{current_season_year} vs {current_season_year - 2}–{current_season_year - 1})",
        language,
    )

    # Use dynamic metrics if player_obj is provided, else fall back to preferred
    preferred = _get_available_metrics(player_obj) if player_obj else [
        "Goals per 90", "Assists per 90", "G+A per 90",
        "Shots per 90", "Shot Accuracy %",
        "Key Passes per 90", "Pass Completion %",
        "Tackles per 90", "Interceptions per 90",
        "Dribble Success %", "Duels Won %",
    ]

    rows = []
    for metric in preferred:
        curr = current_metrics.get(metric)
        prev = prev_metrics.get(metric)
        if curr is None and prev is None:
            continue
        curr_str = f"{curr:.2f}" if curr is not None else "—"
        prev_str = f"{prev:.2f}" if prev is not None else "—"
        rows.append({"Metric": metric, "Current": curr_str, "Previous": prev_str})

    if not rows:
        return _nodata(language)

    df = pd.DataFrame(rows).set_index("Metric")
    return f"{title}\n\n{df.to_markdown(tablefmt='pipe', index=True)}"


# ------------------------------------------------------------------ #
# Profile helpers (unchanged from original)                           #
# ------------------------------------------------------------------ #
def _pretty_style_name(key: str) -> str:
    return PLAY_STYLE_PRETTY.get(key, key)

def _build_style_matrix(
    scout_df: pd.DataFrame,
    positions: list[tuple[str, str | None]],
    styles: list[str],
    style_strength: float = 0.6,
):
    matrix = compute_grade_for_positions_and_styles(
        scout_df, positions=positions, styles=styles, style_strength=style_strength
    )
    role_keys, role_pretty, seen = [], [], set()
    for base, sub in positions:
        key = f"{base}:{sub}" if sub else base
        if key not in seen:
            seen.add(key)
            role_keys.append(key)
            role_pretty.append(label_from_pair(base, sub))

    style_pretty = [_pretty_style_name(s) for s in styles]

    df = pd.DataFrame(index=role_pretty, columns=style_pretty, dtype=float)
    for rk, sp in zip(role_keys, role_pretty):
        for s, sname in zip(styles, style_pretty):
            bd = matrix.get((rk, s))
            df.loc[sp, sname] = round(bd.final_score, 1) if bd else float("nan")
    return df, role_pretty, style_pretty

def _parse_label_value_lines(paragraphs: list[str]) -> list[dict]:
    items: list[dict] = []
    for p in paragraphs or []:
        if ":" in p:
            label, value = p.split(":", 1)
            label, value = label.strip(), value.strip().strip(".")
            if label and value:
                items.append({"label": label, "value": value})
    return items

def _merge_profile_items(profile: dict) -> list[dict]:
    seen = set()
    merged: list[dict] = []
    for a in profile.get("attributes", []):
        label = str(a.get("label", "")).strip()
        value = str(a.get("value", "")).strip()
        if label and value and label.lower() not in seen:
            merged.append({"label": label, "value": value})
            seen.add(label.lower())
    for a in _parse_label_value_lines(profile.get("paragraphs", [])):
        if a["label"].lower() not in seen and a["value"]:
            merged.append(a)
            seen.add(a["label"].lower())
    return merged

def _profile_table_md(full_name: str, items: list[dict], language: str) -> str:
    title = _t("### 👤 Player Presentation", "### 👤 Présentation du Joueur", language)
    if not items:
        no_data = _t("_(No bio details found)_", "_(Aucune information trouvée)_", language)
        return f"{title}\n\n**{full_name}**\n\n{no_data}".strip()
    rows = "\n".join(f"| **{it['label']}** | {it['value']} |" for it in items)
    return f"{title}\n\n**{full_name}**\n\n| Field | Value |\n|---|---|\n{rows}".strip()

# ------------------------------------------------------------------ #
# Shared numeric helper (used by compare.py via import)               #
# ------------------------------------------------------------------ #
def _to_numeric_safely(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    out = pd.to_numeric(cleaned, errors="coerce")
    return out.where(out.notna(), series)


# ------------------------------------------------------------------ #
# Main pipeline                                                        #
# ------------------------------------------------------------------ #
def analyze_player(
    players: list[str],
    language: str = "English",
    styles: list[str] | None = None,
    style_strength: float = 0,
    skip_llm: bool = False,
) -> str:
    """
    Single-player analysis pipeline (API-football edition):
      1. Search player by name via API-football
      2. Fetch league pool for percentile computation
      3. Build scout_df (Metric / Per90 / Percentile)
      4. Deterministic grades (multi-position + style matrix)
      5. Spider graph
      6. Optional LLM narrative (Groq)

    Returns a full Markdown report string.
    """
    if not isinstance(players, list) or len(players) != 1:
        return "⚠️ Please provide exactly one player to analyze."

    query_name = players[0]
    _log(f"🧪 Analyze request for: {query_name}")

    season = current_season()

    # ---- 1. Search player ----
    results = search_player(query_name, season=season)
    if not results:
        # Try previous season as fallback
        results = search_player(query_name, season=season - 1)
        if results:
            season = season - 1
    if not results:
        return f"❌ No player found for: **{query_name}**. Try a more complete name (e.g. full first + last name)."

    player_obj = pick_best_player(results, query_name)
    if not player_obj:
        return f"❌ Could not identify a best match for: **{query_name}**."

    player_info = player_obj.get("player") or {}
    photo_url = player_info.get("photo")
    full_name = player_info.get("name") or query_name.title()
    player_id = player_info.get("id")
    _log(f"✅ Matched player: {full_name} (id={player_id})")

    entry = best_stats_entry(player_obj)
    if not entry:
        return f"❌ No statistics available for **{full_name}** in season {season}."

    league_id    = (entry.get("league") or {}).get("id")
    league_name  = (entry.get("league") or {}).get("name", "Unknown League")
    position_str = get_position_str(player_obj)

    try:
        # ---- 2. Fetch league pool for percentiles ----
        pool: list[dict] = []
        if league_id:
            _log(f"📡 Fetching league pool: league_id={league_id}, season={season}")
            pool = get_league_players(league_id, season, max_pages=5)
            if not pool:
                _log(f"⚠️ No league data available for {league_name} (season {season}). Using empty pool.")
                # Fallback: Use empty pool and skip percentiles
                scout_df = build_scout_df(player_obj, [], position_filter=position_str)
            else:
                _log(f"📊 Pool size: {len(pool)} players")
                scout_df = build_scout_df(player_obj, pool, position_filter=position_str)
        else:
            _log(f"⚠️ No league_id available for {full_name}. Using empty pool.")
            scout_df = build_scout_df(player_obj, [], position_filter=position_str)

        # ---- 4. Build profile ----
        profile = build_profile(player_obj)
        items = _merge_profile_items(profile)
        presentation_md = _profile_table_md(full_name, items, language)

        photo_url = player_info.get("photo")
        player_photo_html = f"![{full_name}]({photo_url})" if photo_url else ""

        # ---- 5. Determine positions ----
        pos_raw = position_str
        for a in profile.get("attributes", []):
            if str(a.get("label", "")).lower().startswith("position"):
                pos_raw = a.get("value", position_str)
                break
        positions = normalize_positions_from_profile(pos_raw)
        if positions:
            role_base, role_sub = positions[0]
        else:
            role_base, role_sub = "mf", None
        role_hint = f"{role_base}:{role_sub}" if role_sub else role_base

        # ---- 6. Compute deterministic grade ----
        grade_bd = compute_grade(scout_df, role_hint=role_hint)
        if not positions:
            if ":" in grade_bd.role:
                b, s = grade_bd.role.split(":", 1)
                positions = [(b, s)]
            else:
                positions = [(grade_bd.role, None)]

        # ---- 7. Multi-position grades ----
        per_pos = compute_grade_for_positions(scout_df, positions)
        per_pos_sorted = dict(sorted(per_pos.items(), key=lambda kv: kv[1].final_score, reverse=True))

        multi_title = _t("### 🧮 Multi-position Grades /100", "### 🧮 Notes par poste /100", language)
        rows = ["| Position | Score/100 | Top drivers | Missing |", "|---|---:|---|---|"]
        for role_key, bd in per_pos_sorted.items():
            top2 = sorted(bd.matched, key=lambda x: x[2], reverse=True)[:2]
            top_str = ", ".join(f"{m} (w={w:.2f})" for m, w, _ in top2) if top2 else "—"
            miss_str = ", ".join(bd.missing[:3]) if bd.missing else "—"
            if ":" in role_key:
                base, sub = role_key.split(":", 1)
                pretty = label_from_pair(base, sub)
            else:
                pretty = label_from_pair(role_key, None)
            rows.append(f"| {pretty} | **{bd.final_score:.1f}** | {top_str} | {miss_str} |")
        multi_md = multi_title + "\n" + "\n".join(rows) + "\n"

        # ---- 8. Style Fit Matrix ----
        styles = styles or list(PLAY_STYLE_PRESETS.keys())
        style_df, *_ = _build_style_matrix(
            scout_df, positions, styles, style_strength=float(style_strength)
        )
        style_title = _t(
            "### 🎛️ Style Fit Matrix — Roles × Team Play Styles",
            "### 🎛️ Adéquation au style — Postes × Styles d'équipe",
            language,
        )
        cols = [str(c).replace("|", "\\|") for c in style_df.columns]
        header = "| Role \\ Style | " + " | ".join(cols) + " |"
        sep    = "|" + "|".join(["---"] + ["---:"] * len(cols)) + "|"

        _valid = style_df.copy()
        valid_cols = _valid.columns[~_valid.isna().all(axis=0)]
        valid_rows = _valid.index[~_valid.isna().all(axis=1)]
        if len(valid_cols) and len(valid_rows):
            row_max = _valid.loc[valid_rows, valid_cols].idxmax(axis=1)
            col_max = _valid.loc[valid_rows, valid_cols].idxmax(axis=0)
        else:
            row_max = pd.Series(index=_valid.index, dtype=object)
            col_max = pd.Series(index=_valid.columns, dtype=object)

        lines = [header, sep]
        for r_label_raw in style_df.index:
            r_label_print = str(r_label_raw).replace("|", "\\|")
            cells = []
            for c_label in style_df.columns:
                v = style_df.loc[r_label_raw, c_label]
                if pd.isna(v):
                    cells.append("—")
                    continue
                is_row_best = (r_label_raw in row_max.index) and (row_max.get(r_label_raw) == c_label)
                is_col_best = (c_label in col_max.index) and (col_max.get(c_label) == r_label_raw)
                txt = f"{float(v):.1f}"
                cells.append(f"**{txt}**" if (is_row_best or is_col_best) else txt)
            lines.append(f"| {r_label_print} | " + " | ".join(cells) + " |")

        style_note = _t(
            "_Bold = best style per role and best role per style. Scores /100._",
            "_Gras = meilleur style par poste et meilleur poste par style. Scores /100._",
            language,
        )
        style_md = style_title + "\n" + "\n".join(lines) + "\n\n" + style_note + "\n"

        # ---- 9. Scouting table ----
        display_df = scout_df[["Per90", "Percentile"]].copy()
        display_df["Percentile"] = pd.to_numeric(display_df["Percentile"], errors="coerce")
        scout_md_title = _t(
            f"### 🧾 Scouting Report — {league_name} {season}",
            f"### 🧾 Rapport de scouting — {league_name} {season}",
            language,
        )
        scout_md = display_df.to_markdown(tablefmt="pipe", index=True)

        # ---- 10. Season comparison / trends ----
        prev_player_objs = get_player_by_id(player_id, season - 1) if player_id else []
        prev_obj = prev_player_objs[0] if prev_player_objs else None
        current_metrics, prev_metrics = build_season_comparison(player_obj, prev_obj)

        std2_md = _season_stats_md(current_metrics, prev_metrics, season, language, player_obj)
        trend_block_md, _ = build_trend_block_for_llm(current_metrics, prev_metrics, language, player_obj)

        # ---- 11. Spider graph ----
        if "Percentile" in scout_df.columns:
            scout_df["Percentile"] = pd.to_numeric(scout_df["Percentile"], errors="coerce")

        spider_fig = create_spider_graph(
            player_data=scout_df,
            player_name=full_name,
            role_hint=role_hint,
            language=language,
        )
        raw_plotly = spider_fig.to_html(full_html=False, include_plotlyjs="inline")
        spider_graph_html = f"<!--PLOTLY_START-->{raw_plotly}<!--PLOTLY_END-->"

        # ---- 12. Fast preview (skip LLM) ----
        if skip_llm:
            _log("⚡ Fast preview: skipping LLM.")
            print("✅ Report Generation Done.")
            return _md(f"""
{player_photo_html}
{presentation_md}
{spider_graph_html}
{SEPARATOR}
{scout_md_title}
{scout_md}
{SEPARATOR}
{multi_md}
{SEPARATOR}
{style_md}
{SEPARATOR}
{std2_md}
""")

        # ---- 13. LLM analysis ----
        top_roles_for_llm = [
            (label_from_pair(*(rk.split(":") if ":" in rk else (rk, None))), round(bd.final_score, 1))
            for rk, bd in list(per_pos_sorted.items())[:3]
        ]
        grade_ctx = {
            "role": grade_bd.role,
            "score": round(grade_bd.final_score, 1),
            "drivers": sorted(grade_bd.matched, key=lambda x: x[2], reverse=True)[:5],
            "missing": grade_bd.missing[:5],
            "per_position_top": top_roles_for_llm,
        }

        llm_text = analyze_single_player_workflow(
            full_name,
            scout_df,
            language=language,
            grade_ctx=grade_ctx,
            multi_style_md=style_md,
            trend_block_md=trend_block_md,
            presentation_md=presentation_md,
        )

        print("✅ Report Generation Done.")
        return _md(f"""
{player_photo_html}
{presentation_md}
{spider_graph_html}
{SEPARATOR}
{scout_md_title}
{scout_md}
{SEPARATOR}
{multi_md}
{SEPARATOR}
{style_md}
{SEPARATOR}
{std2_md}
{SEPARATOR}
{llm_text}
""")

    except Exception as e:
        return f"⚠️ Error analyzing **{full_name}**: {e}"
