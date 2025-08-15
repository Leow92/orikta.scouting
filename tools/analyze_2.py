# tools/analyze.py

from __future__ import annotations

import textwrap
from typing import Iterable, Optional

import pandas as pd

from utils.fbref_scraper import scrape_all_tables, scrape_player_profile
from utils.llm_analysis_player import analyze_single_player  # ← use the optimized light module
from utils.resolve_player_url import search_fbref_url_with_playwright
from tools.grading_v2 import (
    compute_grade,
    rationale_from_breakdown,
    normalize_positions_from_profile,
    compute_grade_for_positions,
    label_from_pair,
)

# ------------------------- #
# Config & small utilities  #
# ------------------------- #

VERBOSE = False  # set True for console diagnostics

def _log(msg: str) -> None:
    if VERBOSE:
        print(msg)

def _is_fr(language: str | None) -> bool:
    return (language or "").strip().lower().startswith("fr")

def _md(s: str) -> str:
    """Dedent and strip a multi-line Markdown string."""
    return textwrap.dedent(s).strip()

SEPARATOR = "\n\n---\n\n"

def _t(en: str, fr: str, language: str) -> str:
    return fr if _is_fr(language) else en

def _nodata(language: str) -> str:
    return "donnée indisponible" if _is_fr(language) else "insufficient data"

# ------------------------------------- #
# FBref Standard Stats helpers (robust) #
# ------------------------------------- #

CANON_STANDARD_HEADERS = [
    "Season","Age","Squad","Country","Comp","LgRank",
    "MP","Starts","Min","90s",
    "Gls","Ast","G+A","G-PK","PK","PKatt","CrdY","CrdR",
    "xG","npxG","xAG","npxG+xAG",
    "PrgC","PrgP","PrgR",
    "Gls/90","Ast/90","G+A/90","G-PK/90","G+A-PK/90",
    "xG/90","xAG/90","xG+xAG/90","npxG/90","npxG+xAG/90",
    "Matches",
]

SEASON_TOKENS = ("season", "seasons", "yr", "year")

def _prefer_stats_standard_key(keys: Iterable[str]) -> Optional[str]:
    """
    Choose the most useful stats_standard* table if multiple exist.
    Preference order: domestic league -> league -> generic -> all comps -> fallback.
    """
    keys = list(keys)
    if not keys:
        return None
    priority = [
        "stats_standard_dom_lg",
        "stats_standard_lg",
        "stats_standard",
        "stats_standard_all",
    ]
    for pref in priority:
        if pref in keys:
            return pref
    return keys[0]

def _to_numeric_safely(series: pd.Series) -> pd.Series:
    """Convert to numeric after removing %, commas, and trimming."""
    cleaned = (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    out = pd.to_numeric(cleaned, errors="coerce")
    return out.where(out.notna(), series)

def _looks_numeric_cols(cols: Iterable) -> bool:
    try:
        cols = list(cols)
        return sum(str(c).isdigit() for c in cols) >= max(3, int(0.7 * len(cols)))
    except Exception:
        return False

def _season_key(v: str) -> int:
    s = str(v or "")
    for sep in ("-", "/", "–", "—"):
        if sep in s and s[:4].isdigit():
            try:
                return int(s.split(sep)[0])
            except Exception:
                return -10**9
    return int(s[:4]) if s[:4].isdigit() else -10**9

def _find_season_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if str(c).strip().lower() in SEASON_TOKENS:
            return c
    if not df.empty:
        first = df.columns[0]
        sample = str(df.iloc[0, 0])
        if any(sep in sample for sep in ("-", "/", "–", "—")) and sample[:4].isdigit():
            return first
    return df.columns[0] if len(df.columns) else None

def _clean_standard_df(std_df: pd.DataFrame) -> pd.DataFrame:
    """
    If columns are 0..N, map to canonical headers (width-trimmed).
    Try to promote first row as header if it looks like a header row.
    Strip commas; drop fully empty rows.
    """
    df = std_df.copy()

    # Try header promotion if first row smells like headers
    if not _looks_numeric_cols(df.columns):
        try:
            first = df.iloc[0].astype(str).str.lower().tolist()
            if any(tok in first for tok in ("season", "age", "squad", "comp", "team")):
                df.columns = df.iloc[0]
                df = df.iloc[1:].reset_index(drop=True)
        except Exception:
            pass

    # Map numeric columns to canonical names
    if _looks_numeric_cols(df.columns):
        w = len(df.columns)
        df.columns = CANON_STANDARD_HEADERS[:w]

    df = df.replace({"": pd.NA}).dropna(how="all")

    for col in df.columns:
        try:
            df[col] = df[col].astype(str).str.replace(",", "", regex=False)
        except Exception:
            pass

    return df

def _last_two_seasons_md(std_df_raw: Optional[pd.DataFrame], language: str) -> str:
    if std_df_raw is None or std_df_raw.empty:
        return _nodata(language)

    df = _clean_standard_df(std_df_raw)
    season_col = _find_season_col(df)
    if not season_col or season_col not in df.columns:
        return _nodata(language)

    df["__sk__"] = df[season_col].apply(_season_key)
    df = df.sort_values("__sk__", ascending=False).drop(columns="__sk__", errors="ignore")

    pref = [
        "Season","Age","Squad","Comp","MP","Starts","Min","90s",
        "Gls","Ast","G+A","G-PK","xG","npxG","xAG","npxG+xAG",
        "PrgC","PrgP","PrgR",
        "Gls/90","Ast/90","G+A/90","xG/90","xAG/90","xG+xAG/90","npxG/90","npxG+xAG/90",
    ]
    cols = [c for c in pref if c in df.columns] or list(df.columns)
    top2 = df[cols].head(2)

    title = _t("### 📚 Standard Stats (last two seasons)",
               "### 📚 Statistiques standards (2 dernières saisons)",
               language)
    return f"{title}\n\n{top2.to_markdown(index=False)}"

# ------------------------- #
# Profile helpers           #
# ------------------------- #

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
    title = _t(f"### 👤 {full_name} Presentation",
               f"### 👤 Présentation de {full_name}",
               language)

    if not items:
        no_data = _t("_(No bio details found)_", "_(Aucune information trouvée)_", language)
        return f"""{title}

**{full_name}**

{no_data}
""".strip()

    rows = "\n".join(f"| **{it['label']}** | {it['value']} |" for it in items)
    return f"""{title}

**{full_name}**

| Field | Value |
|---|---|
{rows}
""".strip()

# ------------------------- #
# Main entry                #
# ------------------------- #

def analyze_player(players: list[str], language: str = "English") -> str:
    """
    Single-player pipeline:
      - Resolve URL
      - Scrape profile + tables
      - Build presentation + scout table + last-two-seasons block
      - Deterministic grades (single + multi-position)
      - LLM analysis (light)
    Returns a full markdown report.
    """
    if not isinstance(players, list) or len(players) != 1:
        return "⚠️ Please provide exactly one player to analyze."

    query_name = players[0]
    _log(f"🧪 Analyze request for: {query_name}")

    url = search_fbref_url_with_playwright(query_name)
    if not url:
        return f"❌ Could not resolve FBref page for: {query_name}"

    full_name = query_name.title()
    _log(f"✅ Using URL for {full_name}: {url}")

    try:
        # 1) Profile + positions
        profile = scrape_player_profile(url)  # {"name","attributes","paragraphs","position_hint"}
        if profile.get("name"):
            full_name = profile["name"]

        pos_raw = None
        for a in profile.get("attributes", []):
            if str(a.get("label", "")).lower().startswith("position"):
                pos_raw = a.get("value")
                break
        positions = normalize_positions_from_profile(pos_raw)  # list of (base, sub|None)

        items = _merge_profile_items(profile)
        presentation_md = _profile_table_md(full_name, items, language)

        # 2) Tables
        tables = scrape_all_tables(url)

        # 2a) Scouting table (primary)
        scout_key = next((k for k in tables.keys() if k.startswith("scout_summary")), None)
        if not scout_key:
            return f"{presentation_md}\n⚠️ {_t('Could not find a scouting table for', 'Tableau scouting introuvable pour', language)} {full_name}."

        _log(f"📄 Using scouting table: {scout_key} for {full_name}")
        scout_df = tables[scout_key]
        if scout_df.shape[1] < 3:
            return f"{presentation_md}\n⚠️ {_t('Scouting table has an unexpected format for', 'Format inattendu du tableau scouting pour', language)} {full_name}."

        # Normalize columns to ["Metric","Per90","Percentile"]
        norm_cols = ["Metric", "Per90", "Percentile"]
        scout_df = scout_df.copy()
        scout_df.columns = norm_cols[: len(scout_df.columns)]
        scout_df.set_index("Metric", inplace=True)

        # Keep only Per90 + Percentile, numeric where possible
        display_df = scout_df[["Per90", "Percentile"]].copy()
        for col in ["Per90", "Percentile"]:
            display_df[col] = _to_numeric_safely(display_df[col])

        # Drop empty metric rows
        def _row_is_empty(s: pd.Series) -> bool:
            return (str(s.name).strip() == "") or s.isna().all() or (s.astype(str).str.strip() == "").all()

        display_df = display_df[~display_df.apply(_row_is_empty, axis=1)]

        scout_md = display_df.to_markdown(tablefmt="pipe", index=True)

        # 2b) Standard Stats: last two seasons block
        standard_keys = [k for k in tables.keys() if k.startswith("stats_standard")]
        std2_md = _nodata(language)
        if standard_keys:
            chosen = _prefer_stats_standard_key(standard_keys)
            std_df_raw = tables.get(chosen)
            if isinstance(std_df_raw, pd.DataFrame) and not std_df_raw.empty:
                std2_md = _last_two_seasons_md(std_df_raw, language)

        # 2c) Deterministic grade (best-guess role)
        role_hint = profile.get("position_hint") or scout_key
        grade_bd = compute_grade(scout_df, role_hint=role_hint)
        grade_md = rationale_from_breakdown(grade_bd, language=language)  # kept if you later want to display it

        # 2d) Multi-position grades
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

        # 2e) Scouting section title
        ctxt = scout_key.replace("scout_summary_", "").upper()
        scout_title = _t(f"### 🧾 Scouting Report ({ctxt})",
                         f"### 🧾 Rapport de scouting ({ctxt})",
                         language)

        # 3) LLM analysis (light) — pass compact grade context
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

        llm_text_light = analyze_single_player(
            full_name,
            scout_df,
            language=language,
            grade_ctx=grade_ctx,
            std_md=std2_md,
        )

        print("✅ Report Generation Done.")

        return _md(f"""
{presentation_md}
{SEPARATOR}
{scout_title}
{scout_md}
{SEPARATOR}
{multi_md}
{SEPARATOR}
{llm_text_light}
""")

    except Exception as e:
        return f"⚠️ Error analyzing {full_name}: {e}"
