# tools/analyze.py

from __future__ import annotations
import textwrap
from typing import Iterable, Optional
import pandas as pd
from utils.fbref_scraper import scrape_all_tables, scrape_player_profile
from utils.llm_analysis_player import analyze_single_player_workflow  # ← use the optimized light module
from utils.resolve_player_url import search_fbref_url_with_playwright
#from tools.grading_v2 import (compute_grade, rationale_from_breakdown, normalize_positions_from_profile, compute_grade_for_positions, label_from_pair)
from tools.grading_v3 import (
    compute_grade_for_positions_and_styles,
    compute_grade,
    rationale_from_breakdown,
    normalize_positions_from_profile,
    compute_grade_for_positions,
    label_from_pair,
    PLAY_STYLE_PRESETS,
    PLAY_STYLE_PRETTY
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

# --- Trend columns to consider (ordered by scouting value) ---
TREND_CANDIDATES = [
    # per 90 & quality
    "G+A/90","Gls/90","Ast/90","xG/90","xAG/90","npxG/90",
    # creation & progression
    "Shot-Creating Actions","Progressive Passes","Progressive Carries",
    # stability/availability
    "90s","Min",
    # accuracy/retention
    "Pass Completion %",
]

def _to_num(x):
    try:
        return pd.to_numeric(str(x).replace(",", "").replace("%","").strip(), errors="coerce")
    except Exception:
        return pd.NA

def _select_last_two_seasons(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by season, keep the most-played row per season (by Min or 90s), then take last two seasons."""
    season_col = _find_season_col(df)
    if not season_col: 
        return pd.DataFrame()

    # coerce helper columns
    df = df.copy()
    if "Min" in df.columns:
        df["__Min__"] = df["Min"].map(_to_num)
    if "90s" in df.columns:
        df["__90s__"] = df["90s"].map(_to_num)

    # season key & sort recent → old
    df["__sk__"] = df[season_col].apply(_season_key)
    df = df[df["__sk__"] > -10**8]  # filter non-season rows
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values("__sk__", ascending=False)

    # choose 1 row per season by max Min, else max 90s, else first
    chosen = []
    for sk, grp in df.groupby("__sk__", sort=False):
        if "__Min__" in grp and grp["__Min__"].notna().any():
            idx = grp["__Min__"].idxmax()
        elif "__90s__" in grp and grp["__90s__"].notna().any():
            idx = grp["__90s__"].idxmax()
        else:
            idx = grp.index[0]
        chosen.append(df.loc[idx])
    df1 = pd.DataFrame(chosen).sort_values("__sk__", ascending=False)

    # last two seasons only
    return df1.head(2).reset_index(drop=True)

def _trend_threshold(col: str) -> float:
    """Minimal change to consider non-noise."""
    c = col.lower()
    if "%" in col or "completion" in c:
        return 1.0   # percentage points
    if "/90" in c:
        return 0.10  # per-90 delta
    if "progressive" in c or "shot-creating" in c:
        return 0.5
    if c in ("min","90s"):
        return 2.0   # two full 90s
    return 0.5

def _compute_trend_deltas(std_df_raw: pd.DataFrame) -> dict[str, float]:
    """Return {metric: recent - previous} for candidate columns present."""
    if std_df_raw is None or std_df_raw.empty:
        return {}
    df = _clean_standard_df(std_df_raw)
    last2 = _select_last_two_seasons(df)
    if last2.shape[0] < 2:
        return {}

    # recent (row 0) vs previous (row 1)
    recent, prev = last2.iloc[0], last2.iloc[1]
    deltas: dict[str, float] = {}
    for col in TREND_CANDIDATES:
        if col in last2.columns:
            r = _to_num(recent[col])
            p = _to_num(prev[col])
            if pd.notna(r) and pd.notna(p):
                deltas[col] = float(r - p)
    return deltas

def _classify_trends(deltas: dict[str, float], top_k: int = 3) -> tuple[list[str], list[str], list[str]]:
    """Split into Improving / Consistent / Declining, sorted by absolute impact and trimmed to top_k."""
    if not deltas:
        return [], [], []
    # apply thresholds
    improving = [(k, v) for k, v in deltas.items() if v >  _trend_threshold(k)]
    declining = [(k, v) for k, v in deltas.items() if v < -_trend_threshold(k)]
    consistent = [k for k, v in deltas.items() if -_trend_threshold(k) <= v <= _trend_threshold(k)]

    # sort by absolute size (desc)
    improving.sort(key=lambda kv: kv[1], reverse=True)
    declining.sort(key=lambda kv: kv[1])  # v is negative; ascending = most negative first

    # keep names only
    improving_names = [k for k, _ in improving[:top_k]]
    declining_names = [k for k, _ in declining[:top_k]]
    consistent_names = sorted(consistent)[:top_k]

    return improving_names, consistent_names, declining_names

def build_trend_block_for_llm(std_df_raw: pd.DataFrame, language: str) -> tuple[str, dict]:
    """
    Produce a number-free, LLM-friendly block + raw lists for internal use.
    The block lists metric names only, ordered by computed effect.
    """
    deltas = _compute_trend_deltas(std_df_raw)
    improving, consistent, declining = _classify_trends(deltas, top_k=3)

    if _is_fr(language):
        title = "### Évolution des performances (2 dernières saisons)"
        block = [
            title, "",
            f"- **En hausse** : {(', '.join(improving)) or '—'}",
            f"- **Stables** : {(', '.join(consistent)) or '—'}",
            f"- **En baisse** : {(', '.join(declining)) or '—'}",
            "",
            "_Utilise ces listes uniquement ; ne cite aucun nombre._"
        ]
    else:
        title = "### Performance Evolution (last two seasons)"
        block = [
            title, "",
            f"- **Improving Metrics**: {(', '.join(improving)) or '—'}",
            f"- **Consistent Metrics**: {(', '.join(consistent)) or '—'}",
            f"- **Declining Metrics**: {(', '.join(declining)) or '—'}",
            "",
            "_Use these lists only; do not cite any numbers._"
        ]

    return "\n".join(block).strip(), {
        "deltas": deltas,
        "improving": improving,
        "consistent": consistent,
        "declining": declining,
    }


# ------------------------- #
# Profile helpers           #
# ------------------------- #

def _pretty_style_name(key: str) -> str:
    return PLAY_STYLE_PRETTY.get(key, key)

def _build_style_matrix(
    scout_df: pd.DataFrame,
    positions: list[tuple[str, str | None]],
    styles: list[str],
    style_strength: float = 0.6,
):
    """
    Returns (matrix_df, role_labels_pretty, style_labels_pretty).
    Index: pretty role labels; Columns: pretty style labels; Values: score/100 (float).
    """
    # Compute full matrix
    matrix = compute_grade_for_positions_and_styles(
        scout_df, positions=positions, styles=styles, style_strength=style_strength
    )
    # Stable order of roles (pretty labels)
    role_keys = []
    role_pretty = []
    seen = set()
    for base, sub in positions:
        key = f"{base}:{sub}" if sub else base
        if key not in seen:
            seen.add(key)
            role_keys.append(key)
            role_pretty.append(label_from_pair(base, sub))

    style_pretty = [_pretty_style_name(s) for s in styles]

    # Fill a dense DataFrame
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

def analyze_player_v0(players: list[str], language: str = "English") -> str:
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
        if not positions:
            if ":" in grade_bd.role:
                base, sub = grade_bd.role.split(":", 1)
                positions = [(base, sub)]
            else:
                positions = [(grade_bd.role, None)]
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
        std_df_raw = None
        if standard_keys:
            chosen = _prefer_stats_standard_key(standard_keys)
            std_df_raw = tables.get(chosen)
            if isinstance(std_df_raw, pd.DataFrame) and not std_df_raw.empty:
                std2_md = _last_two_seasons_md(std_df_raw, language)

        # NEW: compute trends programmatically (number-free lists)
        trend_block_md, trend_debug = build_trend_block_for_llm(std_df_raw, language) if std_df_raw is not None else (_nodata(language), {})

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

        # === Style Fit Matrix (roles × team play styles) ===
        # === Style Fit Matrix (roles × team play styles) ===
        styles = list(PLAY_STYLE_PRESETS.keys())
        style_strength = 0.6

        # Compute matrix (rows = pretty role labels, cols = pretty style labels)
        style_df, roles_pretty, styles_pretty = _build_style_matrix(
            scout_df, positions, styles, style_strength=style_strength
        )

        # Title
        style_title = _t("### 🎛️ Style Fit Matrix — Roles × Team Play Styles",
                        "### 🎛️ Adéquation au style — Postes × Styles d’équipe",
                        language)

        # Escape any '|' in labels (just in case)
        cols = [str(c).replace("|", "\\|") for c in style_df.columns]
        rows_idx = [str(r).replace("|", "\\|") for r in style_df.index]

        # Header and separator (first col left, numeric cols right)
        header = "| Role \\ Style | " + " | ".join(cols) + " |"
        sep_cells = ["---"] + ["---:"] * len(cols)
        sep = "|" + "|".join(sep_cells) + "|"

        # Safe maxima (skip all-NaN)
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
            r_label = str(r_label_raw).replace("|", "\\|")   # NEW: escape
            cells = []
            for c_label in style_df.columns:
                v = style_df.loc[r_label_raw, c_label]
                if pd.isna(v):
                    cells.append("—")
                    continue
                is_row_best = (r_label in row_max.index) and (row_max.get(r_label) == c_label)
                is_col_best = (c_label in col_max.index) and (col_max.get(c_label) == r_label)
                txt = f"{float(v):.1f}"
                cells.append(f"**{txt}**" if (is_row_best or is_col_best) else txt)
            lines.append(f"| {r_label} | " + " | ".join(cells) + " |")

        style_note = _t(
            "_Bold = best style per role and best role per style. Scores are /100. Adjusted with style_strength._",
            "_Gras = meilleur style par poste et meilleur poste par style. Scores sur 100. Pondérés par style_strength._",
            language,
        )

        style_md = style_title + "\n" + "\n".join(lines) + "\n\n" + style_note + "\n"


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

        llm_text_light = analyze_single_player_workflow(
            full_name,
            scout_df,
            language=language,
            grade_ctx=grade_ctx,
            multi_style_md=style_md,
            trend_block_md=trend_block_md,
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
{style_md}
{SEPARATOR}
{std2_md}
{SEPARATOR}
{llm_text_light}
""")

    except Exception as e:
        return f"⚠️ Error analyzing {full_name}: {e}"

def analyze_player(
    players: list[str],
    language: str = "English",
    styles: list[str] | None = None,
    style_strength: float = 0.6,
    skip_llm: bool = False,
) -> str:
    """
    Single-player pipeline:
      - Resolve URL
      - Scrape profile + tables
      - Presentation + scouting + last-two-seasons
      - Deterministic grades (single + multi-position)
      - Style Fit Matrix (roles × play styles)
      - Optional LLM analysis (skip_llm for fast preview)
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
        # 1) Profile + raw positions
        profile = scrape_player_profile(url)  # {"name","attributes","paragraphs","position_hint"}
        if profile.get("name"):
            full_name = profile["name"]

        pos_raw = None
        for a in profile.get("attributes", []):
            if str(a.get("label", "")).lower().startswith("position"):
                pos_raw = a.get("value")
                break
        positions = normalize_positions_from_profile(pos_raw)  # list[(base, sub|None)]

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
        scout_df = scout_df.copy()
        scout_df.columns = ["Metric", "Per90", "Percentile"][: len(scout_df.columns)]
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

        # 2b) Standard Stats: last two seasons block (+ raw df kept for trends)
        standard_keys = [k for k in tables.keys() if k.startswith("stats_standard")]
        std_df_raw = None
        std2_md = _nodata(language)
        if standard_keys:
            chosen = _prefer_stats_standard_key(standard_keys)
            std_df_raw = tables.get(chosen)
            if isinstance(std_df_raw, pd.DataFrame) and not std_df_raw.empty:
                std2_md = _last_two_seasons_md(std_df_raw, language)

        # NEW: compute trends programmatically (number-free lists)
        trend_block_md, trend_debug = (
            build_trend_block_for_llm(std_df_raw, language) if std_df_raw is not None else (_nodata(language), {})
        )

        # 2c) Deterministic grade (best-guess role)
        role_hint = profile.get("position_hint") or scout_key
        grade_bd = compute_grade(scout_df, role_hint=role_hint)

        # Fallback positions if none parsed from profile
        if not positions:
            if ":" in grade_bd.role:
                base, sub = grade_bd.role.split(":", 1)
                positions = [(base, sub)]
            else:
                positions = [(grade_bd.role, None)]

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

        # 2e) Style Fit Matrix (roles × team play styles)
        styles = styles or list(PLAY_STYLE_PRESETS.keys())
        style_strength = float(style_strength)

        style_df, _roles_pretty, _styles_pretty = _build_style_matrix(
            scout_df, positions, styles, style_strength=style_strength
        )

        style_title = _t("### 🎛️ Style Fit Matrix — Roles × Team Play Styles",
                         "### 🎛️ Adéquation au style — Postes × Styles d’équipe",
                         language)

        # Header & separator
        cols = [str(c).replace("|", "\\|") for c in style_df.columns]
        header = "| Role \\ Style | " + " | ".join(cols) + " |"
        sep = "|" + "|".join(["---"] + ["---:"] * len(cols)) + "|"

        # Safe maxima
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
            r_label_print = str(r_label_raw).replace("|", "\\|")  # escape for markdown
            cells = []
            for c_label in style_df.columns:
                v = style_df.loc[r_label_raw, c_label]
                if pd.isna(v):
                    cells.append("—")
                    continue
                # Compare maxima using RAW labels (not escaped)
                is_row_best = (r_label_raw in row_max.index) and (row_max.get(r_label_raw) == c_label)
                is_col_best = (c_label in col_max.index) and (col_max.get(c_label) == r_label_raw)
                txt = f"{float(v):.1f}"
                cells.append(f"**{txt}**" if (is_row_best or is_col_best) else txt)
            lines.append(f"| {r_label_print} | " + " | ".join(cells) + " |")

        style_note = _t(
            "_Bold = best style per role and best role per style. Scores are /100. Adjusted with style_strength._",
            "_Gras = meilleur style par poste et meilleur poste par style. Scores sur 100. Pondérés par style_strength._",
            language,
        )
        style_md = style_title + "\n" + "\n".join(lines) + "\n\n" + style_note + "\n"

        # 2f) Scouting section title
        ctxt = scout_key.replace("scout_summary_", "").upper()
        scout_title = _t(f"### 🧾 Scouting Report ({ctxt})",
                         f"### 🧾 Rapport de scouting ({ctxt})",
                         language)

        # 3) Optionally skip LLM for fast preview
        if skip_llm:
            _log("⚡ Fast preview: skipping LLM.")
            return _md(f"""
{presentation_md}
{SEPARATOR}
{scout_title}
{scout_md}
{SEPARATOR}
{multi_md}
{SEPARATOR}
{style_md}
{SEPARATOR}
{std2_md}
""")

        # 4) LLM analysis (light) — pass compact grade context
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

        llm_text_light = analyze_single_player_workflow(
            full_name,
            scout_df,
            language=language,
            grade_ctx=grade_ctx,
            multi_style_md=style_md,
            trend_block_md=trend_block_md,
        )

        _log("✅ Report Generation Done.")

        return _md(f"""
{presentation_md}
{SEPARATOR}
{scout_title}
{scout_md}
{SEPARATOR}
{multi_md}
{SEPARATOR}
{style_md}
{SEPARATOR}
{std2_md}
{SEPARATOR}
{llm_text_light}
""")

    except Exception as e:
        return f"⚠️ Error analyzing {full_name}: {e}"
