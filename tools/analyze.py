# tools/analyze.py

import pandas as pd
from utils.fbref_scraper import scrape_all_tables, scrape_player_profile
from utils.llm_analysis_single_player import analyze_single_player
from utils.resolve_player_url import search_fbref_url_with_playwright
from tools.grading_v2 import (
    compute_grade, rationale_from_breakdown,
    normalize_positions_from_profile, compute_grade_for_positions, label_from_pair
)

# tools/analyze.py
import textwrap

def _md(s: str) -> str:
    """Dedent and strip a multi-line Markdown string."""
    return textwrap.dedent(s).strip()

SEPARATOR = "\n\n---\n\n"

def _section_title(text_en: str, text_fr: str, language: str) -> str:
    return text_fr if (language or "").lower().startswith("fr") else text_en

def _prefer_stats_standard_key(keys: list[str]) -> str | None:
    """
    Choose the most useful stats_standard* table if multiple exist.
    Preference order (best-effort): domestic league -> all comps -> others.
    """
    if not keys:
        return None
    priority = [
        "stats_standard_dom_lg",
        "stats_standard_lg",
        "stats_standard",
        "stats_standard_all",
    ]
    for pref in priority:
        for k in keys:
            if k == pref:
                return k
    return keys[0]  # fallback

# 🔧 NEW: safe numeric conversion helper (future-proof: no errors="ignore")
def _to_numeric_safely(series: pd.Series) -> pd.Series:
    """
    Attempt to convert a column to numeric:
    - strips %, commas, and whitespace
    - converts to numeric where possible (NaN otherwise)
    - preserves original values where conversion failed
    """
    cleaned = (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    numeric = pd.to_numeric(cleaned, errors="coerce")
    # keep original where conversion failed
    return numeric.where(numeric.notna(), series)

def _parse_label_value_lines(paragraphs: list[str]) -> list[dict]:
    """
    Best-effort: turn 'Label: Value' paragraphs into {'label','value'} pairs.
    Leaves other paragraphs out (or you can collect them separately).
    """
    items = []
    for p in paragraphs or []:
        if ":" in p:
            label, value = p.split(":", 1)
            label = label.strip()
            value = value.strip().strip(".")
            if label and value:
                items.append({"label": label, "value": value})
    return items

def _merge_profile_items(profile: dict) -> list[dict]:
    """
    Merge 'attributes' + parsed paragraphs, de-dup by label (keep first non-empty).
    """
    seen = set()
    merged = []
    # attributes from <p><strong>Label</strong> Value</p>
    for a in profile.get("attributes", []):
        label = str(a.get("label", "")).strip()
        value = str(a.get("value", "")).strip()
        if label and value and label.lower() not in seen:
            merged.append({"label": label, "value": value})
            seen.add(label.lower())
    # paragraphs that look like Label: Value
    for a in _parse_label_value_lines(profile.get("paragraphs", [])):
        label = a["label"]
        if label.lower() not in seen and a["value"]:
            merged.append(a)
            seen.add(label.lower())
    return merged

def _profile_table_md(full_name: str, items: list[dict], language: str) -> str:
    title = (
        f"### 👤 {full_name} Presentation"
        if not (language or "").lower().startswith("fr")
        else f"### 👤 Présentation de {full_name}"
    )

    if not items:
        no_data = (
            "_(No bio details found)_"
            if not (language or "").lower().startswith("fr")
            else "_(Aucune information trouvée)_"
        )
        return f"""{title}

**{full_name}**

{no_data}
""".strip()

    # ensure each key/value is its own row
    rows = "\n".join(f"| **{it['label']}** | {it['value']} |" for it in items)

    return f"""{title}

**{full_name}**

| Field | Value |
|---|---|
{rows}
""".strip()

# --- add near your other helpers in tools/analyze.py ---

# --- FBref "Standard Stats" canonical headers (domestic league) ---
CANON_STANDARD_HEADERS = [
    "Season","Age","Squad","Country","Comp","LgRank",
    "MP","Starts","Min","90s",
    "Gls","Ast","G+A","G-PK","PK","PKatt","CrdY","CrdR",
    "xG","npxG","xAG","npxG+xAG",
    "PrgC","PrgP","PrgR",
    "Gls/90","Ast/90","G+A/90","G-PK/90","G+A-PK/90",
    "xG/90","xAG/90","xG+xAG/90","npxG/90","npxG+xAG/90",
    "Matches"
]

SEASON_TOKENS = ("season","seasons","yr","year")

def _looks_numeric_cols(cols) -> bool:
    try:
        return sum(str(c).isdigit() for c in cols) >= max(3, int(0.7 * len(cols)))
    except Exception:
        return False

def _season_key(v: str) -> int:
    s = str(v or "")
    for sep in ("-","/","–","—"):
        if sep in s and s[:4].isdigit():
            try: return int(s.split(sep)[0])
            except: return -10**9
    return int(s[:4]) if s[:4].isdigit() else -10**9

def _find_season_col(df: pd.DataFrame):
    # 1) direct named
    for c in df.columns:
        if str(c).strip().lower() in SEASON_TOKENS:
            return c
    # 2) if first column values look like seasons, use it
    if not df.empty:
        first = df.columns[0]
        sample = str(df.iloc[0, 0])
        if any(sep in sample for sep in ("-","/","–","—")) and sample[:4].isdigit():
            return first
    # 3) fallback
    return df.columns[0] if len(df.columns) else None

def _clean_standard_df(std_df: pd.DataFrame) -> pd.DataFrame:
    """
    If columns are numeric (0..N), rename with canonical headers (trimmed).
    Also drop fully empty rows and strip commas in numeric-like fields.
    """
    df = std_df.copy()
    # simple header promotion if first row *looks* like text headers
    if _looks_numeric_cols(df.columns):
        # keep raw; many FBref tables are already data rows
        pass
    else:
        # if first row contains tokens like 'season', set as header
        try:
            first = df.iloc[0].astype(str).str.lower().tolist()
            if any(tok in first for tok in ("season","age","squad","comp","team")):
                df.columns = df.iloc[0]
                df = df.iloc[1:].reset_index(drop=True)
        except Exception:
            pass

    # if still numeric columns, map to canonical names (trim to width)
    if _looks_numeric_cols(df.columns):
        w = len(df.columns)
        df.columns = CANON_STANDARD_HEADERS[:w]

    # basic cleanup
    df = df.replace({"": pd.NA})
    df = df.dropna(how="all")
    # strip commas from Minutes etc. (won't error if non-string)
    for col in df.columns:
        try:
            df[col] = df[col].astype(str).str.replace(",", "", regex=False)
        except Exception:
            pass
    return df

def _last_two_seasons_md(std_df_raw: pd.DataFrame, language: str) -> str:
    if std_df_raw is None or std_df_raw.empty:
        return "donnée indisponible" if (language or "").lower().startswith("fr") else "insufficient data"

    df = _clean_standard_df(std_df_raw)
    season_col = _find_season_col(df)
    if season_col is None or season_col not in df.columns:
        return "donnée indisponible" if (language or "").lower().startswith("fr") else "insufficient data"

    # sort recent → oldest
    df["__sk__"] = df[season_col].apply(_season_key)
    df = df.sort_values("__sk__", ascending=False).drop(columns="__sk__", errors="ignore")

    # choose readable subset if present
    pref = [
        "Season","Age","Squad","Comp","MP","Starts","Min","90s",
        "Gls","Ast","G+A","G-PK","xG","npxG","xAG","npxG+xAG",
        "PrgC","PrgP","PrgR",
        "Gls/90","Ast/90","G+A/90","xG/90","xAG/90","xG+xAG/90","npxG/90","npxG+xAG/90"
    ]
    cols = [c for c in pref if c in df.columns] or list(df.columns)

    top2 = df[cols].head(2)
    title = "### 📚 Standard Stats (last two seasons)" if not (language or "").lower().startswith("fr") \
            else "### 📚 Statistiques standards (2 dernières saisons)"
    return f"{title}\n\n{top2.to_markdown(index=False)}"


def analyze_player(players: list, language: str = "English") -> str:
    if not isinstance(players, list) or len(players) != 1:
        return "⚠️ Please provide exactly one player to analyze."

    query_name = players[0]
    print(f"🧪 Analyze request for: {query_name}")

    url = search_fbref_url_with_playwright(query_name)
    if not url:
        return f"❌ Could not resolve FBref page for: {query_name}"

    full_name = query_name.title()
    print(f"✅ Using URL for {full_name}: {url}")

    try:
        # 1) Player Presentation (and capture raw position text)
        profile = scrape_player_profile(url)  # {"name","attributes","paragraphs","position_hint"}
        if profile.get("name"):
            full_name = profile["name"]

        # pull FBref "Position" field for multi‑position grading
        pos_raw = None
        for a in profile.get("attributes", []):
            if str(a.get("label","")).lower().startswith("position"):
                pos_raw = a.get("value")
                break
        positions = normalize_positions_from_profile(pos_raw)  # list of (base, sub|None)

        items = _merge_profile_items(profile)
        presentation_md = _profile_table_md(full_name, items, language)

        # 2) Scrape all tables
        tables = scrape_all_tables(url)

        # 2a) Scouting table (Last 365d vs position group)
        scout_key = next((k for k in tables.keys() if k.startswith("scout_summary")), None)
        print(scout_key)
        if not scout_key:
            return f"{presentation_md}\n⚠️ Could not find a scouting table for {full_name}."

        print(f"📄 Using scouting table: {scout_key} for {full_name}")
        scout_df = tables[scout_key]
        if scout_df.shape[1] < 3:
            return f"{presentation_md}\n⚠️ Scouting table for {full_name} has an unexpected format."

        # normalize percentiles/per90
        scout_df.columns = ["Metric", "Per90", "Percentile"][:len(scout_df.columns)]
        scout_df.set_index("Metric", inplace=True)
        display_df = scout_df[["Per90", "Percentile"]].copy()

        # convert safely, as you already do
        for col in ["Per90", "Percentile"]:
            display_df[col] = _to_numeric_safely(display_df[col])

        # NEW: remove empty metric rows and rows that are all NaN/empty
        def _row_is_empty(s) -> bool:
            return (str(s.name).strip() == "" or s.isna().all() or (s.astype(str).str.strip() == "").all())

        mask = display_df.apply(_row_is_empty, axis=1)
        if mask.any():
            display_df = display_df[~mask]

        # render as pipe table
        scout_md = display_df.to_markdown(tablefmt="pipe", index=True)

        # 2b) Standard Stats (extract last 2 seasons only)
        standard_keys = [k for k in tables.keys() if k.startswith("stats_standard")]
        std2_md = "donnée indisponible" if (language or "").lower().startswith("fr") else "insufficient data"
        if standard_keys:
            chosen = _prefer_stats_standard_key(standard_keys)
            std_df_raw = tables[chosen].copy()
            std2_md = _last_two_seasons_md(std_df_raw, language)

        # 2c) Deterministic grade (single best-guess role)
        role_hint = profile.get("position_hint") or scout_key
        grade_bd = compute_grade(scout_df, role_hint=role_hint)
        grade_md = rationale_from_breakdown(grade_bd, language=language)

        # 🔥 2d) Multi‑position grades (NOW that scout_df exists)
        per_pos = compute_grade_for_positions(scout_df, positions)
        # sort by score desc for the table
        per_pos_sorted = dict(sorted(per_pos.items(), key=lambda kv: kv[1].final_score, reverse=True))

        multi_title = _section_title("### 🧮 Multi‑position Grades /100", "### 🧮 Notes par poste /100", language)
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

        # Titles/markdown
        scout_title = _section_title(
            f"### 🧾 Scouting Report ({scout_key.replace('scout_summary_', '').upper()})",
            f"### 🧾 Rapport de scouting ({scout_key.replace('scout_summary_', '').upper()})",
            language,
        )

        #grade_section_title = _section_title("### 🧾 Grade/100", "### 🧾 Note/100", language)

        # 3) (Optional) pass a compact per‑position context to the LLM as well
        top_roles_for_llm = [
            (label_from_pair(*(rk.split(":") if ":" in rk else (rk, None))), round(bd.final_score, 1))
            for rk, bd in list(per_pos_sorted.items())[:3]  # top-3 roles
        ]
        grade_ctx = {
            "role": grade_bd.role,
            "score": round(grade_bd.final_score, 1),
            "drivers": sorted(grade_bd.matched, key=lambda x: x[2], reverse=True)[:5],
            "missing": grade_bd.missing[:5],
            "per_position_top": top_roles_for_llm,  # e.g., [("Winger", 84.2), ("ST/CF", 79.8), ...]
        }

        # 4) LLM analysis (light)
        llm_text_light = analyze_single_player(
            full_name,
            scout_df,
            language=language,
            grade_ctx=grade_ctx,   # passes best‑guess + top per‑position ranks
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
