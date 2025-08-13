# tools/analyze.py

import pandas as pd
from utils.fbref_scraper import scrape_all_tables, scrape_player_profile
#from utils.llm_analysis_single_player import analyze_single_player
from utils.llm_analysis_single_player_light import analyze_single_player
from utils.resolve_player_url import search_fbref_url_with_playwright
from tools.grading import compute_grade, rationale_from_breakdown

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
    """
    Render name + two-column table (Field | Value) as markdown.
    """
    title = f"### 👤 {full_name} Presentation" if not (language or "").lower().startswith("fr") else f"### 👤 Présentation de {full_name}"
    if not items:
        no_data = "_(No bio details found)_" if not (language or "").lower().startswith("fr") else "_(Aucune information trouvée)_"
        return f"""{title}

**{full_name}**

{no_data}

---
"""
    # Markdown table
    rows = "\n".join(f"| **{it['label']}** | {it['value']} |" for it in items)
    return f"""{title}

**{full_name}**

| Field | Value |
|---|---|
{rows}

---
"""

def analyze_player(players: list, language: str = "English") -> str:
    """
    Analyze a single player's scouting report using FBref tables + LLM summary.
    - Player Presentation (name + <p> lines from #info.players.open)
    - Scouting (Last 365d, vs position group) — 'scout_summary_*'
    - Standard Stats (season-by-season) — first 'stats_standard*' table found
    """
    if not isinstance(players, list) or len(players) != 1:
        return "⚠️ Please provide exactly one player to analyze."

    query_name = players[0]
    print(f"🧪 Analyze request for: {query_name}")

    # Resolve URL via dynamic search
    url = search_fbref_url_with_playwright(query_name)
    if not url:
        return f"❌ Could not resolve FBref page for: {query_name}"

    full_name = query_name.title()
    print(f"✅ Using URL for {full_name}: {url}")

    try:
        '''
        # 1) Player Presentation
        profile = scrape_player_profile(url)  # {"name": ..., "paragraphs": [...]}
        if profile.get("name"):
            full_name = profile["name"]  # prefer exact FBref h1
        pres_title = _section_title("### 👤 Player Presentation", "### 👤 Présentation du joueur", language)
        pres_lines = "\n".join(f"- {line}" for line in profile.get("paragraphs", []))
        presentation_md = f"""{pres_title}

        {full_name}  
        {pres_lines if pres_lines else "_(No bio details found)_"}

        ---
        """
        

        # 1) Player Presentation
        profile = scrape_player_profile(url)  # {"name", "attributes", "paragraphs", "position_hint"}
        if profile.get("name"):
            full_name = profile["name"]  # prefer exact FBref h1

        pres_title = _section_title("### 👤 Player Presentation", "### 👤 Présentation du joueur", language)

        # bullet list of labeled attributes
        attr_lines = "\n".join(
            f"- **{a['label']}**: {a['value']}" for a in profile.get("attributes", [])
        )

        # free-form paragraphs under the attributes (if any)
        bio_lines = "\n".join(f"- {line}" for line in profile.get("paragraphs", []))

        presentation_md = f"""{pres_title}

        {full_name}
        {attr_lines if attr_lines else ""}
        ---
        """
        '''
        profile = scrape_player_profile(url)  # {"name","attributes","paragraphs","position_hint"}
        if profile.get("name"):
            full_name = profile["name"]

        items = _merge_profile_items(profile)
        presentation_md = _profile_table_md(full_name, items, language)
        
        #{bio_lines if bio_lines else ""}

        # 2) Scrape all tables
        tables = scrape_all_tables(url)

        # 2a) Scouting table (Last 365d vs position group)
        scout_key = next((k for k in tables.keys() if k.startswith("scout_summary")), None)
        if not scout_key:
            return f"{presentation_md}\n⚠️ Could not find a scouting table for {full_name}."

        print(f"📄 Using scouting table: {scout_key} for {full_name}")
        scout_df = tables[scout_key]
        if scout_df.shape[1] < 3:
            return f"{presentation_md}\n⚠️ Scouting table for {full_name} has an unexpected format."

        # normalize
        scout_df.columns = ["Metric", "Per90", "Percentile"][:len(scout_df.columns)]
        scout_df.set_index("Metric", inplace=True)

        display_df = scout_df[["Per90", "Percentile"]].copy()

        # ✅ FUTURE-PROOF: safe numeric conversion (no deprecated errors="ignore")
        for col in ["Per90", "Percentile"]:
            display_df[col] = _to_numeric_safely(display_df[col])

        # ✅ NEW: Deterministic grade from the scouting table
        role_hint = profile.get("position_hint") or scout_key  # prefer profile hint
        grade_bd = compute_grade(scout_df, role_hint=role_hint)
        grade_md = rationale_from_breakdown(grade_bd, language=language)

        scout_title = _section_title(
            f"### 🧾 Scouting Report ({scout_key.replace('scout_summary_', '').upper()})",
            f"### 🧾 Rapport de scouting ({scout_key.replace('scout_summary_', '').upper()})",
            language,
        )
        scout_md = display_df.to_markdown()

        grade_section_title = _section_title(
            f"### 🧾 Grade/100",
            f"### 🧾 Note/100",
            language,
        )

        # 2b) Standard Stats (season-by-season)
        standard_keys = [k for k in tables.keys() if k.startswith("stats_standard")]
        if not standard_keys:
            # keep presentation; still allow LLM to run with scout_df only
            std_md = _section_title("### 📚 Standard Stats (season-by-season)", "### 📚 Statistiques standards (saison par saison)", language) + "\n\n" + (
                "insufficient data" if not (language or "").lower().startswith("fr") else "donnée indisponible"
            )
        else:
            print(f"📄 Using standard table: {standard_keys} for {full_name}")
            chosen = _prefer_stats_standard_key(standard_keys)
            std_df = tables[chosen].copy()

            std_title = _section_title(
                "### 📚 Standard Stats (season-by-season)",
                "### 📚 Statistiques standards (saison par saison)",
                language,
            )
            # 👉 send exactly as scraped
            std_md = f"\n\n{std_title}\n\n" + std_df.to_markdown(index=False)

        #print(extra_context_md)

        # 3) LLM analysis with extra context
        '''
        llm_text = analyze_single_player(
            full_name,
            scout_df,
            language=language,
            std_md=std_md
        )
        
        '''
        
        llm_text_light = analyze_single_player(
            full_name,
            scout_df,
            language=language
        )

        print("✅ Report Generation Done.")

        return f"""{presentation_md}

{scout_title}
{scout_md}

{grade_section_title}
{grade_md}

---

{llm_text_light}
"""

    except Exception as e:
        return f"⚠️ Error analyzing {full_name}: {e}"
