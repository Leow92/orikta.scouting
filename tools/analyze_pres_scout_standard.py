# tools/analyze_pres_scout_standard.py

import pandas as pd
from utils.fbref_scraper import scrape_all_tables, scrape_player_profile
from utils.llm_analysis import analyze_single_player_v2
from utils.resolve_player_url import search_fbref_url_with_playwright

def _section_title(text_en: str, text_fr: str, language: str) -> str:
    return text_fr if (language or "").lower().startswith("fr") else text_en

def _prefer_stats_standard_key(keys: list[str]) -> str | None:
    """
    Choose the most useful stats_standard* table if multiple exist.
    Preference order (best-effort): domestic league -> all comps -> others.
    """
    if not keys:
        return None
    # Sort by a manual preference
    priority = [
        "stats_standard_dom_lg",
        "stats_standard_lg",       # some pages
        "stats_standard",          # generic
        "stats_standard_all",      # if present
    ]
    for pref in priority:
        for k in keys:
            if k == pref:
                return k
    # fallback: the first one
    return keys[0]

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

        # 2) Scrape all tables
        tables = scrape_all_tables(url)

        # 2a) Scouting table (Last 365d vs position group)
        scout_key = next((k for k in tables.keys() if k.startswith("scout_summary")), None)
        if not scout_key:
            # we keep presentation, but bail out if no scout table
            return f"{presentation_md}\n⚠️ Could not find a scouting table for {full_name}."

        print(f"📄 Using scouting table: {scout_key} for {full_name}")
        scout_df = tables[scout_key]
        if scout_df.shape[1] < 3:
            return f"{presentation_md}\n⚠️ Scouting table for {full_name} has an unexpected format."

        # normalize
        scout_df.columns = ["Metric", "Per90", "Percentile"][:len(scout_df.columns)]
        scout_df.set_index("Metric", inplace=True)

        display_df = scout_df[["Per90", "Percentile"]].copy()
        # optional numeric cleaning
        for col in ["Per90", "Percentile"]:
            try:
                display_df[col] = pd.to_numeric(display_df[col].astype(str).str.replace("%", ""), errors="ignore")
            except Exception:
                pass

        scout_title = _section_title(
            f"### 🧾 {full_name} — Scouting Report ({scout_key.replace('scout_summary_', '').upper()})",
            f"### 🧾 {full_name} — Rapport de scouting ({scout_key.replace('scout_summary_', '').upper()})",
            language,
        )
        scout_md = display_df.to_markdown()

        # 2b) Standard Stats (season-by-season)
        standard_keys = [k for k in tables.keys() if k.startswith("stats_standard")]
        if not standard_keys:
            # we keep presentation, but bail out if no scout table
            return f"{presentation_md}\n⚠️ Could not find a scouting table for {full_name}."

        print(f"📄 Using scouting table: {standard_keys} for {full_name}")

        std_md = ""
        if standard_keys:
            chosen = _prefer_stats_standard_key(standard_keys)
            std_df = tables[chosen].copy()

            # Heuristic: if first row looks like headers (contains 'Season' or 'Age'), set it as header
            try:
                if not std_df.empty and any(x.lower() in ("season", "age", "squad", "comp") for x in std_df.iloc[0].astype(str).str.lower()):
                    std_df.columns = std_df.iloc[0]
                    std_df = std_df.iloc[1:].reset_index(drop=True)
            except Exception:
                pass

            std_title = _section_title(
                "### 📚 Standard Stats (season-by-season)",
                "### 📚 Statistiques standards (saison par saison)",
                language,
            )
            # keep a reasonable number of rows to avoid overly long outputs
            std_md = f"\n\n{std_title}\n\n" + std_df.to_markdown(index=False)

        # Build LLM extra context (presentation + standard stats)
        extra_context_md = presentation_md + (std_md or "")

        # 3) LLM analysis with extra context
        llm_text = analyze_single_player_v2(
            full_name,
            scout_df,
            language=language,
            extra_context_md=extra_context_md
        )

        return f"""{presentation_md}
{scout_title}

{scout_md}
{std_md}

---

{llm_text}
"""

    except Exception as e:
        return f"⚠️ Error analyzing {full_name}: {e}"
