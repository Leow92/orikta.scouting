# tools/compare.py

import pandas as pd
from utils.fbref_scraper import scrape_all_tables
from utils.llm_analysis_comparison import analyze_comparison
from utils.resolve_player_url import search_fbref_url_with_playwright

def compare_players(players: list, language="English") -> str:
    print("🧪 Received players:", players)

    if not isinstance(players, list) or len(players) != 2:
        return "⚠️ Please provide exactly two players to compare."

    player_tables = {}
    player_scout_data = {}
    full_names = []

    for name in players:
        # Dynamic resolution
        print(f"🌐 {name} not found in index — performing web search...")
        url = search_fbref_url_with_playwright(name)
        if not url:
            return f"❌ Could not resolve FBref page for: {name}"

        full_name = name.title()
        full_names.append(full_name)
        print(f"✅ Found URL for {name}: {url}")

        try:
            tables = scrape_all_tables(url)
            player_tables[full_name] = tables

            # ✅ Find first 'scout_summary_*' key dynamically
            scout_key = next((k for k in tables if k.startswith("scout_summary")), None)
            if not scout_key:
                return f"⚠️ Could not find a scouting table for {full_name}"

            print(f"📄 Using scouting table: {scout_key} for {full_name}")

            scout_df = tables[scout_key]
            if scout_df.shape[1] < 3:
                return f"⚠️ Scouting table for {full_name} has unexpected format."

            scout_df.columns = ["Metric", "Per90", "Percentile"][:len(scout_df.columns)]
            scout_df.set_index("Metric", inplace=True)
            player_scout_data[full_name] = scout_df

        except Exception as e:
            return f"⚠️ Error scraping {full_name}: {e}"

    # LLM analysis & rendering
    try:
        # Full table comparison side-by-side
        comparison_df = pd.DataFrame()
        for name, scout_df in player_scout_data.items():
            per90 = scout_df[["Per90"]].rename(columns={"Per90": name})
            comparison_df = pd.concat([comparison_df, per90], axis=1)

        # 🔍 LLM natural language comparison
        analysis_text = analyze_comparison(
            full_names[0], full_names[1],
            {
                full_names[0]: player_scout_data[full_names[0]],
                full_names[1]: player_scout_data[full_names[1]],
            },
            language=language
        )

        return f"""### 📊 Scouting Report Comparison

{comparison_df.to_markdown()}

---

### 🧠 AI Analysis
{analysis_text}
"""

    except Exception as e:
        return f"❌ Error creating comparison: {e}"
