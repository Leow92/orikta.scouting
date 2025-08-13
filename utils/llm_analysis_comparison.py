# utils/llm_analysis.py

import requests
from datetime import datetime
from zoneinfo import ZoneInfo

OLLAMA_API_URL = "http://localhost:11434/api/chat"

date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
report_header = f"📅 Report generated on {date_str}\n\n"

def make_report_header(language: str = "English") -> str:
    """Return a language-aware report header with Brussels local time."""
    now = datetime.now(ZoneInfo("Europe/Brussels")).strftime("%Y-%m-%d %H:%M")
    if (language or "").lower().startswith("fr"):
        return f"📅 Rapport généré le {now} (Europe/Brussels)\n\n"
    return f"📅 Report generated on {now} (Europe/Brussels)\n\n"

def _lang_block(language: str) -> str:
    # Robust language control — defaults to English if unknown
    lang = (language or "English").strip()
    if lang.lower().startswith("fr"):
        return "Réponds intégralement en **français**. Utilise une terminologie football claire."
    return "Respond entirely in **English**. Use clear, football-specific terminology."

def analyze_comparison(player1: str, player2: str, scout_tables: dict, language: str = "English", extra_context_md: str = "") -> str:
    """
    Generates a football-style natural language analysis comparing two players
    based only on their 'scout_summary_FW' tables.

    Args:
        player1: str - full name of player 1
        player2: str - full name of player 2
        scout_tables: dict - {player_name: scout_df} with only 'scout_summary_FW' tables

    Returns:
        str - natural language analysis
    """

    try:
        # Convert each scouting table to markdown
        table1_md = scout_tables[player1].to_markdown()
        table2_md = scout_tables[player2].to_markdown()
        
        prompt = f"""
        You are a **tactical football analyst and data scout** with deep expertise in advanced football metrics and scouting data.

        {_lang_block(language)}

        Your task is to deliver a **strategic, data-informed comparison** between two football players based on the `scout_summary_FW` tables. Emphasize **differences in playing style, tactical role, strengths, and areas for improvement** using per 90 values and percentile rankings compared to their position group.

        INSTRUCTIONS
        1. Use only the player data provided below.
        2. Compare the following performance categories: shooting, passing, dribbling, defending, and possession.
        3. Go beyond stats: give tactical interpretations and context.
        4. Explain percentiles to highlight **relative strengths and weaknesses**.
        5. Clarify technical terms where necessary.
        6. Add relevant emojis for emphasis 🎯⚽🔥.
        7. Include color coding:
        - 🟢 Green = strong points
        - 🔴 Red = weak points
        - 🟡 Yellow = areas for improvement
        8. Assign a **grade out of 100** to each player based on the data.
        9. Identify the **preferred tactical position** of each player (e.g., "Right Winger in a 4-3-3").

        CONTEXT
        You are comparing two football players using scouting data tables. This analysis is intended to help coaches, analysts, or scouts understand player fit and performance through data. Assume the audience has a solid football knowledge base.

        OUTPUT FORMAT
        Use the following structure, with section headers translated into **{language}**:
        1. **Brief Overview** – A 2–3 sentence summary of key differences.
        2. **Strengths, Weaknesses & Development Areas** – List each player’s:
        - 🟢 Strengths (green)
        - 🔴 Weaknesses (red)
        - 🟡 Points to improve (yellow)
        3. **Statistical Rating** – Grade each player out of 100 based on overall statistical profile.
        4. **Tactical Fit** – Suggest the preferred position and tactical system for each player (e.g., "Left Forward in a 4-2-3-1").
        5. **Final Recommendation** – Optional: suggest which player fits better in a specific tactical context if relevant.

        PLAYER DATA
        {player1}:
        {table1_md}

        {player2}:
        {table2_md}
        """

        payload = {
            "model": "gemma3",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }

        res = requests.post(OLLAMA_API_URL, json=payload)
        res.raise_for_status()
        data = res.json()
        print("✅ Comparison Done.")

        header = make_report_header(language)
        return header + "### 🧠 LLM Analysis\n\n" + data["message"]["content"]

    except Exception as e:
        return f"⚠️ LLM analysis failed: {e}"
    
def analyze_single_player(player: str, scout_tables: dict, language: str = "English") -> str:
    """
    LLM analysis for a single player based on the scout_summary_* table only.
    """
    try:
        table_md = scout_tables.to_markdown()

        prompt = f"""
        You are a **tactical football analyst and data scout** with deep expertise in advanced football metrics and scouting data.

        {_lang_block(language)}

        Your task is to deliver a **strategic, data‑informed scouting report** for **{player}** using ONLY the table provided (per 90 values + percentiles, last 365 days, compared to their position group). Go beyond raw stats: interpret tactically and explain what the numbers say about role, strengths, and weaknesses.

        INSTRUCTIONS
        1) Use only the player data provided below.
        2) Cover the following performance categories: shooting, chance creation/passing, ball carrying/dribbling, defending/pressing, and possession/progression.
        3) Go beyond stats: add tactical context (movement profile, zone occupation, link play, pressing behavior, threat profile).
        4) Explain percentiles to highlight **relative strengths and weaknesses**.
        5) Clarify technical terms briefly (one‑liners).
        6) Add relevant emojis for emphasis 🎯⚽🔥 (light touch).
        7) Use color coding in the write‑up:
        - 🟢 Green = strong points
        - 🔴 Red = weak points
        - 🟡 Yellow = areas for improvement
        8) Assign a **grade out of 100** based on the overall statistical profile.
        9) Recommend a **preferred tactical position/role** (e.g., "Right Winger in a 4‑3‑3", "AM in 4‑2‑3‑1").

        OUTPUT FORMAT (headings must be in {language})
        1. **Brief Overview** – 2–3 sentence snapshot of profile and impact.
        2. **Statistical Highlights** – key per‑90 and percentile callouts (quote figures).
        3. **Strengths / Weaknesses / Development Areas**
        - 🟢 Strengths
        - 🔴 Weaknesses
        - 🟡 Points to improve
        4. **Tactical Fit** – best role(s) and system(s); how to maximize the player.
        5. **Statistical Rating (/100)** – justify the grade.
        6. **Final Recommendation** – concise takeaway for coaches/recruiters.

        PLAYER DATA (do not ignore)
        {player}
        {table_md}
        """

        payload = {
            "model": "mistral",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }

        res = requests.post(OLLAMA_API_URL, json=payload)
        res.raise_for_status()
        data = res.json()
        print("✅ Analysis Done.")

        header = make_report_header(language)
        return header + "### 🧠 LLM Analysis\n\n" + data["message"]["content"]


    except Exception as e:
        return f"⚠️ LLM analysis failed: {e}"
    
def analyze_single_player_v2(
    player: str,
    scout_df,
    language: str = "English",
    extra_context_md: str = ""
) -> str:
    """
    LLM analysis for a single player.
    - Uses scout_summary_* table (required)
    - Optionally includes extra markdown context (player presentation + standard stats)
    """
    try:
        table_md = scout_df.to_markdown()

        prompt = f"""
{report_header}
You are a **tactical football analyst and data scout** with deep expertise in advanced football metrics and scouting data.

{_lang_block(language)}

Use ALL the information below to write a concise, insightful scouting analysis for **{player}**.
Prioritize the `scout_summary` table for conclusions, but you may reference additional context if provided.

### REQUIRED DATA (do not ignore)
#### Scout Summary (per 90 + percentiles, last 365 days)
{table_md}

### ADDITIONAL CONTEXT
{extra_context_md}

### OUTPUT FORMAT (headings must be in {language})
1. **Brief Overview**
2. **Statistical Highlights** – cite a few key per‑90 and percentile figures
3. **Strengths / Weaknesses / Development Areas**
   - 🟢 Strengths
   - 🔴 Weaknesses
   - 🟡 Points to improve
4. **Tactical Fit** – best role(s) & systems
5. **Statistical Rating (/100)** – justify the grade
6. **Final Recommendation**
"""
        payload = {
            "model": "gemma3",  # or another local model you've pulled in Ollama
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }

        res = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
        res.raise_for_status()
        data = res.json()
        return "### 🧠 LLM Analysis\n\n" + data["message"]["content"]

    except Exception as e:
        return f"⚠️ LLM analysis failed: {e}"
    
