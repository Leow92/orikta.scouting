# utils/llm_analysis.py

from __future__ import annotations
import requests
from datetime import datetime

OLLAMA_API_URL = "http://localhost:11434/api/chat"

def report_header() -> str:
    return f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}"

def _lang_block(language: str) -> str:
    lang = (language or "English").lower()
    if lang.startswith("fr"):
        return (
            "Rédige en **français**. Utilise des titres et puces clairs. "
            "Si une donnée est manquante, écris « donnée indisponible » plutôt que d’inventer."
        )
    return (
        "Write in **English**. Use clear headings and bullet points. "
        "If a data point is missing, write 'insufficient data' rather than inventing it."
    )

def analyze_single_player(
    player: str,
    scout_df,
    language: str = "English",
    extra_context_md: str = ""
) -> str:
    """
    LLM analysis for a single player.
    - REQUIRED: scout_summary_* as DataFrame (per-90 + percentile columns)
    - ADDITIONAL: extra_context_md should contain:
        - Player Presentation (presentation_md)
        - Standard Stats table (std_md)
        - Optional deterministic grade section (grade_md) if computed upstream
    """
    try:
        table_md = scout_df.to_markdown()

        # Hard constraints to reduce hallucinations + enforce structure
        constraints = f"""
- Do NOT fabricate clubs, seasons, minutes, or positions not present in the context.
- Quote numeric stats only if present in the scout table or standard stats (include units like per 90, %, etc.).
- If last season's standard stats are not found, state that clearly and proceed.
- Keep each bullet list to **≤ 5 items**.
- When a formation assignment is uncertain, specify the uncertainty explicitly.
- Prefer the provided deterministic grade (if present in context) instead of inventing a new number.
"""

        # Clear section mapping in the user's requested shape
        output_format = f"""
### 1) {"Vue d'ensemble" if language.lower().startswith("fr") else "Brief Overview"}
- Base this section **only** on the Player Presentation paragraphs in the context (bio, role, physical profile, club status, injuries if mentioned).

### 2) {"Analyse du Scouting (365 jours)" if language.lower().startswith("fr") else "Scouting Summary Analysis (last 365 days)"}
- Use the **scout_summary** table (per‑90 + percentiles).
- Provide three subsections as bullet lists:
  - {"🟢 Forces" if language.lower().startswith("fr") else "🟢 Strengths"} (top 3–5 signals with highest percentiles relevant to the player's position)
  - {"🔴 Faiblesses" if language.lower().startswith("fr") else "🔴 Weaknesses"} (lowest percentiles and any negative metrics)
  - {"🟡 Axes d'amélioration" if language.lower().startswith("fr") else "🟡 Points to Improve"} (concrete, metric-based improvements)

### 3) {"Analyse des Statistiques Standards (contexte par saison)" if language.lower().startswith("fr") else "Standard Stats Analysis (season context)"}
- Compare the 365‑day snapshot with **last season** from the standard stats table.
- Call out trends (↑/↓) in key metrics (usage, goal contribution, passing, defensive actions).
- If last season cannot be determined, say {"« donnée indisponible »" if language.lower().startswith("fr") else "'insufficient data'"}.

### 4) {"Astuce tactique" if language.lower().startswith("fr") else "Tactical Tip"}
- For each system, list **best-fit roles** (2–3 max) based on the evidence:
  - **4‑3‑3**: e.g., LW / RW / CF or #6 / #8 / #10, etc.
  - **4‑4‑2**: e.g., ST, SS, RM/LM, DM/CM
  - **3‑5‑2**: e.g., ST, AM, WB, #6/#8
- Tie each suggested role to specific metrics/signals from the scouting or standard stats.

### 5) {"Note (/100) et justification" if language.lower().startswith("fr") else "Rating (/100) & Justification"}
- If a **deterministic grade** exists in the context, **reuse it** and briefly justify with top drivers.
- Otherwise, provide a conservative rating anchored in the top/bottom percentiles, and justify succinctly.

### 6) {"Recommandation finale" if language.lower().startswith("fr") else "Final Recommendation"}
- One short paragraph recommending **fit, risks, and next-step scouting actions** (minutes threshold, leagues, roles to test).
"""

        prompt = f"""
{report_header()}
You are a **tactical football analyst and data scout** with deep expertise in advanced football metrics and scouting data.

{_lang_block(language)}

Follow the constraints below and produce the requested sections exactly.

#### CONSTRAINTS
{constraints}

#### REQUIRED DATA — Scout Summary (per 90 + percentiles, last 365 days)
{table_md}

#### ADDITIONAL CONTEXT (Player Presentation, Deterministic Grade if any, Standard Stats)
{extra_context_md}

#### OUTPUT FORMAT (all headings and bullet labels in {language})
{output_format}
"""

        payload = {
            "model": "gemma3",  # choose the local model you pulled with Ollama
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }

        res = requests.post(OLLAMA_API_URL, json=payload, timeout=90)
        res.raise_for_status()
        data = res.json()
        return "### 🧠 LLM Analysis\n\n" + data["message"]["content"]

    except Exception as e:
        return f"⚠️ LLM analysis failed: {e}"
