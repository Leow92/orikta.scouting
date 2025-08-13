# utils/llm_analysis.py

from __future__ import annotations
import requests, json
from datetime import datetime

OLLAMA_API_URL = "http://localhost:11434/api/chat"

# --- Glossaries ---
SCOUT_METRIC_GLOSSARY_EN = {
    "Non-Penalty Goals": "Goals scored excluding penalties",
    "npxG": "Expected goals, non-penalty",
    "xAG": "Expected assisted goals",
    "Shots Total": "Total shots attempted",
    "Shot-Creating Actions": "Actions leading to a shot",
    "Passes Attempted": "Total passes attempted",
    "Pass Completion %": "Percentage of passes completed",
    "Progressive Passes": "Passes moving the ball significantly towards goal",
    "Progressive Carries": "Carries moving the ball significantly towards goal",
    "Successful Take-Ons": "Dribbles successfully beating an opponent",
    "Touches (Att Pen)": "Touches in opponent’s penalty area",
    "Tackles Won": "Tackles where possession was won",
    "Interceptions": "Interceptions of opponent’s passes",
    "Blocks": "Blocks of shots, passes, or crosses",
    "Clearances": "Clearances from defensive area",
    "Aerials Won": "Aerial duels won by the player",
}

SCOUT_METRIC_GLOSSARY_FR = {
    "Non-Penalty Goals": "Buts marqués hors penalties",
    "npxG": "Buts attendus hors penalty (expected goals, sans penalty)",
    "xAG": "Passes décisives attendues (expected assisted goals)",
    "Shots Total": "Tirs tentés au total",
    "Shot-Creating Actions": "Actions menant à un tir",
    "Passes Attempted": "Passes tentées au total",
    "Pass Completion %": "Pourcentage de passes réussies",
    "Progressive Passes": "Passes faisant progresser nettement vers le but",
    "Progressive Carries": "Conduites de balle progressant nettement vers le but",
    "Successful Take-Ons": "Dribbles réussis (adversaire éliminé)",
    "Touches (Att Pen)": "Touches de balle dans la surface adverse",
    "Tackles Won": "Tacles gagnés (possession récupérée)",
    "Interceptions": "Interceptions de passes adverses",
    "Blocks": "Contres de tirs, passes ou centres",
    "Clearances": "Dégagements depuis la zone défensive",
    "Aerials Won": "Duels aériens gagnés",
}

def _glossary(language: str) -> dict[str, str]:
    return SCOUT_METRIC_GLOSSARY_FR if (language or "").lower().startswith("fr") else SCOUT_METRIC_GLOSSARY_EN

def report_header() -> str:
    return f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}"

def _lang_block(language: str) -> str:
    lang = (language or "English").lower()
    if lang.startswith("fr"):
        return ("Rédige en **français**. Utilise des titres et puces clairs. "
                "Si une donnée est manquante, écris « donnée indisponible ».")
    return ("Write in **English**. Use clear headings and bullet points. "
            "If a data point is missing, write 'insufficient data'.")

def analyze_single_player(
    player: str,
    scout_df,
    language: str = "English",
) -> str:
    try:
        # Table + minimal glossary (only metrics present)
        table_md = scout_df.to_markdown()
        gloss = _glossary(language)

        # normalize index to strings for matching
        idx = [str(i) for i in scout_df.index]
        present = [m for m in gloss.keys() if m in idx]

        glossary_md = "\n".join(f"- **{m}**: {gloss[m]}" for m in present)

        section1 = "Analyse du Scouting (365 jours)" if language.lower().startswith("fr") else "Scouting Summary Analysis (last 365 days)"
        strengths = "🟢 Forces" if language.lower().startswith("fr") else "🟢 Strengths"
        weaknesses = "🔴 Faiblesses" if language.lower().startswith("fr") else "🔴 Weaknesses"
        improve = "🟡 Axes d'amélioration" if language.lower().startswith("fr") else "🟡 Points to Improve"
        tactical = "Astuce tactique" if language.lower().startswith("fr") else "Tactical Tip"

        output_format = f"""
### 1) {section1}
- Provide three subsections as bullet lists:
  - {strengths} (top 3–5 high percentiles relevant to role)
  - {weaknesses} (low percentiles and negative signals)
  - {improve} (concrete, metric‑based improvements)

### 2) {tactical}
- For each system, list 2–3 best‑fit roles:
  - **4‑3‑3**
  - **4‑4‑2**
  - **3‑5‑2**
- Tie each role to specific metrics/signals.
"""

        # Prompt with a tiny glossary section (only if we found any present metrics)
        glossary_title = "### Scouting Metrics Glossary" if not language.lower().startswith("fr") else "### Glossaire des métriques de scouting"
        glossary_block = f"\n{glossary_title}\n{glossary_md}\n" if glossary_md else ""

        prompt = f"""
{report_header()}
You are a **tactical football analyst and data scout** with deep expertise in advanced football metrics.

{_lang_block(language)}

{glossary_block}
#### REQUIRED DATA — Scout Summary (per 90 + percentiles, last 365 days)
{table_md}

#### OUTPUT FORMAT (use {language} headings)
{output_format}
"""
        
        payload = {
            "model": "gemma3",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }

        # Stream response
        chunks: list[str] = []
        with requests.post(OLLAMA_API_URL, json=payload, timeout=300, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if "message" in ev and "content" in ev["message"]:
                    chunks.append(ev["message"]["content"])
                if ev.get("done"):
                    break

        content = "".join(chunks).strip()
        return "### 🧠 LLM Analysis\n\n" + (content or "⚠️ LLM returned no content.")

    except Exception as e:
        return f"⚠️ LLM analysis failed: {e}"

