# utils/llm_analysis.py

from __future__ import annotations
import requests
from datetime import datetime
import json

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
    language: str = "English"
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
- Keep each bullet list to **≤ 5 items**.
- When a formation assignment is uncertain, specify the uncertainty explicitly.
"""

        # Clear section mapping in the user's requested shape
        output_format = f"""
### 1) {"Analyse du Scouting (365 jours)" if language.lower().startswith("fr") else "Scouting Summary Analysis (last 365 days)"}
- Use the **scout_summary** table (per‑90 + percentiles).
- Provide three subsections as bullet lists:
  - {"🟢 Forces" if language.lower().startswith("fr") else "🟢 Strengths"} (top 3–5 signals with highest percentiles relevant to the player's position)
  - {"🔴 Faiblesses" if language.lower().startswith("fr") else "🔴 Weaknesses"} (lowest percentiles and any negative metrics)
  - {"🟡 Axes d'amélioration" if language.lower().startswith("fr") else "🟡 Points to Improve"} (concrete, metric-based improvements)

### 2) {"Astuce tactique" if language.lower().startswith("fr") else "Tactical Tip"}
- For each system, list **best-fit roles** (2–3 max) based on the evidence:
  - **4‑3‑3**
  - **4‑4‑2**
  - **3‑5‑2**
- Tie each suggested role to specific metrics/signals from the scouting or standard stats.
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

#### OUTPUT FORMAT (all headings and bullet labels in {language})
{output_format}
"""

        payload_0 = {
            "model": "gemma2:9b",  # gemma2:9b / gemma3
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }

        payload = {
            "model": "gemma3",
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "keep_alive": "30m",
            "options": {
                "temperature": 0.2,
                "num_ctx": 2048,
                "num_predict": 800,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1,
                "seed": 1
            }
            }

        '''
        res = requests.post(OLLAMA_API_URL, json=payload, timeout=300)
        res.raise_for_status()
        data = res.json()
        return "### 🧠 LLM Analysis\n\n" + data["message"]["content"]
        '''

        chunks = []

        with requests.post(OLLAMA_API_URL, json=payload, timeout=300, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                # The model sends back tokens in "message" events
                if "message" in ev and "content" in ev["message"]:
                    chunks.append(ev["message"]["content"])
                # If Ollama signals done
                if ev.get("done"):
                    break

        full_content = "".join(chunks).strip()

        return "### 🧠 LLM Analysis\n\n" + (full_content or "⚠️ LLM returned no content.")

    except Exception as e:
        return f"⚠️ LLM analysis failed: {e}"
