# utils/llm_analysis_single_player_light.py

from __future__ import annotations
import requests, json
from datetime import datetime
import pandas as pd

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

ROLE_HINTS = {
    "gk": [
        "Save", "PSxG", "Crosses Stopped", "GA/90", "Launch%", "Avg Distance of Def Actions",
        "Pass Completion % (Launched)", "Passes Attempted (Avg Len)"
    ],
    "df": [
        "Tackles", "Interceptions", "Clearances", "Blocks", "Dribblers Tackled", "Aerials", "Ball Recoveries"
    ],
    "mf": [
        "Progressive Passes", "Progressive Carries", "Pass Completion", "Shot-Creating Actions", "xAG"
    ],
    "fw": [
        "Non-Penalty Goals", "Shots Total", "npxG", "xG", "Touches (Att Pen)", "Progressive Passes Received", "Assists"
    ],
}

def _infer_role_from_metrics(index_names: list[str]) -> tuple[str, float]:
    """
    Return (role, confidence 0..1) by counting pattern hits on metric names.
    """
    txt = [s.lower() for s in index_names]
    scores = {}
    for role, keys in ROLE_HINTS.items():
        hits = 0
        for k in keys:
            k_low = k.lower()
            hits += sum(1 for s in txt if k_low in s)
        scores[role] = hits
    role = max(scores, key=scores.get)
    total_hits = sum(scores.values()) or 1
    conf = (scores[role] / total_hits) if total_hits else 0.25
    # if everything is zero, default to mf
    if all(v == 0 for v in scores.values()):
        return "mf", 0.25
    return role, max(0.25, min(1.0, conf))

# --- NEW: rank signals for the LLM (so it can order bullets) ---
def _rank_signals(scout_df: pd.DataFrame, top_n: int = 8) -> tuple[list[tuple[str,float]], list[tuple[str,float]]]:
    """
    Return (top_list, bottom_list) of (metric, percentile) sorted desc/asc.
    """
    if "Percentile" not in scout_df.columns:
        return [], []
    s = pd.to_numeric(scout_df["Percentile"], errors="coerce")
    s = s.dropna()
    if s.empty:
        return [], []
    top = [(m, float(s[m])) for m in s.sort_values(ascending=False).head(top_n).index]
    bot = [(m, float(s[m])) for m in s.sort_values(ascending=True).head(top_n).index]
    return top, bot

# --- NEW: role-specific evaluation guides (EN/FR, short) ---
ROLE_GUIDE_EN = {
    "gk": "GK priority: shot-stopping (Save%, PSxG+/-/90), high claims/crosses, sweeping (Avg Def Action Dist). Distribution quality is secondary to core shot-stopping.",
    "df": "DF priority: defending (Tackles, Int, Blocks, Clearances, Aerials), then build-up (Prog Passes, Pass Completion). Goals have very low weight.",
    "mf": "MF priority: progression and circulation (Prog Passes/Carries, Pass Completion), chance creation (SCA, xAG). Goals are a bonus unless AM/SS.",
    "fw": "FW priority: finishing volume & quality (Shots, npxG/xG, NPG), final-third presence (Touches Att Pen), chance creation (Assists/xAG) as secondary.",
}
ROLE_GUIDE_FR = {
    "gk": "Priorité GB : arrêts (Save%, PSxG+/-/90), sorties aériennes, lecture/sweeper (distance des actions déf.). Relance = secondaire après l’arrêt.",
    "df": "Priorité DEF : actions défensives (Tacles, Int, Blocks, Dégagements, Duels aériens), puis relance (Passes prog., % réussite). Les buts ont très peu de poids.",
    "mf": "Priorité MIL : progression et circulation (Passes/Courses prog., % réussite), création (SCA, xAG). Les buts sont un plus sauf rôle très offensif.",
    "fw": "Priorité ATT : volume/qualité de finition (Tirs, npxG/xG, buts hors pen.), présence zone décisive (Touches surface adv.), création = secondaire.",
}

def _role_guide(role: str, language: str) -> str:
    if (language or "").lower().startswith("fr"):
        return ROLE_GUIDE_FR.get(role, ROLE_GUIDE_FR["mf"])
    return ROLE_GUIDE_EN.get(role, ROLE_GUIDE_EN["mf"])



def analyze_single_player(
    player: str,
    scout_df,
    language: str = "English",
    grade_ctx: dict | None = None,
) -> str:
    try:
        if "Percentile" in scout_df.columns:
            scout_df = scout_df.copy()
            scout_df["Percentile"] = pd.to_numeric(scout_df["Percentile"], errors="coerce")

        # Table markdown
        table_md = scout_df.to_markdown()

        # Minimal glossary filtered to present metrics
        gloss = _glossary(language)
        idx = [str(i) for i in scout_df.index]
        present = [m for m in gloss.keys() if m in idx]
        glossary_md = "\n".join(f"- **{m}**: {gloss[m]}" for m in present)

        # --- NEW: role inference + ranked signals passed to the model ---
        role, conf = _infer_role_from_metrics(idx)
        top_signals, bottom_signals = _rank_signals(scout_df, top_n=8)

        # pack ranked lists as compact lines (Metric — Pctl)
        def fmt_list(pairs): 
            return "\n".join(f"- {m} — {int(p)}" for m, p in pairs)

        ranked_block_en = f"""**Detected role**: {role.upper()} (confidence {conf:.2f})
**Top percentiles (candidates)**:
{fmt_list(top_signals) if top_signals else "- none"}
**Lowest percentiles (candidates)**:
{fmt_list(bottom_signals) if bottom_signals else "- none"}
**Role priorities**: { _role_guide(role, 'English') }"""

        ranked_block_fr = f"""**Poste détecté** : {role.upper()} (confiance {conf:.2f})
**Meilleurs percentiles (candidats)** :
{fmt_list(top_signals) if top_signals else "- aucun"}
**Plus faibles percentiles (candidats)** :
{fmt_list(bottom_signals) if bottom_signals else "- aucun"}
**Priorités du poste** : { _role_guide(role, 'Français') }"""

        ranked_block = ranked_block_fr if (language or "").lower().startswith("fr") else ranked_block_en

        # Labels
        section1 = "Analyse du Scouting (365 jours)" if language.lower().startswith("fr") else "Scouting Summary Analysis (last 365 days)"
        strengths = "🟢 Forces" if language.lower().startswith("fr") else "🟢 Strengths"
        weaknesses = "🔴 Faiblesses" if language.lower().startswith("fr") else "🔴 Weaknesses"
        improve = "🟡 Axes d'amélioration" if language.lower().startswith("fr") else "🟡 Points to Improve"
        tactical = "Astuce tactique" if language.lower().startswith("fr") else "Tactical Tip"

        output_format = f"""
### 1) {section1}
- Provide three subsections as bullet lists:
  - {strengths} (top 3–5 high percentiles relevant to role)
  - {weaknesses} (rank the weaknesses from top priority to low priority depending on player position, low percentiles and negative signals)
  - {improve} (rank the improvements points from top priority to low priority depending on player position, concrete, metric‑based improvements)

### 2) {tactical}
- For each system, list 2–3 best‑fit roles:
  - **4‑3‑3**
  - **4‑4‑2**
  - **3‑5‑2**
- Tie each role to specific metrics/signals.
"""

        # Prompt (short glossary + role guide + ranked candidates)
        glossary_title = "### Scouting Metrics Glossary" if not language.lower().startswith("fr") else "### Glossaire des métriques de scouting"
        glossary_block = f"\n{glossary_title}\n{glossary_md}\n" if glossary_md else ""
        ranked_title = "### Role Context & Ranked Signals" if not language.lower().startswith("fr") else "### Contexte poste & signaux classés"

        grade_block = ""
        # ---- Build grade/per-position context blocks (append, don't overwrite) ----
        grade_parts = []

        # A) Best roles from per-position grades (if provided)
        if grade_ctx and grade_ctx.get("per_position_top"):
            if (language or "").lower().startswith("fr"):
                top_roles_lines = "\n".join(f"- {r} : {s}/100" for r, s in grade_ctx["per_position_top"])
                grade_parts.append(f"**Meilleurs postes (détectés)** :\n{top_roles_lines}")
            else:
                top_roles_lines = "\n".join(f"- {r}: {s}/100" for r, s in grade_ctx["per_position_top"])
                grade_parts.append(f"**Best roles (detected)**:\n{top_roles_lines}")

        # B) Deterministic grade context (role, score, drivers, missing)
        if grade_ctx:
            def _fmt_drivers(drivers):
                lines = []
                for (metric, weight, contrib) in drivers:
                    try:
                        w = f"{float(weight):.2f}"
                    except Exception:
                        w = str(weight)
                    lines.append(f"- {metric} (weight {w})")
                return "\n".join(lines) if lines else "- —"

            role_map = {"fw": "FW", "mf": "MF", "df": "DF", "gk": "GK"}
            role_str = role_map.get(str(grade_ctx.get("role", "")).lower(), str(grade_ctx.get("role", "")).upper())
            score_str = f"{grade_ctx.get('score', '—')}/100"
            drivers_md = _fmt_drivers(grade_ctx.get("drivers", []))
            missing_md = "\n".join(f"- {m}" for m in (grade_ctx.get("missing") or [])) or "- —"

            if (language or "").lower().startswith("fr"):
                grade_parts.append(
                    f"**Contexte de note déterministe**\n"
                    f"- Rôle (grade) : **{role_str}**\n"
                    f"- Note : **{score_str}**\n"
                    f"- Principaux moteurs (pondérations) :\n{drivers_md}\n"
                    f"- Signaux manquants :\n{missing_md}"
                )
            else:
                grade_parts.append(
                    f"**Deterministic Grade Context**\n"
                    f"- Role (grade): **{role_str}**\n"
                    f"- Score: **{score_str}**\n"
                    f"- Top weighted drivers:\n{drivers_md}\n"
                    f"- Missing signals:\n{missing_md}"
                )

        # Join all grade-related blocks once
        grade_block = ("\n\n" + "\n\n".join(grade_parts) + "\n\n") if grade_parts else ""

        prompt = f"""
        {report_header()}
        You are a **tactical football analyst**.

        {_lang_block(language)}

        {ranked_title}
        {ranked_block}
        {grade_block}
        {glossary_block}
        #### REQUIRED DATA — Scout Summary (per 90 + percentiles, last 365 days)
        {table_md}

        #### OUTPUT FORMAT (use {language} headings)
        {output_format}

        **Rules**:
        - Do NOT fabricate values.
        - Order items strictly by **impact for the detected role**; when grade context is present, align priorities with the **top weighted drivers** and **penalize missing signals**.
        - Deprioritize metrics with low role relevance (e.g., goals for DF).
        - Reference metrics with names and percentiles where possible.
        """

        payload = {
            "model": "gemma3",
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,            # <-- make the API stream
            "keep_alive": "30m",
            "options": {
                "temperature": 0.2,
                "num_ctx": 3072,
                "num_predict": 800,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1,
            },
        }

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

