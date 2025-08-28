# utils/llm_analysis_player.py

from __future__ import annotations
import json
from datetime import datetime
from typing import Iterable
import pandas as pd
import requests
from requests.exceptions import ReadTimeout
from utils.lang import _is_fr, _lang_block

OLLAMA_API_URL = "http://localhost:11434/api/chat"

# ----------------------------- #
# Language utils & presentation #
# ----------------------------- #
def report_header() -> str:
    return f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}"

# ----------------------------- #
# Glossary                      #
# ----------------------------- #
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

def _glossary(language: str | None) -> dict[str, str]:
    return SCOUT_METRIC_GLOSSARY_FR if _is_fr(language) else SCOUT_METRIC_GLOSSARY_EN

# ----------------------------- #
# Role hints & guides           #
# ----------------------------- #
ROLE_HINTS = {
    "gk": [
        "Save", "PSxG", "Crosses Stopped", "GA/90", "Launch%", "Avg Distance of Def Actions",
        "Pass Completion % (Launched)", "Passes Attempted (Avg Len)"
    ],
    "df": ["Tackles", "Interceptions", "Clearances", "Blocks", "Dribblers Tackled", "Aerials", "Ball Recoveries"],
    "mf": ["Progressive Passes", "Progressive Carries", "Pass Completion", "Shot-Creating Actions", "xAG"],
    "fw": ["Non-Penalty Goals", "Shots Total", "npxG", "xG", "Touches (Att Pen)", "Progressive Passes Received", "Assists"],
}

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

def _role_guide(role: str, language: str | None) -> str:
    return (ROLE_GUIDE_FR if _is_fr(language) else ROLE_GUIDE_EN).get(role, (ROLE_GUIDE_FR if _is_fr(language) else ROLE_GUIDE_EN)["mf"])

ROLE_CODE_MAP = {"fw": "FW", "mf": "MF", "df": "DF", "gk": "GK"}

# ----------------------------- #
# Signal processing             #
# ----------------------------- #
def _infer_role_from_metrics(index_names: Iterable[str]) -> tuple[str, float]:
    """
    Heuristic (count substring hits) -> (role, confidence 0..1).
    Defaults to ('mf', 0.25) on no signal.
    """
    txt = [str(s).lower() for s in index_names]
    scores: dict[str, int] = {}
    for role, keys in ROLE_HINTS.items():
        rk = sum(sum(1 for s in txt if k.lower() in s) for k in keys)
        scores[role] = rk
    if not scores or all(v == 0 for v in scores.values()):
        return "mf", 0.25
    role = max(scores, key=scores.get)
    total = sum(scores.values()) or 1
    conf = max(0.25, min(1.0, scores[role] / total))
    return role, conf

def _rank_signals(scout_df: pd.DataFrame, top_n: int = 8) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """
    Return (top_list, bottom_list) of (metric, percentile) from the 'Percentile' column.
    """
    if "Percentile" not in scout_df.columns:
        return [], []
    s = pd.to_numeric(scout_df["Percentile"], errors="coerce").dropna()
    if s.empty:
        return [], []
    # s is a Series indexed by metric names
    top = [(idx, float(val)) for idx, val in s.nlargest(top_n).items()]
    bot = [(idx, float(val)) for idx, val in s.nsmallest(top_n).items()]
    return top, bot

# ----------------------------- #
# Formatting helpers            #
# ----------------------------- #
def _fmt_pairs(pairs: list[tuple[str, float]]) -> str:
    return "\n".join(f"- {m} — {int(p)}p" for m, p in pairs) if pairs else "- —"

def _fmt_drivers(drivers: list[tuple[str, float, float]] | None, max_n: int = 5) -> str:
    if not drivers:
        return "- —"
    out = []
    for metric, weight, _contrib in drivers[:max_n]:
        try:
            w = f"{float(weight):.2f}"
        except Exception:
            w = str(weight)
        out.append(f"- {metric} (w={w})")
    return "\n".join(out) if out else "- —"

# ----------------------------- #
# Main entry                    #
# ----------------------------- #
def analyze_single_player_workflow(
    player: str,
    scout_df: pd.DataFrame,
    language: str = "English",
    grade_ctx: dict | None = None,
    multi_style_md: str | None = None,
    trend_block_md: str | None = None,
    presentation_md: str | None = None,
) -> str:
    """
    Multi-call LLM workflow:
      1) Investment Verdict (grade + scouting + trends)
      2) Scouting Analysis (scouting-only)
      3) Performance Evolution (trends-only)
      4) Tactical Fit (scouting + role)
    Returns a stitched markdown block; never raises (returns an error string on failure).
    """
    try:
        # ---------- Prep: numeric safety + render sources ----------
        if "Percentile" in scout_df.columns:
            scout_df = scout_df.copy()
            scout_df["Percentile"] = pd.to_numeric(scout_df["Percentile"], errors="coerce")

        table_md = scout_df.to_markdown()

        if "Percentile" in scout_df.columns:
            scout_pct_only = scout_df[["Percentile"]].copy()
            scout_pct_only_md = scout_pct_only.to_markdown(tablefmt="pipe", index=True)
        else:
            # Fallback if Percentile missing (shouldn't happen in your pipeline)
            scout_pct_only_md = "_No percentile data available_"

        # Glossary filtered to present metrics
        gloss = _glossary(language)
        idx = [str(i) for i in scout_df.index]
        present = [m for m in gloss if m in idx]
        if present:
            glossary_title = "### Scouting Metrics Glossary" if not _is_fr(language) else "### Glossaire des métriques de scouting"
            glossary_md = "\n".join(f"- **{m}**: {gloss[m]}" for m in present)
            glossary_block = f"\n{glossary_title}\n{glossary_md}\n"
        else:
            glossary_block = ""

        # Role inference + ranked signals
        role, conf = _infer_role_from_metrics(idx)
        top_signals, bottom_signals = _rank_signals(scout_df, top_n=8)
        top_signals_md = _fmt_pairs(top_signals)
        bottom_signals_md = _fmt_pairs(bottom_signals)

        # Grade context strings
        grade_role_str = None
        drivers_md = missing_md = ""
        if grade_ctx:
            raw_role = str(grade_ctx.get("role", "")).lower()
            grade_role_str = ROLE_CODE_MAP.get(raw_role, str(grade_ctx.get("role", "")).upper() or "—")
            drivers_md = _fmt_drivers(grade_ctx.get("drivers"))
            missing_md = "\n".join(f"- {m}" for m in (grade_ctx.get("missing") or [])[:5]) or "- —"
        # Fallback if no grade_ctx
        if not grade_role_str:
            grade_role_str = f"Detected role: {role.upper()} (confidence {conf:.2f})"

        # ---------- Common caller ----------
        base_payload_opts = {
            "temperature": 0.15,
            "top_p": 0.9,
            "repeat_penalty": 1.05,
            "num_ctx": 2048,
            "num_predict": 800,
        }

        def _ollama_stream(user_content: str, language: str) -> str:
            payload = {
                "model": "gemma3",
                "messages": [
                    {"role": "system", "content": _lang_block(language)},   # <-- language enforced here
                    {"role": "user",   "content": user_content},
                ],
                "stream": True,
                "keep_alive": "30m",
                "options": {
                    "temperature": 0.15,
                    "top_p": 0.9,
                    "repeat_penalty": 1.05,
                    "num_ctx": 2048,
                    "num_predict": 800,
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
                    if isinstance(ev, dict) and ev.get("error"):
                        raise RuntimeError(ev["error"])
                    msg = ev.get("message", {})
                    if isinstance(msg, dict) and "content" in msg:
                        chunks.append(msg["content"])
                    if ev.get("done"):
                        break
            return "".join(chunks).strip()

        def _call_twice(prompt_text: str) -> str:
            try:
                return _ollama_stream(prompt_text, language)   # <-- pass language from outer scope
            except ReadTimeout:
                return _ollama_stream(prompt_text, language)

        # ---------- Prompt 1: Investment Verdict ----------
        prompt_verdict = f"""
You are an elite tactical football analyst advising a top European club. Your club position itself on {player}. 

# TASK
You will produce an investment verdict (BUY / HOLD / PASS) with a weighted score and confidence, using only the rubric below and the provided data.

# RUBRIC (weights sum to 100)
1) Role Fit & Current Level (30) — from role grade in "Role & Grade".
2) Immediate Impact (20) — mean of top-3 **role-critical** percentiles from the scouting table.
3) Development Upside (15) — from seasonal trends: +5 per clear Improving metric (max +15).
4) Risk Profile (25) — start 100, subtract:
   -10 each role-critical weakness <25p (max −30)
   -5 each Declining metric in trends (max −15)
   -10 if availability risk (low 90s or missing minutes). Normalize 0–100.
5) System Fit / Versatility (10) — up to 10 if ≥2 systems show 2–3 credible positions backed by strong percentiles; else 5 if only one fits; else 0.

Total = weighted sum.
Verdict mapping: BUY (≥80 & no red flag) / HOLD (65–79 or one moderate risk) / PASS (<65 or any red flag).
Red flags: multiple role-critical weaknesses <20p; clear multi-metric decline; availability concern.

Confidence (1–5): start 3; +1 broad scouting coverage; +1 coherent complete trends; −1 key metrics missing; −1 trends absent.

# DATA
## Role & Grade
{grade_role_str}
{(f"- Top weighted drivers:\n{drivers_md}" if grade_ctx else "").strip()}
{(f"- Missing signals:\n{missing_md}" if grade_ctx else "").strip()}

## Scouting Summary (last 365d)
{scout_pct_only_md}

## Seasonal Trends (last two seasons)
{trend_block_md}

Only provide the output standalone.
""".strip()

        #verdict_md = _call_twice(prompt_verdict) or ("insufficient data" if not _is_fr(language) else "donnée indisponible")

        # Build once, reuse for Prompts 2 & 4
        scout_pct_only = scout_df[["Percentile"]].copy()
        scout_pct_only_md = scout_pct_only.to_markdown(tablefmt="pipe", index=True)

        # ---------- Prompt 2: Scouting Analysis (scouting only) ---------- 
        prompt_scoutingv2 = f"""
<role>  
You are a tactical football analyst with expertise in player performance evaluation based on percentile-based scouting data.  
</role>

<task>  
Your task is to analyze {player}’s performance over the last 365 days using percentile-based scouting data and role, position context.  
</task>

<instructions>  
Follow these steps:
1. Read and interpret percentile values from the scouting summary.
2. Rank and categorize metrics into:
   - 🟢 Strengths — 3–4 items, ordered by *role priority**, then by *lowest percentile*.
   - 🔴 Weaknesses — 2–3 items, ordered by *role priority*, then by *lowest percentile*.
   - 🟡 Points to Improve — 2–3 actionable suggestions, ordered by *role priority* and *impact potential*.
3. Write each bullet in this format:  
   `*Metric — XXp*: short, role-specific note (≤ 18 words)`

**Important Rules:**
* Use only the provided scouting table (`Scouting Summary`).
* Do not fabricate any values or interpret beyond the given data.
* Cite percentile values using “XXp” (rounded).
* Prioritize relevance to the player's role using the `Role priorities` guide.
</instructions>

<context>  
- Player: {player}  
- Detected role: **{role.upper()}** (confidence: {conf:.2f})  
- Role priorities: {_role_guide(role, language)}  
- Top metrics (helper list for ranking):  
  {top_signals_md}  
- Lowest metrics (helper list for ranking):  
  {bottom_signals_md}  
</context>

<output-format> 
First, provide a quick explanation of what is a percentile and what it means.

Then, organize output into three clear sections using the following headers:  
🟢 Strengths  

- ...  
- ...  

🔴 Weaknesses  

* ...
* ...

🟡 Points to Improve  

* ...
* ...
</output-format>

<user-input>  
## Scouting Summary (last 365d)  
{scout_pct_only_md}

## Scouting Metrics Glossary  
{glossary_block}

## Presentation of the Player  
{presentation_md}

## Role Context Summary  
{grade_role_str}  
</user-input>
  
{_lang_block(language)}
""".strip()

        # ---------- Prompt 3: Performance Evolution (trends only) ----------
        prompt_trends = f"""
You are a tactical football analyst. You are comparing the performance of {player} between the last two seasons, the data are available in DATA.

# TASK
From seasonal stats only, list up to 3 metrics each:
- Improving Metrics
- Consistent Metrics
- Declining Metrics (add a plausible one-clause reason if relevant: role change, minutes, league strength)
Do not use the scouting table.

# DATA
## Seasonal Trends (last two seasons)
{trend_block_md}

## Scouting Metrics Glossary
{glossary_block}

Only provide the output standalone.
""".strip()

        #trends_md = _call_twice(prompt_trends) or ("insufficient data" if not _is_fr(language) else "donnée indisponible")

        # ---------- Prompt 4: Tactical Fit (scouting + role) ----------
        prompt_tacticalv2 = f"""
<role>  
You are a tactical football analyst with expertise in player profiling and formation fit analysis.  
</role>

<task>  
Your task is to analyze {player}’s profile to identify their best tactical fits across multiple formations, based solely on scouting percentiles and role context.  
</task>

<instructions>  
For each of the following systems — **4-3-3**, **4-4-2**, and **3-5-2** — do the following:
* Recommend **2–3 best-fit positions** using only valid taxonomy positions: `fw`, `mf`, `df`, `gk` and their subroles.
* For each position, provide a concise, one-line explanation that:
   - Cites 1–2 relevant metrics using the format `XXp` (rounded percentiles).
   - Explains *why* the player fits that position, based on their strengths and role alignment.

**Important Rules:**

* Use only data from the provided scouting table — no seasonal stats.
* Ground all position fits in percentile data and role priorities.
* Do **not fabricate** data or interpret outside the given metrics.

</instructions>

<context>  
- Player: {player}  
- Detected role: **{role.upper()}** (confidence: {conf:.2f})  
- Role priorities: {_role_guide(role, language)}  
- Role grading & interpretation:  
  {grade_role_str}  
</context>

<examples>  
**Example for 4-3-3:**  
- `CM (mf/creator)`: High passing (87p) and progressive carries (84p) suit advanced midfield playmaker role.  
- `RW (fw/inverter)`: Acceleration (89p) and xA (85p) match inverted wide-forward profile.  
</examples>

<output-format>  
Structure your answer like this: 
🔷 4-3-3  
- [POSITION]: [Short reason with metric(s) — XXp]  
- ...  

🔷 4-4-2

* [POSITION]: [Short reason with metric(s) — XXp]
* ...

🔷 3-5-2

* [POSITION]: [Short reason with metric(s) — XXp]
* ...
</output-format>

<user-input>  
## Scouting Summary (last 365d)  
{scout_pct_only_md}

## Multi Style Positions Table  
{multi_style_md}

## Scouting Metrics Glossary  
{glossary_block}

## Presentation of the Player  
{presentation_md}
</user-input>

{_lang_block(language)}  
""".strip()

        scouting_md = _call_twice(prompt_scoutingv2) or ("insufficient data" if not _is_fr(language) else "donnée indisponible")
        #tactical_md = _call_twice(prompt_tacticalv2) or ("insufficient data" if not _is_fr(language) else "donnée indisponible")

        prompt_summaryv2 = f"""
<role>  
You are a professional football scout skilled in synthesizing structured data into clear, evaluative scouting reports for recruitment teams.  
</role>

<task>  
Your task is to write a **Scouting Synthesis Text Analysis** (3–5 analytical paragraphs) for a player, based entirely on structured data provided in table format.  
</task>

<instructions>  
Follow these steps:
1. Carefully analyze the following structured inputs:
   - `{{scouting_metrics_table}}`: Percentiles and per90 values across key metrics.
   - `{{style_fit_matrix_table}}`: Fit scores showing player-role compatibility across tactical styles.
   - `{{standard_stats_table}}`: Season-by-season stats (past two seasons).
   - `{{scout_summary_points}}`: Bullet points of strengths, weaknesses, and areas for improvement.
   - `{{presentation_player}}`: Player profile introduction.

2. Write a 3–5 paragraph scouting synthesis. Your analysis must:

   * Present the player's **technical, tactical, physical**, and **mental** traits.
   * Reference key metrics using **percentiles (e.g., “87p”)** or **per90 values** when relevant.
   * Highlight standout strengths and critical weaknesses.
   * Discuss the player’s **positional versatility** and **fit within tactical systems**.
   * Analyze **trends across seasons** from the standard stats table.
   * Integrate insights from `{{scout_summary_points}}` smoothly into the narrative.
   * Conclude with a **Summary Projection** indicating:

     * Optimal role/system.
     * Competitive level suited for.
     * Conditions needed to maximize the player’s impact.

3. **Style & Constraints**:

   * Write in a **concise, professional, and decision-oriented tone**.
   * Avoid restating tables — interpret, compare, and synthesize insights for a recruitment audience.
   * Do not invent any values or extrapolate beyond the given data.

</instructions>

<output-format>  
Structure your output in **five short paragraphs** as follows:  
```
1️⃣ Overview and player identity (position, key traits, basic style).  
2️⃣ Attacking and creative attributes (key metrics + interpretation).  
3️⃣ Defensive and physical qualities, limitations or gaps.  
4️⃣ Tactical fit: roles, formations, and adaptability.  
5️⃣ Summary Projection: optimal level, usage, development conditions.  
```  
</output-format>

<user-input>  
{{scouting_metrics_table}}:  
{scout_pct_only_md}

{{style_fit_matrix_table}}:
{multi_style_md}

{{standard_stats_table}}:
{trend_block_md}

{{scout_summary_points}}:
{scouting_md}

{{presentation_player}}:
{presentation_md}
</user-input>

{_lang_block(language)} 
""".strip()
        
        summary_md = _call_twice(prompt_summaryv2) or ("insufficient data" if not _is_fr(language) else "donnée indisponible")

        # ---------- Assemble final markdown ----------
        #title_verdict = "### 💼 Verdict" if not _is_fr(language) else "### 💼 Verdict"
        title_scout = "### 🧾 Scouting Analysis" if not _is_fr(language) else "### 🧾 Analyse scouting"
        #title_trend = "### 📈 Performance Evolution" if not _is_fr(language) else "### 📈 Évolution des performances"
        title_tactic = "### ♟️ Tactical Fit" if not _is_fr(language) else "### ♟️ Adaptation tactique"
        title_summary = "### ♟️ Overall Summary" if not _is_fr(language) else "### ♟️ Résumé Global"

        final_md = (
            "### 🧠 LLM Analysis\n\n"
            #f"{title_verdict}\n\n{verdict_md}\n\n---\n\n"
            f"{title_scout}\n\n{scouting_md}\n\n---\n\n"
            #f"{title_trend}\n\n{trends_md}\n\n---\n\n"
            #f"{title_tactic}\n\n{tactical_md}\n\n---\n\n"
            f"{title_summary}\n\n{summary_md}"
        )

        return final_md or ("insufficient data" if not _is_fr(language) else "donnée indisponible")

    except Exception as e:
        return f"⚠️ LLM analysis failed: {e}"
