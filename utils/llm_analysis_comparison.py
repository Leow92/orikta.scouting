# utils/llm_analysis_comparison.py
from __future__ import annotations
from typing import Callable
from utils.lang import _lang_block

def _is_fr(language: str | None) -> bool:
    return (language or "").strip().lower().startswith("fr")

def compare_llm_workflow(
    *,
    A_name: str,
    B_name: str,
    language: str,
    role_label: str,
    style_label: str,
    style_influence: float,
    scout_md_A: str,
    scout_md_B: str,
    trend_md_A: str,
    trend_md_B: str,
    style_rows_md: str,
    aligned_diff_md: str,
    similarity_0_100: float,
    glossary_block: str,
    call_fn: Callable[[str, str], str],   # <- inject your _call_twice
) -> str:
    # 1) Executive verdict
    prompt1 = f"""
You are an elite tactical football analyst advising a top European club.

# TASK
Pick the better signing between **{A_name}** and **{B_name}** for the **{role_label}** role
{"with the team style: **"+style_label+"**" if style_label != "—" else "(no fixed team style)"}.
Give a weighted score (/100) and confidence (1–5). Use only the data provided.

# DECISION RULES
- Role Fit & Current Level (35) — compare the percentiles relevant to **{role_label}**.
- Immediate Impact (20) — average of top-3 role-critical percentiles per player.
- Development Upside (15) — from seasonal trend lists (improving > consistent > declining).
- Risk Profile (20) — penalize role-critical weaknesses (<25p), multi-metric decline, and availability flags (in trends).
- Style Fit (10) — use style comparison rows; style influence = {style_influence:.2f}.

Tie-breakers: higher similarity (profile match) may favor like-for-like replacement; else choose better style fit.

# OUTPUT (Markdown, 2 lines)
**Pick — NAME (Total/100, Confidence X/5)**
Short one-sentence rationale (role fit + impact + risk + style).

# DATA
## Role & Style
- Target role: {role_label}
- Team style: {style_label} (influence {style_influence:.2f})
- Similarity (profiles): {similarity_0_100:.1f}/100

## Style head-to-head (summary)
{style_rows_md}

## Scouting Percentiles — {A_name}
{scout_md_A}

## Scouting Percentiles — {B_name}
{scout_md_B}

## Seasonal Trends — {A_name}
{trend_md_A}

## Seasonal Trends — {B_name}
{trend_md_B}

{glossary_block}

{_lang_block(language)}
""".strip()
    
    prompt_verdict_2 = f"""
<role> You are an **elite tactical football analyst** advising a top European football club on transfer decisions. </role> <task> Your task is to **recommend the better signing** between two players (**{A_name}** and **{B_name}**) for the **{role_label}** role, based solely on provided scouting and trend data. </task> <instructions> Follow the evaluation framework below to assess each player and make a decision: * Assign a **total weighted score (out of 100)** and a **confidence rating (1–5)**. * Base your analysis only on the structured data blocks provided. * Prioritize evaluation criteria by the assigned weightings: - **Role Fit & Current Level (35 points)** — Based on percentiles directly related to the {role_label}. - **Immediate Impact (20 points)** — Average of the top 3 role-critical percentiles per player. - **Development Upside (15 points)** — Based on seasonal trends (improving > consistent > declining). - **Risk Profile (20 points)** — Penalize for: - Role-critical weaknesses (<25th percentile), - Negative multi-metric trend patterns, - Availability or fitness concerns. - **Style Fit (10 points)** — Based on the head-to-head style match section. Weight by style influence ({style_influence:.2f}).

Tie-breaking logic:

If scores are very close, consider:

Profile similarity to target role ({similarity_0_100:.1f}/100) for like-for-like replacement.

Superior style fit for system synergy.

</instructions> <context> Your analysis is part of a high-stakes transfer decision with no additional scouting context. Do not assume beyond the data. </context> <output-format> Respond in **Markdown** using exactly **2 lines**: 1. `**Pick — [PLAYER NAME] (Total/100, Confidence X/5)**` 2. A concise sentence explaining the rationale, focusing on role fit, impact, risk, and style. </output-format> <user-input> **Data Provided** - **Target Role**: {role_label} - **Team Style**: {style_label} (Influence: {style_influence:.2f}) - **Similarity Score (Profile Match)**: {similarity_0_100:.1f}/100
Style Comparison (Head-to-Head)

{style_rows_md}

Scouting Percentiles — {A_name}

{scout_md_A}

Scouting Percentiles — {B_name}

{scout_md_B}

Seasonal Trends — {A_name}

{trend_md_A}

Seasonal Trends — {B_name}

{trend_md_B}

{glossary_block}

{_lang_block(language)}
</user-input>
"""

    # 2) Head-to-head scouting
    prompt2 = f"""
You are a tactical football analyst.

# TASK
Head-to-head scouting for **{role_label}** using only percentiles:
- 🟢 Where **{A_name}** leads: 3–4 bullets (*Metric — Δp*: brief role-specific note, ≤ 14 words)
- 🟠 Where **{B_name}** leads: 3–4 bullets (*Metric — Δp*: …)
Order by **role importance**, then absolute gap Δp.

# DATA
## Aligned differences (Δp = {A_name} − {B_name}, role-weighted order)
{aligned_diff_md}

## Scouting — {A_name} (percentiles only)
{scout_md_A}

## Scouting — {B_name} (percentiles only)
{scout_md_B}

{glossary_block}

{_lang_block(language)}
""".strip()

    # 3) System fit (percentiles only)
    prompt4 = f"""
You are a tactical football analyst.

# TASK
For each system — **4-3-3**, **4-4-2**, **3-5-2** — pick **{A_name}** or **{B_name}** (not both) for the **{role_label}** deployment.
Give one line per system with a short “because” citing 1–2 key metrics (Metric — XXp). No seasonal stats.

# DATA
## Scouting — {A_name} (percentiles only)
{scout_md_A}

## Scouting — {B_name} (percentiles only)
{scout_md_B}

{glossary_block}

{_lang_block(language)}
""".strip()

    # Call the injected function (handles language + retry)
    def _fallback() -> str:
        return "donnée indisponible" if _is_fr(language) else "insufficient data"

    exec_md = call_fn(prompt_verdict_2, language) or _fallback()
    scout_h2h_md = call_fn(prompt2, language) or _fallback()
    system_fit_md = call_fn(prompt4, language) or _fallback()

    # Titles localized here (no dependency on _t)
    if _is_fr(language):
        title_exec = f"### 💼 Verdict — {A_name} vs {B_name}"
        title_h2h  = f"### 🧾 Analyse comparée — {A_name} vs {B_name}"
        title_sys  = f"### ♟️ Adaptation tactique — {A_name} vs {B_name}"
    else:
        title_exec = f"### 💼 Executive Verdict — {A_name} vs {B_name}"
        title_h2h  = f"### 🧾 Head-to-Head Scouting — {A_name} vs {B_name}"
        title_sys  = f"### ♟️ System Fit — {A_name} vs {B_name}"

    return (
        "### 🧠 LLM Comparison\n\n"
        f"{title_exec}\n\n{exec_md}\n\n---\n\n"
        f"{title_h2h}\n\n{scout_h2h_md}\n\n---\n\n"
        f"{title_sys}\n\n{system_fit_md}"
    )
