# utils/llm_analysis_comparison.py

from __future__ import annotations
from typing import Callable
from utils.lang import _is_fr
from prompts.render import render


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
    call_fn: Callable[[str, str], str],
) -> str:
    fallback = "donnée indisponible" if _is_fr(language) else "insufficient data"

    shared = dict(
        A_name=A_name,
        B_name=B_name,
        role_label=role_label,
        scout_md_A=scout_md_A,
        scout_md_B=scout_md_B,
        glossary_block=glossary_block,
        language=language,
    )

    p_verdict = render(
        "comparison_verdict.j2",
        style_label=style_label,
        style_influence=style_influence,
        similarity_0_100=similarity_0_100,
        style_rows_md=style_rows_md,
        trend_md_A=trend_md_A,
        trend_md_B=trend_md_B,
        **shared,
    )

    p_diff = render(
        "comparison_diff.j2",
        aligned_diff_md=aligned_diff_md,
        **shared,
    )

    exec_md  = call_fn(p_verdict, language) or fallback
    scout_md = call_fn(p_diff,    language) or fallback

    if _is_fr(language):
        title_exec = f"### 💼 Verdict — {A_name} vs {B_name}"
        title_h2h  = f"### 🧾 Analyse comparée — {A_name} vs {B_name}"
    else:
        title_exec = f"### 💼 Executive Verdict — {A_name} vs {B_name}"
        title_h2h  = f"### 🧾 Head-to-Head Scouting — {A_name} vs {B_name}"

    return (
        "### 🧠 LLM Comparison\n\n"
        f"{title_exec}\n\n{exec_md}\n\n---\n\n"
        f"{title_h2h}\n\n{scout_md}\n\n---\n\n"
    )
