# utils/llm_analysis_comparison.py

from __future__ import annotations
from typing import Callable
from utils.lang import _is_fr
from prompts.render import render
import utils.pipeline_log as pipeline_log


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
    aligned_diff_md: str,
    similarity_0_100: float,
    glossary_block: str,
    call_fn: Callable[[str, str], str],
) -> str:
    fallback = "donnée indisponible" if _is_fr(language) else "insufficient data"

    prompt = render(
        "comparison_deep_v0.2.j2",
        A_name=A_name,
        B_name=B_name,
        role_label=role_label,
        style_label=style_label,
        style_influence=style_influence,
        scout_md_A=scout_md_A,
        scout_md_B=scout_md_B,
        trend_md_A=trend_md_A,
        trend_md_B=trend_md_B,
        aligned_diff_md=aligned_diff_md,
        similarity_0_100=similarity_0_100,
        glossary_block=glossary_block,
        language=language,
    )

    pipeline_log.log(f"[compare_llm] Prompt length: {len(prompt)} chars — calling LLM…")
    analysis_md = call_fn(prompt, language) or fallback
    if analysis_md == fallback:
        pipeline_log.log("[compare_llm] LLM returned empty — using fallback", level="warning")
    else:
        pipeline_log.log(f"[compare_llm] LLM response received ({len(analysis_md)} chars)", level="success")

    if _is_fr(language):
        title = f"### 🧠 Analyse comparative — {A_name} vs {B_name}"
    else:
        title = f"### 🧠 Deep Scouting Analysis — {A_name} vs {B_name}"

    return f"{title}\n\n{analysis_md}\n\n---\n\n"
