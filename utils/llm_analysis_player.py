# utils/llm_analysis_player.py

from __future__ import annotations
from datetime import datetime
import pandas as pd
from requests.exceptions import ReadTimeout
from utils.lang import _is_fr
from utils.llm_client import _groq_chat
from prompts.render import render
from prompts.lang import glossary_block, role_guide, ROLE_CODE_MAP


def report_header() -> str:
    return f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}"


# ----------------------------- #
# Role inference from metrics   #
# ----------------------------- #
_ROLE_HINTS = {
    "gk": ["Save", "Goals Conceded", "Saves per 90", "Save %"],
    "df": ["Tackles", "Interceptions", "Blocks", "Duels Won"],
    "mf": ["Key Passes", "Pass Completion", "Dribbles", "Assists"],
    "fw": ["Goals per 90", "G+A", "Shots per 90", "Shot Accuracy"],
}

def _infer_role_from_metrics(index_names) -> tuple[str, float]:
    txt = [str(s).lower() for s in index_names]
    scores: dict[str, int] = {}
    for role, keys in _ROLE_HINTS.items():
        scores[role] = sum(sum(1 for s in txt if k.lower() in s) for k in keys)
    if not scores or all(v == 0 for v in scores.values()):
        return "mf", 0.25
    best = max(scores, key=scores.get)
    total = sum(scores.values()) or 1
    return best, max(0.25, min(1.0, scores[best] / total))


# ----------------------------- #
# Signal ranking                #
# ----------------------------- #
def _rank_signals(scout_df: pd.DataFrame, top_n: int = 8):
    if "Percentile" not in scout_df.columns:
        return [], []
    s = pd.to_numeric(scout_df["Percentile"], errors="coerce").dropna()
    if s.empty:
        return [], []
    return (
        [(idx, float(val)) for idx, val in s.nlargest(top_n).items()],
        [(idx, float(val)) for idx, val in s.nsmallest(top_n).items()],
    )

def _fmt_pairs(pairs: list[tuple[str, float]]) -> str:
    return "\n".join(f"- {m} — {int(p)}p" for m, p in pairs) if pairs else "- —"

def _fmt_drivers(drivers, max_n: int = 5) -> str:
    if not drivers:
        return "- —"
    return "\n".join(
        f"- {m} (w={float(w):.2f})" for m, w, *_ in drivers[:max_n]
    )


# ----------------------------- #
# Main workflow                 #
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
    try:
        if "Percentile" in scout_df.columns:
            scout_df = scout_df.copy()
            scout_df["Percentile"] = pd.to_numeric(scout_df["Percentile"], errors="coerce")

        scout_pct_only_md = scout_df[["Percentile"]].to_markdown(tablefmt="pipe", index=True)

        # Glossary filtered to present metrics
        idx = [str(i) for i in scout_df.index]
        glossary_block_str = glossary_block(language, idx)

        # Role inference + ranked signals
        role, conf = _infer_role_from_metrics(idx)
        top_signals, bottom_signals = _rank_signals(scout_df)
        top_signals_md    = _fmt_pairs(top_signals)
        bottom_signals_md = _fmt_pairs(bottom_signals)

        # Grade context
        grade_role_str = f"Detected role: {role.upper()} (confidence {conf:.2f})"
        if grade_ctx:
            raw_role = str(grade_ctx.get("role", "")).lower()
            grade_role_str = ROLE_CODE_MAP.get(raw_role, raw_role.upper() or "—")

        def _call(prompt: str) -> str:
            try:
                return _groq_chat(prompt, language)
            except ReadTimeout:
                return _groq_chat(prompt, language)

        fallback = "donnée indisponible" if _is_fr(language) else "insufficient data"

        # ---- Prompt 1: bullet scouting analysis ----
        """ p_scouting = render(
            "player_scouting.j2",
            player=player,
            role=role,
            conf=conf,
            role_guide_str=role_guide(role, language),
            top_signals_md=top_signals_md,
            bottom_signals_md=bottom_signals_md,
            scout_pct_only_md=scout_pct_only_md,
            glossary_block_str=glossary_block_str,
            presentation_md=presentation_md or "",
            grade_role_str=grade_role_str,
            language=language,
        )
        scouting_md = _call(p_scouting) or fallback """

        # ---- Prompt 2: synthesis paragraphs ----
        p_summary = render(
            "player_summary.j2",
            scout_pct_only_md=scout_pct_only_md,
            multi_style_md=multi_style_md or "",
            trend_block_md=trend_block_md or "",
            presentation_md=presentation_md or "",
            language=language,
        )
        summary_md = _call(p_summary) or fallback

        # ---- Assemble ----
        if _is_fr(language):
            title_summary = "### Résumé Global"
            title_analysis = "### Analyse Poussée"
        else:
            title_summary = "### Overall Summary"
            title_analysis = "### Deep Analysis"

        return (
            f"{title_analysis}\n\n"
            f"{title_summary}\n\n{summary_md}"
        )

    except Exception as e:
        return f"⚠️ LLM analysis failed: {e}"
