# prompts/lang.py
#
# Prompt-level language utilities: language-constraint suffix, metric glossary,
# and role guides injected into every LLM call.

from __future__ import annotations
from typing import List
from utils.lang import _is_fr

# ------------------------------------------------------------------ #
# Language constraint (appended to every prompt)                      #
# ------------------------------------------------------------------ #
def lang_constraint(language: str | None) -> str:
    """Return a firm language instruction to append to any LLM prompt."""
    if _is_fr(language):
        return (
            "Write the ENTIRE answer in **French**. "
            "Do not include any English words, headings, labels, or code comments. "
            "If you accidentally use English, **discard and restart in French**. "
            "<answer_language>French</answer_language>"
        )
    return (
        "Write the ENTIRE answer in **English**. "
        "Do not include any French words, headings, labels, or code comments. "
        "If you accidentally use French, **discard and restart in English**. "
        "<answer_language>English</answer_language>"
    )


# ------------------------------------------------------------------ #
# Metric glossary (API-football metric names)                         #
# ------------------------------------------------------------------ #
GLOSSARY_EN: dict[str, str] = {
    "Goals per 90":          "Goals scored per 90 minutes",
    "Assists per 90":        "Goal assists per 90 minutes",
    "G+A per 90":            "Goals + Assists combined per 90 minutes",
    "Shots per 90":          "Total shot attempts per 90 minutes",
    "Shot Accuracy %":       "Percentage of shots on target",
    "Key Passes per 90":     "Passes directly creating a shot attempt per 90 min",
    "Pass Completion %":     "Percentage of passes successfully completed",
    "Tackles per 90":        "Tackles made per 90 minutes",
    "Interceptions per 90":  "Opponent passes intercepted per 90 minutes",
    "Blocks per 90":         "Shots or passes blocked per 90 minutes",
    "Dribbles per 90":       "Dribble attempts per 90 minutes",
    "Dribble Success %":     "Percentage of dribbles successfully completed",
    "Duels Won %":           "Percentage of ground and aerial duels won",
    "Fouls per 90":          "Fouls committed per 90 minutes (lower = better)",
    "Fouls Drawn per 90":    "Fouls won per 90 minutes",
    "Save %":                "Percentage of shots on target saved (GK)",
    "Saves per 90":          "Saves made per 90 minutes (GK)",
    "Goals Conceded per 90": "Goals conceded per 90 minutes (GK, lower = better)",
}

GLOSSARY_FR: dict[str, str] = {
    "Goals per 90":          "Buts marqués par 90 minutes",
    "Assists per 90":        "Passes décisives par 90 minutes",
    "G+A per 90":            "Buts + Passes décisives par 90 minutes",
    "Shots per 90":          "Tirs tentés par 90 minutes",
    "Shot Accuracy %":       "Pourcentage de tirs cadrés",
    "Key Passes per 90":     "Passes menant directement à un tir par 90 min",
    "Pass Completion %":     "Pourcentage de passes réussies",
    "Tackles per 90":        "Tacles réalisés par 90 minutes",
    "Interceptions per 90":  "Interceptions de passes adverses par 90 min",
    "Blocks per 90":         "Tirs ou passes contrés par 90 minutes",
    "Dribbles per 90":       "Tentatives de dribble par 90 minutes",
    "Dribble Success %":     "Pourcentage de dribbles réussis",
    "Duels Won %":           "Pourcentage de duels (au sol et aériens) gagnés",
    "Fouls per 90":          "Fautes commises par 90 min (moins = mieux)",
    "Fouls Drawn per 90":    "Fautes obtenues par 90 minutes",
    "Save %":                "Pourcentage d'arrêts sur tirs cadrés (GB)",
    "Saves per 90":          "Arrêts par 90 minutes (GB)",
    "Goals Conceded per 90": "Buts encaissés par 90 min (GB, moins = mieux)",
}


def glossary_block(language: str | None, present_metrics: List[str]) -> str:
    """Return a formatted glossary string filtered to metrics present in the report."""
    G = GLOSSARY_FR if _is_fr(language) else GLOSSARY_EN
    matches = [m for m in G if m in present_metrics]
    if not matches:
        return ""
    title = "### Glossaire des métriques" if _is_fr(language) else "### Scouting Metrics Glossary"
    lines  = "\n".join(f"- **{m}**: {G[m]}" for m in matches)
    return f"\n{title}\n{lines}\n"


# ------------------------------------------------------------------ #
# Role guides (injected as context into player-analysis prompts)      #
# ------------------------------------------------------------------ #
ROLE_GUIDE_EN: dict[str, str] = {
    "gk": (
        "GK priority: shot-stopping (Save %, Goals Conceded per 90), "
        "aerial command (Duels Won %), distribution (Pass Completion %). "
        "Shot-stopping is paramount."
    ),
    "df": (
        "DF priority: defensive actions (Tackles per 90, Interceptions per 90, "
        "Blocks per 90, Duels Won %), then build-up (Pass Completion %, "
        "Key Passes per 90). Goals have very low weight."
    ),
    "mf": (
        "MF priority: creativity (Key Passes per 90, Assists per 90), "
        "ball retention (Pass Completion %), dribbling (Dribble Success %). "
        "Goals are a bonus unless AM/attacking role."
    ),
    "fw": (
        "FW priority: finishing quality (Goals per 90, G+A per 90, Shot Accuracy %), "
        "shot volume (Shots per 90), dribbling (Dribble Success %). "
        "Assists and key passes are secondary."
    ),
}

ROLE_GUIDE_FR: dict[str, str] = {
    "gk": (
        "Priorité GB : arrêts (Save %, buts encaissés/90), "
        "maîtrise de la surface (Duels Won %), relance (Pass Completion %). "
        "Les arrêts sont primordiaux."
    ),
    "df": (
        "Priorité DEF : actions défensives (Tackles/90, Interceptions/90, "
        "Blocks/90, Duels Won %), puis relance (Pass Completion %, Key Passes/90). "
        "Les buts ont très peu de poids."
    ),
    "mf": (
        "Priorité MIL : créativité (Key Passes/90, Assists/90), "
        "conservation (Pass Completion %), dribble (Dribble Success %). "
        "Les buts sont un plus sauf rôle très offensif."
    ),
    "fw": (
        "Priorité ATT : finition (Goals/90, G+A/90, Shot Accuracy %), "
        "volume de tirs (Shots/90), dribble (Dribble Success %). "
        "Assists et passes clés = secondaire."
    ),
}


def role_guide(role: str, language: str | None) -> str:
    """Return the role priority description for injection into LLM context."""
    G = ROLE_GUIDE_FR if _is_fr(language) else ROLE_GUIDE_EN
    return G.get(role, G["mf"])


ROLE_CODE_MAP: dict[str, str] = {
    "fw": "FW", "mf": "MF", "df": "DF", "gk": "GK",
}
