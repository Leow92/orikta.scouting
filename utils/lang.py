from typing import List

def _is_fr(language: str | None) -> bool:
    return (language or "").strip().lower().startswith("fr")

def _lang_block(language: str | None) -> str:
    if _is_fr(language):
        return ("Only provide the output standalone. Write your answer in **French**")
    return ("Only provide the output standalone.")

def _glossary_block_for(language: str, present_metrics: List[str]) -> str:
    # Minimal glossary like in utils/llm_analysis_player.py
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
        "Tackles": "Tackles made",
        "Interceptions": "Interceptions of opponent’s passes",
        "Blocks": "Blocks of shots, passes, or crosses",
        "Clearances": "Clearances from defensive area",
        "Aerials won %": "Share of aerial duels won",
    }
    SCOUT_METRIC_GLOSSARY_FR = {
        "Non-Penalty Goals": "Buts marqués hors penalties",
        "npxG": "Buts attendus hors penalty",
        "xAG": "Passes décisives attendues",
        "Shots Total": "Tirs tentés",
        "Shot-Creating Actions": "Actions menant à un tir",
        "Passes Attempted": "Passes tentées",
        "Pass Completion %": "Pourcentage de passes réussies",
        "Progressive Passes": "Passes faisant progresser vers le but",
        "Progressive Carries": "Conduites progressives",
        "Successful Take-Ons": "Dribbles réussis",
        "Touches (Att Pen)": "Touches dans la surface adverse",
        "Tackles": "Tacles réalisés",
        "Interceptions": "Interceptions",
        "Blocks": "Contres (tirs/passes/centres)",
        "Clearances": "Dégagements",
        "Aerials won %": "Pourcentage de duels aériens gagnés",
    }
    G = SCOUT_METRIC_GLOSSARY_FR if _is_fr(language) else SCOUT_METRIC_GLOSSARY_EN
    present = [m for m in G if m in present_metrics]
    if not present:
        return ""
    title = "### Scouting Metrics Glossary" if not _is_fr(language) else "### Glossaire des métriques de scouting"
    lines = "\n".join(f"- **{m}**: {G[m]}" for m in present)
    return f"\n{title}\n{lines}\n"
