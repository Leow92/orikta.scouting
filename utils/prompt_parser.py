# utils/prompt_parser.py

import re

def parse_prompt(prompt: str) -> dict:
    """
    Parses user prompt to extract intent and player names.
    Supports English and French for 'compare' and 'analyze'.

    Returns:
        {
            "tool": "compare" | "analyze" | None,
            "players": [list of detected player names]
        }
    """
    prompt_lower = prompt.lower()

    # Supported keywords (EN + FR)
    compare_keywords = [
        "compare", "versus", "vs", "comparer", "contre"
    ]
    analyze_keywords = [
        "analyze", "analyse", "evaluer", "évaluer", "performance", "étudier"
    ]

    # Detect tool
    tool = None
    if any(kw in prompt_lower for kw in compare_keywords):
        tool = "compare"
    elif any(kw in prompt_lower for kw in analyze_keywords):
        tool = "analyze"

    # Extract names: look for capitalized words (single or two-word names)
    player_candidates = re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)?\b", prompt)

    # Exclude common non-name words
    blacklist = {
        "Compare", "Vs", "Versus", "Graph", "Analyze", "Performance",
        "Comparer", "Contre", "Analyse", "Évaluer", "Evaluer", "Étudier", "Joueur"
    }

    players = [name for name in player_candidates if name not in blacklist]

    return {
        "tool": tool,
        "players": players
    }
