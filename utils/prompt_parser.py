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
    prompt = prompt or ""
    prompt_lower = prompt.lower()

    # --- 1️⃣ Tool Detection ---
    compare_keywords = [
        "compare", "versus", "vs", "against", "between",  # English
        "comparer", "contre", "vs.", "versus", "face à", "entre"  # French
    ]
    analyze_keywords = [
        "analyze", "analyse", "evaluate", "scout", "assess", "study", "look at",  # English
        "analyser", "analyse", "évaluer", "evaluer", "étudier", "étudie", "regarder", "analyse de"  # French
    ]
    
    # Detect tool
    tool = None
    if any(kw in prompt_lower for kw in compare_keywords):
        tool = "compare"
    elif any(kw in prompt_lower for kw in analyze_keywords):
        tool = "analyze"

    # ---- Player extraction ----
    # Unicode-aware: allow diacritics, apostrophes and hyphens inside names, 1 or 2 tokens
    # Examples: Mbappé, Lamine Yamal, Jean-Pierre, O'Neill
    name_regex = r"\b[A-ZÀ-ÖØ-Ý][a-zà-öø-ÿ'’-]+(?:\s[A-ZÀ-ÖØ-Ý][a-zà-öø-ÿ'’-]+)?\b"
    player_candidates = re.findall(name_regex, prompt)

    # Exclude/trim command words stuck to names (e.g., "Compare Mbappé" -> "Mbappé")
    blacklist = {
        "compare", "versus", "vs", "graph", "analyze", "performance",
        "comparer", "contre", "analyse", "évaluer", "evaluer", "étudier",
        "joueur", "between", "face", "à"
    }

    cleaned = []
    for cand in player_candidates:
        parts = cand.split()
        # If first token is a command word, drop it and keep the rest (if any)
        if parts and parts[0].lower() in blacklist:
            parts = parts[1:]
        # Skip if nothing remains or it’s just another command word
        if not parts or parts[0].lower() in blacklist:
            continue
        cleaned.append(" ".join(parts))

    # Deduplicate, preserve order
    seen = set()
    players = []
    for n in cleaned:
        if n not in seen:
            seen.add(n)
            players.append(n)

    return {
        "tool": tool,
        "players": players
    }