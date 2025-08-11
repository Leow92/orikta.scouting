# utils/parser.py

import re
from typing import Dict, Optional

COMMON_STOPWORDS = {
    "compare", "analyze", "graph", "show", "me", "a", "of", "chart", "stats", "and", "vs", "versus", "generate"
}

def parse_input(user_input: str) -> dict:
    # Extract the tool command (e.g. @compare)
    command_match = re.search(r"@(\w+)", user_input)
    if not command_match:
        return {"command": None, "args": None}

    command = command_match.group(1).lower()

    # Get text before the command to extract arguments
    args_text = user_input[:command_match.start()].strip()

    # Remove common leading verbs or filler words
    args_text = re.sub(r"^(compare|analyze|show|graph|vs)\s+", "", args_text, flags=re.IGNORECASE)

    # Split on ' and ', ',' or multiple spaces
    args = re.split(r"\s+and\s+|,\s*|\s{2,}", args_text)
    args = [arg.strip() for arg in args if arg.strip()]

    return {"command": command, "args": args}
