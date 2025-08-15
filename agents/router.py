# agents/router.py

from tools.compare import compare_players
from tools.analyze_2 import analyze_player

def route_command(tool_call: dict, language: str = "English"):
    command = tool_call["command"]
    args = tool_call["args"]

    if command == "compare":
        return compare_players(args, language=language)
    elif command == "analyze":
        return analyze_player(args, language=language)  # update later if analyze needs language
    else:
        return f"❌ Unknown command: {command}"
