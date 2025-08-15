# agents/router.py

from tools.compare import compare_players
# Ensure this import matches your actual file name:
from tools.analyze_2 import analyze_player   # ← if your file is tools/analyze.py

def route_command(tool_call: dict, **kwargs):
    """
    Routes a command and accepts extra UI options via **kwargs to avoid signature drift.
    Supported kwargs:
      - language: str = "English"
      - styles: list[str] | None = None
      - style_strength: float = 0.6
      - skip_llm: bool = False
    """
    command = tool_call.get("command")
    args = tool_call.get("args", [])

    language = kwargs.get("language", "English")
    styles = kwargs.get("styles")
    style_strength = kwargs.get("style_strength", 0.6)
    skip_llm = kwargs.get("skip_llm", False)

    if command == "compare":
        return compare_players(args, language=language)

    if command == "analyze":
        return analyze_player(
            args,
            language=language,
            styles=styles,
            style_strength=style_strength,
            skip_llm=skip_llm,
        )

    return f"❌ Unknown command: {command}"
