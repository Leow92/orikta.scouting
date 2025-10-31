from tools.compare import compare_players
from tools.analyze import analyze_player

def route_command(tool_call: dict, language: str = "English",
                  styles: list[str] | None = None,
                  style_strength: float = 0.6,
                  skip_llm: bool = False):
    command = tool_call["command"]
    args = tool_call["args"]

    if command == "compare":
        # ⬇️ Forward everything to comparison too
        return compare_players(
            args,
            language=language,
            #styles=styles,
            #style_influence=style_strength,  # name differs in compare.py
            skip_llm=skip_llm,
        )
    elif command == "analyze":
        return analyze_player(
            args,
            language=language,
            #styles=styles,
            #style_strength=style_strength,
            skip_llm=skip_llm,
        )
    else:
        return f"❌ Unknown command: {command}"
