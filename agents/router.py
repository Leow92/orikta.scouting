# agents/router.py
#
# Agentic router: a single LLM call (Groq, JSON mode) understands the user's
# natural-language query, detects language, chooses the right tool, and
# extracts player names.  Out-of-scope queries get a polite refusal.

from __future__ import annotations
import json
import os
from dotenv import load_dotenv
from groq import Groq

from tools.analyze import analyze_player
from tools.compare import compare_players
from prompts.render import render
import utils.pipeline_log as pipeline_log

load_dotenv()
_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

_ROUTER_SYSTEM = render("router.j2")

# ------------------------------------------------------------------ #
# LLM routing call                                                     #
# ------------------------------------------------------------------ #
def _llm_route(query: str) -> dict:
    """Call Groq (JSON mode) to classify the query. Returns a routing dict."""
    pipeline_log.log("[router] Sending query to LLM router (llama-3.3-70b-versatile)…")
    try:
        resp = _client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _ROUTER_SYSTEM},
                {"role": "user",   "content": query},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=256,
        )
        raw = (resp.choices[0].message.content or "{}").strip()
        result = json.loads(raw)
        pipeline_log.log(
            f"[router] Intent: {result.get('tool', '?')} | "
            f"Players: {result.get('players', [])} | "
            f"Language: {result.get('language', '?')}",
            level="success",
        )
        return result
    except json.JSONDecodeError:
        pipeline_log.log("[router] Malformed JSON from LLM router", level="error")
        return {"tool": "out_of_scope", "players": [], "language": "en",
                "message": "⚠️ Router returned malformed JSON. Please rephrase your query."}
    except Exception as e:
        pipeline_log.log(f"[router] Router error: {e}", level="error")
        return {"tool": "out_of_scope", "players": [], "language": "en",
                "message": f"⚠️ Router error: {e}"}


# ------------------------------------------------------------------ #
# Fallback messages                                                    #
# ------------------------------------------------------------------ #
def _no_player_msg(language: str) -> str:
    if language == "Français":
        return (
            "⚠️ Je n'ai pas pu identifier de nom de joueur dans votre requête.\n\n"
            "Essayez : _Analyser Mbappé_ ou _Comparer Bellingham vs Pedri_."
        )
    return (
        "⚠️ I couldn't identify a player name in your query.\n\n"
        "Try: _Analyze Bellingham_ or _Compare Mbappe vs Yamal_."
    )


def _default_out_of_scope(language: str) -> str:
    if language == "Français":
        return (
            "Je suis un assistant de **scouting football**. Je peux :\n\n"
            "- **Analyser** un joueur : _Analyser Kylian Mbappé_\n"
            "- **Comparer** deux joueurs : _Comparer Bellingham et Pedri_"
        )
    return (
        "I'm a **football scouting assistant**. I can:\n\n"
        "- **Analyze** a single player: _Analyze Jude Bellingham_\n"
        "- **Compare** two players: _Compare Mbappe vs Yamal_"
    )


# ------------------------------------------------------------------ #
# Public entry point                                                   #
# ------------------------------------------------------------------ #
def route_query(user_query: str, skip_llm: bool = False) -> tuple[str, str]:
    """
    Agentic router.  Understands intent, detects language, calls the right tool.

    Returns:
        (response_markdown: str, language: str)
        language is "English" or "Français".
    """
    routing  = _llm_route(user_query)

    lang_code = routing.get("language", "en")
    language  = "Français" if lang_code == "fr" else "English"
    tool      = routing.get("tool", "out_of_scope")
    players   = routing.get("players") or []

    if tool == "analyze":
        if not players:
            pipeline_log.log("[router] No player name detected — cannot dispatch", level="warning")
            return _no_player_msg(language), language
        pipeline_log.log(f"[router] Dispatching to analyze → {players[:1]}")
        return analyze_player(players[:1], language=language, skip_llm=skip_llm), language

    if tool == "compare":
        if len(players) < 2:
            pipeline_log.log("[router] Fewer than 2 players detected — cannot compare", level="warning")
            return _no_player_msg(language), language
        pipeline_log.log(f"[router] Dispatching to compare → {players[:2]}")
        return compare_players(players[:2], language=language, skip_llm=skip_llm), language

    # out_of_scope
    pipeline_log.log("[router] Out of scope — returning guidance message", level="warning")
    msg = (routing.get("message") or "").strip() or _default_out_of_scope(language)
    return msg, language
