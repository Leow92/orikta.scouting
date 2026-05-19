# utils/llm_client.py
#
# Centralized LLM client. Supports Groq and Mistral as providers.
# Provider is selected by the LLM_PROVIDER env var ("groq" or "mistral").
# Models are overridable via env vars — see constants below.

from __future__ import annotations
import os
from dotenv import load_dotenv
from requests.exceptions import ReadTimeout
from prompts.lang import lang_constraint as _lang_block

load_dotenv()

# ------------------------------------------------------------------ #
# Provider & model config (all overridable via .env)                  #
# ------------------------------------------------------------------ #
LLM_PROVIDER = "groq"  # "groq" | "mistral"

GROQ_NARRATIVE_MODEL = os.getenv("GROQ_NARRATIVE_MODEL", "openai/gpt-oss-120b")
GROQ_ROUTER_MODEL    = os.getenv("GROQ_ROUTER_MODEL",    "llama-3.3-70b-versatile")

MISTRAL_NARRATIVE_MODEL = os.getenv("MISTRAL_NARRATIVE_MODEL", "mistral-medium-3-5")
MISTRAL_ROUTER_MODEL    = os.getenv("MISTRAL_ROUTER_MODEL",    "ministral-14b-2512")

# Resolved router model (used for logging)
ACTIVE_ROUTER_MODEL = (
    MISTRAL_ROUTER_MODEL if LLM_PROVIDER == "mistral" else GROQ_ROUTER_MODEL
)

# ------------------------------------------------------------------ #
# Lazy client init                                                     #
# ------------------------------------------------------------------ #
_groq_instance    = None
_mistral_instance = None


def _groq_client():
    global _groq_instance
    if _groq_instance is None:
        from groq import Groq
        _groq_instance = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq_instance


def _mistral_client():
    global _mistral_instance
    if _mistral_instance is None:
        from mistralai.client import Mistral
        _mistral_instance = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    return _mistral_instance


# ------------------------------------------------------------------ #
# Provider implementations — narrative                                 #
# ------------------------------------------------------------------ #
def _chat_groq(user_content: str, language: str, model: str) -> str:
    try:
        stream = _groq_client().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _lang_block(language)},
                {"role": "user",   "content": user_content},
            ],
            temperature=1,
            max_tokens=1500,
            stream=True,
        )
        chunks = []
        event_count = 0
        for event in stream:
            event_count += 1
            if event.choices and event.choices[0].delta and event.choices[0].delta.content:
                chunks.append(event.choices[0].delta.content)

        result = "".join(chunks).strip()
        if not result:
            # Reasoning models emit tokens in delta.reasoning_content during thinking
            # and may produce no delta.content — retry without streaming.
            resp = _groq_client().chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _lang_block(language)},
                    {"role": "user",   "content": user_content},
                ],
                temperature=1,
                max_tokens=1500,
                stream=False,
            )
            result = (resp.choices[0].message.content or "").strip()

        if not result:
            return f"⚠️ LLM returned empty response ({event_count} stream events, model={model})"
        return result

    except ReadTimeout:
        return _chat_groq(user_content, language, model)
    except Exception as e:
        return f"⚠️ Groq request failed: {e}"


def _chat_mistral(user_content: str, language: str, model: str) -> str:
    try:
        resp = _mistral_client().chat.complete(
            model=model,
            messages=[
                {"role": "system", "content": _lang_block(language)},
                {"role": "user",   "content": user_content},
            ],
            temperature=1,
            max_tokens=1500,
        )
        result = (resp.choices[0].message.content or "").strip()
        if not result:
            return f"⚠️ Mistral returned empty response (model={model})"
        return result
    except Exception as e:
        return f"⚠️ Mistral request failed: {e}"


# ------------------------------------------------------------------ #
# Provider implementations — JSON routing                              #
# ------------------------------------------------------------------ #
def _route_groq(messages: list, max_tokens: int = 256) -> str:
    resp = _groq_client().chat.completions.create(
        model=GROQ_ROUTER_MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "{}").strip()


def _route_mistral(messages: list, max_tokens: int = 256) -> str:
    resp = _mistral_client().chat.complete(
        model=MISTRAL_ROUTER_MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "{}").strip()


# ------------------------------------------------------------------ #
# Public API                                                           #
# ------------------------------------------------------------------ #
def llm_chat(user_content: str, language: str, model: str | None = None) -> str:
    """Narrative LLM call. Dispatches to the active provider (LLM_PROVIDER)."""
    if LLM_PROVIDER == "mistral":
        return _chat_mistral(user_content, language, model or MISTRAL_NARRATIVE_MODEL)
    return _chat_groq(user_content, language, model or GROQ_NARRATIVE_MODEL)


def llm_route(messages: list, max_tokens: int = 256) -> str:
    """JSON-mode routing call. Returns the raw JSON string from the LLM."""
    if LLM_PROVIDER == "mistral":
        return _route_mistral(messages, max_tokens)
    return _route_groq(messages, max_tokens)


# Backward-compatibility alias (used by existing callers before centralization)
def _groq_chat(user_content: str, language: str, model: str | None = None) -> str:
    return llm_chat(user_content, language, model)
