# prompts/render.py
#
# Jinja2 environment and single render() entry point.
# All prompt templates live alongside this file as *.j2 files.

from __future__ import annotations
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from utils.lang import _is_fr

_env = Environment(
    loader=FileSystemLoader(str(Path(__file__).parent)),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


def render(template_name: str, **kwargs) -> str:
    """Render a prompt template.

    Automatically injects `is_fr` (bool) derived from the `language` kwarg
    so every template can branch on language without extra boilerplate.
    """
    kwargs.setdefault("is_fr", _is_fr(kwargs.get("language", "English")))
    return _env.get_template(template_name).render(**kwargs).strip()
