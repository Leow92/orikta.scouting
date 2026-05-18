def _is_fr(language: str | None) -> bool:
    return (language or "").strip().lower().startswith("fr")
