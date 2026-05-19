# ui/themes.py
"""Visual themes for orikta.scouting injected via st.markdown CSS."""

THEMES = {
    "☀️ Classic Light": "light",
    "🌙 Classic Dark":  "dark",
    "⚽ World Cup 2026": "worldcup",
}

def get_theme_css(theme_key: str) -> str:
    return f"<style>{_THEMES.get(theme_key, _LIGHT)}</style>"

# ── Shared structural rules ──────────────────────────────────────────────────
_BASE = """
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
hr { border: none; border-top: 1px solid var(--orikta-border, #e5e7eb); margin: 24px 0; }
.small { color: var(--orikta-caption, #6b7280); font-size: 0.9rem; }
.kbd  { padding: 2px 6px; border-radius: 4px; border: 1px solid var(--orikta-border, #d1d5db); }
.stMarkdown table { width: 100%; border-collapse: collapse; }
.stMarkdown th, .stMarkdown td { padding: 6px; border: 1px solid var(--orikta-border, #e5e7eb); }
.stMarkdown th { background: var(--orikta-th-bg, #f8fafc); color: var(--orikta-th-text, #111); font-weight: 600; }
"""

# ── Classic Light ────────────────────────────────────────────────────────────
_LIGHT = _BASE + """
:root {
    --orikta-border:   #e2e8f0;
    --orikta-th-bg:    #f1f5f9;
    --orikta-th-text:  #111827;
    --orikta-caption:  #6b7280;
}
"""

# ── Classic Dark ─────────────────────────────────────────────────────────────
_DARK = _BASE + """
:root {
    --orikta-border:   #374151;
    --orikta-th-bg:    #1f2937;
    --orikta-th-text:  #e5e7eb;
    --orikta-caption:  #9ca3af;
}
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
    background-color: #0e1117 !important;
}
section[data-testid="stSidebar"],
[data-testid="stSidebarContent"] {
    background-color: #111827 !important;
}
header[data-testid="stHeader"] {
    background-color: rgba(14,17,23,0.95) !important;
}
.stMarkdown p,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span:not(.stBadge) { color: #e5e7eb !important; }
h1, h2, h3, h4, h5, h6 { color: #f9fafb !important; }
label { color: #d1d5db !important; }
.stTextInput input {
    background-color: #1f2937 !important;
    color: #f9fafb !important;
    border-color: #374151 !important;
}
"""

# ── World Cup 2026 ───────────────────────────────────────────────────────────
_WORLDCUP = _BASE + """
:root {
    --orikta-border:   #1e3a6e;
    --orikta-th-bg:    #002060;
    --orikta-th-text:  #FFD700;
    --orikta-caption:  #93c5fd;
}
/* App shells */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
    background-color: #00205B !important;
}
section[data-testid="stSidebar"],
[data-testid="stSidebarContent"] {
    background-color: #001440 !important;
    border-right: 2px solid #E8112D !important;
}
header[data-testid="stHeader"] {
    background: linear-gradient(90deg, #00205B 60%, #001440) !important;
    border-bottom: 2px solid #FFD700 !important;
}
/* Typography */
h1 {
    color: #FFD700 !important;
    text-shadow: 0 2px 10px rgba(0,0,0,0.6);
    letter-spacing: 1px;
}
h2, h3 { color: #FFD700 !important; }
h4, h5, h6 { color: #e2e8f0 !important; }
.stMarkdown p,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span:not(.stBadge) { color: #e2e8f0 !important; }
label { color: #93c5fd !important; }
/* Generate / primary button */
[data-testid="baseButton-primary"],
.stFormSubmitButton > button {
    background: linear-gradient(135deg, #E8112D 0%, #FFD700 100%) !important;
    color: #00205B !important;
    font-weight: 800 !important;
    border: none !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    box-shadow: 0 4px 15px rgba(232,17,45,0.4) !important;
}
[data-testid="baseButton-primary"]:hover,
.stFormSubmitButton > button:hover {
    box-shadow: 0 6px 20px rgba(255,215,0,0.5) !important;
    transform: translateY(-1px) !important;
}
/* Secondary / download buttons */
[data-testid="baseButton-secondary"],
[data-testid="baseButton-secondaryFormSubmit"] {
    background-color: transparent !important;
    color: #FFD700 !important;
    border: 1px solid #FFD700 !important;
}
/* Text input */
.stTextInput input,
[data-testid="stTextInput"] input {
    background-color: #001a4d !important;
    color: #ffffff !important;
    border: 1px solid #1e3a6e !important;
}
.stTextInput input:focus,
[data-testid="stTextInput"] input:focus {
    border-color: #FFD700 !important;
    box-shadow: 0 0 0 2px rgba(255,215,0,0.25) !important;
}
/* Tables */
.stMarkdown th {
    background: #002060 !important;
    color: #FFD700 !important;
    border-color: #1e3a6e !important;
}
.stMarkdown td { border-color: #1e3a6e !important; color: #e2e8f0 !important; }
/* Expander */
[data-testid="stExpander"] {
    background-color: #001440 !important;
    border: 1px solid #1e3a6e !important;
}
[data-testid="stExpander"] summary { color: #93c5fd !important; }
/* Sidebar text */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span:not(.stBadge) { color: #e2e8f0 !important; }
/* Caption */
.stCaption,
[data-testid="stCaptionContainer"] { color: #93c5fd !important; }
"""

_THEMES = {"light": _LIGHT, "dark": _DARK, "worldcup": _WORLDCUP}
