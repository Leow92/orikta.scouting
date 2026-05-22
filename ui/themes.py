# ui/themes.py
"""Visual themes for orikta.scouting injected via st.markdown CSS."""

def get_theme_css() -> str:
    return f"<style>{_WORLDCUP}</style>"

_CREATIVE_TITLE = """
[data-testid="stMarkdownContainer"] h1 {
    color: #DFFF00 !important;
    font-style: italic !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
}
"""

# ── Shared structural rules ──────────────────────────────────────────────────
_BASE = """
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
hr { border: none; border-top: 1px solid var(--orikta-border, #e5e7eb); margin: 24px 0; }
.small { color: var(--orikta-caption, #6b7280); font-size: 0.9rem; }
.kbd  { padding: 2px 6px; border-radius: 4px; border: 1px solid var(--orikta-border, #d1d5db); }
.stMarkdown table { width: 100%; border-collapse: collapse; }
.stMarkdown th, .stMarkdown td { padding: 6px; border: 1px solid var(--orikta-border, #e5e7eb); }
.stMarkdown th { background: var(--orikta-th-bg, #f8fafc); color: var(--orikta-th-text, #111); font-weight: 600; }
/* Placeholder visible on both light and dark input backgrounds */
[data-testid="stTextInput"] input::placeholder {
    color: var(--orikta-placeholder, rgba(100, 116, 139, 0.75)) !important;
    opacity: 1 !important;
}

/* Mobile Responsiveness */
@media (max-width: 768px) {
    .block-container { padding-top: 0.5rem; padding-bottom: 1rem; }
    .main > div:first-child { padding-left: 8px !important; padding-right: 8px !important; }
    
    /* Typography - better mobile readability */
    h1 { font-size: 1.5rem !important; letter-spacing: 0.5px !important; }
    h2 { font-size: 1.25rem !important; }
    h3 { font-size: 1.1rem !important; }
    p, span, li { font-size: 0.95rem !important; line-height: 1.5 !important; }
    
    /* Input and buttons - larger touch targets */
    [data-testid="stTextInput"] input { 
        font-size: 1rem !important; 
        padding: 14px 16px !important; 
        min-height: 48px !important; 
    }
    [data-testid="baseButton-primary"] { 
        padding: 14px 24px !important; 
        font-size: 1rem !important; 
        min-height: 48px !important; 
    }
    [data-testid="baseButton-secondary"] { 
        padding: 12px 20px !important; 
        font-size: 0.95rem !important; 
        min-height: 44px !important; 
    }
    
    /* Form layout - stack input and button vertically on mobile */
    [data-testid="stForm"] > div:first-child { 
        flex-direction: column !important; 
        gap: 12px !important; 
    }
    [data-testid="stForm"] [data-testid="stTextInput"] { 
        width: 100% !important; 
        margin-bottom: 12px !important; 
    }
    
    /* Tables - compact for mobile */
    .stMarkdown table { font-size: 0.85rem !important; }
    .stMarkdown th, .stMarkdown td { padding: 4px 6px !important; }
    
    /* Expanders and sidebar */
    [data-testid="stExpander"] { font-size: 0.95rem !important; }
    section[data-testid="stSidebar"] { padding: 12px !important; }
    
    /* Make main content area scrollable */
    [data-testid="stAppViewContainer"] > .main { 
        overflow-y: auto !important; 
        max-height: calc(100vh - 140px) !important; 
        -webkit-overflow-scrolling: touch !important; 
    }
    
    /* Larger touch targets for all interactive elements */
    [data-testid="stRadio"] > div, 
    [data-testid="stToggle"] > div,
    [data-testid="stCheckbox"] > div { 
        min-height: 44px !important; 
        padding: 8px 0 !important; 
    }
    [data-testid="stSelectbox"] > div > div, 
    [data-testid="stButton"] > button { 
        min-height: 44px !important; 
    }
    
    /* Download buttons */
    [data-testid="stDownloadButton"] > button { 
        min-height: 44px !important; 
        padding: 12px 20px !important; 
        font-size: 0.95rem !important; 
    }
    
    /* Adjust spacing */
    .stCaption { font-size: 0.85rem !important; }
    hr { margin: 16px 0 !important; }
}

/* Tablet responsiveness */
@media (max-width: 1024px) and (min-width: 769px) {
    .main > div:first-child { 
        padding-left: 16px !important; 
        padding-right: 16px !important; 
    }
    h1 { font-size: 1.8rem !important; }
    [data-testid="stTextInput"] input { font-size: 0.95rem !important; }
}

/* Small tablets / large phones */
@media (max-width: 768px) and (min-width: 480px) {
    .main > div:first-child { 
        padding-left: 12px !important; 
        padding-right: 12px !important; 
    }
    [data-testid="stForm"] > div:first-child { 
        flex-direction: row !important; 
        flex-wrap: wrap !important; 
    }
    [data-testid="stForm"] [data-testid="stTextInput"] { 
        flex: 1 1 calc(70% - 12px) !important; 
        min-width: 200px !important; 
    }
    [data-testid="stForm"] [data-testid="baseButton-primary"] { 
        flex: 0 0 calc(30% - 12px) !important; 
        min-width: 120px !important; 
    }
}
"""

# ── Classic Light ────────────────────────────────────────────────────────────
_LIGHT = _BASE + _CREATIVE_TITLE + """
:root {
    --orikta-border:   #1e3a6e;
    --orikta-th-bg:    #002060;
    --orikta-th-text:  #FFD700;
    --orikta-caption:  #93c5fd;
}
"""

# ── World Cup 2026 ───────────────────────────────────────────────────────────
_WORLDCUP = _BASE + _CREATIVE_TITLE + """
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
.stTextInput input::placeholder,
[data-testid="stTextInput"] input::placeholder {
    color: rgba(147, 197, 253, 0.55) !important;
    opacity: 1 !important;
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

