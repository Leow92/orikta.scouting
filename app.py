# app.py

import time
import streamlit as st
from agents.router import route_command
from utils.prompt_parser import parse_prompt
from ui.branding import footer_brand
from tools.grading import PLAY_STYLE_PRESETS, PLAY_STYLE_PRETTY

# ---------------- UI Strings (EN/FR) ---------------- #
UI_STRINGS = {
    "en": {
        "title": "orikta.scouting",
        "caption": "Tactical scouting, powered by your local stack.",
        "sidebar_language": "🌐 Language",
        "sidebar_styles_title": "🎛️ Team Play Styles",
        "sidebar_select_styles": "Select styles",
        "sidebar_style_strength": "Style influence",
        "sidebar_fast_preview": "⚡ Fast preview (skip LLM)",
        "sidebar_fast_preview_help": "Show presentation + scouting + grades + style matrix; skip LLM analysis for speed.",
        "sidebar_verbose": "Verbose logs",
        "sidebar_clear_history": "Clear history",
        "sidebar_history": "🕓 Prompt History",
        "sidebar_history_empty": "No history yet.",
        "input_label": "💬 Ask your question",
        "input_placeholder": "e.g. Analyze: Cherki • Compare: Mbappe vs Lamineyamal",
        "generate": "Generate",
        "spinner": "🔎 Building report…",
        "result_title": "🧠 Result",
        "meta_line": "⏱️ Generated in {s:.1f}s • Lang: {lang} • Styles: {styles}",
        "downloads_title": "⬇️ Download the Report",
        
    },
    "fr": {
        "title": "orikta.scouting",
        "caption": "Scouting tactique, propulsé par votre stack locale.",
        "sidebar_language": "🌐 Langue",
        "sidebar_styles_title": "🎛️ Styles de jeu d’équipe",
        "sidebar_select_styles": "Sélectionner les styles",
        "sidebar_style_strength": "Influence du style",
        "sidebar_fast_preview": "⚡ Aperçu rapide (sans LLM)",
        "sidebar_fast_preview_help": "Affiche présentation + scouting + notes + matrice de style ; saute l’analyse LLM.",
        "sidebar_verbose": "Logs détaillés",
        "sidebar_clear_history": "Effacer l’historique",
        "sidebar_history": "🕓 Historique des requêtes",
        "sidebar_history_empty": "Aucun historique.",
        "input_label": "💬 Saisissez votre requête",
        "input_placeholder": "ex. Analyser : Cherki • Comparer : Mbappe vs Lamineyamal",
        "generate": "Générer",
        "spinner": "🔎 Génération du rapport…",
        "result_title": "🧠 Résultat",
        "meta_line": "⏱️ Généré en {s:.1f}s • Langue : {lang} • Styles : {styles}",
        "downloads_title": "⬇️ Téléchargez le Rapport",
    },
}

def _lang_code():
    return "fr" if str(st.session_state.get("language", "English")).lower().startswith("fr") else "en"

def _t(key: str) -> str:
    lang = _lang_code()
    return UI_STRINGS.get(lang, UI_STRINGS["en"]).get(key, UI_STRINGS["en"].get(key, key))

# -------- Page config & CSS (high-contrast table headers) --------
st.set_page_config(page_title="orikta.scouting — Single Player", page_icon="🪨", layout="wide")
st.markdown("""
<style>
/* Base */
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
hr { border: none; border-top: 1px solid #e5e7eb; margin: 24px 0; }
.small { color:#6b7280; font-size: 0.9rem; }
.kbd { padding: 2px 6px; border-radius: 4px; border:1px solid #d1d5db; }

/* Tables (thèmes clair & sombre) */
.stMarkdown table { width: 100%; border-collapse: collapse; }
.stMarkdown th, .stMarkdown td { padding: 6px; border: 1px solid var(--orikta-border, #e5e7eb); }

/* Clair par défaut */
.stMarkdown th { background: var(--orikta-th-bg, #f8fafc); color: var(--orikta-th-text, #111); font-weight: 600; }
/* On NE FORCE PAS la couleur des td : elles héritent du thème */

/* Sombre */
@media (prefers-color-scheme: dark) {
  .stMarkdown th, .stMarkdown td { border-color: #374151; }
  .stMarkdown th { background: #111827; color: #e5e7eb; }
  /* Pas de couleur forcée sur td : le thème fournit déjà un texte clair */
}
</style>
""", unsafe_allow_html=True)

# -------- Sidebar: Language & Options --------
with st.sidebar:
    st.markdown(f"### {_t('sidebar_language')}")
    current_lang = st.session_state.get("language", "English")
    idx = 1 if str(current_lang).lower().startswith("fr") else 0
    selection = st.radio(
        "Language",                             # <= non-empty
        options=["English", "Français"],
        index=idx,
        horizontal=True,
        key="lang_selector",
        label_visibility="collapsed",           # <= hide label in UI
    )

    st.session_state.language = "Français" if selection.startswith("Fr") else "English"
    language = st.session_state.language

    st.markdown("---")
    st.markdown(f"### {_t('sidebar_styles_title')}")
    styles_all = list(PLAY_STYLE_PRESETS.keys())
    style_labels = {k: PLAY_STYLE_PRETTY.get(k, k) for k in styles_all}
    prev_styles = st.session_state.get("_orikta_styles", styles_all)
    styles = st.multiselect(
        "Team play styles",                     # <= non-empty
        options=styles_all,
        default=prev_styles,
        format_func=lambda k: style_labels.get(k, k),
        key="styles",
        label_visibility="collapsed",
    )
    style_strength = st.slider(
        "Style influence",                      # <= non-empty
        0.0, 1.0,
        st.session_state.get("_orikta_style_strength", 0.6),
        0.05,
        label_visibility="collapsed",
    )

    st.markdown("---")
    fast_preview = st.toggle(
        _t("sidebar_fast_preview"),
        value=st.session_state.get("_orikta_fast_preview", False),
        help=_t("sidebar_fast_preview_help")
    )
    verbose = st.toggle(_t("sidebar_verbose"), value=st.session_state.get("_orikta_verbose", False))

    st.markdown("---")
    clear_hist = st.button(_t("sidebar_clear_history"))

# Clear history if requested
if clear_hist:
    st.session_state.history = []
    st.session_state.selected_history_index = 0

# -------- Header --------
st.markdown(f"# {_t('title')}")
st.caption(_t("caption"))

# -------- Session State init --------
st.session_state.setdefault("history", [])
st.session_state.setdefault("last_prompt", "")
st.session_state.setdefault("selected_history_index", 0)

# -------- Input form (single chat bar + Generate) --------
with st.form(key="query_form", clear_on_submit=False):
    user_input = st.text_input(
        _t("input_label"),                 # localized, non-empty label (kept collapsed)
        key="input",
        placeholder=_t("input_placeholder"),  # <- localized placeholder
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button(_t("generate"), type="primary", use_container_width=True)

# -------- Run pipeline on submit --------
if submitted and user_input and user_input != st.session_state.last_prompt:
    st.session_state.last_prompt = user_input

    parsed = parse_prompt(user_input)  # -> {"tool": "analyze"/"compare", "players": [...]}
    tool = parsed.get("tool")
    players = parsed.get("players", [])

    if tool:
        with st.spinner(_t("spinner")):
            # Persist UI options
            st.session_state["_orikta_styles"] = styles
            st.session_state["_orikta_style_strength"] = style_strength
            st.session_state["_orikta_fast_preview"] = fast_preview
            st.session_state["_orikta_verbose"] = verbose

            t0 = time.time()
            try:
                response = route_command(
                    {"command": tool, "args": players},
                    language=language,
                    styles=styles,
                    style_strength=style_strength,
                    skip_llm=fast_preview
                )
                elapsed = time.time() - t0

                st.session_state.history.insert(0, {
                    "prompt": user_input,
                    "response": response,
                    "elapsed": elapsed,
                    "language": language,
                    "styles": styles,
                    "style_strength": style_strength,
                    "skip_llm": fast_preview,
                })
                st.session_state.selected_history_index = 0
            except Exception as e:
                st.session_state.history.insert(0, {"prompt": user_input, "response": f"⚠️ Error: {e}"})
                st.session_state.selected_history_index = 0
    else:
        st.warning("🤖 I couldn’t detect whether you want to **analyze** or **compare**. Try: `Analyze: Player Name`.")

# -------- Sidebar: History --------
with st.sidebar:
    st.markdown(f"### {_t('sidebar_history')}")
    if st.session_state.history:
        prompt_options = [f"{i+1}. {h['prompt']}" for i, h in enumerate(st.session_state.history)]
        sel = st.selectbox(
            _t("sidebar_history"),  # or "Select a past result"
            options=prompt_options,
            index=st.session_state.selected_history_index,
            key="history_selector",
            label_visibility="collapsed",
        )
        st.session_state.selected_history_index = prompt_options.index(sel)
    else:
        st.info(_t("sidebar_history_empty"))

def display_report(report_md: str):
    START = "<!--PLOTLY_START-->"
    END = "<!--PLOTLY_END-->"
    if START in report_md and END in report_md:
        pre, rest = report_md.split(START, 1)
        plotly_html, post = rest.split(END, 1)

        if pre.strip():
            st.markdown(pre)
        st.components.v1.html(plotly_html, height=520, scrolling=False)
        if post.strip():
            st.markdown(post)
        return

    # Fallback to the heuristic if markers are missing:
    s = report_md
    start = s.find('<div id="')
    if start != -1:
        first_close = s.find('</script>', start)
        second_close = s.find('</script>', first_close + 9) if first_close != -1 else -1
        end = (second_close + 9) if second_close != -1 else (first_close + 9 if first_close != -1 else -1)
        if end != -1:
            st.markdown(s[:start])
            st.components.v1.html(s[start:end], height=520, scrolling=False)
            st.markdown(s[end:])
            return

    st.markdown(report_md)

def md_to_html(md_text: str, title: str = "orikta Report") -> str:
        try:
            import markdown
            
            # Split the markdown text at the first instance of a HTML block
            parts = md_text.split('<html>', 1)
            
            # The first part is any markdown text before the HTML.
            body_start = markdown.markdown(parts[0], extensions=["tables", "fenced_code"])
            
            # The second part is the raw HTML of the graph plus the rest of the markdown.
            if len(parts) > 1:
                raw_html_content = '<html>' + parts[1]
                body = raw_html_content
            else:
                body = body_start

        except Exception:
            body = f"<pre>{md_text}</pre>"
            
        css = """
        body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;line-height:1.55;padding:24px;color:#111}
        table{border-collapse:collapse;width:100%}
        th,td{border:1px solid #ddd;padding:6px;text-align:left}
        th{background:#f8fafc;color:#111;font-weight:600}
        h1,h2,h3{margin-top:1.2em}
        code,pre{font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono","Courier New", monospace}
        """
        return f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title><style>{css}</style></head><body>{body}</body></html>"

# -------- Main Display --------
if st.session_state.history:
    chosen = st.session_state.history[st.session_state.selected_history_index]
    st.markdown(f"### {_t('result_title')}")
    
    # Use the new helper function to display the report content.
    display_report(chosen["response"])
    
    styles_chosen = ", ".join(chosen.get("styles", [])) or "—"
    mode = "Fast" if chosen.get("skip_llm") else "Full"
    st.caption(_t("meta_line").format(
        s=chosen.get("elapsed", 0.0),
        lang=chosen.get("language"),
        styles=styles_chosen
    ) + f" • Mode: {mode}")

    # Downloads
    st.divider()
    st.markdown(f"#### {_t('downloads_title')}")

    def _slugify(s: str) -> str:
        return "".join(c.lower() if c.isalnum() else "-" for c in s).strip("-")[:80] or "report"

    fname_base = _slugify(chosen["prompt"])

    # The md_to_html function is now at the top.
    html_bytes = md_to_html(chosen["response"], title="orikta — Scouting Report").encode("utf-8")
    st.download_button(
        _t("HTML"),
        data=html_bytes,
        file_name=f"{fname_base}.html",
        mime="text/html",
        use_container_width=True,
    )

footer_brand()