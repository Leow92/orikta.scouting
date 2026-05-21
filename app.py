import time
import streamlit as st
from agents.router import route_query
from ui.branding import footer_brand
from ui.themes import get_theme_css
import utils.pipeline_log as pipeline_log

# ---------------- UI Strings (EN/FR) ---------------- #
UI_STRINGS = {
    "en": {
        "title": "orikta.scouting",
        "caption": "Tactical scouting, analyse or compare any player(s).",
        "sidebar_language": "🌐 Language",
        "sidebar_fast_preview": "⚡ Fast preview (skip LLM)",
        "sidebar_fast_preview_help": "Skip LLM analysis for speed.",
        "sidebar_verbose": "Verbose logs",
        "sidebar_clear_history": "Clear history",
        "sidebar_history": "🕓 Prompt History",
        "sidebar_history_empty": "No history yet.",
        "input_label": "💬 Ask your question",
        "input_placeholder": "e.g. create the report for Cherki • compare Mbappe vs Yamal",
        "generate": "Generate",
        "spinner": "🔎 Building report…",
        "spinner_done": "✅ Report ready",
        "spinner_error": "❌ Generation failed",
        "pipeline_logs_title": "🪵 Pipeline logs",
        "meta_line": "⏱️ Generated in {s:.1f}s • Lang: {lang} • Styles: {styles}",
        "downloads_title": "⬇️ Download the Report",
    },
    "fr": {
        "title": "orikta.scouting",
        "caption": "Scouting tactique, analysez ou comparez n’importe quel joueur.",
        "sidebar_language": "🌐 Langue",
        "sidebar_fast_preview": "⚡ Aperçu rapide (sans LLM)",
        "sidebar_fast_preview_help": "Saute l’analyse LLM.",
        "sidebar_verbose": "Logs détaillés",
        "sidebar_clear_history": "Effacer l’historique",
        "sidebar_history": "🕓 Historique des requêtes",
        "sidebar_history_empty": "Aucun historique.",
        "input_label": "💬 Saisissez votre requête",
        "input_placeholder": "ex. génère le rapport de Cherki • compare Mbappe et Yamal",
        "generate": "Générer",
        "spinner": "🔎 Génération du rapport…",
        "spinner_done": "✅ Rapport prêt",
        "spinner_error": "❌ Échec de la génération",
        "pipeline_logs_title": "🪵 Logs pipeline",
        "meta_line": "⏱️ Généré en {s:.1f}s • Langue : {lang} • Styles : {styles}",
        "downloads_title": "⬇️ Téléchargez le Rapport",
    },
}

_LEVEL_ICON = {
    "info":    "ℹ️",
    "success": "✅",
    "warning": "⚠️",
    "error":   "❌",
}

def _lang_code():
    return "fr" if str(st.session_state.get("language", "English")).lower().startswith("fr") else "en"

def _t(key: str) -> str:
    lang = _lang_code()
    return UI_STRINGS.get(lang, UI_STRINGS["en"]).get(key, UI_STRINGS["en"].get(key, key))

# -------- Page config & theme --------
st.set_page_config(
    page_title="orikta.scouting — Single Player",
    page_icon="⚽",
    layout="wide",
)

# Mobile responsiveness - inject viewport meta tag
st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
""", unsafe_allow_html=True)
st.markdown(get_theme_css(), unsafe_allow_html=True)

# -------- Sidebar: Language & Options --------
with st.sidebar:
    st.markdown(f"### {_t('sidebar_language')}")
    current_lang = st.session_state.get("language", "English")
    idx = 1 if str(current_lang).lower().startswith("fr") else 0
    selection = st.radio(
        "Language",
        options=["English", "Français"],
        index=idx,
        horizontal=True,
        key="lang_selector",
        label_visibility="collapsed",
    )

    st.session_state.language = "Français" if selection.startswith("Fr") else "English"
    language = st.session_state.language

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
        _t("input_label"),
        key="input",
        placeholder=_t("input_placeholder"),
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button(_t("generate"), type="primary", use_container_width=True)

# -------- Run pipeline on submit --------
_generation_run = False

# Define _push_live_logs before it is used
def _push_live_logs(entries):
    lines = [
        f"[+{e.elapsed:>6.2f}s] {_LEVEL_ICON.get(e.level, 'ℹ️')}  {e.message}"
        for e in entries
    ]
    _log_slot.code("\n".join(lines), language=None)

if submitted and user_input and user_input != st.session_state.last_prompt:
    st.session_state.last_prompt = user_input
    st.session_state["_orikta_fast_preview"] = fast_preview
    st.session_state["_orikta_verbose"] = verbose
    _generation_run = True

    with st.status(_t("spinner"), expanded=True) as _status:
        _log_slot = st.empty()
        t0 = time.time()
        pipeline_log.reset()
        pipeline_log.set_ui_callback(_push_live_logs)
        try:
            response, detected_lang = route_query(user_input, skip_llm=fast_preview)
            elapsed = time.time() - t0
            pipeline_log.set_ui_callback(None)

            # Update session language to match what the router detected
            st.session_state.language = detected_lang

            st.session_state.history.insert(0, {
                "prompt": user_input,
                "response": response,
                "elapsed": elapsed,
                "language": detected_lang,
                "skip_llm": fast_preview,
                "logs": pipeline_log.get_logs(),
            })
            st.session_state.selected_history_index = 0
            _status.update(label=_t("spinner_done"), state="complete", expanded=verbose)
        except Exception as e:
            pipeline_log.set_ui_callback(None)
            st.session_state.history.insert(0, {
                "prompt": user_input,
                "response": f"⚠️ Error: {e}",
                "logs": pipeline_log.get_logs(),
            })
            st.session_state.selected_history_index = 0
            _status.update(label=_t("spinner_error"), state="error", expanded=True)

# -------- Sidebar: History --------
with st.sidebar:
    st.markdown(f"### {_t('sidebar_history')}")
    if st.session_state.history:
        prompt_options = [f"{i+1}. {h['prompt']}" for i, h in enumerate(st.session_state.history)]
        sel = st.selectbox(
            _t("sidebar_history"),
            options=prompt_options,
            index=st.session_state.selected_history_index,
            key="history_selector",
            label_visibility="collapsed",
        )
        st.session_state.selected_history_index = prompt_options.index(sel)
    else:
        st.info(_t("sidebar_history_empty"))

def display_pipeline_logs(logs, expanded: bool = False) -> None:
    if not logs:
        return
    with st.expander(_t("pipeline_logs_title"), expanded=expanded):
        lines = [
            f"[+{e.elapsed:>6.2f}s] {_LEVEL_ICON.get(e.level, 'ℹ️')}  {e.message}"
            for e in logs
        ]
        st.code("\n".join(lines), language=None)

def display_report(report_md: str):
    START = "<!--PLOTLY_START-->"
    END = "<!--PLOTLY_END-->"
    if START in report_md and END in report_md:
        pre, rest = report_md.split(START, 1)
        plotly_html, post = rest.split(END, 1)

        if pre.strip():
            st.markdown(pre, unsafe_allow_html=True)
        st.components.v1.html(plotly_html, height=520, scrolling=False)
        if post.strip():
            st.markdown(post, unsafe_allow_html=True)
        return

    # Fallback to the heuristic if markers are missing:
    s = report_md
    start = s.find('<div id="')
    if start != -1:
        first_close = s.find('</script>', start)
        second_close = s.find('</script>', first_close + 9) if first_close != -1 else -1
        end = (second_close + 9) if second_close != -1 else (first_close + 9 if first_close != -1 else -1)
        if end != -1:
            st.markdown(s[:start], unsafe_allow_html=True)
            st.components.v1.html(s[start:end], height=520, scrolling=False)
            st.markdown(s[end:], unsafe_allow_html=True)
            return

    st.markdown(report_md, unsafe_allow_html=True)

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

    if not _generation_run:
        display_pipeline_logs(chosen.get("logs", []), expanded=verbose)

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

    html_bytes = md_to_html(chosen["response"], title="orikta — Scouting Report").encode("utf-8")
    st.download_button(
        _t("HTML"),
        data=html_bytes,
        file_name=f"{fname_base}.html",
        mime="text/html",
        use_container_width=True,
    )

footer_brand()