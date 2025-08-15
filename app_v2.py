# app.py

import time
import streamlit as st
from agents.router_v2 import route_command
from utils.prompt_parser import parse_prompt
from ui.branding import footer_brand
from tools.grading_v3 import PLAY_STYLE_PRESETS, PLAY_STYLE_PRETTY

# -------- Page config & light CSS --------
st.set_page_config(page_title="onix.scouting — Single Player", page_icon="🪨", layout="wide")
st.markdown("""
<style>
:root{
  --onix-fg:#111;            /* text */
  --onix-bg:#ffffff;         /* page bg */
  --onix-th-bg:#f3f4f6;      /* header bg (light gray) */
  --onix-border:#e5e7eb;     /* borders */
}
@media (prefers-color-scheme: dark){
  :root{
    --onix-fg:#e5e7eb;
    --onix-bg:#0b0f19;
    --onix-th-bg:#1f2937;    /* darker header bg */
    --onix-border:#374151;
  }
}

/* page padding */
.block-container { padding-top: 1rem; padding-bottom: 2rem; }

/* force table styles inside Markdown blocks */
div[data-testid="stMarkdown"] table{
  width:100%;
  border-collapse:collapse;
  background:var(--onix-bg) !important;
  color:var(--onix-fg) !important;
}
div[data-testid="stMarkdown"] table th,
div[data-testid="stMarkdown"] table td{
  border:1px solid var(--onix-border) !important;
  padding:6px;
  color:var(--onix-fg) !important;
}
div[data-testid="stMarkdown"] table thead th{
  background:var(--onix-th-bg) !important;
  font-weight:600;
}

/* generic hr + small text */
hr{ border:none; border-top:1px solid var(--onix-border); margin:24px 0; }
.small{ color:#6b7280; font-size:0.9rem; }

/* keep code readable in both themes */
code, pre{
  color:var(--onix-fg) !important;
}
</style>
""", unsafe_allow_html=True)

# -------- Sidebar: Language & Options --------
with st.sidebar:
    st.markdown("### 🌐 Language")
    current_lang = st.session_state.get("language", "English")
    idx = 1 if str(current_lang).lower().startswith("fr") else 0
    selection = st.radio("Choose language", options=["English", "Français"], index=idx, horizontal=True, key="lang_selector")
    st.session_state.language = "Français" if selection.startswith("Fr") else "English"
    language = st.session_state.language

    st.markdown("---")
    st.markdown("### 🎛️ Team Play Styles")
    styles_all = list(PLAY_STYLE_PRESETS.keys())
    style_labels = {k: PLAY_STYLE_PRETTY.get(k, k) for k in styles_all}
    # Remember last selection
    prev_styles = st.session_state.get("_onix_styles", styles_all)
    styles = st.multiselect("Select styles", options=styles_all, default=prev_styles, format_func=lambda k: style_labels.get(k, k), key="styles")
    style_strength = st.slider("Style influence", 0.0, 1.0, st.session_state.get("_onix_style_strength", 0.6), 0.05)

    st.markdown("---")
    fast_preview = st.toggle("⚡ Fast preview (skip LLM)", value=st.session_state.get("_onix_fast_preview", False),
                             help="Show presentation + scouting + grades + style matrix; skip LLM analysis for speed.")
    verbose = st.toggle("Verbose logs", value=st.session_state.get("_onix_verbose", False))

    st.markdown("---")
    clear_hist = st.button("Clear history")

# Clear history if requested
if clear_hist:
    st.session_state.history = []
    st.session_state.selected_history_index = 0

# -------- Header --------
st.markdown("# 🪨 onix.scouting")
st.caption("Tactical scouting tool.")

# -------- Session State init --------
st.session_state.setdefault("history", [])
st.session_state.setdefault("last_prompt", "")
st.session_state.setdefault("selected_history_index", 0)

# -------- Input form (no accidental re-runs) --------
with st.form(key="query_form", clear_on_submit=False):
    user_input = st.text_input("💬 Ask your question (e.g., `Analyze: Cherki` or `Compare: Mbappé vs Salah`)",
                               value="", placeholder="e.g. Analyze Player: Rayan Cherki")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        examples = st.selectbox("Examples", ["", "Analyze: Cherki", "Analyze: Yamal", "Compare: Mbappé vs Salah"])
    with col_b:
        fill = st.form_submit_button("Use example")
        if fill and examples:
            user_input = examples

    submitted = st.form_submit_button("Generate", type="primary", use_container_width=True)

# -------- Run pipeline on submit --------
if submitted and user_input and user_input != st.session_state.last_prompt:
    st.session_state.last_prompt = user_input

    parsed = parse_prompt(user_input)  # -> {"tool": "analyze"/"compare", "players": [...]}
    tool = parsed.get("tool")
    players = parsed.get("players", [])

    if tool:
        with st.spinner("🔎 Building report…"):
            # Persist UI options for next run & for pipeline side-reading if needed
            st.session_state["_onix_styles"] = styles
            st.session_state["_onix_style_strength"] = style_strength
            st.session_state["_onix_fast_preview"] = fast_preview
            st.session_state["_onix_verbose"] = verbose

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

                # Save new result (top of history)
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
    st.markdown("### 🕓 Prompt History")
    if st.session_state.history:
        prompt_options = [f"{i+1}. {h['prompt']}" for i, h in enumerate(st.session_state.history)]
        sel = st.selectbox("Select a past result", options=prompt_options, index=st.session_state.selected_history_index, key="history_selector")
        st.session_state.selected_history_index = prompt_options.index(sel)
    else:
        st.info("No history yet.")

# -------- Main Display --------
if st.session_state.history:
    chosen = st.session_state.history[st.session_state.selected_history_index]
    st.markdown("### 🧠 Result")
    # Render as Markdown (the report is Markdown already)
    st.markdown(chosen["response"])
    st.caption(f"⏱️ Generated in {chosen.get('elapsed', 0):.1f}s • Lang: {chosen.get('language')} • Styles: {', '.join(chosen.get('styles', [])) or '—'}")

    # Downloads
    st.divider()
    st.markdown("#### ⬇️ Download")

    def _slugify(s: str) -> str:
        return "".join(c.lower() if c.isalnum() else "-" for c in s).strip("-")[:80] or "report"

    fname_base = _slugify(chosen["prompt"])

    st.download_button(
        "Download Markdown (.md)",
        data=chosen["response"],
        file_name=f"{fname_base}.md",
        mime="text/markdown",
        use_container_width=True,
    )

    def md_to_html(md_text: str, title: str = "Onix Report") -> str:
        try:
            import markdown  # pip install markdown
            body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
        except Exception:
            body = f"<pre>{md_text}</pre>"
        css = """
        body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;line-height:1.55;padding:24px;color:#111}
        table{border-collapse:collapse;width:100%}
        th,td{border:1px solid #ddd;padding:6px;text-align:left}
        h1,h2,h3{margin-top:1.2em}
        code,pre{font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono","Courier New", monospace}
        """
        return f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title><style>{css}</style></head><body>{body}</body></html>"

    html_bytes = md_to_html(chosen["response"], title="Onix — Scouting Report").encode("utf-8")
    st.download_button(
        "Download HTML (.html)",
        data=html_bytes,
        file_name=f"{fname_base}.html",
        mime="text/html",
        use_container_width=True,
    )

footer_brand()
