# app.py

import streamlit as st
from agents.router import route_command
from utils.prompt_parser import parse_prompt
from ui.branding import footer_brand

# --- Language selector (sidebar) ---
with st.sidebar:
    st.markdown("### 🌐 Language")
    # Keep existing value as default
    current_lang = st.session_state.get("language", "English")
    idx = 1 if str(current_lang).lower().startswith("fr") else 0
    selection = st.radio(
        "Choose language",
        options=["English", "Français"],
        index=idx,
        horizontal=True,
        key="lang_selector",
    )
    # Persist in session
    st.session_state.language = "Français" if selection.startswith("Fr") else "English"

# Always read from session_state to propagate to tools
language = st.session_state.language

# top header (optional)
st.markdown("# 🪨 onix.scouting")
st.caption("Tactical scouting, powered by your local stack.")

# Session State
if "history" not in st.session_state:
    st.session_state.history = []

if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = ""

if "selected_history_index" not in st.session_state:
    st.session_state.selected_history_index = 0  # default to latest

# Sidebar: History Select
with st.sidebar:
    st.markdown("### 🕓 Prompt History")

    if st.session_state.history:
        prompt_options = [entry["prompt"] for entry in st.session_state.history]
        selected = st.selectbox(
            "Select a past prompt to view the result:",
            options=prompt_options,
            index=st.session_state.selected_history_index,
            key="history_selector"
        )
        st.session_state.selected_history_index = prompt_options.index(selected)
    else:
        st.info("No history yet.")

# Chat Input (Send on Enter)
user_input = st.text_input(
    "💬 Ask your question (e.g. Compare Player 1 and Player2 OR Analyze Player3)",
    key="input",
    placeholder="e.g. Mbappé vs Salah",
)

# New prompt triggers tool
if user_input and user_input != st.session_state.last_prompt:
    st.session_state.last_prompt = user_input

    parsed = parse_prompt(user_input)
    tool = parsed.get("tool")
    players = parsed.get("players", [])

    if tool:
        with st.spinner("🔎 Thinking..."):
            try:
                response = route_command({"command": tool, "args": players}, language=language)

                # Save new result to history (top)
                st.session_state.history.insert(0, {
                    "prompt": user_input,
                    "response": response
                })

                # Update selected index to newest
                st.session_state.selected_history_index = 0

            except Exception as e:
                st.session_state.history.insert(0, {
                    "prompt": user_input,
                    "response": f"⚠️ Error: {e}"
                })
                st.session_state.selected_history_index = 0
    else:
        st.warning("🤖 Could not detect whether you want to compare or analyze. Please be more specific.")

# Main Display: Show selected result from history
if st.session_state.history:
    selected_entry = st.session_state.history[st.session_state.selected_history_index]
    selected_prompt = selected_entry["prompt"]
    selected_response = selected_entry["response"]

    st.markdown("### 🧠 Result")
    st.markdown(selected_response, unsafe_allow_html=True)

    # --- Download buttons ---
    st.divider()
    st.markdown("#### ⬇️ Download")

    # Slugify a filename from the prompt
    def _slugify(s: str) -> str:
        return "".join(c.lower() if c.isalnum() else "-" for c in s).strip("-")[:80] or "report"

    fname_base = _slugify(selected_prompt)

    # 2a) Download as Markdown
    st.download_button(
        label="Download Markdown (.md)",
        data=selected_response,
        file_name=f"{fname_base}.md",
        mime="text/markdown",
        use_container_width=True,
    )

    # 2b) Download as HTML (best effort)
    def md_to_html(md_text: str, title: str = "Onix Report") -> str:
        # Try proper markdown → HTML conversion; fall back to <pre> if package missing
        try:
            import markdown  # pip install markdown
            body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
        except Exception:
            body = f"<pre>{md_text}</pre>"
        css = """
        body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;
             line-height:1.55; padding:24px; color:#111}
        table{border-collapse:collapse; width:100%}
        th,td{border:1px solid #ddd; padding:6px; text-align:left}
        h1,h2,h3{margin-top:1.2em}
        code,pre{font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono",
                 "Courier New", monospace}
        """
        return f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title><style>{css}</style></head><body>{body}</body></html>"

    html_bytes = md_to_html(selected_response, title="Onix — Scouting Report").encode("utf-8")

    st.download_button(
        label="Download HTML (.html)",
        data=html_bytes,
        file_name=f"{fname_base}.html",
        mime="text/html",
        use_container_width=True,
    )

footer_brand()
