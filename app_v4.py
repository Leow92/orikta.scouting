# app_v4.py

import streamlit as st
from agents.router import route_command
from utils.prompt_parser import parse_prompt

from ui.branding import set_page_config, sidebar_brand, footer_brand

# must be called before other Streamlit calls
set_page_config()

# language selector (you already have this; keep your logic)
language = st.session_state.get("language", "English")

# sidebar header
sidebar_brand(language=language)

# top header (optional)
st.markdown("# 🪨 Onix")
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
    selected_response = st.session_state.history[st.session_state.selected_history_index]["response"]
    st.markdown("### 🧠 Result")
    st.markdown(selected_response, unsafe_allow_html=True)

footer_brand()
