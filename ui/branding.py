# ui/branding.py
import streamlit as st
from config import APP_NAME, TAGLINE_EN, TAGLINE_FR, VERSION

def set_page_config():
    st.set_page_config(
        page_title=f"{APP_NAME} — Scouting",
        page_icon="🪨",  # Onix rock emoji
        layout="wide"
    )

def sidebar_brand(language: str = "English"):
    st.sidebar.markdown(f"## 🪨 {APP_NAME}")
    st.sidebar.caption(TAGLINE_FR if language.lower().startswith("fr") else TAGLINE_EN)
    st.sidebar.divider()
    st.sidebar.caption(f"v{VERSION}")

def footer_brand():
    st.markdown(
        "<div style='text-align:center;opacity:0.7;font-size:12px'>"
        "Built with <b>Onix</b> · Local-first scouting"
        "</div>",
        unsafe_allow_html=True
    )
