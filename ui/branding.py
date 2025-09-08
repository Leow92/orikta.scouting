# ui/branding.py
import streamlit as st

def footer_brand():
    st.markdown(
        "<div style='text-align:center;opacity:0.7;font-size:12px'>"
        "Built with <b>orikta</b> · Local-first scouting"
        "</div>",
        unsafe_allow_html=True
    )
