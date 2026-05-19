import streamlit as st

def footer_brand():
    st.markdown(
        "<div style='text-align:center;padding:2rem 0 0.5rem;'>"
        "<span style='opacity:0.35;font-size:10px;font-weight:800;letter-spacing:0.2em;text-transform:uppercase;"
        "color:var(--orikta-caption,#6b7280);'>"
        "orikta &nbsp;&middot;&nbsp; local-first scouting"
        "</span> &nbsp; "
        "<span style='opacity:0.3;font-size:10px;'>"
        "| &nbsp;<a href='https://github.com/Leow92' target='_blank' "
        "style='color:inherit;text-decoration:none;letter-spacing:0.1em;'>GitHub</a>"
        " &nbsp;|&nbsp; "
        "<a href='https://www.linkedin.com/in/leonard-baesen-wagner/' target='_blank' "
        "style='color:inherit;text-decoration:none;letter-spacing:0.1em;'>LinkedIn</a>"
        "</span></div>",
        unsafe_allow_html=True
    )