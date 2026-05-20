# pages/team_builder.py
#
# Team Builder — select a formation, fill player slots, get team grades.

import streamlit as st

from ui.themes import THEMES, get_theme_css
from ui.branding import footer_brand
from ui.pitch import create_pitch_figure
from tools.team_builder import (
    FORMATIONS, ZONE_ORDER, SlotResult,
    fetch_slot, compute_team_scores,
)
from utils.api_football import current_season

st.set_page_config(
    page_title="orikta — Team Builder",
    page_icon="🏗️",
    layout="wide",
)

# ── Sidebar: same theme + language controls as app.py ───────────────
# Using identical session state keys ensures both pages share the setting.
st.session_state.setdefault("theme_selector", "⚽ World Cup 2026")
st.session_state.setdefault("language", "English")

with st.sidebar:
    st.markdown("### 🌐 Language")
    current_lang = st.session_state.get("language", "English")
    selection = st.radio(
        "Language",
        options=["English", "Français"],
        index=1 if str(current_lang).lower().startswith("fr") else 0,
        horizontal=True,
        key="lang_selector",
        label_visibility="collapsed",
    )
    st.session_state.language = "Français" if selection.startswith("Fr") else "English"

    st.markdown("### 🎨 Theme")
    st.selectbox(
        "Theme",
        options=list(THEMES.keys()),
        key="theme_selector",
        label_visibility="collapsed",
    )

st.markdown(
    get_theme_css(THEMES.get(st.session_state.theme_selector, "worldcup")),
    unsafe_allow_html=True,
)

# ── Language helper ──────────────────────────────────────────────────
_lang = "fr" if str(st.session_state.get("language", "English")).lower().startswith("fr") else "en"

def _t(en: str, fr: str) -> str:
    return fr if _lang == "fr" else en

# ── Session state init ───────────────────────────────────────────────
st.session_state.setdefault("tb_formation", "4-3-3")
st.session_state.setdefault("tb_team_name", "")
st.session_state.setdefault("tb_results", [])
st.session_state.setdefault("tb_built", False)

# ── Header ───────────────────────────────────────────────────────────
st.markdown("# 🏗️ " + _t("Team Builder", "Constructeur d'Équipe"))
st.caption(_t(
    "Choose a formation, assign players to each position, then build your team rating.",
    "Choisissez une formation, assignez des joueurs à chaque poste, puis évaluez votre équipe.",
))

# ── Top controls: team name + formation ─────────────────────────────
col_name, col_form = st.columns([2, 1])
with col_name:
    new_team_name = st.text_input(
        _t("Team name", "Nom de l'équipe"),
        value=st.session_state.tb_team_name,
        placeholder=_t("My dream team…", "Mon équipe de rêve…"),
    )
    st.session_state.tb_team_name = new_team_name

with col_form:
    formation_options = list(FORMATIONS.keys())
    prev_formation = st.session_state.tb_formation
    new_formation = st.selectbox(
        _t("Formation", "Formation"),
        options=formation_options,
        index=formation_options.index(st.session_state.tb_formation),
    )
    if new_formation != prev_formation:
        st.session_state.tb_formation = new_formation
        st.session_state.tb_built = False
        st.session_state.tb_results = []

st.divider()

# ── Player inputs grouped by zone ───────────────────────────────────
slots = FORMATIONS[st.session_state.tb_formation]

ZONE_LABELS = {
    "gk":       _t("🥅 Goalkeeper",   "🥅 Gardien"),
    "defense":  _t("🛡️ Defense",       "🛡️ Défense"),
    "midfield": _t("⚙️ Midfield",      "⚙️ Milieu"),
    "attack":   _t("⚡ Attack",         "⚡ Attaque"),
}

for zone in ZONE_ORDER:
    zone_slots = [s for s in slots if s.zone == zone]
    if not zone_slots:
        continue
    st.markdown(f"**{ZONE_LABELS[zone]}**")
    n = len(zone_slots)
    cols = st.columns(n if n <= 4 else 4)
    for i, slot in enumerate(zone_slots):
        with cols[i % len(cols)]:
            key = f"tb_slot_{slot.id}"
            st.session_state.setdefault(key, "")
            st.text_input(
                slot.label,
                key=key,
                placeholder=_t("Player name…", "Nom du joueur…"),
                label_visibility="visible",
            )

st.divider()

# ── Build button ─────────────────────────────────────────────────────
if st.button(
    _t("⚽  Build Team", "⚽  Construire l'Équipe"),
    type="primary",
    use_container_width=True,
):
    season = current_season()
    results: list[SlotResult] = []

    progress = st.progress(0.0, text=_t("Fetching players…", "Récupération des joueurs…"))
    for i, slot in enumerate(slots):
        query = st.session_state.get(f"tb_slot_{slot.id}", "").strip()
        progress.progress(
            i / len(slots),
            text=_t(f"[{i+1}/{len(slots)}] {slot.label}: {query or '(empty)'}",
                    f"[{i+1}/{len(slots)}] {slot.label}: {query or '(vide)'}"),
        )
        r = fetch_slot(slot, query, season=season)
        results.append(r)

    progress.progress(1.0, text=_t("Done ✅", "Terminé ✅"))
    progress.empty()

    st.session_state.tb_results = results
    st.session_state.tb_built = True

# ── Results ──────────────────────────────────────────────────────────
if st.session_state.tb_built and st.session_state.tb_results:
    results: list[SlotResult] = st.session_state.tb_results
    team_scores = compute_team_scores(results)
    overall = team_scores["overall"]
    zone_scores = team_scores["zones"]

    st.divider()

    col_score, col_pitch = st.columns([1, 2], gap="large")

    # ── Left: scores + player table ──────────────────────────────────
    with col_score:
        team_label = st.session_state.tb_team_name or _t("Team", "Équipe")
        st.markdown(f"### {team_label}")

        if overall is not None:
            grade_color = (
                "#00C853" if overall >= 75 else
                "#F39C12" if overall >= 60 else
                "#E74C3C"
            )
            st.markdown(
                f'<div style="font-size:3.2rem;font-weight:900;color:{grade_color};'
                f'line-height:1.1">{overall:.0f}'
                f'<span style="font-size:1.1rem;color:#aaa;font-weight:400"> /100</span></div>'
                f'<div style="color:#aaa;font-size:0.85rem;margin-bottom:16px">'
                f'{_t("Overall Team Rating", "Note Globale de l\'Équipe")}</div>',
                unsafe_allow_html=True,
            )

        zone_display = {
            "gk":       _t("🥅 GK",        "🥅 GB"),
            "defense":  _t("🛡️ Defense",   "🛡️ Défense"),
            "midfield": _t("⚙️ Midfield",  "⚙️ Milieu"),
            "attack":   _t("⚡ Attack",     "⚡ Attaque"),
        }

        for zone in ZONE_ORDER:
            z = zone_scores.get(zone)
            if z is None:
                continue
            bar_color = "#00C853" if z >= 75 else "#F39C12" if z >= 60 else "#E74C3C"
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;margin-bottom:2px">'
                f'<span><b>{zone_display[zone]}</b></span>'
                f'<span style="color:{bar_color};font-weight:700">{z:.0f}</span></div>'
                f'<div style="background:rgba(255,255,255,0.12);border-radius:4px;height:7px;margin-bottom:12px">'
                f'<div style="background:{bar_color};width:{int(z)}%;height:7px;border-radius:4px"></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("##### " + _t("Player Grades", "Notes des Joueurs"))

        table_rows = ["| Pos | Player | Grade |", "|---|---|---:|"]
        for r in results:
            name_display = r.found_name or (r.query if r.query else "—")
            if r.error and not r.found_name:
                grade_display = "⚠️"
            elif r.grade is not None:
                g = r.grade
                color = "#00C853" if g >= 75 else "#F39C12" if g >= 60 else "#E74C3C"
                grade_display = f'<span style="color:{color};font-weight:700">{g:.0f}</span>'
            else:
                grade_display = "—"
            table_rows.append(f"| **{r.slot.label}** | {name_display} | {grade_display} |")

        st.markdown("\n".join(table_rows), unsafe_allow_html=True)

        # Warnings for unfound players
        errors = [(r.slot.label, r.query, r.error) for r in results if r.error]
        if errors:
            st.markdown("---")
            st.markdown("**⚠️ " + _t("Issues", "Problèmes") + "**")
            for label, query, err in errors:
                st.caption(f"{label} ({query}): {err}")

    # ── Right: pitch diagram ─────────────────────────────────────────
    with col_pitch:
        fig = create_pitch_figure(
            slots=slots,
            results=results,
            formation_name=(
                f"{st.session_state.tb_formation}"
                + (f" — {st.session_state.tb_team_name}" if st.session_state.tb_team_name else "")
            ),
            height=620,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Grade legend
        st.markdown(
            '<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;'
            'font-size:0.78rem;margin-top:-8px">'
            '<span>🟩 ≥82</span>'
            '<span>🟢 70–81</span>'
            '<span>🟡 60–69</span>'
            '<span>🟠 48–59</span>'
            '<span>🔴 &lt;48</span>'
            '<span style="color:#607D8B">⬛ n/a</span>'
            '</div>',
            unsafe_allow_html=True,
        )

st.divider()
footer_brand()
