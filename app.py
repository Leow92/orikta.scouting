import time
import streamlit as st
from agents.router import route_query
from ui.branding import footer_brand
from ui.themes import get_theme_css
import utils.pipeline_log as pipeline_log

_LEVEL_ICON = {
    "info":    "ℹ️",
    "success": "✅",
    "warning": "⚠️",
    "error":   "❌",
}

# -------- Page config & theme --------
st.set_page_config(
    page_title="orikta.scouting — Single Player",
    page_icon="⚽",
    layout="wide",
)

st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
""", unsafe_allow_html=True)
st.markdown(get_theme_css(), unsafe_allow_html=True)

# -------- Sidebar: Options --------
with st.sidebar:
    fast_preview = st.toggle(
        "⚡ Fast preview (skip LLM)",
        value=st.session_state.get("_orikta_fast_preview", False),
        help="Skip LLM analysis for speed.",
    )
    verbose = st.toggle("Verbose logs", value=st.session_state.get("_orikta_verbose", False))

    st.markdown("---")
    clear_hist = st.button("Clear history")

# Clear history if requested
if clear_hist:
    st.session_state.history = []
    st.session_state.selected_history_index = 0

# -------- Header --------
st.markdown("# orikta.scouting")
st.caption("Tactical scouting, analyse or compare any player(s).")

# -------- Session State init --------
st.session_state.setdefault("history", [])
st.session_state.setdefault("last_prompt", "")
st.session_state.setdefault("selected_history_index", 0)

# -------- Input form --------
with st.form(key="query_form", clear_on_submit=False):
    user_input = st.text_input(
        "💬 Ask your question",
        key="input",
        placeholder="e.g. Is Bellingham good for a CL side? • Compare Mbappe vs Yamal",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Generate", type="primary", use_container_width=True)

# -------- Typewriter placeholder animation --------
# st.markdown does not execute <script> tags (inserted via innerHTML).
# st.components.v1.html() runs in a real iframe; window.parent gives us the
# Streamlit page DOM where the actual text input lives.
st.components.v1.html("""
<script>
(function() {
    var par = window.parent;
    if (par._oriktaTyper) return;
    par._oriktaTyper = true;

    var Q = [
        "Is Mbappe better as a number 9 or as a left winger?",
        "Who fits a transition play style better — Bellingham or Vinícius Jr.?",
        "Is Bellingham better as a box-to-box or a deep-lying playmaker?",
        "Who would thrive more in a crossing-heavy system — Saka or Salah?",
        "Can Pedri adapt to a high-press system?",
        "Should we sign Haaland or Mbappé for a possession-based side?",
        "Is Yamal ready to lead the line at a Champions League club?",
        "Who fits a low-block counter-attack better — Osimhen or Lukaku?",
        "Is Tchouaméni better suited as a pivot or a box-to-box midfielder?",
        "Compare Pedri vs Bellingham for a 4-3-3 pressing system",
        "Which striker profiles best for a false 9 role?",
        "Is Vinicius Jr. better suited for counter-attack or possession play?",
    ];

    var qi = 0, ci = 0, del = false, inp = null;

    function find() {
        return par.document.querySelector('[data-testid="stTextInput"] input');
    }

    function tick() {
        if (!inp || !inp.isConnected) inp = find();
        if (!inp) { setTimeout(tick, 400); return; }

        if (par.document.activeElement === inp || inp.value) {
            setTimeout(tick, 400);
            return;
        }

        var q = Q[qi];
        if (!del) {
            if (ci < q.length) {
                inp.placeholder = q.slice(0, ++ci);
                setTimeout(tick, 52 + Math.random() * 36);
            } else {
                setTimeout(function() { del = true; tick(); }, 2200);
            }
        } else {
            if (ci > 0) {
                inp.placeholder = q.slice(0, --ci);
                setTimeout(tick, 26 + Math.random() * 18);
            } else {
                del = false;
                qi = (qi + 1) % Q.length;
                setTimeout(tick, 450);
            }
        }
    }

    setTimeout(tick, 900);
})();
</script>
""", height=0)

# -------- Run pipeline on submit --------
_generation_run = False

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

    with st.status("🔎 Building report…", expanded=True) as _status:
        _log_slot = st.empty()
        t0 = time.time()
        pipeline_log.reset()
        pipeline_log.set_ui_callback(_push_live_logs)
        try:
            response, _ = route_query(user_input, skip_llm=fast_preview)
            elapsed = time.time() - t0
            pipeline_log.set_ui_callback(None)

            st.session_state.history.insert(0, {
                "prompt": user_input,
                "response": response,
                "elapsed": elapsed,
                "skip_llm": fast_preview,
                "logs": pipeline_log.get_logs(),
            })
            st.session_state.selected_history_index = 0
            _status.update(label="✅ Report ready", state="complete", expanded=verbose)
        except Exception as e:
            pipeline_log.set_ui_callback(None)
            st.session_state.history.insert(0, {
                "prompt": user_input,
                "response": f"⚠️ Error: {e}",
                "logs": pipeline_log.get_logs(),
            })
            st.session_state.selected_history_index = 0
            _status.update(label="❌ Generation failed", state="error", expanded=True)

# -------- Sidebar: History --------
with st.sidebar:
    st.markdown("### 🕓 Prompt History")
    if st.session_state.history:
        prompt_options = [f"{i+1}. {h['prompt']}" for i, h in enumerate(st.session_state.history)]
        sel = st.selectbox(
            "Prompt History",
            options=prompt_options,
            index=st.session_state.selected_history_index,
            key="history_selector",
            label_visibility="collapsed",
        )
        st.session_state.selected_history_index = prompt_options.index(sel)
    else:
        st.info("No history yet.")

def display_pipeline_logs(logs, expanded: bool = False) -> None:
    if not logs:
        return
    with st.expander("🪵 Pipeline logs", expanded=expanded):
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
        parts = md_text.split('<html>', 1)
        body_start = markdown.markdown(parts[0], extensions=["tables", "fenced_code"])
        if len(parts) > 1:
            body = '<html>' + parts[1]
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

    mode = "Fast" if chosen.get("skip_llm") else "Full"
    st.caption(
        f"⏱️ Generated in {chosen.get('elapsed', 0.0):.1f}s • Mode: {mode}"
    )

    st.divider()
    st.markdown("#### ⬇️ Download the Report")

    def _slugify(s: str) -> str:
        return "".join(c.lower() if c.isalnum() else "-" for c in s).strip("-")[:80] or "report"

    fname_base = _slugify(chosen["prompt"])

    html_bytes = md_to_html(chosen["response"], title="orikta — Scouting Report").encode("utf-8")
    st.download_button(
        "HTML",
        data=html_bytes,
        file_name=f"{fname_base}.html",
        mime="text/html",
        use_container_width=True,
    )

footer_brand()
