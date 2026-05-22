# orikta.scouting

Ask a real scouting question. Get a data-backed answer.

orikta.scouting is a football scouting app built with Streamlit. 

You type a question in plain English or French — *"Is Mbappe better as a 9 or a left winger?"* — and the app fetches live player statistics from **API-football**, runs deterministic grading and percentile ranking against the full league pool, then feeds your exact question plus all that data to an LLM (**Mistral** or **Groq**). The scout report opens by directly answering what you asked, then backs it up with metric-level evidence.

---

## How it works

1. **You ask a question** in natural language — the router classifies intent and call the right tool (analyze / compare) and extracts player names.
2. **The pipeline fetches data** — per-90 stats, percentile grades against the league pool, season-over-season trends, role fit scores.
3. **Your question + all the data** is passed to the LLM — the report opens by answering your specific question, then delivers the full scouting synthesis.
4. **Every claim is data-backed** — inline metric citations (`Progressive Passes — 92p`), a ceiling verdict, and a recruitment action (Sign / Monitor / Reject).

---

## Features

- **Conversational interface**: ask tactical questions, get tailored answers grounded in real data.
- Single-player scouting or two-player head-to-head comparison.
- **Team Builder**: pick a formation, slot in players, get role-fit grades on a visual pitch diagram.
- Deterministic layer: per-90 stats, percentile grades, role fit score, head-to-head deltas, profile similarity, play-style fit.
- **Fast preview**: skip the LLM call for instant deterministic output.
- English UI; LLM narrative output supports English & Français.
- Pluggable LLM backend: Mistral (default) or Groq.
- Works **locally** or **online** (deployed Streamlit app).

---

## Requirements

- Python 3.11+ (tested on 3.12)
- API-football key (`API_FOOTBALL_KEY`)
- At least one LLM provider key: `MISTRAL_API_KEY` (default) or `GROQ_API_KEY`

---

## Quick Start

```bash
# 1) Clone & enter
git clone https://github.com/Leow92/orikta.scouting.git
cd orikta.scouting

# 2) Install dependencies
pip install -r requirements.txt

# 3) Create a .env file with your API keys
cp .env.example .env   # then fill in your keys

# 4) Run the app
streamlit run app.py
```

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `API_FOOTBALL_KEY` | Yes | API-football v3 key |
| `MISTRAL_API_KEY` | When using Mistral (default) | Mistral API key |
| `GROQ_API_KEY` | When using Groq | Groq API key |
| `LLM_PROVIDER` | No | `"mistral"` (default) or `"groq"` |
| `MISTRAL_NARRATIVE_MODEL` | No | Override narrative model (default: `mistral-small-2506`) |
| `MISTRAL_ROUTER_MODEL` | No | Override router model (default: `ministral-14b-2512`) |
| `GROQ_NARRATIVE_MODEL` | No | Override Groq narrative model |
| `GROQ_ROUTER_MODEL` | No | Override Groq router model |

---

## Usage

### Sidebar options

- **Team play styles** + style influence slider
- **Fast preview (skip LLM)** for deterministic-only mode
- **Theme** selector (default: World Cup 2026)

### Ask anything about a player or a duel

The app routes your question automatically — no commands, no forms.

**Single player**
```
Is Mbappe better as a number 9 or as a left winger?
Is Bellingham better as a box-to-box or a deep-lying playmaker?
Can Pedri adapt to a high-press system?
Is Yamal ready to lead the line at a Champions League club?
```

**Head-to-head comparison**
```
Who fits a transition play style better — Bellingham or Vinícius Jr.?
Who would thrive more in a crossing-heavy system — Saka or Salah?
Should we sign Haaland or Mbappé for a possession-based side?
Compare Pedri vs Bellingham for a 4-3-3 pressing system
```

### What you get

- Per-90 stats table and percentile grades vs. the full league pool
- Spider / radar chart (single or dual overlay)
- LLM scouting report — opens by answering your question directly, backs every claim with `Metric — XXp` citations, closes with a ceiling verdict and recruitment action
- Download as **Markdown** or **HTML**

---

## Team Builder

Navigate to the **Team Builder** page (sidebar) to:

1. Pick a formation (4-3-3, 4-4-2, more soon...)
2. Enter a player name for each slot — the app fetches their stats and computes a role-fit grade
3. View the squad on an interactive pitch diagram with per-zone and overall team scores

---

## Deterministic vs LLM

| Layer | What it does |
|---|---|
| **Deterministic** | Per-90 stats, percentile grades, role fit score, head-to-head deltas, spider chart — no LLM, fully reproducible |
| **LLM narrative** | Answers your question, then delivers a metric-backed tactical report with ceiling verdict and recruitment action |
| **Fast preview** | Skips the LLM call entirely — instant deterministic results |

---

## Project Layout

```
app.py                              # Streamlit UI, session state, history
pages/team_builder.py               # Team Builder multi-page app
agents/router.py                    # LLM-based intent router (analyze / compare / out_of_scope)
tools/analyze.py                    # Single-player pipeline
tools/compare.py                    # Two-player pipeline
tools/grading.py                    # Role weights, grading, play-style presets
tools/team_builder.py               # Formation definitions, slot fetch & grade, team scoring
utils/api_football.py               # REST client for API-football v3
utils/percentile_engine.py          # Per-90 conversion, percentile computation, scout DataFrame
utils/llm_client.py                 # Mistral / Groq LLM client (provider-agnostic API)
utils/llm_analysis_player.py        # Single-player LLM workflow
utils/llm_analysis_comparison.py    # Comparison LLM workflow
utils/lang.py                       # Bilingual helpers (_t, _is_fr)
utils/pipeline_log.py               # Prompt/response size logging
prompts/router.j2                   # Router system prompt (Jinja2)
prompts/player_summary_v0.3.j2      # Single-player narrative template (active)
prompts/comparison_deep.j2          # Comparison narrative template
prompts/lang.py                     # lang_constraint(), glossary_block(), role_guide()
prompts/render.py                   # Jinja2 renderer (auto-injects is_fr)
ui/graph.py                         # Spider / radar charts (Plotly)
ui/pitch.py                         # Interactive pitch diagram (Plotly)
ui/themes.py                        # CSS themes (World Cup 2026, light)
ui/branding.py                      # Footer / branding
```

---

## Deployment

The app runs both **locally** and on **Streamlit Cloud**.

- Store all env vars as **Secrets** in your Streamlit Cloud app settings.
- `packages.txt` provides any required Linux system libraries.

---

## Credits

Built with:

- [Streamlit](https://streamlit.io)
- [API-football](https://www.api-football.com) — player statistics
- [Mistral AI](https://mistral.ai) — default LLM provider
- [Groq](https://groq.com) — alternative LLM provider

Built by [Léonard Baesen-Wagner](https://www.linkedin.com/in/leonard-baesen-wagner/)
