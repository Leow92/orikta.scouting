# orikta.scouting

A local or cloud-deployed, privacy-first football scouting app built with Streamlit.  
It fetches player statistics from the **API-football** REST API, computes deterministic performance grades, and generates tactical scouting narratives via a pluggable LLM backend (**Mistral** or **Groq**) — in English or French.

---

## Features

- Analyze a single player or compare two players side-by-side.
- **Team Builder**: select a formation, assign players to slots, and get team-wide grades on a visual pitch diagram.
- Deterministic scores: role fit, head-to-head, profile similarity, style fit matrix.
- **Fast preview**: skip LLM for instant, deterministic output.
- Bilingual UI / LLM output: English & Français.
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
| `MISTRAL_NARRATIVE_MODEL` | No | Override narrative model (default: `mistral-medium-3-5`) |
| `MISTRAL_ROUTER_MODEL` | No | Override router model (default: `ministral-14b-2512`) |
| `GROQ_NARRATIVE_MODEL` | No | Override Groq narrative model |
| `GROQ_ROUTER_MODEL` | No | Override Groq router model |

---

## Usage

### Sidebar options

- **Language** (English / Français)
- **Team play styles** + style influence
- **Fast preview (skip LLM)** for deterministic-only mode
- **Theme** selector (default: World Cup 2026)

### Prompt examples

```
Analyze Cherki
Compare Mbappe Mohamedsalah
Mbappe vs Mohamedsalah
```

### Output

- Deterministic data tables with per-90 stats and percentile grades
- Spider / radar charts (single or dual overlay)
- Optional LLM-generated narrative scouting report
- Download as **Markdown** or **HTML**

---

## Team Builder

Navigate to the **Team Builder** page (sidebar) to:

1. Pick a formation (4-3-3, 4-4-2, 4-2-3-1, 3-5-2, …)
2. Enter a player name for each slot — the app fetches their stats and computes a role-fit grade
3. View the squad on an interactive pitch diagram with per-zone and overall team scores

---

## Deterministic vs LLM

| Mode | Description |
|---|---|
| **Deterministic** | Per-90 stats, percentile grades, role fit score, head-to-head, spider chart |
| **LLM narrative** | Tactical interpretation, scouting story, comparison verdict |
| **Fast preview** | Skips the LLM call for instant local results |

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
prompts/player_summary.j2           # Single-player narrative template
prompts/comparison_deep.j2          # Comparison narrative template
prompts/lang.py                     # lang_constraint(), glossary_block(), role_guide()
prompts/render.py                   # Jinja2 renderer (auto-injects is_fr)
ui/graph.py                         # Spider / radar charts (Plotly)
ui/pitch.py                         # Interactive pitch diagram (Plotly)
ui/themes.py                        # CSS themes (World Cup 2026, light)
ui/branding.py                      # Footer / branding
```

---

## Manual Tests

```bash
python test_groq.py            # sanity-check Groq connection
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
