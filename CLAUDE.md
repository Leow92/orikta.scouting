# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

orikta.scouting is a football (soccer) player scouting application built with Streamlit. It scrapes player statistics from FBref, computes deterministic performance grades, and optionally generates tactical scouting narratives via the Groq LLM API. The UI is fully bilingual (English/French).

## Setup & Running

```bash
pip install -r requirements.txt
cp .env.example .env   # add GROQ_API_KEY and API_FOOTBALL_KEY
streamlit run app.py
```

**Required env vars:** `GROQ_API_KEY`, `API_FOOTBALL_KEY`

## Architecture & Data Flow

**Entry point:** [app.py](app.py) — Streamlit UI, session state, language toggle, history. Calls `parse_prompt()` to detect intent, then dispatches via `route_command()` in [agents/router.py](agents/router.py).

**Two pipelines:**

1. **Analyze** (`tools/analyze.py`):  
   API-football name search across major leagues → fetch league player pool → `build_scout_df()` (per-90 + percentile computation) → `normalize_positions_from_profile()` → `compute_grade()` → spider chart → optional Groq narrative

2. **Compare** (`tools/compare.py`):  
   Runs `_fetch_player()` for each player (same as analyze), then `_align_metrics()` finds common metrics, `_head_to_head_score()` weights by role, `_cosine_similarity()` computes profile similarity → dual spider chart → optional Groq comparison

**Data acquisition (API-football):**  
- [utils/api_football.py](utils/api_football.py): REST client for `v3.football.api-sports.io`. Key functions: `search_player()` (scans top leagues by name), `get_league_players()` (paginated pool fetch), `best_stats_entry()` (picks major-league stats entry). All cached via `@st.cache_data`.  
- [utils/percentile_engine.py](utils/percentile_engine.py): `stats_entry_to_per90()` converts raw API stats to per-90 metrics; `compute_percentiles()` ranks against league pool; `build_scout_df()` returns the `(Metric / Per90 / Percentile)` DataFrame consumed by grading and graphs; `build_profile()` builds the player bio dict.

**Free plan constraint:** `search_player()` requires `league + search` — name-only search is not available. The function iterates over `SEARCH_LEAGUES` (top 10 leagues) until it finds a match. League pools are cached 24h. Max 100 API requests/day; a cold analysis costs ~11 requests (up to 5 league searches + 5 pool pages + 1 prev-season).

**Grading system** (`tools/grading.py`):  
- Role taxonomy: `fw` / `mf` / `df` / `gk` as base roles; sub-roles `st`, `w`, `dm`, `cm`, `am`, `wm`, `cb`, `fb`  
- `SUBROLE_BLEND = 0.60` interpolates base + sub-role weights  
- All metric names match `stats_entry_to_per90()` output (e.g. `"Goals per 90"`, `"Dribble Success %"`)  
- `PLAY_STYLE_PRESETS` apply weight deltas for styles like "possession_high", "high_press_transition", "low_block_counter", "crossing_wide"  
- Score = `Σ(weight × percentile) / Σ(weight)` across role-matched metrics  
- Multi-position: grades computed for all detected positions, ranked by score

**LLM integration** (`utils/llm_client.py`, `utils/llm_analysis_player.py`, `utils/llm_analysis_comparison.py`):  
- Groq API, model `openai/gpt-oss-20b`  
- Trend analysis feeds metric names only (no raw numbers) to LLM  
- Bilingual prompt templates with strict language constraints  
- Fast preview mode skips LLM entirely for instant results

**Bilingual utilities** (`utils/lang.py`):  
- `_t(en, fr, language)` — inline language switch  
- `_is_fr(language)` — boolean helper  
- Glossaries for football terminology translation

**Visualization** (`ui/graph.py`):  
- `create_spider_graph()` — single player Plotly radar chart  
- `create_spider_graph_duo()` — dual player overlay radar chart

## Key Conventions

- All pipeline functions return strings (Markdown or error text) rather than raising exceptions, for UI stability.
- Metric aliasing: `ALIASES` dict in `grading.py` normalizes FBref naming variations before matching against weights.
- `_to_numeric_safely()` handles percentages, commas, and missing values throughout scrapers and grading.
- Streamlit session state (`st.session_state`) holds conversation history, language setting, and cached results.
- All reports are Markdown with embedded Plotly HTML (returned as a string, rendered via `st.markdown(..., unsafe_allow_html=True)`).

## Adding New Roles, Styles, or Metrics

- **New sub-role:** Add entry to `ROLE_TAXONOMY`, `SUBROLE_WEIGHTS`, and `ROLE_ALIASES` in [tools/grading.py](tools/grading.py)
- **New play style:** Add a `PLAY_STYLE_PRESETS` entry with metric weight deltas
- **New metric alias:** Add to `ALIASES` dict so it maps to the canonical API-football metric name from `stats_entry_to_per90()`
