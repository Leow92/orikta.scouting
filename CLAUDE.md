# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

orikta.scouting is a football (soccer) player scouting application built with Streamlit. It fetches player statistics from API-football, computes deterministic performance grades, and optionally generates tactical scouting narratives via the Groq LLM API. The UI is fully bilingual (English/French).

## Setup & Running

```bash
pip install -r requirements.txt
cp .env.example .env   # add GROQ_API_KEY and API_FOOTBALL_KEY
streamlit run app.py
```

**Required env vars:** `GROQ_API_KEY`, `API_FOOTBALL_KEY`  
**Python version:** 3.11+ (tested on 3.12)

## Manual Tests

```bash
python test_groq.py       # sanity-check Groq connection
python test_scraping.py   # test FBref scraping (legacy, not in main pipeline)
```

## Architecture & Data Flow

**Entry point:** [app.py](app.py) — Streamlit UI, session state, language toggle, history. Submits user text to `route_query()` in [agents/router.py](agents/router.py).

**Routing — LLM-based:** `route_query()` calls Groq (`llama-3.3-70b-versatile`, JSON mode) with the `router.j2` system prompt to classify intent (`analyze` / `compare` / `out_of_scope`), detect language (`en`/`fr`), and extract player names. It then dispatches to the appropriate tool. `utils/prompt_parser.py` is a legacy regex-based parser and is no longer used by the main flow.

**Two pipelines:**

1. **Analyze** (`tools/analyze.py`):  
   API-football name search across major leagues → fetch league player pool → `build_scout_df()` (per-90 + percentile computation) → `normalize_positions_from_profile()` → `compute_grade()` → spider chart → optional Groq narrative

2. **Compare** (`tools/compare.py`):  
   Runs `_fetch_player()` for each player (same as analyze), then `_align_metrics()` finds common metrics, `_head_to_head_score()` weights by role, `_cosine_similarity()` computes profile similarity → dual spider chart → optional Groq comparison

**Data acquisition (API-football):**  
- [utils/api_football.py](utils/api_football.py): REST client for `v3.football.api-sports.io`. Key functions: `search_player()` (scans `SEARCH_LEAGUES` — top 10 leagues — by name), `get_league_players()` (paginated pool fetch), `best_stats_entry()` (picks major-league stats entry). All cached via `@st.cache_data`.  
- [utils/percentile_engine.py](utils/percentile_engine.py): `stats_entry_to_per90()` converts raw API stats to per-90 metrics; `compute_percentiles()` ranks against league pool; `build_scout_df()` returns the `(Metric / Per90 / Percentile)` DataFrame consumed by grading and graphs; `build_profile()` builds the player bio dict.

**Free plan constraint:** `search_player()` requires `league + search` — name-only search is not available. Max 100 API requests/day; a cold analysis costs ~11 requests (up to 5 league searches + 5 pool pages + 1 prev-season). League pools are cached 24h.

**Prompt templates** (`prompts/`):  
- Jinja2 templates (`.j2` files) rendered via `prompts/render.py::render()`.  
- `render()` auto-injects `is_fr` (bool) from the `language` kwarg so templates can branch on language without boilerplate.  
- `router.j2` is the system prompt for the routing LLM call. `player_scouting.j2`, `player_summary.j2`, `comparison_*.j2` are the narrative prompts. `_lang_constraint.j2` is a shared language-enforcement partial.

**Grading system** (`tools/grading.py`):  
- Role taxonomy: `fw` / `mf` / `df` / `gk` as base roles; sub-roles `st`, `w`, `dm`, `cm`, `am`, `wm`, `cb`, `fb`  
- `SUBROLE_BLEND = 0.60` interpolates base + sub-role weights  
- All metric names match `stats_entry_to_per90()` output (e.g. `"Goals per 90"`, `"Dribble Success %"`)  
- `PLAY_STYLE_PRESETS` apply weight deltas for styles like `"possession_high"`, `"high_press_transition"`, `"low_block_counter"`, `"crossing_wide"`  
- Score = `Σ(weight × percentile) / Σ(weight)` across role-matched metrics  
- Multi-position: grades computed for all detected positions, ranked by score

**LLM integration:**  
- Router: Groq `llama-3.3-70b-versatile` (JSON mode, temperature 0) — intent/language/entity extraction only  
- Narratives: Groq `openai/gpt-oss-20b` (streaming, temperature 1) via `utils/llm_client.py`; workflows in `utils/llm_analysis_player.py` and `utils/llm_analysis_comparison.py`  
- Fast preview mode (`skip_llm=True`) skips the narrative LLM call entirely

**Bilingual utilities** (`utils/lang.py`):  
- `_t(en, fr, language)` — inline language switch  
- `_is_fr(language)` — boolean helper  
- Glossaries for football terminology translation

**Visualization** (`ui/graph.py`):  
- `create_spider_graph()` — single player Plotly radar chart  
- `create_spider_graph_duo()` — dual player overlay radar chart

## Key Conventions

- All pipeline functions return strings (Markdown or error text) rather than raising exceptions, for UI stability.
- Metric aliasing: `ALIASES` dict in `grading.py` normalizes naming variations before matching against weights.
- `_to_numeric_safely()` handles percentages, commas, and missing values throughout the data layer.
- Streamlit session state (`st.session_state`) holds conversation history, language setting, and cached results.
- All reports are Markdown with embedded Plotly HTML (returned as a string, rendered via `st.markdown(..., unsafe_allow_html=True)`).
- `utils/fbref_scraper.py` and `utils/fbref_scraper_api.py` are legacy files kept for reference; the active pipeline uses API-football exclusively.

## Adding New Roles, Styles, or Metrics

- **New sub-role:** Add entry to `ROLE_TAXONOMY`, `SUBROLE_WEIGHTS`, and `ROLE_ALIASES` in [tools/grading.py](tools/grading.py)
- **New play style:** Add a `PLAY_STYLE_PRESETS` entry with metric weight deltas
- **New metric alias:** Add to `ALIASES` dict so it maps to the canonical metric name from `stats_entry_to_per90()`
- **New prompt template:** Add a `.j2` file in `prompts/`, then call `render("your_template.j2", **kwargs)` — `is_fr` is injected automatically
