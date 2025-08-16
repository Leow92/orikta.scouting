# onix.scouting — README

A local, privacy-first football scouting app built with Streamlit. It scrapes FBref player pages, computes deterministic grades and style fits, and (optionally) asks a local LLM (via Ollama) to produce concise scouting notes — in English or French.

## Features

* Analyze a single player or compare two players.
* Deterministic scores: role fit, head-to-head, profile similarity, style fit matrix.
* **Fast preview**: skip LLM for instant, deterministic output.
* Bilingual UI/LLM: English / Français.
* Local LLM via **Ollama** (`gemma3` by default).

## Requirements

* Python 3.11+ (tested on 3.12)
* Playwright (for FBref fetching)
* Ollama running locally (with a pulled model, e.g. `gemma3`)

## Quick Start

```bash
# 1) Clone & enter
git clone <your-repo-url>
cd onix.scouting

# 2) Python deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3) Playwright browsers
python -m playwright install

# 4) Ollama (install separately)
ollama pull gemma3
ollama serve  # keep running

# 5) Run the app
streamlit run app.py
```

> The app expects Ollama at `http://localhost:11434/api/chat`. If you change it, update:
>
> * `utils/llm_analysis_player.py`
> * `tools/compare.py` (or `utils/llm_analysis_comparison.py`, if used)

## Usage

* **Sidebar**

  * Language (English/Français)
  * Team play styles + style influence
  * **Fast preview (skip LLM)**
* **Prompt examples**

  * Single: `Analyze: Rayan Cherki`
  * Compare: `Compare: CJ Egan-Riley vs Joel Ordoñez`
* Download results as **Markdown** or **HTML**.

## Deterministic vs LLM

* **Deterministic**: profile, scouting table, multi-position grades, style fit matrix, head-to-head tables.
* **LLM (optional)**: narrative analysis, tactical fit, comparison verdicts.
  Toggle **Fast preview** to skip LLM for both analyze and compare.

## Project Layout

```
app.py                          # Streamlit UI
agents/router.py                # Routes "analyze" / "compare"
tools/analyze.py                # Single-player pipeline
tools/compare.py                # Two-player pipeline
tools/grading.py                # Role weights, grading, style presets
utils/resolve_player_url.py     # FBref player search (Playwright)
utils/fbref_scraper.py          # HTML → DataFrames
utils/llm_analysis_player.py    # Single-player LLM workflow
utils/llm_analysis_comparison.py# (optional) comparison LLM workflow
ui/branding.py                  # Footer/branding
```

## Troubleshooting

* **Streamlit label warning**: ensure labels are non-empty; hide with `label_visibility="collapsed"`.
* **Dark theme text invisible**: don’t force `td` text color; CSS already respects theme.
* **LLM outputs wrong language**: confirm `_lang_block(language)` is sent as a **system** message and the selected UI language is passed through.

## License

Add your preferred license (MIT/Apache-2.0/etc.).
