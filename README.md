# orikta.scouting

A local or cloud-deployed, privacy-first football scouting app built with Streamlit.  
It scrapes FBref player pages, computes deterministic grades and style fits, and (optionally) uses the **Groq LLM Provider** to produce concise scouting notes — in English or French.

---

## 🚀 Features

* Analyze a single player or compare two players.
* Deterministic scores: role fit, head-to-head, profile similarity, style fit matrix.
* **Fast preview**: skip LLM for instant, deterministic output with the grading tool.
* Bilingual UI / LLM: English 🇬🇧 & Français 🇫🇷.
* Works **locally** or **online** (deployed Streamlit app).

---

## 🧩 Requirements

* Python 3.11+ (tested on 3.12)
* Playwright (for FBref fetching)
* Groq API key (`GROQ_API_KEY`) in a local `.env` file

---

## ⚙️ Quick Start

```bash
# 1) Clone & enter
git clone https://github.com/Leow92/orikta.scouting.git
cd orikta.scouting

# 2) Install dependencies
pip install -r requirements.txt

# 3) Install Playwright browsers
python -m playwright install

# 4) Create a .env file with your Groq API key
echo "GROQ_API_KEY=sk-your-key-here" > .env

# 5) Run the app
streamlit run app.py
````

> The app uses the **Groq API** ([https://api.groq.com/openai/v1/chat/completions](https://api.groq.com/openai/v1/chat/completions)).
> You can configure the model (default: `mixtral-8x7b-32768`) inside:
>
> * `utils/llm_client.py` — Groq API client
> * `utils/llm_analysis_player.py` — single-player LLM workflow
> * `utils/llm_analysis_comparison.py` — comparison LLM workflow

---

## 🧠 Usage

### Sidebar options

* 🌐 **Language** (English / Français)
* 🎛️ **Team play styles** + style influence
* ⚡ **Fast preview (skip LLM)** for deterministic mode

### Prompt examples

```text
Analyze Cherki
Compare Mbappe Mohamedsalah
Mbappe vs Mohamedsalah
```

### Output

* Deterministic data tables
* Optional Groq-based narrative analysis
* Download as **Markdown** or **HTML**

---

## 🧮 Deterministic vs LLM

| Mode              | Description                                                     |
| ----------------- | --------------------------------------------------------------- |
| **Deterministic** | Profile data, grades, style matrix, head-to-head stats          |
| **LLM (Groq)**    | Tactical interpretation, narrative scouting, comparison verdict |
| **Fast preview**  | Skips Groq call for instant local results                       |

---

## 🗂️ Project Layout

```
app.py                          # Streamlit UI
agents/router.py                # Routes "analyze" / "compare"
tools/analyze.py                # Single-player pipeline
tools/compare.py                # Two-player pipeline
tools/grading.py                # Role weights, grading, style presets
utils/resolve_player_url.py     # FBref player search (Playwright)
utils/prompt_parser.py          # Parse user prompt to detect intent
utils/lang.py                   # Handle language switch for LLM output
utils/fbref_scraper.py          # HTML → DataFrames
utils/llm_client.py             # Groq LLM API client
utils/llm_analysis_player.py    # Single-player LLM workflow
utils/llm_analysis_comparison.py# Comparison LLM workflow
ui/branding.py                  # Footer / branding
```

---

## ☁️ Deployment

The app runs both **locally** and on **Streamlit Cloud**.

**Environment variables**

* `GROQ_API_KEY` → stored in `.env` locally
* On Streamlit Cloud → add as a **Secret** in your app settings

**Playwright setup**

* Handled automatically on first run (no sudo needed)
* `packages.txt` provides all required Linux libraries

---

## 🏁 Credits

Built with ❤️ using:

* [Streamlit](https://streamlit.io)
* [Playwright](https://playwright.dev/python/)
* [Groq LLM API](https://groq.com)
* [FBref](https://fbref.com) data sources

