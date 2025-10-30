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
