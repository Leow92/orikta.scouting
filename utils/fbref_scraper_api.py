# utils/fbref_scraper_api.py

import os
import requests
import pandas as pd
from bs4 import BeautifulSoup, Comment, NavigableString, Tag
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

SCRAPER_API_KEY = os.getenv("SCRAPERAPI_KEY")
if not SCRAPER_API_KEY:
    raise RuntimeError("❌ Missing SCRAPERAPI_KEY in secrets.toml or .env file")

# -------------------------------------------------------------
# 🧠 Fetch and cache the HTML
# -------------------------------------------------------------
@st.cache_data(ttl=6 * 60 * 60)
def fetch_rendered_html(url: str, render: bool = True, timeout: int = 60) -> str:
    """Use ScraperAPI to fetch a fully rendered FBref player page."""
    api_url = "https://api.scraperapi.com/"
    params = {
        "api_key": SCRAPER_API_KEY,
        "url": url,
        "render": "true" if render else "false",
        "wait_for": "div#meta",  # helps ScraperAPI stop earlier
    }

    resp = requests.get(api_url, params=params, timeout=timeout)
    resp.raise_for_status()

    html = resp.text
    if "Cloudflare" in html or "verify you are a human" in html:
        raise RuntimeError("⚠️ ScraperAPI returned a Cloudflare block page.")
    return html

# -------------------------------------------------------------
# 🔍 Extract tables and profile from provided HTML
# -------------------------------------------------------------
def scrape_all_tables_from_html(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    tables = {}
    for table in soup.find_all("table"):
        tid = table.get("id", "no_id")
        rows = table.select("tbody tr")
        data = [
            [cell.get_text(strip=True) for cell in row.select("th, td")]
            for row in rows
        ]
        if data:
            df = pd.DataFrame(data)
            tables[tid] = df
    return tables


def _extract_label_value_from_p(p: Tag) -> tuple[str, str] | None:
    strong = p.find("strong")
    if not strong:
        return None
    label = strong.get_text(" ", strip=True).rstrip(":").strip()
    parts = []
    for child in p.children:
        if isinstance(child, Tag) and child.name == "strong":
            continue
        if isinstance(child, NavigableString):
            parts.append(str(child))
        elif isinstance(child, Tag):
            parts.append(child.get_text(" ", strip=True))
    value = " ".join(s.strip() for s in parts if s and s.strip())
    if value.startswith(":"):
        value = value[1:].strip()
    if value.startswith("-"):
        value = value[1:].strip()
    return (label, value)


def _infer_position_hint(attributes):
    for attr in attributes:
        if attr["label"].lower() in ("position", "positions", "primary position"):
            txt = attr["value"].lower()
            if any(k in txt for k in ["gk", "goalkeeper", "keeper"]):
                return "gk"
            if any(k in txt for k in ["defender", "cb", "rb", "lb", "rwb", "lwb"]):
                return "df"
            if any(k in txt for k in ["midfielder", "dm", "cm", "am", "#6", "#8", "#10"]):
                return "mf"
            if any(k in txt for k in ["forward", "striker", "winger", "st", "cf", "lw", "rw"]):
                return "fw"
    return None


def scrape_player_profile_from_html(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    container = soup.select_one("div#meta") or soup.select_one("div#info.players")
    if not container:
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            try:
                cs = BeautifulSoup(comment, "html.parser")
                container = cs.select_one("div#meta") or cs.select_one("div#info.players")
                if container:
                    break
            except Exception:
                continue

    if not container:
        return {"name": None, "attributes": [], "paragraphs": [], "position_hint": None}

    h1 = container.find("h1")
    name = h1.get_text(" ", strip=True) if h1 else None

    attributes, paragraphs = [], []
    for p in container.find_all("p", recursive=True):
        label_value = _extract_label_value_from_p(p)
        if label_value:
            label, value = label_value
            if value:
                attributes.append({"label": label, "value": value})
        else:
            txt = p.get_text(" ", strip=True)
            if txt:
                paragraphs.append(txt)

    position_hint = _infer_position_hint(attributes)
    return {
        "name": name,
        "attributes": attributes,
        "paragraphs": paragraphs,
        "position_hint": position_hint,
    }
