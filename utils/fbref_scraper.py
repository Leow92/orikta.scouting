# utils/fbref_scraper.py

import pandas as pd
from bs4 import BeautifulSoup, Comment, NavigableString, Tag
from playwright.sync_api import sync_playwright
import time
from typing import List, Dict, Any
import subprocess, os, shutil

# Ensure Playwright browsers are available on Streamlit Cloud
try:
    cache_path = os.path.expanduser("~/.cache/ms-playwright")
    if not os.path.exists(cache_path) or not any(
        os.path.isdir(os.path.join(cache_path, d)) for d in os.listdir(cache_path)
    ):
        # install browsers only (no sudo, no --with-deps)
        subprocess.run(
            ["playwright", "install", "chromium", "firefox", "webkit"],
            check=True
        )
        print("✅ Playwright browsers installed successfully.")
    else:
        print("✅ Playwright browsers already present.")
except Exception as e:
    print(f"⚠️ Playwright browser install skipped or failed: {e}")


def fetch_rendered_html(url: str, wait_time: float = 3.5) -> str:
    """Uses Playwright to fetch fully rendered HTML."""
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        time.sleep(wait_time)  # Let JS finish loading
        html = page.content()
        browser.close()
    return html

def scrape_all_tables(url: str) -> dict:
    """Scrape all tables from a fully rendered FBref player page."""
    html = fetch_rendered_html(url)
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
    """
    Given a <p>, if it contains a <strong>LABEL</strong>, return (label, value).
    Value = text of the same <p> excluding the <strong> node.
    """
    strong = p.find("strong")
    if not strong:
        return None

    label = strong.get_text(" ", strip=True).rstrip(":").strip()

    # Build value from non-<strong> children
    parts: List[str] = []
    for child in p.children:
        if isinstance(child, Tag) and child.name == "strong":
            continue
        if isinstance(child, NavigableString):
            parts.append(str(child))
        elif isinstance(child, Tag):
            parts.append(child.get_text(" ", strip=True))
    value = " ".join(s.strip() for s in parts if s and s.strip())

    # clean common leading separators
    if value.startswith(":"):
        value = value[1:].strip()
    if value.startswith("-"):
        value = value[1:].strip()

    return (label, value)

def _infer_position_hint(attributes: List[Dict[str, str]]) -> str | None:
    """
    Try to infer a coarse role (fw/mf/df/gk) from attributes like 'Position'.
    """
    for attr in attributes:
        if attr["label"].lower() in ("position", "positions", "primary position"):
            txt = attr["value"].lower()
            if any(k in txt for k in ["gk", "goalkeeper", "keeper"]):
                return "gk"
            if any(k in txt for k in ["defender", "cb", "rb", "lb", "rwb", "lwb", "centre-back", "center-back"]):
                return "df"
            if any(k in txt for k in ["midfielder", "dm", "cm", "am", "#6", "#8", "#10", "wing-back"]):
                return "mf"
            if any(k in txt for k in ["forward", "striker", "winger", "st", "cf", "lw", "rw"]):
                return "fw"
    return None

def scrape_player_profile(url: str) -> dict:
    """
    Return structured player profile from FBref:
      {
        "name": str | None,
        "attributes": [{"label": str, "value": str}, ...],  # from <p><strong>Label</strong> Value</p>
        "paragraphs": [str, ...],                            # <p> without <strong>
        "position_hint": "fw"|"mf"|"df"|"gk"|None
      }
    Searches both visible DOM and HTML comments (FBref sometimes hides blocks).
    """
    html = fetch_rendered_html(url)
    soup = BeautifulSoup(html, "html.parser")

    container = soup.select_one("div#info.players.open") or soup.select_one("div#info.players")
    if not container:
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            try:
                cs = BeautifulSoup(comment, "html.parser")
                container = cs.select_one("div#info.players.open") or cs.select_one("div#info.players")
                if container:
                    break
            except Exception:
                continue

    if not container:
        return {"name": None, "attributes": [], "paragraphs": [], "position_hint": None}

    # name
    h1 = container.find("h1")
    name = h1.get_text(" ", strip=True) if h1 else None

    attributes: List[Dict[str, str]] = []
    paragraphs: List[str] = []

    # Gather direct <p> children first; if empty, fall back to all <p> descendants
    ps = container.find_all("p", recursive=True)
    for p in ps:
        label_value = _extract_label_value_from_p(p)
        if label_value:
            label, value = label_value
            if value:  # avoid empty lines
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
