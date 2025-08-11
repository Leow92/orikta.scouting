# utils/fbref_scraper.py

import pandas as pd
from bs4 import BeautifulSoup, Comment
from playwright.sync_api import sync_playwright
import time

def fetch_rendered_html(url: str, wait_time: float = 3.5) -> str:
    """Uses Playwright to fetch fully rendered HTML."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
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

def scrape_player_profile(url: str) -> dict:
    """
    Return the player's name (from <h1>) and all <p> contents under:
      <div id="info" class="players open"> ... </div>
    Falls back to <div id="info" class="players"> if 'open' isn't present,
    and also searches inside HTML comments (FBref sometimes hides blocks).
    """
    html = fetch_rendered_html(url)
    soup = BeautifulSoup(html, "html.parser")

    # Try direct containers first
    container = soup.select_one("div#info.players.open") or soup.select_one("div#info.players")

    # If not found, search inside HTML comments
    if not container:
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            try:
                cs = BeautifulSoup(comment, "html.parser")
                container = cs.select_one("div#info.players.open") or cs.select_one("div#info.players")
                if container:
                    break
            except Exception:
                continue

    # If still nothing, return empty structure (but not None to keep UI stable)
    if not container:
        return {"name": None, "paragraphs": []}

    # Extract <h1> name
    h1 = container.find("h1")
    name = h1.get_text(" ", strip=True) if h1 else None

    # Extract all <p> texts
    paragraphs = [
        p.get_text(" ", strip=True)
        for p in container.find_all("p")
        if p.get_text(strip=True)
    ]

    return {"name": name, "paragraphs": paragraphs}