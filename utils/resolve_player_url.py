# utils/playwright_search.py

from playwright.sync_api import sync_playwright
import time

def search_fbref_url_with_playwright(player_name: str) -> str:
    query = f"site:fbref.com {player_name}"
    search_url = f"https://duckduckgo.com/?q={query.replace(' ', '+')}"

    print(f"🔍 Launching browser for query: {query}")
    print(search_url)

    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()
        page.goto(search_url, timeout=15000)
        print("📥 Page loaded.")

        try:
            # ✅ Wait for the correct link
            page.wait_for_selector('a[data-testid="result-title-a"]', timeout=8000)
            first_result = page.query_selector('a[data-testid="result-title-a"]')

            if first_result:
                href = first_result.get_attribute("href")
                print(f"✅ Found link: {href}")

                if href and "/players/" in href:
                    print(f"✅ Valid FBref player URL: {href}")
                    return href
                else:
                    print("⚠️ Link found but not a valid FBref player page.")
            else:
                print("❌ No result-title-a link found.")
        
        except Exception as e:
            print(f"❌ Exception while scraping: {e}")

        browser.close()
        return None
