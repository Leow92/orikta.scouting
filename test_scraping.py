from utils.fbref_scraper_api import scrape_all_tables

tables = scrape_all_tables("https://fbref.com/en/players/42fd9c7f/Kylian-Mbappe")
print([k for k in tables.keys() if "scout" in k])