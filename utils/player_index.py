import pandas as pd
from pathlib import Path

PLAYER_INDEX_CACHE = Path("data/player_index.csv")

def lookup_player_url(force_refresh: bool = False) -> pd.DataFrame:
    if not PLAYER_INDEX_CACHE.exists():
        raise FileNotFoundError("Missing player_index.csv in data/cache/")
    
    return pd.read_csv(PLAYER_INDEX_CACHE)
