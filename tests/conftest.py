"""
Shared fixtures and test infrastructure.

The streamlit stub must be installed before any module that imports streamlit
is collected — conftest.py is guaranteed to run first.
"""
from __future__ import annotations
import sys
import types
import pytest
import pandas as pd

# ---------------------------------------------------------------------------
# Stub streamlit so modules decorated with @st.cache_data load without a
# running Streamlit server.  setdefault preserves the real module if it was
# already imported (e.g. by an IDE plugin).
# ---------------------------------------------------------------------------
if "streamlit" not in sys.modules:
    _st = types.ModuleType("streamlit")
    _st.cache_data = lambda **kw: (lambda f: f)
    sys.modules["streamlit"] = _st


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_scout_df() -> pd.DataFrame:
    """Scout DataFrame with 50th-percentile values for all common metrics."""
    metrics = [
        "Goals per 90", "Assists per 90", "G+A per 90",
        "Shots per 90", "Shot Accuracy %", "Key Passes per 90",
        "Pass Completion %", "Dribble Success %", "Dribbles per 90",
        "Tackles per 90", "Interceptions per 90", "Duels Won %",
    ]
    return pd.DataFrame({"Percentile": [50.0] * len(metrics)}, index=metrics)


@pytest.fixture
def fw_scout_df() -> pd.DataFrame:
    """Scout DataFrame with elite forward metrics (high attacking, low defensive)."""
    metrics = {
        "Goals per 90": 90.0, "Assists per 90": 80.0, "G+A per 90": 88.0,
        "Shots per 90": 75.0, "Shot Accuracy %": 85.0, "Key Passes per 90": 60.0,
        "Pass Completion %": 70.0, "Dribble Success %": 80.0,
        "Dribbles per 90": 70.0, "Tackles per 90": 30.0,
        "Interceptions per 90": 25.0, "Duels Won %": 55.0,
    }
    return pd.DataFrame({"Percentile": list(metrics.values())}, index=list(metrics.keys()))


@pytest.fixture
def gk_scout_df() -> pd.DataFrame:
    """Scout DataFrame with goalkeeper metrics."""
    metrics = {
        "Save %": 90.0,
        "Goals Conceded per 90": 20.0,
        "Pass Completion %": 85.0,
        "Duels Won %": 70.0,
    }
    return pd.DataFrame({"Percentile": list(metrics.values())}, index=list(metrics.keys()))


@pytest.fixture
def raw_outfield_entry() -> dict:
    """Minimal API-football statistics entry for an outfield player (2 700 min)."""
    return {
        "games": {"minutes": 2700, "position": "Midfielder"},
        "goals": {"total": 10, "assists": 8, "saves": None, "conceded": None},
        "shots": {"total": 50, "on": 25},
        "passes": {"key": 60, "accuracy": 85.0},
        "tackles": {"total": 45, "blocks": 10, "interceptions": 30},
        "duels": {"total": 120, "won": 70},
        "dribbles": {"attempts": 40, "success": 30},
        "fouls": {"committed": 20, "drawn": 25},
    }


@pytest.fixture
def raw_gk_entry() -> dict:
    """Minimal API-football statistics entry for a goalkeeper."""
    return {
        "games": {"minutes": 3060, "position": "Goalkeeper"},
        "goals": {"total": None, "assists": None, "saves": 90, "conceded": 24},
        "shots": {}, "passes": {"key": None, "accuracy": 72.0},
        "tackles": {}, "duels": {}, "dribbles": {}, "fouls": {},
    }
