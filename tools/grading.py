# tools/grading.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import math
import pandas as pd

# ---- Configuration: position-aware weights (sum ≈ 1.0 per role) ----
# Keys must match (case-insensitive, stripped) index labels in the scout table.
# Add / adjust freely as you iterate.
DEFAULT_WEIGHTS: Dict[str, Dict[str, float]] = {
    "fw": {
        "Non-Penalty Goals": 0.18,
        "npxG": 0.12,
        "Shots Total": 0.08,
        "xG": 0.06,
        "Shot-Creating Actions": 0.12,
        "Touches (Att Pen)": 0.08,         # sometimes named: "Touches (Att Pen)"
        "Progressive Passes Received": 0.10,
        "Aerials won %": 0.06,
        "Assists": 0.08,
        "Non-Penalty xG + xAG": 0.12,
    },
    "mf": {
        "Progressive Passes": 0.14,
        "Passes Completed (Short/Med/Long)": 0.08,  # umbrella if you split later
        "Pass Completion %": 0.10,
        "Progressive Carries": 0.12,
        "Shot-Creating Actions": 0.12,
        "xAG": 0.08,
        "Assists": 0.08,
        "Tackles": 0.10,
        "Interceptions": 0.10,
        "Ball Recoveries": 0.08,
    },
    "df": {
        "Tackles": 0.16,
        "Interceptions": 0.14,
        "Clearances": 0.10,
        "Blocks": 0.08,
        "Aerials won %": 0.14,
        "Dribblers Tackled %": 0.08,      # "Tackles (Dribblers Tkl%)" varies; map in alias table below if needed
        "Ball Recoveries": 0.10,
        "Progressive Passes": 0.10,
        "Pass Completion %": 0.10,
    },
    "gk": {
        "Post-Shot xG +/- per 90": 0.22,  # PSxG+/- per 90
        "Save%": 0.18,
        "Crosses Stopped %": 0.12,
        "Clean Sheets %": 0.10,
        "Goals Against per 90": 0.06,     # negative metric → see NEGATIVE_KEYS
        "Launch%": 0.06,
        "Passes Attempted (Avg Len)": 0.06,
        "Pass Completion % (Launched)": 0.10,
        "Avg Distance of Def Actions": 0.10,
    },
}

# ---- Aliases (FBref labels vary slightly between seasons/tables) ----
ALIASES: Dict[str, List[str]] = {
    "Touches (Att Pen)": ["Touches (Att Pen)", "Touches (Att Pen Area)", "Touches (Att Penalty Area)"],
    "Dribblers Tackled %": ["Dribblers Tackled %", "Tkl% (Dribblers)"],
    "Save%": ["Save%", "Save %"],
    "Clean Sheets %": ["Clean Sheets %", "CS %"],
    "Aerials won %": ["Aerials won %", "Aerial Win %", "Aerial Wins %"],
    "Goals Against per 90": ["Goals Against per 90", "GA/90"],
    "Post-Shot xG +/- per 90": ["Post-Shot xG +/- per 90", "PSxG+/-/90", "PSxG +/- per 90"],
    "Passes Completed (Short/Med/Long)": [
        "Passes Completed (Short/Med/Long)",
        "Cmp (Short/Med/Long)",
        "Cmp (S/M/L)"
    ],
    "Ball Recoveries": ["Ball Recoveries", "Recoveries"],
}

# ---- Metrics that are "bad when high" (we invert their percentile) ----
NEGATIVE_KEYS: List[str] = [
    "Fouls", "Dispossessed", "Miscontrols", "Errors", "Times Tackled", "Goals Against per 90"
]

@dataclass
class GradeBreakdown:
    role: str
    matched: List[Tuple[str, float, float]]  # (metric_name, weight, contribution_0_100)
    missing: List[str]
    raw_score: float  # 0..100 before clipping
    final_score: float  # clipped 0..100

def _normalize_role(role_hint: str | None) -> str:
    r = (role_hint or "").lower()
    for key in ("fw", "mf", "df", "gk", "forward", "striker", "midfielder", "defender", "goalkeeper"):
        if key in r:
            if key.startswith("f"): return "fw"
            if key.startswith("m"): return "mf"
            if key.startswith("d"): return "df"
            if key.startswith("g"): return "gk"
    return "mf"  # neutral default

def _match_metric_name(index_names: List[str], desired: str) -> str | None:
    """
    Try exact & alias matches against the row index labels coming from FBref.
    Returns the actual index label when found.
    """
    # Normalize index names once
    norm_index = {name.lower().strip(): name for name in index_names}
    # exact
    if desired.lower().strip() in norm_index:
        return norm_index[desired.lower().strip()]
    # aliases
    for alias in ALIASES.get(desired, []):
        if alias.lower().strip() in norm_index:
            return norm_index[alias.lower().strip()]
    # fuzzy: startswith / contains (last resort)
    for k, orig in norm_index.items():
        if desired.lower().strip() in k:
            return orig
    return None

def _invert_if_negative(metric: str, pct: float) -> float:
    for neg in NEGATIVE_KEYS:
        if neg.lower() in metric.lower():
            return 100.0 - pct
    return pct

def compute_grade(
    scout_df: pd.DataFrame,
    role_hint: str | None = None,
    weights: Dict[str, Dict[str, float]] | None = None,
) -> GradeBreakdown:
    """
    Compute a 0..100 deterministic score from the scouting percentile table.
    - scout_df: index = metrics, columns include 'Percentile' (0..100)
    - role_hint: e.g., 'scout_summary_fw' or 'FW' or 'forward'; we auto-normalize
    - weights: override per-role weights if desired
    """
    role = _normalize_role(role_hint)
    W = (weights or DEFAULT_WEIGHTS).get(role, {})
    if not W:
        # fallback to MF if role not configured
        role = "mf"
        W = (weights or DEFAULT_WEIGHTS)[role]

    # Ensure we operate on a clean copy
    df = scout_df.copy()
    if "Percentile" not in df.columns:
        raise ValueError("scout_df must include a 'Percentile' column (0..100).")
    # coerce to numeric safely
    df["__pct__"] = pd.to_numeric(df["Percentile"], errors="coerce")

    index_names = list(df.index.astype(str))
    matched: List[Tuple[str, float, float]] = []
    missing: List[str] = []
    score_sum = 0.0
    weight_sum = 0.0

    for desired_metric, w in W.items():
        actual = _match_metric_name(index_names, desired_metric)
        if not actual:
            missing.append(desired_metric)
            continue
        pct = df.loc[actual, "__pct__"]
        if pd.isna(pct):
            missing.append(desired_metric)
            continue
        pct = float(pct)
        pct = max(0.0, min(100.0, pct))
        pct = _invert_if_negative(actual, pct)
        contribution = w * pct
        score_sum += contribution
        weight_sum += w
        matched.append((actual, w, contribution))

    # If some metrics were missing, renormalize by the sum of weights we actually used
    raw = (score_sum / weight_sum) if weight_sum > 0 else 0.0
    final = max(0.0, min(100.0, raw))

    return GradeBreakdown(
        role=role,
        matched=matched,
        missing=missing,
        raw_score=raw,
        final_score=final,
    )

def rationale_from_breakdown(bd: GradeBreakdown, language: str = "English") -> str:
    """
    Short human-readable rationale (EN/FR) summarizing drivers of the grade.
    """
    # Top positive drivers
    top = sorted(bd.matched, key=lambda x: x[2], reverse=True)[:3]
    drivers = ", ".join([f"{m},weight={w:.2f})" for m, w, _ in top]) if top else "—"

    # Missing signals
    missing = ", ".join(bd.missing[:5]) if bd.missing else "—"

    if (language or "").lower().startswith("fr"):
        return (
            f"Note déterministe: **{bd.final_score:.1f}/100** (rôle: {bd.role.upper()}).\n\n"
            f"Principaux moteurs: {drivers}.\n"
            f"Principaux points d'amélioration, faiblesses: {missing}.\n"
            f"La note est calculée sur des percentiles FBref (365 jours) pondérés par le poste, "
            f"avec inversion des métriques défavorables (ex: pertes de balle, fautes)."
        )
    else:
        return (
            f"Deterministic grade: **{bd.final_score:.1f}/100** (role: {bd.role.upper()}).\n\n"
            f"Top drivers: {drivers}.\n"
            f"Principals improvements points, weaknesses: {missing}.\n"
            f"The grade uses FBref 365‑day percentiles, weighted by position, "
            f"inverting negative metrics (e.g., fouls, dispossessed)."
        )
