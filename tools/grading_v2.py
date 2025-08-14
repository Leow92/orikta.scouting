# tools/grading.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import pandas as pd
import re

# -----------------------------
# Role taxonomy & normalization
# -----------------------------
# Canonical base roles
BASE_ROLES = {"fw", "mf", "df", "gk"}

# Canonical subroles (per base role)
SUBROLES_BY_BASE = {
    "df": {"cb", "fb"},                 # center back, fullback/wingback
    "mf": {"dm", "cm", "am", "wm"},     # defensive, central, attacking, wide
    "fw": {"st", "w"},                  # striker, winger
    "gk": set(),                        # (none for now)
}

# Free-text aliases → (base, sub or None)
ROLE_ALIASES: Dict[str, Tuple[str, Optional[str]]] = {
    # Base roles
    "defender": ("df", None), "df": ("df", None), "cb": ("df", "cb"), "center back": ("df", "cb"),
    "centre back": ("df", "cb"), "rb": ("df", "fb"), "lb": ("df", "fb"),
    "rwb": ("df", "fb"), "lwb": ("df", "fb"), "fullback": ("df", "fb"), "wingback": ("df", "fb"),

    "midfielder": ("mf", None), "mf": ("mf", None),
    "dm": ("mf", "dm"), "defensive midfielder": ("mf", "dm"), "no. 6": ("mf", "dm"),
    "cm": ("mf", "cm"), "central midfielder": ("mf", "cm"), "no. 8": ("mf", "cm"),
    "am": ("mf", "am"), "attacking midfielder": ("mf", "am"), "no. 10": ("mf", "am"),
    "wm": ("mf", "wm"), "wide midfielder": ("mf", "wm"),

    "forward": ("fw", None), "fw": ("fw", None),
    "st": ("fw", "st"), "striker": ("fw", "st"), "cf": ("fw", "st"), "center forward": ("fw", "st"),
    "winger": ("fw", "w"), "rw": ("fw", "w"), "lw": ("fw", "w"),

    "goalkeeper": ("gk", None), "gk": ("gk", None), "keeper": ("gk", None),
}

def _normalize_role(role_hint: Optional[str]) -> Tuple[str, Optional[str]]:
    """
    Return (base, subrole or None). Falls back to ("mf", None).
    Accepts inputs like 'scout_summary_cb', 'Right Winger', 'DF', 'AM', etc.
    """
    if not role_hint:
        return "mf", None
    s = role_hint.lower().strip()

    # extract tokens
    tokens = [t for t in (
        s.replace("_", " ").replace("-", " ").replace("/", " ").split()
    ) if t]

    # direct alias match
    for t in tokens + [s]:
        if t in ROLE_ALIASES:
            return ROLE_ALIASES[t]

    # weak heuristics
    if "gk" in s or "keeper" in s:
        return "gk", None
    if any(k in s for k in ("cb", "center back", "centre back")):
        return "df", "cb"
    if any(k in s for k in ("rb", "lb", "rwb", "lwb", "fullback", "wingback")):
        return "df", "fb"
    if any(k in s for k in ("dm", "no. 6", "defensive mid")):
        return "mf", "dm"
    if any(k in s for k in ("am", "no. 10", "attacking mid")):
        return "mf", "am"
    if any(k in s for k in ("cm", "no. 8", "central mid")):
        return "mf", "cm"
    if "wide" in s or "wm" in s:
        return "mf", "wm"
    if any(k in s for k in ("st", "striker", "cf", "center forward")):
        return "fw", "st"
    if "winger" in s or " rw" in s or " lw" in s:
        return "fw", "w"
    if "df" in s or "defender" in s:
        return "df", None
    if "fw" in s or "forward" in s:
        return "fw", None

    return "mf", None

# ----------------------------------
# Weights: base roles + specific add
# ----------------------------------
# Base role weights (coarse, ~1.0 total)
DEFAULT_WEIGHTS: Dict[str, Dict[str, float]] = {
    "fw": {
        "Non-Penalty Goals": 0.18,
        "npxG": 0.12,
        "Shots Total": 0.08,
        "xG": 0.06,
        "Shot-Creating Actions": 0.12,
        "Touches (Att Pen)": 0.08,
        "Progressive Passes Received": 0.10,
        "Aerials won %": 0.06,
        "Assists": 0.08,
        "Non-Penalty xG + xAG": 0.12,
    },
    "mf": {
        "Progressive Passes": 0.14,
        "Passes Completed (Short/Med/Long)": 0.08,
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
        "Dribblers Tackled %": 0.08,
        "Ball Recoveries": 0.10,
        "Progressive Passes": 0.10,
        "Pass Completion %": 0.10,
    },
    "gk": {
        "Post-Shot xG +/- per 90": 0.22,
        "Save%": 0.18,
        "Crosses Stopped %": 0.12,
        "Clean Sheets %": 0.10,
        "Goals Against per 90": 0.06,
        "Launch%": 0.06,
        "Passes Attempted (Avg Len)": 0.06,
        "Pass Completion % (Launched)": 0.10,
        "Avg Distance of Def Actions": 0.10,
    },
}

# Sub-role additive/override weights (targeted deltas; do not need to sum to 1)
SUBROLE_WEIGHTS: Dict[str, Dict[str, float]] = {
    # --- Defenders ---
    "cb": {
        "Aerials won %": 0.10,
        "Clearances": 0.08,
        "Blocks": 0.06,
        "Tackles": 0.06,
        "Interceptions": 0.06,
        # downweight ball progression vs fullbacks:
        "Progressive Passes": 0.04,
    },
    "fb": {
        "Progressive Carries": 0.10,
        "Progressive Passes": 0.10,
        "Shot-Creating Actions": 0.06,
        "Crosses": 0.08,  # may appear in some FBref variants; aliases below
        "Aerials won %": 0.03,
        "Tackles": 0.05,
        "Interceptions": 0.05,
    },

    # --- Midfielders ---
    "dm": {
        "Tackles": 0.10,
        "Interceptions": 0.10,
        "Ball Recoveries": 0.08,
        "Pressures": 0.06,
        "Pass Completion %": 0.08,
        "Progressive Passes": 0.06,
        # deprioritize direct scoring:
        "Non-Penalty Goals": 0.02,
    },
    "cm": {
        "Progressive Passes": 0.10,
        "Progressive Carries": 0.08,
        "Pass Completion %": 0.08,
        "Shot-Creating Actions": 0.06,
        "xAG": 0.05,
        "Assists": 0.05,
        "Tackles": 0.05,
        "Interceptions": 0.05,
    },
    "am": {
        "Shot-Creating Actions": 0.12,
        "xAG": 0.10,
        "Assists": 0.10,
        "Progressive Carries": 0.08,
        "Non-Penalty xG + xAG": 0.08,
        "Touches (Att Pen)": 0.06,
        "Non-Penalty Goals": 0.06,
    },
    "wm": {
        "Progressive Carries": 0.10,
        "Successful Take-Ons": 0.08,
        "Crosses": 0.08,
        "Shot-Creating Actions": 0.08,
        "xAG": 0.06,
        "Touches (Att Pen)": 0.06,
    },

    # --- Forwards ---
    "st": {
        "Non-Penalty Goals": 0.14,
        "npxG": 0.12,
        "Shots Total": 0.10,
        "Non-Penalty xG + xAG": 0.10,
        "Touches (Att Pen)": 0.08,
        "Aerials won %": 0.06,
        "Progressive Passes Received": 0.10,
    },
    "w": {
        "Progressive Carries": 0.12,
        "Successful Take-Ons": 0.10,
        "Shot-Creating Actions": 0.10,
        "xAG": 0.08,
        "Assists": 0.08,
        "Touches (Att Pen)": 0.08,
        "Non-Penalty xG + xAG": 0.06,
    },
}

# ---- Aliases (FBref labels vary slightly) ----
ALIASES: Dict[str, List[str]] = {
    "Touches (Att Pen)": ["Touches (Att Pen)", "Touches (Att Pen Area)", "Touches (Att Penalty Area)"],
    "Dribblers Tackled %": ["Dribblers Tackled %", "Tkl% (Dribblers)"],
    "Save%": ["Save%", "Save %"],
    "Clean Sheets %": ["Clean Sheets %", "CS %"],
    "Aerials won %": ["Aerials won %", "Aerial Win %", "Aerial Wins %"],
    "Goals Against per 90": ["Goals Against per 90", "GA/90"],
    "Post-Shot xG +/- per 90": ["Post-Shot xG +/- per 90", "PSxG+/-/90", "PSxG +/- per 90"],
    "Passes Completed (Short/Med/Long)": [
        "Passes Completed (Short/Med/Long)", "Cmp (Short/Med/Long)", "Cmp (S/M/L)"
    ],
    "Ball Recoveries": ["Ball Recoveries", "Recoveries"],
    "Crosses": ["Crosses", "Crosses Completed", "Cmp (Crosses)", "Crosses into Pen Area"],
}

# ---- Metrics that are "bad when high" ----
NEGATIVE_KEYS: List[str] = [
    "Fouls", "Dispossessed", "Miscontrols", "Errors", "Times Tackled", "Goals Against per 90"
]

@dataclass
class GradeBreakdown:
    role: str                # base or base:sub (e.g., "df:cb")
    matched: List[Tuple[str, float, float]]  # (metric_name, weight, contribution_0_100)
    missing: List[str]
    raw_score: float
    final_score: float

# -----------------------------
# Weight blending & utilities
# -----------------------------
SUBROLE_BLEND = 0.60  # fraction of weight coming from subrole when present; base contributes (1 - SUBROLE_BLEND)

def _match_metric_name(index_names: List[str], desired: str) -> str | None:
    norm_index = {name.lower().strip(): name for name in index_names}
    # exact
    dl = desired.lower().strip()
    if dl in norm_index:
        return norm_index[dl]
    # aliases
    for alias in ALIASES.get(desired, []):
        al = alias.lower().strip()
        if al in norm_index:
            return norm_index[al]
    # fuzzy contains
    for k, orig in norm_index.items():
        if dl in k:
            return orig
    return None

def _invert_if_negative(metric: str, pct: float) -> float:
    for neg in NEGATIVE_KEYS:
        if neg.lower() in metric.lower():
            return 100.0 - pct
    return pct

def _blend_weights(base_w: Dict[str, float], sub_w: Optional[Dict[str, float]]) -> Dict[str, float]:
    """
    Blend base and subrole weights. We don't require sums to be exactly 1.0;
    we rescale at the end so total = 1.0 over the union of keys.
    """
    if not sub_w:
        # Normalize base only
        total = sum(base_w.values()) or 1.0
        return {k: v / total for k, v in base_w.items()}

    out: Dict[str, float] = {}
    for k in set(base_w) | set(sub_w):
        b = base_w.get(k, 0.0) * (1.0 - SUBROLE_BLEND)
        s = sub_w.get(k, 0.0) * SUBROLE_BLEND
        out[k] = b + s

    total = sum(out.values()) or 1.0
    return {k: v / total for k, v in out.items()}

# -----------------------------
# Public API
# -----------------------------
def compute_grade(
    scout_df: pd.DataFrame,
    role_hint: str | None = None,
    weights: Dict[str, Dict[str, float]] | None = None,
) -> GradeBreakdown:
    """
    Compute a 0..100 deterministic score from the scouting percentile table.
    - scout_df: index = metrics, columns include 'Percentile' (0..100)
    - role_hint: any string like 'CB', 'Right Winger', 'scout_summary_fw', etc.
    - weights: optional override for base roles; subrole weights come from SUBROLE_WEIGHTS.
    """
    base, sub = _normalize_role(role_hint)
    base_w_all = (weights or DEFAULT_WEIGHTS)
    base_w = base_w_all.get(base, DEFAULT_WEIGHTS["mf"])
    sub_w = SUBROLE_WEIGHTS.get(sub) if sub in (SUBROLES_BY_BASE.get(base) or set()) else None
    W = _blend_weights(base_w, sub_w)

    # Ensure we operate on a clean copy
    df = scout_df.copy()
    if "Percentile" not in df.columns:
        raise ValueError("scout_df must include a 'Percentile' column (0..100).")
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

    raw = (score_sum / weight_sum) if weight_sum > 0 else 0.0
    final = max(0.0, min(100.0, raw))
    role_label = base if sub is None else f"{base}:{sub}"

    return GradeBreakdown(
        role=role_label,
        matched=matched,
        missing=missing,
        raw_score=raw,
        final_score=final,
    )

def rationale_from_breakdown(bd: GradeBreakdown, language: str = "English") -> str:
    """
    Short human-readable rationale (EN/FR) summarizing drivers of the grade.
    """
    top = sorted(bd.matched, key=lambda x: x[2], reverse=True)[:3]
    drivers = ", ".join([f"{m}, weight={w:.2f}" for m, w, _ in top]) if top else "—"
    missing = ", ".join(bd.missing[:5]) if bd.missing else "—"

    role_disp = bd.role.upper()
    if (language or "").lower().startswith("fr"):
        return (
            f"Note déterministe: **{bd.final_score:.1f}/100** (rôle: {role_disp}).\n\n"
            f"Principaux moteurs: {drivers}.\n\n"
            f"Principaux points d'amélioration/absents: {missing}.\n\n"
            f"Les pondérations combinent le poste général et le poste spécifique (ex: DF:CB) ; "
            f"les métriques défavorables sont inversées."
        )
    else:
        return (
            f"Deterministic grade: **{bd.final_score:.1f}/100** (role: {role_disp}).\n\n"
            f"Top drivers: {drivers}.\n\n"
            f"Key missing/low-weighted signals: {missing}.\n\n"
            f"Weights blend base role and specific role (e.g., DF:CB); negative metrics are inverted."
        )

# Public map for UI (optional): canonical role label -> pretty
ROLE_PRETTY = {
    "df:cb": "CB",
    "df:fb": "FB/WB",
    "mf:dm": "DM (No.6)",
    "mf:cm": "CM (No.8)",
    "mf:am": "AM/10",
    "mf:wm": "Wide Mid",
    "fw:st": "ST/CF",
    "fw:w":  "Winger",
    "df": "DEF", "mf": "MID", "fw": "FWD", "gk": "GK",
}

def normalize_positions_from_profile(raw: str | list[str] | None) -> list[tuple[str, str | None]]:
    """
    Take the FBref 'Position' field (string like 'FW-MF ▪ Right' or 'DF (CB, FB)')
    or a list of tokens, and return a unique list of (base, subrole|None).
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        text = " ".join(str(x) for x in raw)
    else:
        text = str(raw)

    # remove side markers and punctuation
    text = re.sub(r"[▪•·\(\)\[\]]", " ", text)
    text = text.replace("/", " ").replace("|", " ").replace(",", " ").replace(";", " ").replace("-", " ")
    tokens = [t.strip().lower() for t in text.split() if t.strip()]

    found: list[tuple[str, str | None]] = []
    seen = set()
    # try sequences like "right winger" first
    joined = " " .join(tokens)
    for alias, (base, sub) in ROLE_ALIASES.items():
        if f" {alias} " in f" {joined} ":
            key = (base, sub)
            if key not in seen:
                found.append(key); seen.add(key)

    # fallback: token-wise lookup (single tokens like 'cb','dm','rw')
    for t in tokens:
        if t in ROLE_ALIASES:
            base, sub = ROLE_ALIASES[t]
            key = (base, sub)
            if key not in seen:
                found.append(key); seen.add(key)

    # keep at least the base role if nothing matched
    if not found:
        base, sub = _normalize_role(joined)
        found.append((base, sub))
    return found

def label_from_pair(base: str, sub: str | None) -> str:
    key = f"{base}:{sub}" if sub else base
    return ROLE_PRETTY.get(key, key.upper())

def compute_grade_for_positions(
    scout_df: pd.DataFrame,
    positions: list[tuple[str, str | None]],
    weights: Dict[str, Dict[str, float]] | None = None,
) -> dict[str, GradeBreakdown]:
    """
    Compute a GradeBreakdown for every (base,sub) position in `positions`.
    Returns { role_label: GradeBreakdown } where role_label like 'DF:CB'.
    """
    out: dict[str, GradeBreakdown] = {}
    for base, sub in positions:
        role_hint = f"{base}:{sub}" if sub else base
        bd = compute_grade(scout_df, role_hint=role_hint, weights=weights)
        out[bd.role] = bd
    return out
