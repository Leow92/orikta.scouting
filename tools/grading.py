# tools/grading.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterable
import pandas as pd
import re
import math

# -----------------------------
# Configuration Constants
# -----------------------------

LEAGUE_TIERS: Dict[str, float] = {
    # Top 5 leagues
    "premier league": 1.00, "la liga": 1.00, "bundesliga": 1.00,
    "serie a": 0.98, "ligue 1": 0.95,
    # Second tier
    "er_divisie": 0.90, "primeira liga": 0.88, "belgian pro league": 0.85,
    "scottish premiership": 0.82, "turkish super lig": 0.80,
    # Third tier
    "austrian bundesliga": 0.75, "swiss super league": 0.75,
    "danish superliga": 0.72, "norwegian eliteserien": 0.70,
    # Default for unknown leagues
    "other": 0.75,
}

PEAK_AGES: Dict[str, int] = {
    "gk": 28,
    "df": 27,
    "mf": 27,
    "fw": 26,
}

# Multi-dimensional pillar weights (attacking, technical, defensive, physical)
PILLAR_WEIGHTS: Dict[str, Dict[str, float]] = {
    "fw": {"attacking": 0.40, "technical": 0.25, "defensive": 0.10, "physical": 0.25},
    "mf": {"attacking": 0.30, "technical": 0.35, "defensive": 0.20, "physical": 0.15},
    "df": {"attacking": 0.10, "technical": 0.20, "defensive": 0.45, "physical": 0.25},
    "gk": {"attacking": 0.00, "technical": 0.15, "defensive": 0.60, "physical": 0.25},
}

# Pillar metric mappings
PILLAR_METRICS: Dict[str, List[str]] = {
    "attacking": [
        "Goals per 90", "Assists per 90", "G+A per 90", 
        "Shots per 90", "Shot Accuracy %", "Key Passes per 90",
        "Penalties Won per 90", "Prog. Carries per 90",
    ],
    "technical": [
        "Pass Completion %", "Dribble Success %", 
        "Dribbles per 90", "Fouls Drawn per 90",
        "Prog. Passes per 90", "Through Balls per 90",
    ],
    "defensive": [
        "Tackles per 90", "Interceptions per 90", "Blocks per 90",
        "Duels Won %", "Tackles Won %", "Pressures per 90",
        "Aerial Duels Won %", "Clean Sheets %",
    ],
    "physical": [
        "Duels Won %", "Aerial Duels Won %", 
        "Pressures per 90", "Fouls Drawn per 90",
    ],
}

# -----------------------------
# Role taxonomy & normalization
# -----------------------------
BASE_ROLES = {"fw", "mf", "df", "gk"}

SUBROLES_BY_BASE = {
    "df": {"cb", "fb"},
    "mf": {"dm", "cm", "am", "wm"},
    "fw": {"st", "w"},
    "gk": set(),
}

ROLE_ALIASES: Dict[str, Tuple[str, Optional[str]]] = {
    # Defenders
    "defender": ("df", None), "df": ("df", None), "cb": ("df", "cb"), "center back": ("df", "cb"),
    "centre back": ("df", "cb"), "rb": ("df", "fb"), "lb": ("df", "fb"),
    "rwb": ("df", "fb"), "lwb": ("df", "fb"), "fullback": ("df", "fb"), "wingback": ("df", "fb"),
    # Midfielders
    "midfielder": ("mf", None), "mf": ("mf", None),
    "dm": ("mf", "dm"), "defensive midfielder": ("mf", "dm"), "no. 6": ("mf", "dm"),
    "cm": ("mf", "cm"), "central midfielder": ("mf", "cm"), "no. 8": ("mf", "cm"),
    "am": ("mf", "am"), "attacking midfielder": ("mf", "am"), "no. 10": ("mf", "am"),
    "wm": ("mf", "wm"), "wide midfielder": ("mf", "wm"),
    # Forwards / Attackers
    "forward": ("fw", None), "fw": ("fw", None), "attacker": ("fw", None),
    "st": ("fw", "st"), "striker": ("fw", "st"), "cf": ("fw", "st"), "center forward": ("fw", "st"),
    "winger": ("fw", "w"), "rw": ("fw", "w"), "lw": ("fw", "w"),
    # Goalkeepers
    "goalkeeper": ("gk", None), "gk": ("gk", None), "keeper": ("gk", None),
}


# -----------------------------
# Configuration Dataclasses
# -----------------------------

@dataclass
class GradingConfig:
    """Configurable parameters for the grading system."""
    subrole_blend: float = 0.60
    min_minutes_threshold: int = 500
    min_confidence_threshold: float = 0.5
    league_weight: float = 0.15
    age_weight: float = 0.10
    style_strength: float = 0.5
    use_multi_dimension: bool = True
    
    def __post_init__(self):
        # Ensure values are in valid ranges
        self.subrole_blend = max(0.0, min(1.0, self.subrole_blend))
        self.min_confidence_threshold = max(0.0, min(1.0, self.min_confidence_threshold))
        self.league_weight = max(0.0, min(0.5, self.league_weight))
        self.age_weight = max(0.0, min(0.3, self.age_weight))
        self.style_strength = max(0.0, min(1.0, self.style_strength))


@dataclass
class MultiGrade:
    """Multi-dimensional grade breakdown."""
    overall: float
    attacking: float
    technical: float
    defensive: float
    physical: float
    role: str
    minutes: int
    confidence: float
    league_adjusted: float
    age_adjusted: float
    matched: List[Tuple[str, float, float]]
    missing: List[str]
    pillar_scores: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "overall": round(self.overall, 2),
            "attacking": round(self.attacking, 2),
            "technical": round(self.technical, 2),
            "defensive": round(self.defensive, 2),
            "physical": round(self.physical, 2),
            "role": self.role,
            "minutes": self.minutes,
            "confidence": round(self.confidence, 3),
            "league_adjusted": round(self.league_adjusted, 2),
            "age_adjusted": round(self.age_adjusted, 2),
        }

def _normalize_role(role_hint: Optional[str]) -> Tuple[str, Optional[str]]:
    if not role_hint:
        return "mf", None
    s = role_hint.lower().strip()

    # Fast path: explicit "base:subrole" notation (e.g. "df:fb", "mf:cm")
    if ":" in s:
        base_part, sub_part = s.split(":", 1)
        base_part, sub_part = base_part.strip(), sub_part.strip()
        if base_part in BASE_ROLES and sub_part in (SUBROLES_BY_BASE.get(base_part) or set()):
            return base_part, sub_part

    tokens = [t for t in (s.replace("_", " ").replace("-", " ").replace("/", " ").split()) if t]

    for t in tokens + [s]:
        if t in ROLE_ALIASES:
            return ROLE_ALIASES[t]

    if "gk" in s or "keeper" in s: return "gk", None
    if any(k in s for k in ("cb", "center back", "centre back")): return "df", "cb"
    if any(k in s for k in ("rb", "lb", "rwb", "lwb", "fullback", "wingback")): return "df", "fb"
    if any(k in s for k in ("dm", "no. 6", "defensive mid")): return "mf", "dm"
    if any(k in s for k in ("am", "no. 10", "attacking mid")): return "mf", "am"
    if any(k in s for k in ("cm", "no. 8", "central mid")): return "mf", "cm"
    if "wide" in s or "wm" in s: return "mf", "wm"
    if any(k in s for k in ("st", "striker", "cf", "center forward")): return "fw", "st"
    if "winger" in s or " rw" in s or " lw" in s: return "fw", "w"
    if any(k in s for k in ("attacker",)): return "fw", None
    if "df" in s or "defender" in s: return "df", None
    if "fw" in s or "forward" in s: return "fw", None
    return "mf", None

# ----------------------------------
# Weights: base roles + subrole add
# ----------------------------------
# All metric names match the output of utils/percentile_engine.stats_entry_to_per90()

DEFAULT_WEIGHTS: Dict[str, Dict[str, float]] = {
    "fw": {
        # Primary output — goals and overall contribution
        "Goals per 90":      0.25,
        "G+A per 90":        0.13,
        "Shot Accuracy %":   0.12,  # finishing efficiency
        "Shots per 90":      0.11,
        "Assists per 90":    0.10,
        "Dribble Success %": 0.10,
        "Key Passes per 90": 0.08,  # lower: creation is a secondary fw trait
        "Duels Won %":       0.07,
        "Dribbles per 90":   0.04,
    },
    "mf": {
        "Key Passes per 90":    0.18,
        "Pass Completion %":    0.14,
        "Assists per 90":       0.12,
        "Tackles per 90":       0.12,
        "Interceptions per 90": 0.10,
        "Dribble Success %":    0.10,
        "Duels Won %":          0.10,
        "Goals per 90":         0.07,
        "Dribbles per 90":      0.07,
    },
    "df": {
        "Tackles per 90":       0.20,
        "Interceptions per 90": 0.18,
        "Blocks per 90":        0.14,
        "Duels Won %":          0.18,
        "Pass Completion %":    0.12,
        "Fouls per 90":         0.10,   # negative metric — inverted in scoring
        "Key Passes per 90":    0.08,
    },
    "gk": {
        # Save % measures shot-stopping quality; Goals Conceded is the outcome.
        # Saves per 90 is removed — it rewards GKs behind poor defences (more shots faced).
        "Save %":                  0.50,
        "Goals Conceded per 90":   0.35,  # negative metric — inverted
        "Pass Completion %":       0.10,  # sweeper-keeper distribution
        "Duels Won %":             0.05,  # 1v1 situations — minor
    },
}

SUBROLE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "cb": {
        # Aerial/physical dominance + positional reading; build-up secondary
        "Duels Won %":          0.22,  # headers, challenges — core CB metric
        "Interceptions per 90": 0.18,  # reading the game
        "Blocks per 90":        0.14,  # shot-blocking / last-ditch
        "Tackles per 90":       0.12,  # ground challenges
        "Pass Completion %":    0.06,  # contributing to build-up
    },
    "fb": {
        # Modern full-back: offensive width + defensive resilience
        "Key Passes per 90":    0.16,  # chance creation from wide
        "Dribble Success %":    0.12,  # quality of attacking runs
        "Assists per 90":       0.12,  # end product
        "Dribbles per 90":      0.10,  # volume of attacking overlaps
        "Tackles per 90":       0.06,  # defensive duty
        "Duels Won %":          0.04,  # defensive resilience
    },
    "dm": {
        # Defensive screen: protect the backline, recycle possession
        "Tackles per 90":       0.20,  # primary defensive duty
        "Interceptions per 90": 0.16,  # reading attacks
        "Duels Won %":          0.14,  # physicality in midfield
        "Pass Completion %":    0.10,  # connecting defence to attack
        "Blocks per 90":        0.08,  # screening backline
        # Key Passes removed: DM creation is secondary, leaks from mf base
    },
    "cm": {
        # Box-to-box: balance creation, distribution, defensive coverage
        "Key Passes per 90":    0.14,
        "Pass Completion %":    0.10,
        "Assists per 90":       0.08,
        "Tackles per 90":       0.08,  # box-to-box defensive contribution
        "Interceptions per 90": 0.07,
        "Dribble Success %":    0.07,  # quality over raw dribble count
        "Goals per 90":         0.04,
    },
    "am": {
        # Creative hub + goalscoring threat; shooting efficiency added
        "Key Passes per 90":  0.18,   # primary creative output
        "Assists per 90":     0.14,
        "Goals per 90":       0.13,   # modern AMs are expected to score
        "Dribble Success %":  0.10,
        "G+A per 90":         0.08,
        "Shot Accuracy %":    0.07,   # clinical when shooting
        "Dribbles per 90":    0.04,
    },
    "wm": {
        # Wide midfielder: dribbling threat + scoring + creating
        "Dribble Success %":  0.16,   # elite wide players win their dribbles
        "Goals per 90":       0.12,   # wide mids contribute heavily to goals
        "Dribbles per 90":    0.10,
        "Key Passes per 90":  0.10,
        "Assists per 90":     0.10,
        "Shot Accuracy %":    0.08,   # cutting inside to shoot
        "G+A per 90":         0.06,
    },
    "st": {
        # Pure striker: goals, efficiency, threat volume, physicality
        "Goals per 90":      0.26,   # the primary striker metric
        "Shot Accuracy %":   0.16,   # clinical finishing
        "G+A per 90":        0.14,   # holistic output
        "Shots per 90":      0.12,   # volume of threat
        "Duels Won %":       0.10,   # hold-up play / aerial presence
        "Assists per 90":    0.04,   # secondary — STs do assist occasionally
        # Dribbles per 90 removed: not a primary ST metric
    },
    "w": {
        # Winger: pace, dribbling quality, goals + creation
        "Dribble Success %":   0.20,  # the defining winger trait
        "Goals per 90":        0.16,  # modern wingers are primary scorers
        "Dribbles per 90":     0.12,
        "Key Passes per 90":   0.10,
        "Assists per 90":      0.10,
        "Shot Accuracy %":     0.08,  # cutting inside / finishing
        "G+A per 90":          0.06,
    },
}

# Aliases: allows the metric-matching function to find metrics even if the
# exact string differs slightly. Since API-football metric names are now
# standardized by percentile_engine, this list is intentionally short.
ALIASES: Dict[str, List[str]] = {
    "Goals per 90":         ["Goals per 90", "Goals/90"],
    "Assists per 90":       ["Assists per 90", "Assists/90"],
    "G+A per 90":           ["G+A per 90", "Goals+Assists per 90"],
    "Shots per 90":         ["Shots per 90", "Shots Total per 90"],
    "Shot Accuracy %":      ["Shot Accuracy %", "Shot On Target %"],
    "Key Passes per 90":    ["Key Passes per 90", "Key Passes/90"],
    "Pass Completion %":    ["Pass Completion %", "Pass Accuracy %"],
    "Tackles per 90":       ["Tackles per 90", "Tackles/90"],
    "Interceptions per 90": ["Interceptions per 90", "Interceptions/90"],
    "Blocks per 90":        ["Blocks per 90", "Blocks/90"],
    "Dribbles per 90":      ["Dribbles per 90", "Dribble Attempts per 90"],
    "Dribble Success %":    ["Dribble Success %", "Dribbles Won %"],
    "Duels Won %":          ["Duels Won %", "Duels Won %"],
    "Fouls per 90":         ["Fouls per 90", "Fouls Committed per 90"],
    "Save %":               ["Save %", "Save%", "Saves %"],
    "Saves per 90":         ["Saves per 90", "Saves/90"],
    "Goals Conceded per 90": ["Goals Conceded per 90", "GA/90", "Goals Against per 90"],
}

NEGATIVE_KEYS: List[str] = [
    "Fouls per 90",
    "Goals Conceded per 90",
]

@dataclass
class GradeBreakdown:
    role: str
    matched: List[Tuple[str, float, float]]  # (metric_name, weight, contribution_0_100)
    missing: List[str]
    raw_score: float
    final_score: float
    minutes: int = 0
    confidence: float = 1.0
    league: str = ""
    age: Optional[int] = None
    league_adjusted_score: float = 0.0
    age_adjusted_score: float = 0.0

SUBROLE_BLEND = 0.60  # base ⊕ subrole blend factor


# -----------------------------
# New Helper Functions for Elite Grading
# -----------------------------

def _confidence_factor(minutes: int, min_threshold: int = 500) -> float:
    """
    Sigmoid curve for confidence based on minutes played.
    - 0min = 0.50 confidence
    - 500min = ~0.62 confidence
    - 1000min = ~0.73 confidence
    - 2000min = ~0.88 confidence
    - 5000min = ~0.97 confidence
    """
    if minutes <= 0:
        return 0.5
    # Sigmoid: 1 / (1 + exp(-k * (x - x0)))
    # Calibrated so that:
    # - At 0 minutes: ~0.5
    # - At 2000 minutes: ~0.88
    # - At 5000 minutes: ~0.97
    k = 0.001
    x0 = 1500  # Midpoint
    raw = 1.0 / (1.0 + math.exp(-k * (minutes - x0)))
    # Scale from 0-1 to 0.5-0.97 range
    return 0.5 + (raw) * 0.47


def _get_league_tier(league: str) -> float:
    """Get league tier multiplier (0-1)."""
    if not league:
        return 1.0
    league_lower = league.lower().strip()
    for key, value in LEAGUE_TIERS.items():
        if key in league_lower:
            return value
    return LEAGUE_TIERS.get("other", 0.75)


def _age_adjustment_factor(age: Optional[int], base_role: str) -> float:
    """
    Adjust score based on age relative to peak.
    - Young players (<21): bonus for potential
    - Peak age (26-28): no adjustment
    - Older players (>30): penalty for decline
    Returns multiplier (1.0 = no change)
    """
    if age is None:
        return 1.0
    
    peak = PEAK_AGES.get(base_role, 27)
    
    if age < 21:
        # U21: linear bonus up to +20% at age 18
        years_from_21 = 21 - age
        return 1.0 + 0.2 * (years_from_21 / 3.0)
    elif age > 30:
        # 30+: linear penalty of -5% per year
        years_over_30 = age - 30
        return 1.0 - 0.05 * years_over_30
    else:
        # At or near peak: slight bonus for being in prime
        years_from_peak = abs(age - peak)
        if years_from_peak <= 2:
            return 1.0 + 0.02 * (1.0 - years_from_peak / 2.0)
        return 1.0


def _adjust_percentile_for_context(
    percentile: float,
    minutes: int,
    league: str,
    age: Optional[int],
    base_role: str,
    config: GradingConfig,
) -> Tuple[float, float, float]:
    """
    Adjust raw percentile based on context factors.
    Returns: (league_adjusted, age_adjusted, confidence)
    """
    confidence = _confidence_factor(minutes, config.min_minutes_threshold)
    
    # League adjustment
    league_tier = _get_league_tier(league)
    league_adjusted = percentile * (1.0 - config.league_weight + config.league_weight * league_tier)
    
    # Age adjustment
    age_factor = _age_adjustment_factor(age, base_role)
    age_adjusted = percentile * (1.0 - config.age_weight + config.age_weight * age_factor)
    
    return league_adjusted, age_adjusted, confidence


def validate_scout_df(scout_df: pd.DataFrame) -> None:
    """Validate that the scout DataFrame has required columns and valid values."""
    required = ["Percentile"]
    missing = [col for col in required if col not in scout_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    percentiles = pd.to_numeric(scout_df["Percentile"], errors="coerce")
    if not 0 <= percentiles.min() <= percentiles.max() <= 100:
        raise ValueError("Percentile values must be 0-100")


def check_percentile_distribution(scout_df: pd.DataFrame) -> Dict[str, str]:
    """
    Analyze percentile distribution and warn about potential issues.
    Returns dict of warnings/suggestions.
    """
    warnings = {}
    percentiles = pd.to_numeric(scout_df["Percentile"], errors="coerce")
    
    # Check for suspicious values
    low_percentiles = percentiles[percentiles < 20]
    high_percentiles = percentiles[percentiles > 80]
    
    if len(low_percentiles) > len(percentiles) * 0.7:
        warnings["low_percentiles"] = (
            f"{len(low_percentiles)}/{len(percentiles)} metrics below 20th percentile. "
            "This player may be below average, or the input might not be percentiles."
        )
    
    if len(high_percentiles) > len(percentiles) * 0.7:
        warnings["high_percentiles"] = (
            f"{len(high_percentiles)}/{len(percentiles)} metrics above 80th percentile. "
            "This is expected for elite players."
        )
    
    if percentiles.mean() < 30:
        warnings["low_average"] = (
            f"Average percentile is {percentiles.mean():.1f}. "
            "For a world-class player, expect 70+. Check if input is actually percentiles (0-100)."
        )
    
    # Check if values look like they might be ratios (0-1) instead of percentiles (0-100)
    ratio_like = percentiles[(percentiles > 0) & (percentiles < 5)]
    if len(ratio_like) > len(percentiles) * 0.3:
        warnings["possible_ratio"] = (
            f"{len(ratio_like)} metrics are between 0-5. "
            "If these are meant to be ratios (0-1), multiply by 100 to convert to percentiles."
        )
    
    return warnings


def get_expected_grade_range(role_hint: str) -> Tuple[float, float]:
    """
    Return expected grade range for different quality levels.
    Based on the position's typical weight distribution.
    """
    base, sub = _normalize_role(role_hint)
    
    # Expected ranges based on player quality
    ranges = {
        "elite": (85, 100),      # World's best (Mbappé, Haaland, De Bruyne)
        "starter": (70, 85),      # Regular starter in top 5 league
        "rotation": (55, 70),     # Rotation/squad player
        "prospect": (40, 55),     # Young prospect or backup
        "limited": (0, 40),       # Not good enough for top level
    }
    
    return ranges


# -----------------------------
# Reference Player Benchmarks
# -----------------------------

REFERENCE_PLAYERS = {
    "fw:st": {
        "name": "Kylian Mbappé",
        "expected_grade": (85, 92),
        "metrics": {
            "Goals per 90": 99, "Assists per 90": 90, "G+A per 90": 99,
            "Shots per 90": 95, "Shot Accuracy %": 85, "Key Passes per 90": 75,
            "Pass Completion %": 75, "Dribble Success %": 90, "Dribbles per 90": 85,
            "Tackles per 90": 30, "Interceptions per 90": 25, "Blocks per 90": 10,
            "Duels Won %": 55, "Fouls per 90": 20, "Fouls Drawn per 90": 90,
        }
    },
    "fw:w": {
        "name": "Mohamed Salah",
        "expected_grade": (82, 88),
        "metrics": {
            "Goals per 90": 95, "Assists per 90": 90, "G+A per 90": 95,
            "Shots per 90": 85, "Shot Accuracy %": 80, "Key Passes per 90": 80,
            "Pass Completion %": 80, "Dribble Success %": 85, "Dribbles per 90": 90,
            "Tackles per 90": 40, "Interceptions per 90": 35, "Blocks per 90": 15,
            "Duels Won %": 60, "Fouls per 90": 25, "Fouls Drawn per 90": 85,
        }
    },
    "mf:am": {
        "name": "Kevin De Bruyne",
        "expected_grade": (80, 88),
        "metrics": {
            "Goals per 90": 85, "Assists per 90": 99, "G+A per 90": 98,
            "Shots per 90": 80, "Shot Accuracy %": 85, "Key Passes per 90": 99,
            "Pass Completion %": 90, "Dribble Success %": 80, "Dribbles per 90": 70,
            "Tackles per 90": 60, "Interceptions per 90": 80, "Blocks per 90": 40,
            "Duels Won %": 65, "Fouls per 90": 15, "Fouls Drawn per 90": 70,
        }
    },
    "mf:cm": {
        "name": "Joshua Kimmich",
        "expected_grade": (72, 80),
        "metrics": {
            "Goals per 90": 60, "Assists per 90": 75, "G+A per 90": 70,
            "Shots per 90": 50, "Shot Accuracy %": 75, "Key Passes per 90": 80,
            "Pass Completion %": 92, "Dribble Success %": 75, "Dribbles per 90": 60,
            "Tackles per 90": 85, "Interceptions per 90": 80, "Blocks per 90": 50,
            "Duels Won %": 70, "Fouls per 90": 20, "Fouls Drawn per 90": 60,
        }
    },
    "mf:dm": {
        "name": "Rodri",
        "expected_grade": (72, 80),
        "metrics": {
            "Goals per 90": 30, "Assists per 90": 50, "G+A per 90": 40,
            "Shots per 90": 25, "Shot Accuracy %": 70, "Key Passes per 90": 75,
            "Pass Completion %": 94, "Dribble Success %": 70, "Dribbles per 90": 40,
            "Tackles per 90": 85, "Interceptions per 90": 90, "Blocks per 90": 70,
            "Duels Won %": 85, "Fouls per 90": 10, "Fouls Drawn per 90": 40,
        }
    },
    "df:cb": {
        "name": "Virgil van Dijk",
        "expected_grade": (83, 89),
        "metrics": {
            "Goals per 90": 10, "Assists per 90": 15, "G+A per 90": 12,
            "Shots per 90": 10, "Shot Accuracy %": 50, "Key Passes per 90": 60,
            "Pass Completion %": 92, "Dribble Success %": 70, "Dribbles per 90": 30,
            "Tackles per 90": 85, "Interceptions per 90": 90, "Blocks per 90": 95,
            "Duels Won %": 90, "Fouls per 90": 5, "Fouls Drawn per 90": 40,
        }
    },
    "df:fb": {
        "name": "Trent Alexander-Arnold",
        "expected_grade": (70, 78),
        "metrics": {
            "Goals per 90": 30, "Assists per 90": 95, "G+A per 90": 80,
            "Shots per 90": 40, "Shot Accuracy %": 70, "Key Passes per 90": 90,
            "Pass Completion %": 85, "Dribble Success %": 80, "Dribbles per 90": 75,
            "Tackles per 90": 60, "Interceptions per 90": 70, "Blocks per 90": 40,
            "Duels Won %": 60, "Fouls per 90": 25, "Fouls Drawn per 90": 65,
        }
    },
    "gk": {
        "name": "Alisson Becker",
        "expected_grade": (85, 92),
        "metrics": {
            "Save %": 95,
            "Goals Conceded per 90": 5,  # Inverted - lower is better
            "Pass Completion %": 85,
            "Duels Won %": 70,
        }
    },
}


def test_benchmark_player(role_hint: str, metrics: Optional[Dict[str, float]] = None) -> GradeBreakdown:
    """
    Test with benchmark metrics for a reference player.
    Useful for validating the grading system.
    """
    ref = REFERENCE_PLAYERS.get(role_hint, REFERENCE_PLAYERS.get(role_hint.split(":")[0]))
    if ref is None:
        ref = REFERENCE_PLAYERS["mf"]
    
    if metrics is None:
        metrics = ref["metrics"]
    
    df = pd.DataFrame({k: [v] for k, v in metrics.items()}, index=["Percentile"]).T
    
    # Use 5000 minutes for max confidence
    bd = compute_grade(df, role_hint=role_hint, minutes=5000, league="Premier League", age=28)
    
    expected_min, expected_max = ref["expected_grade"]
    if bd.final_score < expected_min:
        print(f"WARNING: {ref['name']} ({role_hint}) scored {bd.final_score:.1f}, expected {expected_min}-{expected_max}")
    
    return bd


def run_benchmark_tests() -> Dict[str, Dict]:
    """
    Run all benchmark tests and return results.
    """
    results = {}
    
    for role_hint, ref in REFERENCE_PLAYERS.items():
        try:
            bd = test_benchmark_player(role_hint)
            results[role_hint] = {
                "name": ref["name"],
                "final_score": bd.final_score,
                "raw_score": bd.raw_score,
                "confidence": bd.confidence,
                "expected_range": ref["expected_grade"],
                "within_range": ref["expected_grade"][0] <= bd.final_score <= ref["expected_grade"][1],
            }
        except Exception as e:
            results[role_hint] = {"error": str(e)}
    
    return results

# -----------------------------
# Play-style presets (additive weight deltas)
# -----------------------------
PLAY_STYLE_PRESETS: Dict[str, Dict[str, Dict[str, float]]] = {
    # Positional play — ball circulation, high line, build-up through GK
    "possession_high": {
        "df":    {"Key Passes per 90": 0.08, "Pass Completion %": 0.06, "Dribbles per 90": 0.04},
        "df:cb": {"Key Passes per 90": 0.06, "Pass Completion %": 0.06, "Duels Won %": -0.02},
        "df:fb": {"Dribbles per 90": 0.10, "Key Passes per 90": 0.08, "Assists per 90": 0.06},
        "mf":    {"Key Passes per 90": 0.10, "Pass Completion %": 0.06, "Dribbles per 90": 0.06},
        "fw":    {"Assists per 90": 0.04, "Key Passes per 90": 0.06, "G+A per 90": 0.04},
        "gk":    {"Pass Completion %": 0.10, "Save %": 0.04},
    },
    # High press + fast vertical transitions
    "high_press_transition": {
        "df":    {"Tackles per 90": 0.06, "Interceptions per 90": 0.06, "Fouls Drawn per 90": 0.04},
        "df:fb": {"Fouls Drawn per 90": 0.06, "Dribbles per 90": 0.06, "Dribble Success %": 0.04},
        "mf":    {"Fouls Drawn per 90": 0.08, "Tackles per 90": 0.08, "Interceptions per 90": 0.08, "Dribbles per 90": 0.06},
        "fw":    {"Dribbles per 90": 0.06, "Dribble Success %": 0.06, "G+A per 90": 0.06},
        "gk":    {"Pass Completion %": 0.06},
    },
    # Deep block + rapid counter-attack
    "low_block_counter": {
        "df":    {"Blocks per 90": 0.10, "Duels Won %": 0.10, "Interceptions per 90": 0.06, "Tackles per 90": 0.06,
                  "Key Passes per 90": -0.06, "Pass Completion %": -0.04},
        "df:cb": {"Blocks per 90": 0.08, "Duels Won %": 0.08},
        "df:fb": {"Tackles per 90": 0.06, "Interceptions per 90": 0.06},
        "mf":    {"Tackles per 90": 0.08, "Interceptions per 90": 0.06, "Key Passes per 90": -0.04},
        "fw":    {"Goals per 90": 0.06, "G+A per 90": 0.06},
        "gk":    {"Save %": 0.08, "Goals Conceded per 90": -0.10},
    },
    # Wide overloads, crossing-heavy system
    "crossing_wide": {
        "df:fb": {"Assists per 90": 0.12, "Key Passes per 90": 0.06, "Dribbles per 90": 0.06},
        "mf:wm": {"Assists per 90": 0.10, "Key Passes per 90": 0.08},
        "fw:w":  {"Assists per 90": 0.10, "Key Passes per 90": 0.08, "Dribble Success %": 0.06, "G+A per 90": 0.06},
    },
}

PLAY_STYLE_PRETTY = {
    "possession_high":       "Positional / High Possession",
    "high_press_transition": "High Press & Transition",
    "low_block_counter":     "Low Block & Counter",
    "crossing_wide":         "Crossing / Wide Overloads",
}

# -----------------------------
# Internal utilities
# -----------------------------
def _match_metric_name(index_names: List[str], desired: str) -> str | None:
    norm_index = {name.lower().strip(): name for name in index_names}
    dl = desired.lower().strip()
    if dl in norm_index:
        return norm_index[dl]
    for alias in ALIASES.get(desired, []):
        al = alias.lower().strip()
        if al in norm_index:
            return norm_index[al]
    for k, orig in norm_index.items():
        if dl in k:
            return orig
    return None

def _invert_if_negative(metric: str, pct: float) -> float:
    for neg in NEGATIVE_KEYS:
        if neg.lower() in metric.lower():
            return 100.0 - pct
    return pct

def _blend_weights(
    base_w: Dict[str, float], 
    sub_w: Optional[Dict[str, float]],
    blend_factor: float = SUBROLE_BLEND
) -> Dict[str, float]:
    if not sub_w:
        total = sum(base_w.values()) or 1.0
        return {k: v / total for k, v in base_w.items()}
    out: Dict[str, float] = {}
    for k in set(base_w) | set(sub_w):
        b = base_w.get(k, 0.0) * (1.0 - blend_factor)
        s = sub_w.get(k, 0.0) * blend_factor
        out[k] = b + s
    total = sum(out.values()) or 1.0
    return {k: v / total for k, v in out.items()}


def _compute_pillar_scores(
    scout_df: pd.DataFrame,
    role_label: str,
) -> Dict[str, float]:
    """
    Compute scores for each pillar (attacking, technical, defensive, physical).
    Returns dict of pillar_name -> score (0-100)
    """
    base_role = role_label.split(":")[0]
    pillar_weights = PILLAR_WEIGHTS.get(base_role, PILLAR_WEIGHTS["mf"])
    
    df = scout_df.copy()
    if "Percentile" not in df.columns:
        return {}
    df["__pct__"] = pd.to_numeric(df["Percentile"], errors="coerce")
    
    index_names = list(df.index.astype(str))
    pillar_scores: Dict[str, float] = {}
    
    for pillar, metrics in PILLAR_METRICS.items():
        scores = []
        weights_sum = 0.0
        
        for metric in metrics:
            actual = _match_metric_name(index_names, metric)
            if not actual:
                continue
            pct = df.loc[actual, "__pct__"]
            if pd.isna(pct):
                continue
            pct = float(max(0.0, min(100.0, pct)))
            pct = _invert_if_negative(actual, pct)
            scores.append(pct)
            weights_sum += 1.0
        
        if scores:
            pillar_scores[pillar] = sum(scores) / len(scores)
        else:
            pillar_scores[pillar] = 0.0
    
    return pillar_scores


def _score_from_weight_map(
    scout_df: pd.DataFrame,
    weight_map: Dict[str, float],
    role_label: str,
    minutes: int = 0,
    league: str = "",
    age: Optional[int] = None,
    config: Optional[GradingConfig] = None,
) -> GradeBreakdown:
    """
    Enhanced scoring with context awareness.
    """
    if config is None:
        config = GradingConfig()
    
    df = scout_df.copy()
    validate_scout_df(df)
    df["__pct__"] = pd.to_numeric(df["Percentile"], errors="coerce")

    index_names = list(df.index.astype(str))
    matched: List[Tuple[str, float, float]] = []
    missing: List[str] = []
    score_sum = 0.0
    weight_sum = 0.0

    for desired_metric, w in weight_map.items():
        actual = _match_metric_name(index_names, desired_metric)
        if not actual:
            missing.append(desired_metric)
            continue
        pct = df.loc[actual, "__pct__"]
        if pd.isna(pct):
            missing.append(desired_metric)
            continue
        pct = float(max(0.0, min(100.0, pct)))
        pct = _invert_if_negative(actual, pct)
        contribution = w * pct
        score_sum += contribution
        weight_sum += w
        matched.append((actual, w, contribution))

    raw = (score_sum / weight_sum) if weight_sum > 0 else 0.0
    
    base_role = role_label.split(":")[0]
    league_adj, age_adj, confidence = _adjust_percentile_for_context(
        raw, minutes, league, age, base_role, config
    )
    
    # Scale to /80: 100th-percentile average → 80, elite (~90th avg) → ~72-78.
    # Confidence is kept in the breakdown for reference but does not reduce the
    # displayed grade — a small sample should show its actual per-90 level.
    final = max(0.0, min(80.0, raw * 0.8))

    return GradeBreakdown(
        role=role_label,
        matched=matched,
        missing=missing,
        raw_score=raw,
        final_score=final,
        minutes=minutes,
        confidence=confidence,
        league=league,
        age=age,
        league_adjusted_score=league_adj,
        age_adjusted_score=age_adj,
    )

def _normalize_weights(W: Dict[str, float]) -> Dict[str, float]:
    cleaned = {k: max(0.0, float(v)) for k, v in W.items()}
    total = sum(cleaned.values())
    if total <= 0:
        n = max(1, len(cleaned))
        return {k: 1.0 / n for k in cleaned}
    return {k: v / total for k, v in cleaned.items()}

def _build_role_weights(
    role_hint: str | None,
    weights: Dict[str, Dict[str, float]] | None,
    config: Optional[GradingConfig] = None
) -> Tuple[str, Dict[str, float]]:
    base, sub = _normalize_role(role_hint)
    base_w_all = (weights or DEFAULT_WEIGHTS)
    base_w = base_w_all.get(base, DEFAULT_WEIGHTS["mf"])
    sub_w = SUBROLE_WEIGHTS.get(sub) if sub in (SUBROLES_BY_BASE.get(base) or set()) else None
    blend = config.subrole_blend if config else SUBROLE_BLEND
    W = _blend_weights(base_w, sub_w, blend)
    role_label = base if sub is None else f"{base}:{sub}"
    return role_label, W

def _apply_style_deltas(
    role_label: str,
    W: Dict[str, float],
    play_style: Optional[str],
    style_strength: float = 0.5,
    config: Optional[GradingConfig] = None,
) -> Dict[str, float]:
    effective_strength = config.style_strength if config else style_strength
    if not play_style or play_style not in PLAY_STYLE_PRESETS or effective_strength <= 0:
        return W

    base = role_label.split(":")[0]
    deltas: Dict[str, float] = {}
    presets = PLAY_STYLE_PRESETS[play_style]

    for scope in ("*", base, role_label):
        d = presets.get(scope, {})
        if d:
            for k, v in d.items():
                deltas[k] = deltas.get(k, 0.0) + float(v)

    if not deltas:
        return W

    out = dict(W)
    for k, v in deltas.items():
        out[k] = out.get(k, 0.0) + effective_strength * float(v)

    return _normalize_weights(out)

# -----------------------------
# Public API
# -----------------------------
def compute_grade(
    scout_df: pd.DataFrame,
    role_hint: str | None = None,
    weights: Dict[str, Dict[str, float]] | None = None,
    minutes: int = 0,
    league: str = "",
    age: Optional[int] = None,
    config: Optional[GradingConfig] = None,
) -> GradeBreakdown:
    """
    Compute grade with optional context (minutes, league, age).
    
    Args:
        scout_df: DataFrame with 'Percentile' column and metric names as index
        role_hint: Position hint (e.g., "df:cb", "striker")
        weights: Custom weights per position
        minutes: Minutes played (for confidence weighting)
        league: League name (for tier adjustment)
        age: Player age (for potential adjustment)
        config: GradingConfig for custom parameters
    
    Returns:
        GradeBreakdown with all scoring details
    """
    role_label, W = _build_role_weights(role_hint, weights, config)
    return _score_from_weight_map(
        scout_df, W, role_label, minutes=minutes, league=league, age=age, config=config
    )

def compute_grade_with_playstyle(
    scout_df: pd.DataFrame,
    role_hint: str | None = None,
    play_style: Optional[str] = None,
    style_strength: float = 0.5,
    weights: Dict[str, Dict[str, float]] | None = None,
    minutes: int = 0,
    league: str = "",
    age: Optional[int] = None,
    config: Optional[GradingConfig] = None,
) -> GradeBreakdown:
    """
    Compute grade with play style adjustment and context.
    
    Args:
        scout_df: DataFrame with 'Percentile' column and metric names as index
        role_hint: Position hint
        play_style: One of PLAY_STYLE_PRESETS keys
        style_strength: Strength of play style adjustment (0-1)
        weights: Custom weights per position
        minutes: Minutes played
        league: League name
        age: Player age
        config: GradingConfig for custom parameters
    
    Returns:
        GradeBreakdown with play style adjustments
    """
    role_label, W = _build_role_weights(role_hint, weights, config)
    W_styled = _apply_style_deltas(role_label, W, play_style, style_strength, config)
    return _score_from_weight_map(
        scout_df, W_styled, role_label, 
        minutes=minutes, league=league, age=age, config=config
    )

def compute_grade_for_styles(
    scout_df: pd.DataFrame,
    role_hint: str | None,
    styles: Iterable[str],
    style_strength: float = 0.5,
    weights: Dict[str, Dict[str, float]] | None = None,
    minutes: int = 0,
    league: str = "",
    age: Optional[int] = None,
    config: Optional[GradingConfig] = None,
) -> Dict[str, GradeBreakdown]:
    """Compute grades for multiple play styles."""
    out: Dict[str, GradeBreakdown] = {}
    for s in styles:
        out[s] = compute_grade_with_playstyle(
            scout_df, role_hint=role_hint,
            play_style=s, style_strength=style_strength, 
            weights=weights, minutes=minutes, league=league, 
            age=age, config=config
        )
    return out

def compute_grade_for_positions(
    scout_df: pd.DataFrame,
    positions: list[tuple[str, str | None]],
    weights: Dict[str, Dict[str, float]] | None = None,
    minutes: int = 0,
    league: str = "",
    age: Optional[int] = None,
    config: Optional[GradingConfig] = None,
) -> dict[str, GradeBreakdown]:
    """Compute grades for multiple positions."""
    out: dict[str, GradeBreakdown] = {}
    for base, sub in positions:
        role_hint = f"{base}:{sub}" if sub else base
        bd = compute_grade(
            scout_df, role_hint=role_hint, weights=weights,
            minutes=minutes, league=league, age=age, config=config
        )
        out[bd.role] = bd
    return out

def compute_grade_for_positions_and_styles(
    scout_df: pd.DataFrame,
    positions: list[tuple[str, str | None]],
    styles: Iterable[str],
    style_strength: float = 0.5,
    weights: Dict[str, Dict[str, float]] | None = None,
    minutes: int = 0,
    league: str = "",
    age: Optional[int] = None,
    config: Optional[GradingConfig] = None,
) -> Dict[Tuple[str, str], GradeBreakdown]:
    """Compute grades for multiple positions and play styles."""
    out: Dict[Tuple[str, str], GradeBreakdown] = {}
    for base, sub in positions:
        role_hint = f"{base}:{sub}" if sub else base
        role_label, W = _build_role_weights(role_hint, weights, config)
        for s in styles:
            W_styled = _apply_style_deltas(role_label, W, s, style_strength, config)
            out[(role_label, s)] = _score_from_weight_map(
                scout_df, W_styled, role_label,
                minutes=minutes, league=league, age=age, config=config
            )
    return out


# -----------------------------
# Multi-Dimensional Grading API
# -----------------------------

def compute_multi_grade(
    scout_df: pd.DataFrame,
    role_hint: str | None = None,
    weights: Dict[str, Dict[str, float]] | None = None,
    minutes: int = 0,
    league: str = "",
    age: Optional[int] = None,
    config: Optional[GradingConfig] = None,
) -> MultiGrade:
    """
    Compute multi-dimensional grade with pillar scores.
    
    Returns a MultiGrade with:
    - Overall score (weighted average of pillars)
    - Individual pillar scores (attacking, technical, defensive, physical)
    - Context-aware adjustments (minutes, league, age)
    """
    if config is None:
        config = GradingConfig()
    
    # Get base role for pillar weights
    base, sub = _normalize_role(role_hint)
    base_role = base if base else "mf"
    role_label = base_role if sub is None else f"{base_role}:{sub}"
    
    # Compute pillar scores
    pillar_scores = _compute_pillar_scores(scout_df, role_label)
    
    # Get pillar weights for this role
    pillar_weights = PILLAR_WEIGHTS.get(base_role, PILLAR_WEIGHTS["mf"])
    
    # Compute weighted overall score from pillars
    weighted_pillar_sum = sum(
        score * weight for score, weight in zip(pillar_scores.values(), pillar_weights.values())
    )
    total_weight = sum(pillar_weights.values())
    overall_pillar_score = weighted_pillar_sum / total_weight if total_weight > 0 else 0.0
    
    # Also compute traditional weighted score
    role_label_final, W = _build_role_weights(role_hint, weights, config)
    traditional_bd = _score_from_weight_map(
        scout_df, W, role_label_final,
        minutes=minutes, league=league, age=age, config=config
    )
    
    # Combine scores: use traditional as base, adjust with pillars
    base_score = traditional_bd.raw_score
    
    # Adjust confidence
    confidence = traditional_bd.confidence
    
    # Apply context adjustments
    league_adj, age_adj, _ = _adjust_percentile_for_context(
        base_score, minutes, league, age, base_role, config
    )
    
    # Final scores
    final_overall = max(0.0, min(100.0, base_score * confidence))
    
    # Individual pillar scores adjusted by confidence
    final_pillars = {k: max(0.0, min(100.0, v * confidence)) for k, v in pillar_scores.items()}
    
    return MultiGrade(
        overall=final_overall,
        attacking=final_pillars.get("attacking", 0.0),
        technical=final_pillars.get("technical", 0.0),
        defensive=final_pillars.get("defensive", 0.0),
        physical=final_pillars.get("physical", 0.0),
        role=role_label_final,
        minutes=minutes,
        confidence=confidence,
        league_adjusted=league_adj,
        age_adjusted=age_adj,
        matched=traditional_bd.matched,
        missing=traditional_bd.missing,
        pillar_scores=pillar_scores,
    )

def compute_multi_grade_for_positions(
    scout_df: pd.DataFrame,
    positions: list[tuple[str, str | None]],
    minutes: int = 0,
    league: str = "",
    age: Optional[int] = None,
    config: Optional[GradingConfig] = None,
) -> dict[str, MultiGrade]:
    """Compute multi-dimensional grades for multiple positions."""
    out: dict[str, MultiGrade] = {}
    for base, sub in positions:
        role_hint = f"{base}:{sub}" if sub else base
        mg = compute_multi_grade(
            scout_df, role_hint=role_hint,
            minutes=minutes, league=league, age=age, config=config
        )
        out[mg.role] = mg
    return out

def rationale_from_breakdown(bd: GradeBreakdown, language: str = "English") -> str:
    top = sorted(bd.matched, key=lambda x: x[2], reverse=True)[:3]
    drivers = ", ".join([f"{m} ({c:.0f})" for m, w, c in top]) if top else "—"
    missing = ", ".join(bd.missing[:5]) if bd.missing else "—"
    role_disp = label_from_pair(*_normalize_role(bd.role))
    
    # Build context notes
    context_notes = []
    if bd.minutes > 0:
        conf_pct = bd.confidence * 100
        context_notes.append(f"{bd.minutes} mins, confidence: {conf_pct:.0f}%")
        if bd.minutes < 1000:
            context_notes.append("⚠️ Small sample size")
    if bd.league:
        tier = _get_league_tier(bd.league)
        context_notes.append(f"League: {bd.league} ({tier:.0%})")
    if bd.age:
        context_notes.append(f"Age: {bd.age}")
    
    context_str = " | ".join(context_notes) if context_notes else ""
    
    if (language or "").lower().startswith("fr"):
        return (
            f"Note déterministe: **{bd.final_score:.1f}/100** (rôle: {role_disp}).\n"
            f"{context_str}\n\n" if context_str else "\n"
            f"Principaux moteurs: {drivers}.\n\n"
            f"Points à améliorer/absents: {missing}.\n\n"
            f"Pondérations: poste général ⊕ poste spécifique; métriques « négatives » inversées."
        )
    else:
        return (
            f"Deterministic grade: **{bd.final_score:.1f}/100** (role: {role_disp}).\n"
            f"{context_str}\n\n" if context_str else "\n"
            f"Top drivers: {drivers}.\n\n"
            f"Key missing/low-weighted signals: {missing}.\n\n"
            f"Weights blend base and subrole; negative metrics are inverted."
        )


def rationale_from_multi_grade(mg: MultiGrade, language: str = "English") -> str:
    """Generate human-readable rationale from MultiGrade."""
    role_disp = label_from_pair(*_normalize_role(mg.role))
    
    # Context
    context_parts = []
    if mg.minutes > 0:
        context_parts.append(f"{mg.minutes} mins")
        context_parts.append(f"confidence: {mg.confidence:.0%}")
    # League adjusted score is stored in the MultiGrade
    if mg.league_adjusted > 0:
        context_parts.append(f"league-adjusted: {mg.league_adjusted:.1f}")
    if mg.age_adjusted > 0:
        context_parts.append(f"age-adjusted: {mg.age_adjusted:.1f}")
    context_str = ", ".join(context_parts) if context_parts else ""
    
    # Pillar scores
    pillars = [
        ("Attacking", mg.attacking),
        ("Technical", mg.technical),
        ("Defensive", mg.defensive),
        ("Physical", mg.physical),
    ]
    pillar_str = ", ".join(f"{name}: {score:.0f}" for name, score in pillars if score > 0)
    
    # Letter grade
    letter = percentile_to_letter(mg.overall)
    
    if (language or "").lower().startswith("fr"):
        return (
            f"Note multi-dimensionnelle: **{mg.overall:.1f}/100 [{letter}]** (rôle: {role_disp}).\n"
            f"Contexte: {context_str}\n\n" if context_str else "\n"
            f"Piliers: {pillar_str}\n\n"
            f"Note globale pondérée par les minutes et le contexte."
        )
    else:
        return (
            f"Multi-dimensional grade: **{mg.overall:.1f}/100 [{letter}]** (role: {role_disp}).\n"
            f"Context: {context_str}\n\n" if context_str else "\n"
            f"Pillars: {pillar_str}\n\n"
            f"Overall score weighted by minutes and context adjustments."
        )


def percentile_to_letter(percentile: float) -> str:
    """Convert numeric percentile to letter grade."""
    if percentile >= 95: return "A+"
    if percentile >= 90: return "A"
    if percentile >= 85: return "A-"
    if percentile >= 80: return "B+"
    if percentile >= 75: return "B"
    if percentile >= 70: return "B-"
    if percentile >= 65: return "C+"
    if percentile >= 60: return "C"
    if percentile >= 50: return "C-"
    if percentile >= 40: return "D+"
    if percentile >= 30: return "D"
    return "F"


def percentile_to_tier(percentile: float) -> str:
    """Convert numeric percentile to tier classification."""
    if percentile >= 90: return "Elite"
    if percentile >= 80: return "World Class"
    if percentile >= 70: return "Starter"
    if percentile >= 60: return "Rotation"
    if percentile >= 50: return "Squad"
    if percentile >= 40: return "Project"
    return "Limited"

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
    if raw is None:
        return []
    if isinstance(raw, list):
        text = " ".join(str(x) for x in raw)
    else:
        text = str(raw)
    text = re.sub(r"[▪•·\(\)\[\]]", " ", text)
    text = text.replace("/", " ").replace("|", " ").replace(",", " ").replace(";", " ").replace("-", " ")
    tokens = [t.strip().lower() for t in text.split() if t.strip()]

    found: list[tuple[str, str | None]] = []
    seen = set()
    joined = " ".join(tokens)
    for alias, (base, sub) in ROLE_ALIASES.items():
        if f" {alias} " in f" {joined} ":
            key = (base, sub)
            if key not in seen:
                found.append(key); seen.add(key)
    for t in tokens:
        if t in ROLE_ALIASES:
            base, sub = ROLE_ALIASES[t]
            key = (base, sub)
            if key not in seen:
                found.append(key); seen.add(key)
    if not found:
        base, sub = _normalize_role(joined)
        found.append((base, sub))
    return found

def label_from_pair(base: str, sub: str | None) -> str:
    key = f"{base}:{sub}" if sub else base
    return ROLE_PRETTY.get(key, key.upper())
