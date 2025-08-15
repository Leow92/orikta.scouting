# tools/compare.py

from __future__ import annotations
from typing import Tuple, List, Dict, Optional
import json
import math
import pandas as pd
import requests
from requests.exceptions import ReadTimeout
import numpy as np

# --- Your project helpers ---
from utils.resolve_player_url import search_fbref_url_with_playwright
from utils.fbref_scraper import scrape_all_tables, scrape_player_profile
from tools.analyze import (
    _to_numeric_safely, _prefer_stats_standard_key, _clean_standard_df,
    build_trend_block_for_llm, _t, _nodata
)
from tools.grading import (
    compute_grade, label_from_pair,
    DEFAULT_WEIGHTS, SUBROLE_WEIGHTS, SUBROLE_BLEND,
    NEGATIVE_KEYS, ALIASES,
    PLAY_STYLE_PRESETS, PLAY_STYLE_PRETTY
)
from utils.llm_analysis_comparison import compare_llm_workflow
from utils.lang import _is_fr, _lang_block, _glossary_block_for

# --- LLM basics (same as your single-player workflow) ---
OLLAMA_API_URL = "http://localhost:11434/api/chat"
BASE_OPTS = {
    "temperature": 0.15,
    "top_p": 0.9,
    "repeat_penalty": 1.05,
    "num_ctx": 2048,
    "num_predict": 800,
}


# -----------------------------
# Deterministic helpers
# -----------------------------
def _invert_if_negative(metric: str, pct: float) -> float:
    for neg in NEGATIVE_KEYS:
        if neg.lower() in metric.lower():
            return 100.0 - pct
    return pct

def _blend_role_weights(base_w: Dict[str, float], sub_w: Optional[Dict[str, float]]) -> Dict[str, float]:
    if not sub_w:
        total = sum(base_w.values()) or 1.0
        return {k: v/total for k, v in base_w.items()}
    out: Dict[str, float] = {}
    for k in set(base_w) | set(sub_w):
        out[k] = base_w.get(k,0.0)*(1-SUBROLE_BLEND) + sub_w.get(k,0.0)*SUBROLE_BLEND
    tot = sum(out.values()) or 1.0
    return {k: v/tot for k, v in out.items()}

def _style_reweight(
    W_role: Dict[str, float],
    style_key: Optional[str],
    style_influence: float,
    base: Optional[str] = None,
    sub: Optional[str] = None,
) -> Dict[str, float]:
    """
    Blend role weights with style deltas: W' = normalize( W_role + style_influence * delta ).
    All keys are lower-cased; non-numeric leaves are ignored.
    """
    Wb = _normalize_weight_keys(W_role)
    if not style_key:
        return Wb

    delta = _resolve_style_preset(style_key, base, sub)
    delta = _normalize_weight_keys(delta)

    keys = set(Wb) | set(delta)
    out = {k: max(0.0, Wb.get(k, 0.0) + style_influence * delta.get(k, 0.0)) for k in keys}

    tot = sum(out.values()) or 1.0
    return {k: v / tot for k, v in out.items()}


def _map_index_aliases(idx: List[str]) -> Dict[str, str]:
    """Return mapping canonical_key(lower) -> actual index name in df, using ALIASES and exact/lower matches."""
    norm = {name.lower(): name for name in idx}
    for canonical, aliases in ALIASES.items():
        for al in aliases:
            if al.lower() in norm:
                norm[canonical.lower()] = norm[al.lower()]
    # Allow canonical labels present exactly
    for canonical in ALIASES.keys():
        if canonical.lower() in norm:
            norm[canonical.lower()] = norm[canonical.lower()]
    return norm

def _align_metrics(dfA: pd.DataFrame, dfB: pd.DataFrame) -> Tuple[pd.Series, pd.Series, List[str]]:
    """
    Return aligned Percentile series (A,B) over common canonical metrics.
    - Deduplicates exact duplicate labels by mean().
    - Resolves aliases to canonical keys defined in ALIASES.
    """
    # Percentiles (ensure scalar dtype) + dedupe exact duplicates by mean
    sA = pd.to_numeric(dfA["Percentile"], errors="coerce")
    sB = pd.to_numeric(dfB["Percentile"], errors="coerce")
    sA.index = dfA.index.astype(str)
    sB.index = dfB.index.astype(str)
    sA = sA.groupby(level=0).mean()  # collapse duplicate labels → scalar
    sB = sB.groupby(level=0).mean()

    # Build alias maps: canonical(lower) -> actual label present in series
    def _map_index_aliases_present(idx: List[str]) -> Dict[str, str]:
        present = {name.lower(): name for name in idx}
        out: Dict[str, str] = {}
        # prefer canonical keys if present; else first alias seen
        for canonical, aliases in ALIASES.items():
            cl = canonical.lower()
            if cl in present:
                out[cl] = present[cl]
                continue
            for al in aliases:
                al_l = al.lower()
                if al_l in present:
                    out[cl] = present[al_l]
                    break
        # also keep any labels that are not in ALIASES as-is (self-canonical)
        for name in idx:
            nl = name.lower()
            out.setdefault(nl, present[nl])
        return out

    mapA = _map_index_aliases_present(sA.index.tolist())
    mapB = _map_index_aliases_present(sB.index.tolist())

    # Common canonical keys
    common = sorted(set(mapA.keys()) & set(mapB.keys()))
    if not common:
        return pd.Series(dtype=float), pd.Series(dtype=float), []

    # Build aligned series using canonical index
    common = sorted(set(mapA.keys()) & set(mapB.keys()))
    A_vals, B_vals = {}, {}
    for k in common:
        vA = sA.get(mapA[k], np.nan)
        vB = sB.get(mapB[k], np.nan)
        # vA/vB are scalars after groupby; still guard just in case
        if isinstance(vA, pd.Series):
            vA = float(np.nanmean(vA.values))
        if isinstance(vB, pd.Series):
            vB = float(np.nanmean(vB.values))
        canon = str(k).lower()
        A_vals[canon] = float(vA) if pd.notna(vA) else np.nan
        B_vals[canon] = float(vB) if pd.notna(vB) else np.nan


    A = pd.Series(A_vals, name="A", dtype=float).dropna()
    B = pd.Series(B_vals, name="B", dtype=float).dropna()
    # Final common index after dropping NaNs
    common_idx = sorted(set(A.index) & set(B.index))
    return A.reindex(common_idx), B.reindex(common_idx), common_idx

def _head_to_head_score(A: pd.Series, B: pd.Series, W: Dict[str,float]) -> Tuple[float,float, pd.DataFrame]:
    rows = []
    for canon in A.index:
        w = float(W.get(canon, 0.0))
        a_raw = A.loc[canon]
        b_raw = B.loc[canon]
        # Guard if anything weird slips in
        if isinstance(a_raw, pd.Series):
            a_raw = float(np.nanmean(a_raw.values))
        if isinstance(b_raw, pd.Series):
            b_raw = float(np.nanmean(b_raw.values))
        a = _invert_if_negative(canon, float(a_raw))
        b = _invert_if_negative(canon, float(b_raw))
        rows.append((canon, w, a, b, w*a, w*b, a-b))
    det = pd.DataFrame(rows, columns=["Metric","w","A_pct","B_pct","A_contrib","B_contrib","Δp"])
    wsum = det["w"].sum() or 1.0
    scoreA = det["A_contrib"].sum() / wsum
    scoreB = det["B_contrib"].sum() / wsum
    return float(scoreA), float(scoreB), det.sort_values("w", ascending=False)

def _cosine_similarity(A: pd.Series, B: pd.Series) -> float:
    """Cosine similarity of aligned percentile vectors (0..100). Returns 0..100 scale."""
    a = A.values; b = B.values
    na = math.sqrt((a*a).sum()); nb = math.sqrt((b*b).sum())
    if na == 0 or nb == 0:
        return 0.0
    sim = float((a*b).sum() / (na*nb))
    return max(0.0, min(1.0, (sim + 1) / 2.0)) * 100.0  # map [-1,1] -> [0,1] then to 0..100

def _normalize_weight_keys(W: Dict[str, float]) -> Dict[str, float]:
    """Lower-case keys and keep only numeric weights."""
    out: Dict[str, float] = {}
    for k, v in (W or {}).items():
        if isinstance(v, (int, float)):
            out[str(k).lower()] = float(v)
    return out

def _resolve_style_preset(style_key: Optional[str], base: Optional[str], sub: Optional[str]) -> Dict[str, float]:
    """
    Turn PLAY_STYLE_PRESETS[style_key] into a flat {metric_lower: delta_float}.
    Supports:
      - flat dict of numeric deltas
      - role-scoped dicts: "_all", "<base>", "<base:sub>"
    Silently ignores non-numeric leaves.
    """
    P = PLAY_STYLE_PRESETS.get(style_key, {}) if style_key else {}
    flat: Dict[str, float] = {}

    if not isinstance(P, dict):
        return flat

    # 1) accept any top-level numeric entries
    for k, v in P.items():
        if isinstance(v, (int, float)):
            flat[str(k).lower()] = flat.get(str(k).lower(), 0.0) + float(v)

    # 2) role-scoped dicts
    role_layers = []
    if "_all" in P and isinstance(P["_all"], dict):
        role_layers.append(P["_all"])
    if base and base in P and isinstance(P[base], dict):
        role_layers.append(P[base])
    if sub and f"{base}:{sub}" in P and isinstance(P[f"{base}:{sub}"], dict):
        role_layers.append(P[f"{base}:{sub}"])

    for layer in role_layers:
        for k, v in layer.items():
            if isinstance(v, (int, float)):
                flat[str(k).lower()] = flat.get(str(k).lower(), 0.0) + float(v)

    return flat


# -----------------------------
# Fetch & normalize one player
# -----------------------------
def _fetch_player(name: str, language: str = "English"):
    """
    Returns dict with:
      full_name, url, profile, scout_df (Metric/Per90/Percentile), std_df_raw, trend_block_md
    """
    url = search_fbref_url_with_playwright(name)
    if not url:
        raise RuntimeError(f"Could not resolve FBref page for: {name}")
    profile = scrape_player_profile(url)
    full_name = profile.get("name") or name.title()

    tables = scrape_all_tables(url)
    scout_key = next((k for k in tables.keys() if k.startswith("scout_summary")), None)
    if not scout_key:
        raise RuntimeError(f"No scouting table for: {full_name}")

    scout_df = tables[scout_key].copy()
    scout_df.columns = ["Metric","Per90","Percentile"][:len(scout_df.columns)]
    scout_df.set_index("Metric", inplace=True)

    # Numeric safety + clip to 0..100
    for col in ["Per90","Percentile"]:
        scout_df[col] = _to_numeric_safely(scout_df[col])
    scout_df["Percentile"] = pd.to_numeric(scout_df["Percentile"], errors="coerce").astype(float)
    scout_df["Percentile"] = scout_df["Percentile"].clip(lower=0.0, upper=100.0)

    # Standard stats raw (for trends)
    standard_keys = [k for k in tables.keys() if k.startswith("stats_standard")]
    std_df_raw = None
    if standard_keys:
        chosen = _prefer_stats_standard_key(standard_keys)
        std_df_raw = tables.get(chosen)

    trend_block_md, trend_debug = build_trend_block_for_llm(std_df_raw, language) if isinstance(std_df_raw, pd.DataFrame) else (_nodata(language), {})

    return {
        "full_name": full_name,
        "url": url,
        "profile": profile,
        "scout_df": scout_df,
        "std_df_raw": std_df_raw,
        "trend_block_md": trend_block_md,
        "scout_key": scout_key
    }

# -----------------------------
# LLM streaming
# -----------------------------
def _ollama_stream(user_content: str, language: str) -> str:
    payload = {
        "model": "gemma3",
        "messages": [
            {"role": "system", "content": _lang_block(language)},   # <-- language enforced here
            {"role": "user",   "content": user_content},
        ],
        "stream": True,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.15,
            "top_p": 0.9,
            "repeat_penalty": 1.05,
            "num_ctx": 2048,
            "num_predict": 800,
        },
    }
    chunks: list[str] = []
    with requests.post(OLLAMA_API_URL, json=payload, timeout=300, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if isinstance(ev, dict) and ev.get("error"):
                raise RuntimeError(ev["error"])
            msg = ev.get("message", {})
            if isinstance(msg, dict) and "content" in msg:
                chunks.append(msg["content"])
            if ev.get("done"):
                break
    return "".join(chunks).strip()

def _call_twice(prompt_text: str, language: str) -> str:
    try:
        return _ollama_stream(prompt_text, language)   # <-- pass language from outer scope
    except ReadTimeout:
        return _ollama_stream(prompt_text, language)



# -----------------------------
# Public API
# -----------------------------
def compare_players(
    players: list[str],
    language: str = "English",
    target_role: str | None = None,     # e.g., "df:cb"
    styles: list[str] | None = None,    # optional styles to display in rows
    style_influence: float = 0.6,
    skip_llm: bool = False,
) -> str:
    """
    Head-to-head report between two players:
      - Align metrics; compute role- & style-aware scores
      - Style head-to-head mini table
      - Trend contrast (programmatic)
      - Optional LLM comparison workflow (4 short prompts)
    Returns markdown string; never raises (returns error text on failure).
    """
    try:
        assert isinstance(players, list) and len(players) == 2
        A_raw, B_raw = players[0], players[1]

        # 1) Fetch & normalize
        A = _fetch_player(A_raw, language=language)
        B = _fetch_player(B_raw, language=language)

        A_name, B_name = A["full_name"], B["full_name"]
        scoutA, scoutB = A["scout_df"], B["scout_df"]

        # 2) Align percentiles
        A_p, B_p, common_idx = _align_metrics(scoutA, scoutB)
        if len(common_idx) == 0:
            return _t("⚠️ No common metrics to compare.", "⚠️ Aucune métrique commune à comparer.", language)

        # 3) Target role (infer from A if not provided)
        if not target_role:
            bdA = compute_grade(scoutA, role_hint=A.get("scout_key"))
            target_role = bdA.role  # e.g., "mf:am"
        base = target_role.split(":")[0]
        sub  = target_role.split(":")[1] if ":" in target_role else None
        role_label = label_from_pair(base, sub)

        # 4) Build weights (role → optional style reweight)
        base_w = DEFAULT_WEIGHTS.get(base, DEFAULT_WEIGHTS["mf"])
        sub_w  = SUBROLE_WEIGHTS.get(sub) if sub else None
        W_role = _blend_role_weights(base_w, sub_w)
        W_role = _normalize_weight_keys(W_role)  # <<< add this


        # 5) Head-to-head scoring for a "primary style row" preview (first style or None)
        styles = styles or list(PLAY_STYLE_PRESETS.keys())
        style_rows: List[str] = []
        best_style_label = "—"
        best_abs_margin = -1.0  # any real edge will beat this

        for s in styles:
            W = _style_reweight(W_role, s, style_influence, base=base, sub=sub)
            scoreA, scoreB, det = _head_to_head_score(A_p, B_p, W)
            edge = scoreA - scoreB
            s_label = PLAY_STYLE_PRETTY.get(s, s)
            winner = A_name if edge >= 0 else B_name
            row = f"| {s_label} | {scoreA:.1f} | {scoreB:.1f} | **{winner}** |"
            style_rows.append(row)
            if abs(edge) > best_abs_margin:
                best_abs_margin = abs(edge)
                best_style_label = s_label


        # 6) Build aligned differences table (role-weighted ordering)
        W_default = _style_reweight(W_role, None, 0.0)  # pure role for ordering
        details = []
        for m in common_idx:
            w = float(W_default.get(m, 0.0))
            a = float(A_p[m]); b = float(B_p[m])
            dp = a - b
            impact = abs(w * dp)
            details.append((m, w, a, b, dp, impact))
        det_df = pd.DataFrame(details, columns=["Metric","w","A_pct","B_pct","Δp","impact"]).sort_values(
            ["w","impact"], ascending=[False, False]
        )

        # Use player names in the markdown header
        diff_rows = [f"| Metric | w | {A_name} (p) | {B_name} (p) | Δp ({A_name}−{B_name}) |",
                    "|---|---:|---:|---:|---:|"]
        for _, r in det_df.head(12).iterrows():
            diff_rows.append(f"| {r['Metric']} | {r['w']:.2f} | {r['A_pct']:.0f} | {r['B_pct']:.0f} | {r['Δp']:.0f} |")
        aligned_diff_md = "\n".join(diff_rows)

        # 7) Percentiles-only tables for LLM (each)
        scout_pct_A = pd.DataFrame({"Percentile": A_p}).to_markdown(tablefmt="pipe", index=True)
        scout_pct_B = pd.DataFrame({"Percentile": B_p}).to_markdown(tablefmt="pipe", index=True)

        # 8) Style head-to-head mini-table (rows)
        style_header = f"| {_t('Style','Style',language)} | {A_name} (p/100) | {B_name} (p/100) | {_t('Winner','Vainqueur',language)} |"
        style_section_title = _t("#### Style head-to-head (summary)", "#### Duel de styles (résumé)", language)
        style_sep    = "|---|---:|---:|---|"
        style_rows_md = "\n".join([style_header, style_sep] + style_rows)

        # 9) Similarity (cosine, 0..100)
        similarity = _cosine_similarity(A_p, B_p)

        # 10) Assemble deterministic header
        title = f"### 🆚 Head-to-Head — {A_name} vs {B_name}"
        role_line = _t("**Target role**", "**Poste cible**", language) + f": **{role_label}**"
        style_line = _t("**Best style context**", "**Meilleur style**", language) + f": **{best_style_label}**  · " + \
                     _t("style influence", "influence style", language) + f" = {style_influence:.2f}"
        sim_line = _t("**Profile similarity**", "**Similarité de profil**", language) + f": {similarity:.1f}/100"

        deterministic_md = f"""{title}

{role_line}  
{style_line}  
{sim_line}

{style_section_title}
{style_rows_md}

#### {_t('Key role-critical differences (top 12 by role weight × gap)','Différences clés (top 12 par poids × écart)', language)}
{aligned_diff_md}
"""

        # Build glossary here (keeps utils module pure)
        present_metrics = []
        for md in (scout_pct_A + "\n" + scout_pct_B).splitlines():
            if md.startswith("|") and not md.startswith("|---"):
                parts = [p.strip() for p in md.strip("|").split("|")]
                if parts:
                    present_metrics.append(parts[0])
        glossary_block = _glossary_block_for(language, list(dict.fromkeys(present_metrics)))

        # Respect Fast Preview
        if skip_llm:
            deterministic_md += "\n\n> ⚡ **Fast preview:** LLM analysis skipped."
            return deterministic_md

        # LLM comparison (inject call function + glossary)
        compare_llm_md = compare_llm_workflow(
            A_name=A_name, B_name=B_name,
            language=language,
            role_label=role_label,
            style_label=best_style_label,
            style_influence=style_influence,
            scout_md_A=scout_pct_A, scout_md_B=scout_pct_B,
            trend_md_A=A["trend_block_md"], trend_md_B=B["trend_block_md"],
            style_rows_md=style_rows_md,
            aligned_diff_md=aligned_diff_md,
            similarity_0_100=similarity,
            glossary_block=glossary_block,
            call_fn=_call_twice,     # <- inject your function that wraps system-lang + retry
        )

        print("✅ Report Generation Done.")
        
        return deterministic_md + "\n\n---\n\n" + compare_llm_md

    except Exception as e:
        return f"⚠️ Compare failed: {e}"
