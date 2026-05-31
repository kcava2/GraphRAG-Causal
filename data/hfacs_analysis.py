"""
HFACS Extraction Results — Data Analysis (Stage 2 output)
=========================================================
Analyzes the new ``hfacs_results.csv`` produced by hfacs_extractor.py, whose
columns are:

    ev_id, entities_json, hfacs_json, relationships_json, extraction_status

``hfacs_json`` is a JSON object ``{tier: [subcategory, ...]}`` validated against
the 15-tier ``HFACS_SCHEMA``. ``relationships_json`` is a JSON list of
``{subject, relation, object, evidence}`` directed causal edges. This script
imports ``HFACS_SCHEMA`` directly from hfacs_extractor (single source of truth).

Outputs (to figures/):
  hfacs_distributions.png  — subcategory frequency, one panel per tier
  hfacs_cooccurrence.png   — pairwise subcategory co-occurrence (observed subs)
  hfacs_combinations.png   — top-15 tier combinations per record
  hfacs_coverage.png       — per-tier coverage + tiers-per-row + status counts
  hfacs_relationships.png  — relation-type counts + top causal edges
  hfacs_severity.png       — severity distribution (joined from ntsb_clean.csv)
  Console: full numeric summary

Usage:
    python data/hfacs_analysis.py
    python data/hfacs_analysis.py --input data/hfacs_results.csv
"""

import argparse
import json
import os
import sys
import traceback
from collections import Counter
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Import the schema from the extractor (same directory) — single source of truth.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from hfacs_extractor import HFACS_SCHEMA, VALID_RELATIONS  # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TIER_ORDER: list[str] = list(HFACS_SCHEMA.keys())
ALL_SUBS: list[str] = [s for subs in HFACS_SCHEMA.values() for s in subs]
SUB_TO_TIER: dict[str, str] = {s: t for t, subs in HFACS_SCHEMA.items() for s in subs}

TIER_LABELS = {
    "org_climate":         "Organizational Climate",
    "resource_mgmt":       "Resource Management",
    "org_process":         "Organizational Process",
    "supervisory":         "Supervisory",
    "situational_phys":    "Situational — Physical Env",
    "situational_tech":    "Situational — Technological",
    "operator_mental":     "Operator — Mental State",
    "operator_physical":   "Operator — Physiological",
    "operator_limits":     "Operator — Limitations",
    "personnel_crm":       "Personnel — CRM",
    "personnel_readiness": "Personnel — Readiness",
    "unsafe_skill":        "Unsafe Acts — Skill",
    "unsafe_decision":     "Unsafe Acts — Decision",
    "unsafe_perception":   "Unsafe Acts — Perception",
    "unsafe_violation":    "Unsafe Acts — Violation",
}

# A distinct colour per tier; subcategories inherit their tier colour.
_TIER_COLORS = {
    "org_climate": "#5A189A", "resource_mgmt": "#7B2CBF", "org_process": "#9D4EDD",
    "supervisory": "#3A0CA3", "situational_phys": "#4361EE",
    "situational_tech": "#4895EF", "operator_mental": "#DD8452",
    "operator_physical": "#C44E52", "operator_limits": "#E76F51",
    "personnel_crm": "#8338EC", "personnel_readiness": "#3A86FF",
    "unsafe_skill": "#2D6A4F", "unsafe_decision": "#55A868",
    "unsafe_perception": "#95D5B2", "unsafe_violation": "#E9C46A",
}


# ---------------------------------------------------------------------------
# Load & parse
# ---------------------------------------------------------------------------

def _parse_json(cell, default):
    try:
        v = json.loads(cell) if isinstance(cell, str) and cell.strip() else default
        return v if v is not None else default
    except (json.JSONDecodeError, TypeError):
        return default


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    required = {"ev_id", "hfacs_json", "relationships_json", "extraction_status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in {path}: {sorted(missing)}")

    df["hfacs"] = df["hfacs_json"].apply(lambda c: _parse_json(c, {}))
    df["relationships"] = df["relationships_json"].apply(lambda c: _parse_json(c, []))
    if "entities_json" in df.columns:
        df["entities"] = df["entities_json"].apply(lambda c: _parse_json(c, []))
    else:
        df["entities"] = [[] for _ in range(len(df))]
    return df


def sub_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Binary (rows × subcategory) presence matrix from the hfacs dicts."""
    data = np.zeros((len(df), len(ALL_SUBS)), dtype=int)
    col_idx = {s: i for i, s in enumerate(ALL_SUBS)}
    for r, hfacs in enumerate(df["hfacs"]):
        if not isinstance(hfacs, dict):
            continue
        for tier, subs in hfacs.items():
            if not isinstance(subs, list):
                continue
            for s in subs:
                if s in col_idx:
                    data[r, col_idx[s]] = 1
    return pd.DataFrame(data, columns=ALL_SUBS)


# ---------------------------------------------------------------------------
# 1. Subcategory distribution — one panel per tier
# ---------------------------------------------------------------------------

def plot_distributions(df: pd.DataFrame, mat: pd.DataFrame, out_dir: Path) -> None:
    total = max(len(df), 1)
    fig, axes = plt.subplots(4, 4, figsize=(22, 16))
    fig.suptitle("HFACS Subcategory Frequency by Tier", fontsize=16,
                 fontweight="bold", y=1.00)
    flat = axes.flatten()

    for ax, tier in zip(flat, TIER_ORDER):
        subs = HFACS_SCHEMA[tier]
        counts = [int(mat[s].sum()) for s in subs]
        color = _TIER_COLORS.get(tier, "#999999")
        bars = ax.barh(subs[::-1], counts[::-1], color=color,
                       edgecolor="white", linewidth=0.5)
        max_c = max(counts) if counts else 1
        for bar, cnt in zip(bars, counts[::-1]):
            ax.text(bar.get_width() + max_c * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{cnt:,} ({100*cnt/total:.1f}%)",
                    va="center", ha="left", fontsize=8)
        ax.set_title(TIER_LABELS.get(tier, tier), fontsize=11, fontweight="bold")
        ax.set_xlim(0, max_c * 1.4 if max_c else 1)
        ax.tick_params(axis="y", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in flat[len(TIER_ORDER):]:
        ax.set_visible(False)

    plt.tight_layout()
    out = out_dir / "hfacs_distributions.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# 2. Pairwise co-occurrence among OBSERVED subcategories
# ---------------------------------------------------------------------------

def plot_cooccurrence(mat: pd.DataFrame, out_dir: Path) -> None:
    observed = [s for s in ALL_SUBS if mat[s].sum() > 0]
    if len(observed) < 2:
        print("  skip co-occurrence: fewer than 2 observed subcategories")
        return
    sub = mat[observed].to_numpy()
    co = sub.T @ sub

    fig, ax = plt.subplots(figsize=(max(10, len(observed) * 0.5),
                                    max(8, len(observed) * 0.5)))
    im = ax.imshow(co, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(observed)))
    ax.set_yticks(range(len(observed)))
    ax.set_xticklabels(observed, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(observed, fontsize=7)
    ax.set_title("HFACS Pairwise Subcategory Co-occurrence (counts)",
                 fontsize=13, fontweight="bold")

    vmax = co.max() if co.size else 1
    for i in range(len(observed)):
        for j in range(len(observed)):
            v = co[i, j]
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=6,
                        color="white" if v > vmax * 0.6 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out = out_dir / "hfacs_cooccurrence.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# 3. Top tier combinations per record
# ---------------------------------------------------------------------------

def _tier_combo(hfacs: dict) -> str:
    if not isinstance(hfacs, dict):
        return "(none)"
    tiers = [TIER_LABELS.get(t, t) for t in TIER_ORDER if hfacs.get(t)]
    return " + ".join(tiers) if tiers else "(none)"


def plot_combinations(df: pd.DataFrame, out_dir: Path, top_n: int = 15) -> None:
    combos = df["hfacs"].apply(_tier_combo)
    counts = combos.value_counts().head(top_n)
    total = max(len(df), 1)

    fig, ax = plt.subplots(figsize=(16, 8))
    bars = ax.barh(counts.index[::-1], counts.values[::-1],
                   color="#4C72B0", edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, counts.values[::-1]):
        ax.text(bar.get_width() + counts.max() * 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,} ({100*val/total:.1f}%)",
                va="center", ha="left", fontsize=9)
    ax.set_title(f"Top {top_n} HFACS Tier Combinations per Record",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Count")
    ax.set_xlim(0, counts.max() * 1.3 if len(counts) else 1)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = out_dir / "hfacs_combinations.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# 4. Coverage + tiers-per-row + extraction status
# ---------------------------------------------------------------------------

def plot_coverage(df: pd.DataFrame, out_dir: Path) -> None:
    total = max(len(df), 1)
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle("Extraction Coverage Overview", fontsize=14, fontweight="bold")

    # Left: per-tier "% rows with >=1 subcategory"
    ax = axes[0]
    labels = [TIER_LABELS.get(t, t) for t in TIER_ORDER]
    has = [100 * df["hfacs"].apply(lambda h: bool(h.get(t))).mean()
           for t in TIER_ORDER]
    y = np.arange(len(labels))
    ax.barh(y, has, color="#55A868")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("% of records with ≥1 subcategory")
    ax.set_xlim(0, 100)
    ax.set_title("Per-tier coverage")
    for i, v in enumerate(has):
        ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=8)

    # Middle: tiers-per-row histogram
    ax2 = axes[1]
    n_tiers = df["hfacs"].apply(lambda h: sum(1 for t in TIER_ORDER if h.get(t)))
    bins = np.arange(0, len(TIER_ORDER) + 2) - 0.5
    ax2.hist(n_tiers, bins=bins, color="#4C72B0", edgecolor="white")
    ax2.axvline(n_tiers.mean(), color="#C44E52", linestyle="--", lw=1.5,
                label=f"mean = {n_tiers.mean():.2f}")
    ax2.set_xlabel("Number of tiers per record")
    ax2.set_ylabel("Record count")
    ax2.set_title("Tiers-per-record distribution")
    ax2.legend(fontsize=9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Right: extraction status
    ax3 = axes[2]
    status = df["extraction_status"].value_counts()
    colors = {"success": "#55A868", "empty": "#E9C46A", "parse_error": "#C44E52"}
    bars = ax3.bar(status.index, status.values,
                   color=[colors.get(s, "#999999") for s in status.index],
                   edgecolor="white")
    for bar, v in zip(bars, status.values):
        ax3.text(bar.get_x() + bar.get_width() / 2, v + total * 0.005,
                 f"{v:,}\n({100*v/total:.1f}%)", ha="center", va="bottom",
                 fontsize=9)
    ax3.set_title("Extraction status")
    ax3.set_ylabel("Record count")
    ax3.set_ylim(0, status.max() * 1.18 if len(status) else 1)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    plt.tight_layout()
    out = out_dir / "hfacs_coverage.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# 5. Relationship analysis
# ---------------------------------------------------------------------------

def _all_relationships(df: pd.DataFrame) -> list[dict]:
    rels = []
    for lst in df["relationships"]:
        if isinstance(lst, list):
            rels.extend(r for r in lst if isinstance(r, dict))
    return rels


def plot_relationships(df: pd.DataFrame, out_dir: Path, top_n: int = 20) -> None:
    rels = _all_relationships(df)
    if not rels:
        print("  skip relationships: none extracted")
        return

    rel_types = Counter(r.get("relation") for r in rels)
    edges = Counter(
        f"{r.get('subject')}  →  {r.get('object')}"
        for r in rels
        if r.get("relation") == "LEADS_TO"
    )
    top_edges = edges.most_common(top_n)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8),
                             gridspec_kw={"width_ratios": [1, 2]})
    fig.suptitle("HFACS Causal Relationship Extraction", fontsize=14,
                 fontweight="bold")

    ax = axes[0]
    rt_labels = list(rel_types.keys())
    rt_vals = [rel_types[k] for k in rt_labels]
    ax.bar(rt_labels, rt_vals, color=["#2D6A4F", "#4895EF"][:len(rt_labels)],
           edgecolor="white")
    for i, v in enumerate(rt_vals):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=10)
    ax.set_title(f"Relation types (n={len(rels):,})")
    ax.set_ylabel("Count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax2 = axes[1]
    if top_edges:
        labels = [e for e, _ in top_edges][::-1]
        vals = [c for _, c in top_edges][::-1]
        ax2.barh(labels, vals, color="#55A868", edgecolor="white")
        for i, v in enumerate(vals):
            ax2.text(v, i, f" {v}", va="center", fontsize=8)
        ax2.set_title(f"Top {len(top_edges)} LEADS_TO edges")
        ax2.tick_params(axis="y", labelsize=7)
    else:
        ax2.text(0.5, 0.5, "No LEADS_TO edges", ha="center", va="center")
        ax2.set_axis_off()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    out = out_dir / "hfacs_relationships.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# 6. Severity distribution (joined from ntsb_clean.csv on ev_id)
# ---------------------------------------------------------------------------

SEVERITY_LABELS = {0: "0 - Incident", 1: "1 - No injury", 2: "2 - 1 injury",
                   3: "3 - 2–3 injuries", 4: "4 - 4+ / fatal"}


def plot_severity(df: pd.DataFrame, out_dir: Path) -> None:
    clean_path = Path(_HERE) / "ntsb_clean.csv"
    if not clean_path.exists():
        print("  skip severity: ntsb_clean.csv not found")
        return
    clean = pd.read_csv(clean_path, dtype=str)[["ev_id", "severity_class"]]
    merged = df.merge(clean, on="ev_id", how="left")
    sev = pd.to_numeric(merged["severity_class"], errors="coerce").dropna().astype(int)
    if sev.empty:
        print("  skip severity: no severity_class values joined")
        return
    counts = sev.value_counts().sort_index()
    total = max(len(sev), 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    levels = sorted(counts.index.tolist())
    labels = [SEVERITY_LABELS.get(int(lv), str(lv)) for lv in levels]
    values = [int(counts[lv]) for lv in levels]
    colors = ["#CCCCCC", "#95D5B2", "#55A868", "#DD8452", "#C44E52"]
    bars = ax.bar(labels, values, color=colors[:len(levels)], edgecolor="white")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(values) * 0.01,
                f"{v:,}\n({100*v/total:.1f}%)", ha="center", va="bottom",
                fontsize=10)
    ax.set_title("Severity Distribution of Extracted Records",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of records")
    ax.set_ylim(0, max(values) * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = out_dir / "hfacs_severity.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, mat: pd.DataFrame) -> None:
    total = max(len(df), 1)
    print(f"\n{'='*66}")
    print(f"  HFACS Extraction Analysis  --  {len(df):,} rows")
    print(f"{'='*66}\n")

    status = df["extraction_status"].value_counts()
    print("  Extraction status:")
    for s, c in status.items():
        print(f"    {s:<14} {c:6,}  ({100*c/total:5.1f}%)")
    print()

    for tier in TIER_ORDER:
        subs = HFACS_SCHEMA[tier]
        any_sub = df["hfacs"].apply(lambda h: bool(h.get(tier))).sum()
        print(f"  {TIER_LABELS.get(tier, tier)}  "
              f"[>=1: {any_sub:,} ({100*any_sub/total:.1f}%)]")
        for s in subs:
            c = int(mat[s].sum())
            print(f"      {s:<42} {c:6,}  ({100*c/total:5.1f}%)")
        print()

    rels = _all_relationships(df)
    print(f"  {'='*60}")
    print(f"  Relationships: {len(rels):,} total")
    rt = Counter(r.get("relation") for r in rels)
    for k in sorted(VALID_RELATIONS):
        print(f"    {k:<16} {rt.get(k, 0):,}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _safe(fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except Exception as e:
        print(f"  ERROR in {fn.__name__}: {e}")
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="HFACS results data analysis")
    here = Path(__file__).parent
    parser.add_argument("--input", default=str(here / "hfacs_results.csv"))
    parser.add_argument("--out-dir", default=str(here.parent / "figures"))
    args = parser.parse_args()

    # Windows consoles default to cp1252, which can't encode em-dashes/arrows
    # used in tier labels; force UTF-8 so the summary prints cleanly.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load(args.input)
    mat = sub_matrix(df)
    print(f"Loaded {len(df):,} rows from {args.input}")

    _safe(print_summary, df, mat)
    _safe(plot_distributions, df, mat, out_dir)
    _safe(plot_cooccurrence, mat, out_dir)
    _safe(plot_combinations, df, out_dir)
    _safe(plot_coverage, df, out_dir)
    _safe(plot_relationships, df, out_dir)
    _safe(plot_severity, df, out_dir)

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
