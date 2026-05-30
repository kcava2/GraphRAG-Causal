"""
HFACS Extraction Results — Data Analysis
=========================================
Analyzes hfacs_results.csv produced by hfacs_extractor.py.

The extractor emits one boolean column (YES/NO) per HFACS subcategory across
all five tiers in the consolidated HFACS_SCHEMA (Organizational Climate,
Supervisory Conditions, Personnel Conditions, Operator Conditions,
Unsafe Acts). This script summarizes that output.

Outputs:
  figures/hfacs_distributions.png  — YES-rate bar chart per tier (5 panels)
  figures/hfacs_cooccurrence.png   — pairwise YES co-occurrence heatmap
  figures/hfacs_combinations.png   — top-15 YES-subcategory combinations
  figures/hfacs_coverage.png       — per-tier coverage + YES-count histogram
  figures/hfacs_severity.png       — severity-level distribution
  Console: full numeric summary

Usage:
    python data/hfacs_analysis.py
    python data/hfacs_analysis.py --input data/hfacs_results.csv
"""

import argparse
import os
import sys
import traceback
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from config.dag_config import HFACS_SCHEMA
from models.lstm.severity import derive_severity_label

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ALL_SUBS: list[str] = [s for tier in HFACS_SCHEMA.values() for s in tier["subs"]]
TIER_ORDER: list[str] = list(HFACS_SCHEMA.keys())

COLORS = {
    # Organizational Climate
    "Safety Culture":                      "#5A189A",
    "Structure":                           "#9D4EDD",
    # Supervisory
    "Inadequate Supervision":              "#4C72B0",
    "Planned Inappropriate Operations":    "#0077B6",
    "Failed to Correct Known Problem":     "#023E8A",
    # Personnel
    "Crew Resource Management":            "#8338EC",
    "Personal Readiness":                  "#3A86FF",
    # Operator
    "Adverse Mental State":                "#DD8452",
    "Adverse Physiological State":         "#C44E52",
    # Unsafe Acts
    "Decision Errors":                     "#55A868",
    "Skill-based Errors":                  "#2D6A4F",
    "Perceptual Errors":                   "#95D5B2",
    "Routine Violations":                  "#E76F51",
}


# ---------------------------------------------------------------------------
# Load & validate
# ---------------------------------------------------------------------------

def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin1")
    missing = [s for s in ALL_SUBS if s not in df.columns]
    if missing:
        raise ValueError(f"Missing subcategory columns in results file: {missing}")

    for s in ALL_SUBS:
        df[s] = df[s].astype(str).str.strip().str.upper()
        bad = ~df[s].isin({"YES", "NO"})
        n_bad = int(bad.sum())
        if n_bad:
            print(f"  warning: {n_bad} non-YES/NO values in column '{s}' "
                  f"(treated as NO)")
            df.loc[bad, s] = "NO"
    return df


def _yes_count(df: pd.DataFrame, sub: str) -> int:
    return int((df[sub] == "YES").sum())


# ---------------------------------------------------------------------------
# 1. Distribution bar charts — one panel per tier
# ---------------------------------------------------------------------------

def plot_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    total = len(df)
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle("HFACS Subcategory YES Rates by Tier",
                 fontsize=16, fontweight="bold", y=1.00)

    flat_axes = axes.flatten()
    for ax, tier_id in zip(flat_axes, TIER_ORDER):
        tier = HFACS_SCHEMA[tier_id]
        subs = tier["subs"]
        counts = [_yes_count(df, s) for s in subs]
        rates = [100 * c / total for c in counts]
        colors = [COLORS.get(s, "#999999") for s in subs]

        bars = ax.barh(subs[::-1], counts[::-1], color=colors[::-1],
                       edgecolor="white", linewidth=0.5)

        for bar, cnt, rate in zip(bars, counts[::-1], rates[::-1]):
            ax.text(
                bar.get_width() + max(counts) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{cnt:,}  ({rate:.1f}%)",
                va="center", ha="left", fontsize=9,
            )

        ax.set_title(tier["label"], fontsize=12, fontweight="bold")
        ax.set_xlabel("YES count")
        ax.set_xlim(0, max(counts) * 1.35 if max(counts) else 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=9)

    # Hide unused panels (we have 5 tiers, 6 cells)
    for ax in flat_axes[len(TIER_ORDER):]:
        ax.set_visible(False)

    plt.tight_layout()
    out = out_dir / "hfacs_distributions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# 2. Pairwise co-occurrence heatmap (all 13 subcategories)
# ---------------------------------------------------------------------------

def plot_cooccurrence(df: pd.DataFrame, out_dir: Path) -> None:
    yes_mat = (df[ALL_SUBS] == "YES").astype(int).to_numpy()
    co = yes_mat.T @ yes_mat  # (n_subs, n_subs) co-occurrence counts

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(co, cmap="Blues", aspect="auto")

    ax.set_xticks(range(len(ALL_SUBS)))
    ax.set_yticks(range(len(ALL_SUBS)))
    ax.set_xticklabels(ALL_SUBS, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ALL_SUBS, fontsize=8)
    ax.set_title("HFACS Pairwise YES Co-occurrence (counts)",
                 fontsize=14, fontweight="bold")

    # Draw separator lines between tiers
    boundary = 0
    for tier_id in TIER_ORDER[:-1]:
        boundary += len(HFACS_SCHEMA[tier_id]["subs"])
        ax.axhline(boundary - 0.5, color="black", lw=0.6, alpha=0.5)
        ax.axvline(boundary - 0.5, color="black", lw=0.6, alpha=0.5)

    vmax = co.max() if co.size else 1
    for i in range(len(ALL_SUBS)):
        for j in range(len(ALL_SUBS)):
            val = co[i, j]
            color = "white" if val > vmax * 0.6 else "black"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=7, color=color)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out = out_dir / "hfacs_cooccurrence.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# 3. Top YES-subcategory combinations
# ---------------------------------------------------------------------------

def _row_combo(row: pd.Series) -> str:
    yes_subs = [s for s in ALL_SUBS if row[s] == "YES"]
    return " + ".join(yes_subs) if yes_subs else "(none)"


def plot_combinations(df: pd.DataFrame, out_dir: Path, top_n: int = 15) -> None:
    combos = df.apply(_row_combo, axis=1)
    counts = combos.value_counts().head(top_n)
    total = len(df)

    fig, ax = plt.subplots(figsize=(16, 8))
    bars = ax.barh(counts.index[::-1], counts.values[::-1],
                   color="#4C72B0", edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, counts.values[::-1]):
        pct = 100 * val / total
        ax.text(
            bar.get_width() + counts.max() * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}  ({pct:.1f}%)",
            va="center", ha="left", fontsize=9,
        )

    ax.set_title(f"Top {top_n} HFACS YES-Subcategory Combinations",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Count")
    ax.set_xlim(0, counts.max() * 1.3)
    ax.tick_params(axis="y", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = out_dir / "hfacs_combinations.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# 4. Coverage — per-tier any-YES rate + YES-count histogram
# ---------------------------------------------------------------------------

def plot_coverage(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Extraction Coverage Overview", fontsize=14, fontweight="bold")

    # Left: per-tier "% rows with >=1 YES in tier"
    ax = axes[0]
    tier_labels = [HFACS_SCHEMA[t]["label"] for t in TIER_ORDER]
    any_yes = []
    no_yes = []
    for t in TIER_ORDER:
        subs = HFACS_SCHEMA[t]["subs"]
        has_yes = (df[subs] == "YES").any(axis=1)
        any_yes.append(100 * has_yes.mean())
        no_yes.append(100 - any_yes[-1])

    x = np.arange(len(tier_labels))
    w = 0.55
    b1 = ax.bar(x, any_yes, w, label=">=1 YES", color="#55A868")
    b2 = ax.bar(x, no_yes, w, bottom=any_yes, label="all NO", color="#CCCCCC")

    ax.set_xticks(x)
    ax.set_xticklabels(tier_labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Percentage of rows (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Per-tier YES coverage")
    ax.legend(loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, val in zip(b1, any_yes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                f"{val:.1f}%", ha="center", va="center", fontsize=9,
                color="white", fontweight="bold")
    for bar, val, base in zip(b2, no_yes, any_yes):
        if val > 2:
            ax.text(bar.get_x() + bar.get_width() / 2, base + val / 2,
                    f"{val:.1f}%", ha="center", va="center",
                    fontsize=9, color="#555555")

    # Right: histogram of YES count per row
    ax2 = axes[1]
    yes_count = (df[ALL_SUBS] == "YES").sum(axis=1)
    bins = np.arange(0, len(ALL_SUBS) + 2) - 0.5
    ax2.hist(yes_count, bins=bins, color="#4C72B0", edgecolor="white")
    ax2.set_xlabel("Number of YES subcategories per row")
    ax2.set_ylabel("Row count")
    ax2.set_title(f"YES-count distribution per row (of {len(ALL_SUBS)} subs)")
    ax2.set_xticks(range(0, len(ALL_SUBS) + 1))
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    mean_yes = yes_count.mean()
    ax2.axvline(mean_yes, color="#C44E52", linestyle="--", lw=1.5,
                label=f"mean = {mean_yes:.2f}")
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    out = out_dir / "hfacs_coverage.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# 5. Event severity distribution
# ---------------------------------------------------------------------------

SEVERITY_LABELS = {1: "1 - IA (incident)", 2: "2 - CA", 3: "3 - LA", 4: "4 - MA/FA (fatal/major)"}


def plot_severity(df: pd.DataFrame, out_dir: Path) -> None:
    if "NtsbNo" not in df.columns:
        print("  skip severity: 'NtsbNo' column not present")
        return

    severities = df["NtsbNo"].map(derive_severity_label)
    total = len(severities)
    counts = severities.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    levels = sorted(counts.index.tolist())
    labels = [SEVERITY_LABELS.get(int(lv), str(lv)) for lv in levels]
    values = [int(counts[lv]) for lv in levels]
    colors = ["#95D5B2", "#55A868", "#DD8452", "#C44E52"][: len(levels)]

    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, values):
        pct = 100 * val / total
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            f"{val:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=10,
        )

    ax.set_title("Event Severity Distribution (from NtsbNo suffix)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of events")
    ax.set_ylim(0, max(values) * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = out_dir / "hfacs_severity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    total = len(df)
    print(f"\n{'='*64}")
    print(f"  HFACS Extraction Analysis  --  {total:,} total rows")
    print(f"{'='*64}\n")

    for tier_id in TIER_ORDER:
        tier = HFACS_SCHEMA[tier_id]
        print(f"  {tier['label']}")
        subs = tier["subs"]
        any_yes = (df[subs] == "YES").any(axis=1).sum()
        print(f"    Rows with >=1 YES in tier: {any_yes:4d} "
              f"({100*any_yes/total:5.1f}%)")
        for sub in subs:
            cnt = _yes_count(df, sub)
            print(f"      {sub:<40}  {cnt:4d}  ({100*cnt/total:5.1f}%)")
        print()

    print(f"  {'='*60}")
    print(f"  Top 10 YES-subcategory combinations")
    print(f"  {'='*60}")
    combos = df.apply(_row_combo, axis=1)
    for combo_label, cnt in combos.value_counts().head(10).items():
        truncated = combo_label if len(combo_label) <= 80 else combo_label[:77] + "..."
        print(f"    {cnt:4d}  ({100*cnt/total:4.1f}%)  {truncated}")
    print()

    if "NtsbNo" in df.columns:
        sev = df["NtsbNo"].map(derive_severity_label)
        sev_counts = sev.value_counts().sort_index()
        print(f"  {'='*60}")
        print(f"  Event severity distribution (from NtsbNo)")
        print(f"  {'='*60}")
        for lv, c in sev_counts.items():
            label = SEVERITY_LABELS.get(int(lv), str(lv))
            print(f"    {label:<30}  {c:4d}  ({100*c/total:5.1f}%)")
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
    _here = Path(__file__).parent
    parser.add_argument("--input",   default=str(_here / "hfacs_results.csv"))
    parser.add_argument("--out-dir", default=str(_here.parent / "figures"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load(args.input)
    print(f"Loaded {len(df):,} rows from {args.input}")

    _safe(print_summary, df)
    _safe(plot_distributions, df, out_dir)
    _safe(plot_cooccurrence, df, out_dir)
    _safe(plot_combinations, df, out_dir)
    _safe(plot_coverage, df, out_dir)
    _safe(plot_severity, df, out_dir)

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
