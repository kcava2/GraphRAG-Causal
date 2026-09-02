"""
hfacs_tier_counts.py  —  event-count figure for the tier-level HFACS extraction
===============================================================================
Counts how many EVENTS have each HFACS tier present (>=1 extracted factor) and
rolls them up into condition groups:
  Operator conditions   = operator_mental | operator_physical | operator_limits
  Personnel conditions  = personnel_crm | personnel_readiness
  Technological cond.   = situational_tech
  Unsafe acts           = unsafe_skill | unsafe_decision | unsafe_perception | unsafe_violation

Writes figures/hfacs_tier_counts.png. Re-run after any re-extraction.

Usage:
  python data/hfacs_tier_counts.py            # uses data/hfacs_results.csv
  python data/hfacs_tier_counts.py --input data/ntsb_results_pass1.csv
"""

import argparse
import json
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from ntsbdataloader import PRECOND_TIERS, UNSAFE_TIERS  # noqa: E402

FIG = os.path.join(_HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)

OPER = ["operator_mental", "operator_physical", "operator_limits"]
PERS = ["personnel_crm", "personnel_readiness"]
TIER_COLOR = {**{t: "#4C72B0" for t in OPER}, "situational_tech": "#DD8452",
              **{t: "#55A868" for t in PERS},
              **{t: "#C44E52" for t in UNSAFE_TIERS}}


def compute(path):
    hf = pd.read_csv(path, dtype=str)
    ok = hf[hf["extraction_status"] == "success"]
    tiers = PRECOND_TIERS + UNSAFE_TIERS
    tier = {t: 0 for t in tiers}
    grp = {"Operator\nconditions": 0, "Personnel\nconditions": 0,
           "Technological\nconditions": 0, "Unsafe\nacts": 0}
    n = 0
    for j in ok["hfacs_json"].fillna("{}"):
        try:
            c = json.loads(j)
        except (json.JSONDecodeError, TypeError):
            c = {}
        n += 1
        for t in tiers:
            if c.get(t):
                tier[t] += 1
        if any(c.get(t) for t in OPER):
            grp["Operator\nconditions"] += 1
        if any(c.get(t) for t in PERS):
            grp["Personnel\nconditions"] += 1
        if c.get("situational_tech"):
            grp["Technological\nconditions"] += 1
        if any(c.get(t) for t in UNSAFE_TIERS):
            grp["Unsafe\nacts"] += 1
    return n, tier, grp


def main():
    ap = argparse.ArgumentParser(description="HFACS tier/group event-count figure")
    ap.add_argument("--input", default=os.path.join(_HERE, "hfacs_results.csv"))
    ap.add_argument("--out", default=os.path.join(FIG, "hfacs_tier_counts.png"))
    args = ap.parse_args()

    n, tier, grp = compute(args.input)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # --- grouped condition presence ---
    gk = list(grp); gv = [grp[k] for k in gk]
    colors = ["#4C72B0", "#55A868", "#DD8452", "#C44E52"]
    bars = ax1.bar(gk, gv, color=colors)
    for b, v in zip(bars, gv):
        ax1.text(b.get_x() + b.get_width() / 2, v, f"{v}\n({100*v/max(n,1):.0f}%)",
                 ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, n * 1.15); ax1.set_ylabel("events")
    ax1.set_title(f"Events with condition group present  (n={n})", fontweight="bold")
    ax1.spines[["top", "right"]].set_visible(False)

    # --- per-tier presence ---
    tiers = list(tier)
    vals = [tier[t] for t in tiers]
    y = range(len(tiers))
    ax2.barh(list(y), vals, color=[TIER_COLOR[t] for t in tiers])
    ax2.set_yticks(list(y)); ax2.set_yticklabels(tiers, fontsize=9)
    ax2.invert_yaxis()
    for i, v in enumerate(vals):
        ax2.text(v, i, f" {v} ({100*v/max(n,1):.0f}%)", va="center", fontsize=8)
    ax2.set_xlim(0, n * 1.12); ax2.set_xlabel("events")
    ax2.set_title("Events with each tier present", fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("NTSB HFACS text-mining — tier presence by event",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"Saved {os.path.normpath(args.out)}  (n={n} success records)")
    for k in gk:
        print(f"  {k.replace(chr(10),' '):26} {grp[k]} ({100*grp[k]/max(n,1):.0f}%)")


if __name__ == "__main__":
    main()
