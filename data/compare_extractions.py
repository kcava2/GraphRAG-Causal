"""
Compare two HFACS extraction runs (Stage 2)
===========================================
Prints per-tier prevalence and the all-zero precondition rate for a baseline and a
candidate extraction, so a model swap can be judged before committing to a full run.

Both files are restricted to the **NTSB corpus** before comparing. This matters:
``hfacs_results.csv`` accumulates rows from every source the pipeline has mined
(NTSB plus the ASIAS/ASRS KG corpus), while a pilot run is NTSB-only. Comparing
them unfiltered inflates the baseline column and makes a worse model look better.

    python data/compare_extractions.py --baseline data/hfacs_results.qwen25-7b.bak.csv --candidate data/pilot_gemma4.csv

Read the output as: the ZERO row should FALL and the rare tiers (operator_limits,
personnel_readiness) should RISE off the floor. If every tier inflates by a similar
amount, that is over-extraction, not better recall.
"""

import argparse
import json
import os

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
NTSB_CLEAN = os.path.join(_HERE, "ntsb_clean.csv")

PRECOND_TIERS = ["operator_mental", "operator_physical", "operator_limits",
                 "personnel_crm", "personnel_readiness", "situational_tech"]
UNSAFE_TIERS = ["unsafe_skill", "unsafe_decision", "unsafe_perception",
                "unsafe_violation"]


def load(path: str, keep_ids: set | None) -> list[set]:
    """-> one set of extracted tiers per successful record."""
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df = df[df["extraction_status"] == "success"]
    df = df.drop_duplicates("ev_id", keep="last")
    if keep_ids is not None:
        df = df[df["ev_id"].astype(str).isin(keep_ids)]
    out = []
    for raw in df["hfacs_json"].fillna("{}"):
        try:
            obj = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            obj = {}
        out.append(set(obj) if isinstance(obj, dict) else set())
    return out


def rate(records: list[set], tier: str) -> float:
    return sum(tier in r for r in records) / len(records) if records else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True, help="Previous run's results CSV.")
    p.add_argument("--candidate", required=True, help="New run's results CSV.")
    p.add_argument("--all-sources", action="store_true",
                   help="Skip the NTSB filter and compare every row in both files. "
                        "Only meaningful when both runs cover the same corpus.")
    a = p.parse_args()

    keep = None
    if not a.all_sources:
        keep = set(pd.read_csv(NTSB_CLEAN, dtype=str,
                               low_memory=False)["ev_id"].astype(str))

    base, cand = load(a.baseline, keep), load(a.candidate, keep)
    if not base or not cand:
        raise SystemExit("One of the files has no usable rows after filtering.")

    scope = "all sources" if a.all_sources else "NTSB only"
    print(f"\nscope: {scope}   baseline n={len(base)}   candidate n={len(cand)}")
    print(f"\n{'tier':<24}{'baseline':>10}{'candidate':>11}{'delta':>9}")
    print("-" * 54)

    for group, tiers in (("PRECONDITIONS", PRECOND_TIERS), ("UNSAFE ACTS", UNSAFE_TIERS)):
        print(f"{group}")
        for t in tiers:
            b, c = rate(base, t), rate(cand, t)
            print(f"  {t:<22}{b:>9.1%}{c:>10.1%}{c - b:>+9.1%}")

    zb = sum(not (r & set(PRECOND_TIERS)) for r in base) / len(base)
    zc = sum(not (r & set(PRECOND_TIERS)) for r in cand) / len(cand)
    print("-" * 54)
    print(f"  {'ZERO preconditions':<22}{zb:>9.1%}{zc:>10.1%}{zc - zb:>+9.1%}")

    mb = sum(len(r & set(PRECOND_TIERS)) for r in base) / len(base)
    mc = sum(len(r & set(PRECOND_TIERS)) for r in cand) / len(cand)
    print(f"  {'mean precond tiers/rec':<22}{mb:>9.2f}{mc:>10.2f}{mc - mb:>+9.2f}")
    print()


if __name__ == "__main__":
    main()
