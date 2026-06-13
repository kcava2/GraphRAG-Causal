"""
select_subset.py  —  curate a high-quality subset per source for a full run
===========================================================================
Scores every record in data/{ntsb,asias,asrs}_clean.csv and writes the top-N
per source to data/{ntsb,asias,asrs}_subset.csv (same schema). The goal is the
"most complete" data: least Unknown/NaN in the shared features and the richest
narratives (best chance of a high-population text-mined extraction).

Score per record = completeness + richness:
  completeness = fraction of the source's *populatable* shared features that are
                 non-Unknown / non-empty. Structural Unknowns are excluded from
                 the denominator (ASRS pilot_hours_bracket is ALWAYS Unknown;
                 ASIAS visual_condition is ~63% Unknown by source design — kept
                 but low-weight, never a hard filter).
  richness     = normalized narrative length (0..1, clipped) + small bonus for
                 source-specific context fields used by LLM extraction.

This is a deliberately curated (non-random) subset for a methods run.

Usage:
    python select_subset.py --ntsb 1000 --asias 500 --asrs 500
"""

import argparse
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_HERE, "data")

# Shared features that CAN be populated per source (used for completeness).
POPULATABLE = {
    "NTSB": ["visual_condition", "light_conditions", "person_involved", "pilot_hours_bracket"],
    # ASIAS visual_condition is mostly Unknown by source design -> kept but the
    # completeness fraction tolerates it (soft, not a filter).
    "ASIAS": ["visual_condition", "light_conditions", "person_involved", "pilot_hours_bracket"],
    # ASRS has no flight-hours field -> pilot_hours_bracket is always Unknown,
    # so it is excluded from ASRS completeness.
    "ASRS": ["visual_condition", "light_conditions", "person_involved"],
}

# Narrative + context-field columns per source.
NARRATIVE_COL = {"NTSB": ["combined_text"],
                 "ASIAS": ["combined_narrative"],
                 "ASRS": ["narrative", "synopsis"]}
CONTEXT_COLS = {"NTSB": ["finding_description_agg"],
                "ASIAS": ["cause_factor", "cause_subcategory"],
                "ASRS": ["human_factors", "anomaly", "primary_problem"]}


def _is_populated(val) -> bool:
    s = "" if val is None else str(val).strip()
    return s != "" and s.lower() not in ("unknown", "nan", "none")


def _narr_len(row, source) -> int:
    return sum(len(str(row.get(c, "") or "")) for c in NARRATIVE_COL[source])


def _context_bonus(row, source) -> float:
    cols = CONTEXT_COLS[source]
    have = sum(1 for c in cols if _is_populated(row.get(c)))
    return have / max(len(cols), 1)


def score_source(df: pd.DataFrame, source: str) -> pd.DataFrame:
    feats = POPULATABLE[source]
    completeness = df.apply(
        lambda r: np.mean([_is_populated(r.get(f)) for f in feats]), axis=1)

    narr = df.apply(lambda r: _narr_len(r, source), axis=1).astype(float)
    # normalize length to 0..1 with a soft cap at the 95th percentile
    cap = max(np.percentile(narr, 95), 1.0)
    richness = np.clip(narr / cap, 0, 1)
    ctx = df.apply(lambda r: _context_bonus(r, source), axis=1)

    df = df.copy()
    df["_completeness"] = completeness
    df["_narr_len"] = narr.astype(int)
    # weighted: completeness dominates, then narrative richness, small ctx nudge
    df["_score"] = 0.5 * completeness + 0.4 * richness + 0.1 * ctx
    return df.sort_values("_score", ascending=False)


def select(source: str, in_name: str, out_name: str, n: int):
    path = os.path.join(DATA, in_name)
    df = pd.read_csv(path, dtype=str)
    scored = score_source(df, source)
    take = min(n, len(scored))
    sub = scored.head(take).drop(columns=["_completeness", "_narr_len", "_score"])
    out = os.path.join(DATA, out_name)
    sub.to_csv(out, index=False)

    top = scored.head(take)
    print(f"\n{source}: selected {take}/{len(df)} -> {out_name}")
    print(f"  mean completeness: {top['_completeness'].mean():.2f} "
          f"(non-Unknown fraction over {POPULATABLE[source]})")
    print(f"  mean narrative length: {top['_narr_len'].mean():.0f} chars "
          f"(min {top['_narr_len'].min()}, max {top['_narr_len'].max()})")
    # per-feature non-Unknown rate in the chosen subset
    for f in POPULATABLE[source]:
        rate = 100 * top[f].apply(_is_populated).mean()
        print(f"    {f:<22} populated: {rate:5.1f}%")


def main():
    ap = argparse.ArgumentParser(description="Curate a high-quality subset per source")
    ap.add_argument("--ntsb", type=int, default=1000)
    ap.add_argument("--asias", type=int, default=500)
    ap.add_argument("--asrs", type=int, default=500)
    args = ap.parse_args()

    select("NTSB", "ntsb_clean.csv", "ntsb_subset.csv", args.ntsb)
    select("ASIAS", "asias_clean.csv", "asias_subset.csv", args.asias)
    select("ASRS", "asrs_clean.csv", "asrs_subset.csv", args.asrs)
    print("\nDone. Subset CSVs written to data/.")


if __name__ == "__main__":
    main()
