"""
sdr_defect_rate.py  —  FAA Service Difficulty Reports → maintenance-reliability brackets
=======================================================================================
SDRs are per-part mechanical defect reports (~25M rows across SDR-YYYY.csv), NOT
events. This aggregates them into a size-normalized maintenance-reliability signal
that the KG can attach to accident events as a TechnologicalContextNode:

    defects_per_tail(make, year) = SDR rows / distinct reporting aircraft (N-number)

normalizing for fleet size (a big fleet files more reports regardless of quality),
then discretizes to tertile brackets (low / medium / high). Output:

    data/sdr_defect_brackets.csv   columns: make, year, defects_per_tail, bracket

kg_builder reads this lookup (keyed by the event's manufacturer + year) to add a
maintenance_defect_rate context node. RATE not count; ONLY used as context — never
an LSTM training feature, and the temporal cutoff (KG attaches by event year) keeps
it from leaking future maintenance history.

Usage:
  python data/sdr_defect_rate.py            # all SDR-*.csv in data/rawdata
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from standardize import normalize_make                    # noqa: E402
RAW = os.path.join(_HERE, "rawdata")
OUT = os.path.join(_HERE, "sdr_defect_brackets.csv")
NTSB_CLEAN = os.path.join(_HERE, "ntsb_clean.csv")
ASIAS_CLEAN = os.path.join(_HERE, "asias_clean.csv")

USECOLS = ["DifficultyDate", "AircraftMake", "RegistryNNumber"]


def _commercial_makes() -> set:
    """First-token manufacturer names that actually appear in the commercial
    accident data (NTSB acft_make + ASIAS manufacturer). Tertiles are cut over
    ONLY these so 'low/med/high' spans the commercial fleet, not GA-heavy SDR."""
    makes = set()
    for fp, col in ((NTSB_CLEAN, "acft_make"), (ASIAS_CLEAN, "manufacturer")):
        if os.path.exists(fp):
            s = pd.read_csv(fp, dtype=str)[col].dropna().map(normalize_make)
            makes |= set(s[s != ""])
    return makes


def _make_key(s: pd.Series) -> pd.Series:
    """Canonical manufacturer key (SDR code) via the shared alias map."""
    return s.map(normalize_make)


def _year(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.year


def main():
    files = sorted(glob.glob(os.path.join(RAW, "SDR-*.csv")))
    if not files:
        raise SystemExit(f"No SDR-*.csv found in {RAW}")

    # Memory-safe: per file, accumulate (make,year) SDR counts and the DISTINCT
    # (make,year,tail) triples (far smaller than the 25M raw rows). n_tails is then
    # the distinct-tail count per cell; n_sdr the summed row count.
    counts = None                                  # Series indexed by (make, year)
    triples = []                                   # distinct (make, year, tail) per file
    for fp in files:
        try:
            df = pd.read_csv(fp, dtype=str, usecols=lambda c: c in USECOLS,
                             on_bad_lines="skip", engine="python")
        except Exception as e:
            print(f"  skip {os.path.basename(fp)}: {e}")
            continue
        if not set(USECOLS).issubset(df.columns):
            print(f"  skip {os.path.basename(fp)}: missing columns {set(USECOLS)-set(df.columns)}")
            continue
        df["make"] = _make_key(df["AircraftMake"])
        df["year"] = _year(df["DifficultyDate"])
        df = df[(df["make"] != "") & df["year"].notna()]
        df["year"] = df["year"].astype(int)
        c = df.groupby(["make", "year"]).size()
        counts = c if counts is None else counts.add(c, fill_value=0)
        triples.append(df[["make", "year", "RegistryNNumber"]].drop_duplicates())
        print(f"  {os.path.basename(fp)}: {len(df)} usable rows")

    tails = pd.concat(triples, ignore_index=True).drop_duplicates()
    n_tails = tails.groupby(["make", "year"]).size()
    g = pd.DataFrame({"n_sdr": counts, "n_tails": n_tails}).reset_index()
    g = g[g["n_tails"] >= 3]                       # drop tiny, noisy cells
    g["defects_per_tail"] = g["n_sdr"] / g["n_tails"]

    # Restrict to commercial makes (those in the accident data); the data has no GA.
    comm = _commercial_makes()
    if comm:
        before = len(g)
        g = g[g["make"].isin(comm)]
        print(f"  commercial-only: kept {len(g)}/{before} cells "
              f"({g['make'].nunique()} makes)")

    # MAKE-RELATIVE rate: each make-year's defects/tail vs that make's own median
    # across years → "unusually (un)reliable FOR THIS TYPE this year". This strips
    # the manufacturer baseline (Boeing/Airbus are absolutely high simply because
    # they're big complex jets), leaving a reliability *deviation* decoupled from
    # which aircraft it is. Tertiles are then cut on the relative rate.
    g["make_median"] = g.groupby("make")["defects_per_tail"].transform("median")
    g["rel_rate"] = g["defects_per_tail"] / g["make_median"].replace(0, np.nan)
    g = g[g["rel_rate"].notna()]
    q1, q2 = g["rel_rate"].quantile([1 / 3, 2 / 3])
    g["bracket"] = np.where(g["rel_rate"] <= q1, "low",
                   np.where(g["rel_rate"] <= q2, "medium", "high"))

    g[["make", "year", "defects_per_tail", "rel_rate", "bracket"]].to_csv(OUT, index=False)
    print(f"\nWrote {OUT}: {len(g)} (make, year) cells "
          f"(make-relative tertiles at {q1:.2f}/{q2:.2f} × make-median)")
    print("bracket dist:", g["bracket"].value_counts().to_dict())


if __name__ == "__main__":
    main()
