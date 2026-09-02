"""
build_lstm_corpus.py  —  commercial NTSB + a DISJOINT ASIAS slice for the LSTM
=============================================================================
Combines the commercial NTSB target (ntsb_clean.csv) with the ASIAS records that
are NOT in the knowledge graph (asias_clean.csv minus asias_subset.csv, the KG's
ASIAS source) into one LSTM train/test corpus. Leakage-safe by construction: the
ASIAS rows added here are never retrieved (they are disjoint from the KG slice).

ASIAS is mapped to the NTSB schema; rows with missing severity are dropped (the
dataloader drops them anyway). A `_source` column is kept for per-source analysis.

NOTE: ASIAS severity is gravity-based ('serious'=2) and 61% missing, vs NTSB's
injury-count scale — so the combined severity distribution shifts (printed below).

Output: data/lstm_corpus.csv  — use it as --input for extraction / training / eval.

Usage:
  python data/build_lstm_corpus.py
"""

import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import data_assembler as DA                                  # noqa: E402

NTSB = os.path.join(_HERE, "ntsb_clean.csv")
ASIAS = os.path.join(_HERE, "asias_clean.csv")
ASIAS_KG = os.path.join(_HERE, "asias_subset.csv")
OUT = os.path.join(_HERE, "lstm_corpus.csv")
OUT_ASIAS = os.path.join(_HERE, "asias_corpus.csv")   # ASIAS-only slice, for two-pass extraction


def main():
    ntsb = pd.read_csv(NTSB, dtype=str)
    ntsb["_source"] = "NTSB"

    ac = pd.read_csv(ASIAS, dtype=str)
    kg_ids = set(pd.read_csv(ASIAS_KG, dtype=str)["accident_id"].astype(str))
    disj = ac[~ac["accident_id"].astype(str).isin(kg_ids)].copy()
    sev = pd.to_numeric(disj["severity_class"], errors="coerce")
    disj = disj[sev.notna()].copy()                          # usable severity only
    print(f"NTSB: {len(ntsb)} | ASIAS disjoint-from-KG with severity: {len(disj)}")

    g = lambda c: disj[c] if c in disj.columns else ""
    m = pd.DataFrame({
        "ev_id": disj["accident_id"].astype(str),
        "severity_class": disj["severity_class"],
        "invest_type_binary": "",
        "visual_condition": g("visual_condition"),
        "light_conditions": g("light_conditions"),
        "sky_conditions": "UNK",
        "person_involved": g("person_involved"),
        "pilot_hours_bracket": g("pilot_hours_bracket"),
        "year": disj["year"], "month": disj["month"],
        "combined_text": disj["combined_narrative"],
        "acft_make": g("manufacturer"), "acft_model": g("model"),
        "crew_age_mean": "",
        "finding_description_agg": (g("cause_factor").fillna("").astype(str) + ";"
                                    + g("cause_subcategory").fillna("").astype(str)).str.strip(";"),
        "occurrence_description_agg": "",
        "employment_qoq_pct": g("employment_qoq_pct"),
        "fuel_cost_qoq_pct": g("fuel_cost_qoq_pct"),
        "operating_revenue_qoq_pct": g("operating_revenue_qoq_pct"),
        "load_factor_qoq_pct": g("load_factor_qoq_pct"),
        "employment_bracket": g("employment_bracket"),
        "fuel_bracket": g("fuel_bracket"),
        "revenue_bracket": g("revenue_bracket"),
        "loadfactor_bracket": g("loadfactor_bracket"),
        "_source": "ASIAS",
    })[DA.NTSB_COLUMNS + ["_source"]]

    combined = pd.concat([ntsb[DA.NTSB_COLUMNS + ["_source"]], m], ignore_index=True)
    combined = combined.drop_duplicates("ev_id", keep="first")
    combined.to_csv(OUT, index=False)
    m.to_csv(OUT_ASIAS, index=False)                     # ASIAS-only, for the two-pass run
    print(f"Wrote {OUT_ASIAS}: {len(m)} ASIAS-only rows (extraction input)")

    print(f"\nWrote {OUT}: {len(combined)} records "
          f"(NTSB {int((combined._source=='NTSB').sum())} + "
          f"ASIAS {int((combined._source=='ASIAS').sum())})")
    sev = pd.to_numeric(combined["severity_class"], errors="coerce")
    print("combined severity_class dist:", combined["severity_class"].value_counts(dropna=False).to_dict())
    print(f"binary severity (>=3 = high): high {int((sev>=3).sum())} / low {int((sev<3).sum())} "
          f"({100*(sev>=3).mean():.0f}% high)")


if __name__ == "__main__":
    main()
