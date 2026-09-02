"""
build_carol_ntsb.py  —  rebuild ntsb_clean.csv as COMMERCIAL (Part 121)
======================================================================
Sources:
  rawdata/nstb_carol_aviation_investigations_summary.csv  — NTSB CAROL, Part-121
        filtered event list (NtsbNo, injuries, FAR, ...).
  rawdata/avall.mdb     — NTSB Access DB, 2008+.
  rawdata/pre2008.mdb   — NTSB Access DB, 1948-2007.
        Together they cover the CAROL Part-121 events; each supplies narratives,
        findings, crew, flight-time, weather/light, injuries.

Joins CAROL's Part-121 events (NtsbNo -> events.ntsb_no -> ev_id) to the avall
detail tables, aggregates to one row per ev_id, and applies the SAME transforms
as data_assembler.build_ntsb_clean (severity, brackets, standardize, combined_text,
economics) so the output schema matches the rest of the pipeline.

Output: data/ntsb_clean.csv  (overwrites the GA version — commercial is now the
LSTM target). Re-run select_subset -> extraction -> KG -> training afterward.

Usage:
  pip install access_parser
  python data/build_carol_ntsb.py
"""

import os
import sys

import numpy as np
import pandas as pd
from access_parser import AccessParser

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from standardize import (standardize_visual_condition, standardize_light_condition,  # noqa: E402
                         encode_ntsb_severity_gravity, map_ntsb_crew_category, bracket_pilot_hours)
import data_assembler as DA                                  # noqa: E402

RAW = os.path.join(_HERE, "rawdata")
MDBS = [os.path.join(RAW, "avall.mdb"), os.path.join(RAW, "pre2008.mdb")]
CAROL = os.path.join(RAW, "nstb_carol_aviation_investigations_summary.csv")
OUT = os.path.join(_HERE, "ntsb_clean.csv")


def _clean(v):
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.lower() in ("none", "nan") else s


def _table(db, name, cols):
    """Parse an avall table into a str DataFrame with only the needed columns."""
    t = db.parse_table(name)
    return pd.DataFrame({c: [_clean(x) for x in t.get(c, [])] for c in cols})


def _extract_base(mdb_path, ntsb_nos):
    """Aggregate one row per ev_id from a single NTSB Access DB for the events
    whose ntsb_no is in `ntsb_nos`. Returns a DataFrame indexed by ev_id with the
    raw fields needed by the downstream transforms."""
    db = AccessParser(mdb_path)
    ev = _table(db, "events", ["ev_id", "ntsb_no", "ev_type", "ev_year", "ev_month",
                               "wx_cond_basic", "light_cond", "inj_tot_t",
                               "inj_tot_f", "ev_highest_injury"])
    ev = ev[ev["ntsb_no"].isin(ntsb_nos)].drop_duplicates("ev_id")
    ev_ids = set(ev["ev_id"])
    print(f"  {os.path.basename(mdb_path)}: matched {len(ev_ids)} events")
    if not ev_ids:
        return None

    nar = _table(db, "narratives", ["ev_id", "narr_accp", "narr_accf", "narr_cause"])
    nar_g = nar[nar.ev_id.isin(ev_ids)].groupby("ev_id").agg(
        {"narr_accp": DA._join_unique, "narr_accf": DA._join_unique,
         "narr_cause": DA._join_unique})

    ac = _table(db, "aircraft", ["ev_id", "acft_make", "acft_model", "damage"])
    ac_g = ac[ac.ev_id.isin(ev_ids)].drop_duplicates("ev_id").set_index("ev_id")

    fc = _table(db, "Flight_Crew", ["ev_id", "crew_category", "crew_age"])
    fc = fc[fc.ev_id.isin(ev_ids)].copy()
    fc["_role"] = fc["crew_category"].apply(map_ntsb_crew_category)
    fc["_age"] = pd.to_numeric(fc["crew_age"], errors="coerce")
    fc_g = fc.groupby("ev_id").agg(person_involved=("_role", DA._mode_non_unknown),
                                   crew_age_mean=("_age", "mean"))

    ft = _table(db, "flight_time", ["ev_id", "flight_hours"])
    ft = ft[ft.ev_id.isin(ev_ids)].copy()
    ft["_fh"] = pd.to_numeric(ft["flight_hours"], errors="coerce")
    ft["_fh"] = ft["_fh"].mask(ft["_fh"] < 0).clip(upper=50000)
    ft_g = ft.groupby("ev_id")["_fh"].max()

    fd = _table(db, "Findings", ["ev_id", "finding_description"])
    fd_g = fd[fd.ev_id.isin(ev_ids)].groupby("ev_id")["finding_description"].agg(DA._join_unique)

    base = ev.set_index("ev_id").join(nar_g).join(ac_g).join(fc_g)
    base["_pilot_hours_raw"] = ft_g
    base["finding_description_agg"] = fd_g
    return base


def main():
    # ---- CAROL: Part-121 event list (the whole file is the Part-121 export;
    # FAR is multi-valued like "121; 121" so we keep any row mentioning 121). ----
    carol = pd.read_csv(CAROL, dtype=str)
    carol = carol[carol["FAR"].astype(str).str.contains("121", na=False)]
    ntsb_nos = set(carol["NtsbNo"].astype(str).str.strip())
    print(f"CAROL Part-121 events: {len(ntsb_nos)}")

    # ---- union events across both NTSB databases (2008+ and pre-2008) ----
    parts = [b for b in (_extract_base(p, ntsb_nos) for p in MDBS) if b is not None]
    base = pd.concat(parts)
    base = base[~base.index.duplicated(keep="first")]      # ev_id unique across DBs
    print(f"total matched events: {len(base)}")

    base["pilot_hours_bracket"] = base["_pilot_hours_raw"].apply(bracket_pilot_hours)
    base["occurrence_description_agg"] = ""     # no occurrence text in the mdbs

    # Size-invariant severity: worst-outcome gravity (ev_highest_injury) + aircraft
    # damage. HIGH (>=3) = fatal injury OR aircraft destroyed (not injury count).
    base["severity_class"] = [encode_ntsb_severity_gravity(hi, dmg)
                              for hi, dmg in zip(base["ev_highest_injury"],
                                                 base.get("damage", ""))]
    base["combined_text"] = ["\n\n".join(x for x in (DA._s(a), DA._s(b), DA._s(c)) if x)
                             for a, b, c in zip(base["narr_accp"], base["narr_accf"],
                                                base["narr_cause"])]
    base["year"] = pd.to_numeric(base["ev_year"], errors="coerce").astype("Int64")
    base["month"] = pd.to_numeric(base["ev_month"], errors="coerce").astype("Int64")
    base["visual_condition"] = base["wx_cond_basic"].apply(
        lambda v: standardize_visual_condition(v, "NTSB"))
    base["light_conditions"] = base["light_cond"].apply(
        lambda v: standardize_light_condition(v, "NTSB"))
    base["sky_conditions"] = "UNK"
    base["invest_type_binary"] = base["ev_type"].apply(
        lambda v: {"ACC": 0, "INC": 1}.get(DA._s(v).upper()))
    base["acft_make"] = base.get("acft_make", "").apply(DA._s)
    base["acft_model"] = base.get("acft_model", "").apply(DA._s)

    base = base.reset_index()
    base = DA.attach_economics(base, DA.load_economics())

    out = base[DA.NTSB_COLUMNS].copy()
    sev = pd.to_numeric(out["severity_class"], errors="coerce")
    out = out[sev.notna() & (sev >= 0)]               # drop invalid severity (-1)
    out.to_csv(OUT, index=False)

    print(f"\nWrote {OUT}: {len(out)} commercial (Part-121) NTSB records")
    print("severity_class dist:", out["severity_class"].value_counts(dropna=False).to_dict())
    fat = pd.to_numeric(base.set_index('ev_id').loc[out['ev_id'], 'inj_tot_f'], errors="coerce")
    print(f"events with >=1 fatality (inj_tot_f>0): {int((fat > 0).sum())} / {len(out)} "
          f"({100*(fat > 0).mean():.0f}%)  [for a future fatal/non-fatal target]")
    print("combined_text non-empty:", int((out['combined_text'].str.len() > 0).sum()))


if __name__ == "__main__":
    main()
