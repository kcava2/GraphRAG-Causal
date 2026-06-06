"""
data_assembler.py
=================
Parses every raw source and joins them into the three clean CSVs that all
downstream stages consume:

    data/ntsb_clean.csv   — sole LSTM training corpus (never enters the KG)
    data/asias_clean.csv  — KG / RAG only (never enters LSTM training)
    data/asrs_clean.csv   — KG / RAG only (never enters LSTM training)

Raw inputs are never modified. Every standardization is delegated to
data/standardize.py — no transformation logic lives here. No model training,
no KG writes, and no LLM calls happen in this stage.

QoQ note: "quarter-over-quarter" percentage change is computed on the
chronologically-sorted monthly series as pct_change(periods=3) * 100, i.e. each
month versus the same value three months earlier.
"""

import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

# Allow the sibling `standardize` import whether run as a script or imported as
# data.data_assembler.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from standardize import (
    standardize_visual_condition,
    standardize_light_condition,
    encode_ntsb_severity,
    encode_asias_severity,
    map_ntsb_crew_category,
    clean_person_involved,
    bracket_pilot_hours,
    bracket_qoq,
    split_semicolon_field,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(_HERE, "rawdata")
OUT = _HERE  # write clean CSVs alongside the data package

NTSB_CSV = os.path.join(RAW, "ntsb accident data.csv")
EMPLOYMENT_XLS = os.path.join(RAW, "employment.xls")
FUEL_PDF = os.path.join(RAW, "OST_R _ BTS _ Transtats.pdf")

_MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
_MONTH_TO_INT = {m: i for i, m in enumerate(_MONTH_NAMES, 1)}


# ---------------------------------------------------------------------------
# Tiny local helpers (string/number hygiene)
# ---------------------------------------------------------------------------

def _s(value) -> str:
    """Clean string; '' for missing."""
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    s = str(value).strip()
    return "" if s.lower() in ("nan", "none") else s


def _num(value):
    """Float or None."""
    s = _s(value).replace(",", "")
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _int_or_none(value):
    n = _num(value)
    return int(n) if n is not None else None


# ---------------------------------------------------------------------------
# Economic series (employment HTML + fuel PDF) → (year, month) → QoQ %
# ---------------------------------------------------------------------------

def load_employment() -> tuple[dict, dict]:
    """
    Parse employment.xls (HTML) → {(year, month): employment_qoq_pct}.

    Monthly rows have the shape [Month, Year, Full-time, Part-time, Grand Total]
    with comma-formatted integers. A trailing unrelated carrier table is
    ignored by requiring an integer Month in 1..12 and a 4-digit Year.
    """
    with open(EMPLOYMENT_XLS, "r", encoding="latin-1") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    records = []
    for tr in soup.find_all("tr"):
        cells = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
        if len(cells) < 5:
            continue
        month_s, year_s, grand = cells[0], cells[1], cells[4]
        if not (month_s.isdigit() and year_s.isdigit()):
            continue
        month, year = int(month_s), int(year_s)
        if not (1 <= month <= 12 and 1900 <= year <= 2100):
            continue
        grand_val = _num(grand)
        if grand_val is None:
            continue
        records.append((year, month, grand_val))

    emp = pd.DataFrame(records, columns=["year", "month", "grand_total"])
    emp = emp.sort_values(["year", "month"]).reset_index(drop=True)
    emp["qoq"] = emp["grand_total"].pct_change(periods=3) * 100.0
    qoq_map = {
        (int(r.year), int(r.month)): (None if pd.isna(r.qoq) else float(r.qoq))
        for r in emp.itertuples()
    }
    raw_map = {
        (int(r.year), int(r.month)): float(r.grand_total) for r in emp.itertuples()
    }
    return qoq_map, raw_map


def load_fuel() -> tuple[dict, dict]:
    """
    Extract the fuel table from OST_R _ BTS _ Transtats.pdf →
    {(year, month): fuel_cost_qoq_pct}.

    The PDF's table extraction is unreliable, so lines are parsed from
    extract_text(). Each calendar-month row matches
    `<year> <MonthName> <numbers...>`; a left-nav menu sometimes prepends
    text, so the year+month pair is matched anywhere in the line. The final
    numeric token is the Total cost-per-gallon (dollars). Annual 'Total' rows
    do not match (month name required) and are skipped.
    """
    import pdfplumber

    pattern = re.compile(
        r"\b(19\d{2}|20\d{2})\s+(" + "|".join(_MONTH_NAMES)
        + r")\s+([\d.,]+(?:\s+[\d.,]+){2,})"
    )

    records = []
    with pdfplumber.open(FUEL_PDF) as pdf:
        for page in pdf.pages:
            for line in (page.extract_text() or "").split("\n"):
                m = pattern.search(line)
                if not m:
                    continue
                cost_per_gallon = _num(m.group(3).split()[-1])
                if cost_per_gallon is None:
                    continue
                records.append((int(m.group(1)), _MONTH_TO_INT[m.group(2)],
                                cost_per_gallon))

    fuel = pd.DataFrame(records, columns=["year", "month", "cpg"])
    fuel = fuel.drop_duplicates(["year", "month"])
    fuel = fuel.sort_values(["year", "month"]).reset_index(drop=True)
    fuel["qoq"] = fuel["cpg"].pct_change(periods=3) * 100.0
    qoq_map = {
        (int(r.year), int(r.month)): (None if pd.isna(r.qoq) else float(r.qoq))
        for r in fuel.itertuples()
    }
    raw_map = {
        (int(r.year), int(r.month)): float(r.cpg) for r in fuel.itertuples()
    }
    return qoq_map, raw_map


def attach_economics(df: pd.DataFrame, emp_qoq: dict, emp_raw: dict,
                     fuel_qoq: dict, fuel_raw: dict) -> pd.DataFrame:
    """
    LEFT JOIN employment & fuel on integer (year, month).

    Adds QoQ % (bucketed via bracket_qoq) plus the raw level features
    `industry_total` (employment GrandTotal) and `fuel_cost_per_gallon`
    (Total CostPerGallon). Fuel has no pre-2000 data, so unmatched rows get
    fuel_cost_qoq_pct = 0.0 and fuel_cost_per_gallon = 0.0.
    """
    pairs = list(zip(df["year"], df["month"]))

    def _lookup(table, y, m, default):
        if pd.isna(y) or pd.isna(m):
            return default
        v = table.get((int(y), int(m)))
        return default if v is None else v

    df["employment_qoq_pct"] = [_lookup(emp_qoq, y, m, np.nan) for y, m in pairs]
    df["fuel_cost_qoq_pct"] = [_lookup(fuel_qoq, y, m, 0.0) for y, m in pairs]
    df["industry_total"] = [_lookup(emp_raw, y, m, np.nan) for y, m in pairs]
    df["fuel_cost_per_gallon"] = [_lookup(fuel_raw, y, m, 0.0) for y, m in pairs]
    df["employment_bracket"] = df["employment_qoq_pct"].apply(bracket_qoq)
    df["fuel_bracket"] = df["fuel_cost_qoq_pct"].apply(bracket_qoq)
    return df


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _mode_non_unknown(series) -> str:
    """Most common non-'Unknown' value; 'Unknown' if none."""
    vals = [v for v in series if v and v != "Unknown"]
    if not vals:
        return "Unknown"
    return pd.Series(vals).value_counts().index[0]


def _join_unique(series) -> str:
    """';'-join unique, non-missing, stripped strings preserving first-seen order."""
    seen = []
    for v in series:
        s = _s(v)
        if s and s not in seen:
            seen.append(s)
    return ";".join(seen)


def _clamp_injuries(value):
    """inj_tot count clamped: >50000 or <0 → None."""
    n = _num(value)
    if n is None or n < 0 or n > 50000:
        return None
    return n


# ---------------------------------------------------------------------------
# Output 1 — ntsb_clean.csv
# ---------------------------------------------------------------------------

NTSB_COLUMNS = [
    "ev_id", "severity_class", "invest_type_binary",
    "visual_condition", "light_conditions", "sky_conditions",
    "person_involved", "pilot_hours_bracket",
    "year", "month", "combined_text",
    "acft_make", "acft_model", "crew_age_mean",
    "finding_description_agg", "occurrence_description_agg",
    "employment_qoq_pct", "fuel_cost_qoq_pct",
    "industry_total", "fuel_cost_per_gallon",
    "employment_bracket", "fuel_bracket",
]


def build_ntsb_clean(emp_qoq: dict, emp_raw: dict, fuel_qoq: dict, fuel_raw: dict) -> pd.DataFrame:
    """NTSB training corpus — one row per ev_id after crew-level aggregation."""
    df = pd.read_csv(NTSB_CSV, encoding="latin-1", dtype=str)

    M = "merged_air_crash_data."
    df["_role"] = df[M + "crew_category"].apply(map_ntsb_crew_category)

    fh = pd.to_numeric(df[M + "flight_hours"], errors="coerce")
    fh = fh.mask(fh < 0).clip(upper=50000)        # <0 → NaN, cap at 50,000
    df["_fh"] = fh
    df["_age"] = pd.to_numeric(df[M + "crew_age"], errors="coerce")

    grp = df.groupby("ev_id", sort=False)
    agg = pd.DataFrame({
        "person_involved": grp["_role"].agg(_mode_non_unknown),
        "_pilot_hours_raw": grp["_fh"].max(),
        "crew_age_mean": grp["_age"].mean(),
        "finding_description_agg": grp[M + "finding_description"].agg(_join_unique),
        "occurrence_description_agg": grp[M + "Occurrence_Description"].agg(_join_unique),
    })
    agg["pilot_hours_bracket"] = agg["_pilot_hours_raw"].apply(bracket_pilot_hours)

    base = df.drop_duplicates("ev_id", keep="first").set_index("ev_id")
    base = base.join(agg)

    base["severity_class"] = [
        encode_ntsb_severity(et, _clamp_injuries(it))
        for et, it in zip(base[M + "ev_type"], base[M + "inj_tot_t"])
    ]

    base["combined_text"] = [
        "\n\n".join(x for x in (_s(a), _s(b), _s(c)) if x)
        for a, b, c in zip(
            base["narr_accp"], base["narr_accf"], base[M + "narr_cause"]
        )
    ]

    dt = pd.to_datetime(base[M + "ev_date"], format="mixed", errors="coerce")
    base["year"] = dt.dt.year.astype("Int64")
    base["month"] = dt.dt.month.astype("Int64")

    base["visual_condition"] = base[M + "wx_cond_basic"].apply(
        lambda v: standardize_visual_condition(v, "NTSB")
    )
    base["light_conditions"] = base[M + "light_cond"].apply(
        lambda v: standardize_light_condition(v, "NTSB")
    )
    base["sky_conditions"] = "UNK"
    base["invest_type_binary"] = base[M + "ev_type"].apply(
        lambda v: {"ACC": 0, "INC": 1}.get(_s(v).upper())
    )
    base["acft_make"] = base[M + "acft_make"].apply(_s)
    base["acft_model"] = base[M + "acft_model"].apply(_s)

    base = base.reset_index()
    base = attach_economics(base, emp_qoq, emp_raw, fuel_qoq, fuel_raw)
    return base[NTSB_COLUMNS]


# ---------------------------------------------------------------------------
# Output 2 — asias_clean.csv
# ---------------------------------------------------------------------------

# ASIAS c-code → decoded name. Phase (c152), mission (c103) and pilot
# certificate (c40) are intentionally absent — never read into this stage.
ASIAS_FIELDS = {
    "c5": "accident_id", "c6": "year", "c7": "month", "c8": "day",
    "c2": "_c2",  # FAR operating part — filter only (c101 retired after ~2009)
    "c10": "local_time", "c13": "state_code", "c14": "city",
    "c23": "manufacturer", "c24": "model",
    "c104": "_c104", "c105": "_c105", "c107": "_c107", "c109": "_c109",
    "c56": "_c56", "c81": "_c81", "c99": "_c99", "c77": "_c77",
    "c101": "_c101", "c119": "_c119",
}

ASIAS_COLUMNS = [
    "accident_id", "year", "month", "severity_class",
    "visual_condition", "light_conditions", "person_involved",
    "pilot_hours_bracket", "weather_factor",
    "manufacturer", "model", "state_code", "city", "local_time",
    "cause_factor", "cause_subcategory", "combined_narrative",
    "employment_qoq_pct", "fuel_cost_qoq_pct",
    "employment_bracket", "fuel_bracket",
]


def _read_asias_structured() -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(os.path.join(RAW, "a[0-9]*.txt"))):
        d = pd.read_csv(path, sep="\t", dtype=str, encoding="latin-1",
                        on_bad_lines="skip")
        d.columns = [c.strip() for c in d.columns]
        keep = {code: name for code, name in ASIAS_FIELDS.items() if code in d.columns}
        frames.append(d[list(keep)].rename(columns=keep))
    out = pd.concat(frames, ignore_index=True)
    # Ensure every expected column exists even if some files lacked it.
    for name in ASIAS_FIELDS.values():
        if name not in out.columns:
            out[name] = np.nan
    return out


def _read_asias_narratives() -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(os.path.join(RAW, "e[0-9]*.txt"))):
        d = pd.read_csv(path, sep="\t", dtype=str, encoding="latin-1",
                        on_bad_lines="skip")
        d.columns = [c.strip() for c in d.columns]
        if {"c5", "remark"}.issubset(d.columns):
            frames.append(d[["c5", "remark"]])
    if not frames:
        return pd.DataFrame(columns=["c5", "remark"])
    return pd.concat(frames, ignore_index=True).drop_duplicates("c5")


def build_asias_clean(emp_qoq: dict, emp_raw: dict, fuel_qoq: dict, fuel_raw: dict) -> pd.DataFrame:
    """ASIAS Scheduled-Air-Carrier records for the KG / RAG (never LSTM)."""
    df = _read_asias_structured()
    # Scheduled-Air-Carrier / Part 121-125 selection. c101 (purpose-of-flight)
    # was retired in the source after ~2009, so union it with the FAR-part code
    # c2 (matching the ASRS 121/125 method) to keep coverage through 2026.
    def _is_scheduled(c101, c2) -> bool:
        c2s = _s(c2)
        return (_s(c101) == "Scheduled Air Carrie") or ("121" in c2s) or ("125" in c2s)

    df = df[[_is_scheduled(a, b) for a, b in zip(df["_c101"], df["_c2"])]].copy()

    narr = _read_asias_narratives()
    df = df.merge(narr, how="left", left_on="accident_id", right_on="c5")

    df["combined_narrative"] = [
        "\n\n".join(x for x in (_s(c119), _s(remark)) if x)
        for c119, remark in zip(df["_c119"], df.get("remark", ""))
    ]

    df["severity_class"] = df["_c104"].apply(encode_asias_severity)
    df["visual_condition"] = df["_c105"].apply(
        lambda v: standardize_visual_condition(v, "ASIAS")
    )
    df["light_conditions"] = [
        standardize_light_condition(c109, "ASIAS", hhmm=_int_or_none(c10))
        for c109, c10 in zip(df["_c109"], df["local_time"])
    ]
    df["person_involved"] = df["_c81"].apply(clean_person_involved)
    df["pilot_hours_bracket"] = df["_c56"].apply(bracket_pilot_hours)
    df["weather_factor"] = df["_c107"].apply(_s)
    df["cause_factor"] = df["_c99"].apply(_s)
    df["cause_subcategory"] = df["_c77"].apply(_s)

    df["year"] = df["year"].apply(_int_or_none).astype("Int64")
    df["month"] = df["month"].apply(_int_or_none).astype("Int64")
    # Restrict to year >= 2000 so ASIAS aligns with the NTSB corpus (2000+).
    df = df[df["year"] >= 2000].copy()
    for col in ("accident_id", "state_code", "city", "manufacturer",
                "model", "local_time"):
        df[col] = df[col].apply(_s)

    df = attach_economics(df, emp_qoq, emp_raw, fuel_qoq, fuel_raw)
    return df[ASIAS_COLUMNS]


# ---------------------------------------------------------------------------
# Output 3 — asrs_clean.csv
# ---------------------------------------------------------------------------

# (group, sub) header pairs (stripped) → decoded name. Flight Phase / Mission
# are intentionally excluded.
ASRS_FIELDS = {
    ("", "ACN"): "acn",
    ("Time", "Date"): "date",
    ("Time", "Local Time Of Day"): "local_time_of_day",
    ("Environment", "Flight Conditions"): "flight_conditions",
    ("Environment", "Light"): "light",
    ("Aircraft 1", "Operating Under FAR Part"): "far_part",
    ("Person 1", "Human Factors"): "human_factors",
    ("Person 1", "Function"): "reporter_function",
    ("Assessments", "Contributing Factors / Situations"): "contributing_factors",
    ("Assessments", "Primary Problem"): "primary_problem",
    ("Events", "Anomaly"): "anomaly",
    ("Events", "Result"): "result",
    ("Report 1", "Narrative"): "narrative",
    ("Report 1", "Synopsis"): "synopsis",
}

ASRS_COLUMNS = [
    "acn", "year", "month",
    "visual_condition", "light_conditions", "person_involved",
    "pilot_hours_bracket", "far_part",
    "human_factors", "anomaly", "contributing_factors", "result",
    "primary_problem", "narrative", "synopsis",
    "employment_qoq_pct", "fuel_cost_qoq_pct",
    "employment_bracket", "fuel_bracket",
]


def map_asrs_function(value) -> str:
    """ASRS reporter Function → shared person_involved vocabulary."""
    s = _s(value).lower()
    if s == "":
        return "Unknown"
    if "captain" in s or "pic" in s:
        return "PIC"
    if "first officer" in s or "copilot" in s or "co-pilot" in s:
        return "CoPilot"
    if "maintenance" in s or "mechanic" in s:
        return "Maintenance"
    if "controller" in s or "atc" in s:
        return "ATC"
    return "Other"


def _find_col(columns, group: str, sub: str):
    for c in columns:
        if str(c[0]).strip() == group and str(c[1]).strip() == sub:
            return c
    return None


def build_asrs_clean(emp_qoq: dict, emp_raw: dict, fuel_qoq: dict, fuel_raw: dict) -> pd.DataFrame:
    """ASRS FAR 121/125 reports for the KG / RAG (never LSTM)."""
    frames = []
    for path in sorted(glob.glob(os.path.join(RAW, "ASRS_DBOnline_*.csv")),
                       key=lambda p: int(re.search(r"_(\d+)\.csv$", p).group(1))):
        raw = pd.read_csv(path, header=[0, 1], dtype=str, encoding="latin-1",
                          low_memory=False)
        out = pd.DataFrame()
        for (group, sub), name in ASRS_FIELDS.items():
            col = _find_col(raw.columns, group, sub)
            out[name] = raw[col] if col is not None else np.nan
        frames.append(out)

    df = pd.concat(frames, ignore_index=True)
    df = df[df["acn"].apply(lambda v: _s(v) != "")]                 # drop blank rows
    df = df[df["far_part"].apply(
        lambda v: ("121" in _s(v)) or ("125" in _s(v))
    )].copy()

    date_int = df["date"].apply(_int_or_none)
    df["year"] = [d // 100 if d is not None else None for d in date_int]
    df["month"] = [d % 100 if d is not None else None for d in date_int]
    df["year"] = df["year"].astype("Int64")
    df["month"] = df["month"].astype("Int64")
    # Restrict to year >= 2000 so ASRS aligns with the NTSB corpus (2000+).
    df = df[df["year"] >= 2000].copy()

    df["visual_condition"] = df["flight_conditions"].apply(
        lambda v: standardize_visual_condition(v, "ASRS")
    )
    df["light_conditions"] = [
        standardize_light_condition(light, "ASRS", time_slot=slot)
        for light, slot in zip(df["light"], df["local_time_of_day"])
    ]
    df["person_involved"] = df["reporter_function"].apply(map_asrs_function)
    df["pilot_hours_bracket"] = "Unknown"

    # Normalize semicolon-list fields (LLM context); re-join for CSV storage.
    for col in ("human_factors", "anomaly", "contributing_factors", "result"):
        df[col] = df[col].apply(lambda v: ";".join(split_semicolon_field(v)))
    for col in ("acn", "far_part", "primary_problem", "narrative", "synopsis"):
        df[col] = df[col].apply(_s)

    df = attach_economics(df, emp_qoq, emp_raw, fuel_qoq, fuel_raw)
    return df[ASRS_COLUMNS]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _summary(name, df, extra_cols=()):
    print(f"\n=== {name}  ({len(df):,} rows) ===")
    for col in ("severity_class", "visual_condition", "light_conditions",
                "person_involved", "pilot_hours_bracket", *extra_cols):
        if col in df.columns:
            vc = df[col].value_counts(dropna=False).head(8)
            print(f"  {col}: " + ", ".join(f"{k}={v}" for k, v in vc.items()))


def main():
    print("Loading economic series …")
    emp_qoq, emp_raw = load_employment()
    fuel_qoq, fuel_raw = load_fuel()
    print(f"  employment months: {len(emp_qoq)} | fuel months: {len(fuel_qoq)}")

    print("\nBuilding ntsb_clean.csv …")
    ntsb = build_ntsb_clean(emp_qoq, emp_raw, fuel_qoq, fuel_raw)
    ntsb.to_csv(os.path.join(OUT, "ntsb_clean.csv"), index=False)
    _summary("ntsb_clean", ntsb)

    print("\nBuilding asias_clean.csv …")
    asias = build_asias_clean(emp_qoq, emp_raw, fuel_qoq, fuel_raw)
    asias.to_csv(os.path.join(OUT, "asias_clean.csv"), index=False)
    _summary("asias_clean", asias)

    print("\nBuilding asrs_clean.csv …")
    asrs = build_asrs_clean(emp_qoq, emp_raw, fuel_qoq, fuel_raw)
    asrs.to_csv(os.path.join(OUT, "asrs_clean.csv"), index=False)
    _summary("asrs_clean", asrs, extra_cols=("far_part",))

    print("\nDone. Wrote 3 clean CSVs to", OUT)


if __name__ == "__main__":
    main()
