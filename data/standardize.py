"""
standardize.py
==============
All cross-source standardization functions for Stage 1 data preparation.

These are **pure functions** — no file I/O, no global state. Every raw-source
value (NTSB, FAA ASIAS, ASRS) is funneled through exactly one of these so that
the knowledge graph and the LSTM training corpus share an identical feature
vocabulary. No business logic outside this file performs these transforms.

Shared feature domains produced here:
    - visual_condition   : 'VMC' | 'IMC' | 'Unknown'
    - light_conditions   : 'Daylight' | 'Night' | 'Dusk' | 'Dawn' | 'Unknown'
    - person_involved    : 'PIC' | 'CoPilot' | 'Maintenance' | 'ATC'
                           | 'Other' | 'Unknown'
    - pilot_hours_bracket: '<500' | '500-2000' | '2000-5000' | '5000+'
                           | 'Unknown'
    - severity_class     : ordinal int (NTSB & ASIAS encoders align)
    - employment/fuel QoQ bracket : see bracket_qoq()
"""

import math


# ---------------------------------------------------------------------------
# Small internal helpers
# ---------------------------------------------------------------------------

def _is_missing(value) -> bool:
    """True for None, NaN, or blank/`nan`/`none` strings."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    s = str(value).strip()
    return s == "" or s.lower() in ("nan", "none")


def _clean(value) -> str:
    """Strip to a clean string; '' for missing values."""
    if _is_missing(value):
        return ""
    return str(value).strip()


def _to_float(value):
    """Coerce to float; None on failure or missing."""
    if _is_missing(value):
        return None
    try:
        return float(str(value).strip().replace(",", ""))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Visual (weather) condition
# ---------------------------------------------------------------------------

def standardize_visual_condition(raw, source: str) -> str:
    """
    Returns exactly one of: 'VMC' | 'IMC' | 'Unknown'.

    source='NTSB' : 'VMC'->'VMC', 'IMC'->'IMC', 'Unk'->'Unknown', else->'Unknown'
    source='ASIAS': 'VFR'->'VMC', 'IFR'->'IMC', else->'Unknown'
    source='ASRS' : 'VMC'->'VMC', 'IMC'/'Mixed'/'Marginal'->'IMC', else->'Unknown'
    """
    s = _clean(raw)
    if s == "":
        return "Unknown"
    su = s.upper()

    if source == "NTSB":
        if su == "VMC":
            return "VMC"
        if su == "IMC":
            return "IMC"
        return "Unknown"

    if source == "ASIAS":
        if su == "VFR":
            return "VMC"
        if su == "IFR":
            return "IMC"
        return "Unknown"

    if source == "ASRS":
        if su == "VMC":
            return "VMC"
        if su in ("IMC", "MIXED", "MARGINAL"):
            return "IMC"
        return "Unknown"

    return "Unknown"


# ---------------------------------------------------------------------------
# Light condition
# ---------------------------------------------------------------------------

def infer_light_from_astronomy(event_date, lat, lon, hhmm) -> str:
    """
    Civil-twilight light classification from date + location (ASIAS only).

    Returns 'Daylight' | 'Night' | 'Dusk' | 'Dawn' | 'Unknown'.
    Returns 'Unknown' if lat/lon is NaN/None, if hhmm is missing, or if the
    astral computation fails for any reason. Astral is imported lazily so the
    module still imports cleanly even when astral is unavailable.

    hhmm is an integer like 1530 -> 15:30. astral returns UTC-aware datetimes;
    they are shifted to local solar time using the longitude offset (lon / 15).
    """
    if _is_missing(lat) or _is_missing(lon) or _is_missing(hhmm):
        return "Unknown"
    if event_date is None:
        return "Unknown"
    try:
        from astral import LocationInfo
        from astral.sun import sun

        lat_f = float(lat)
        lon_f = float(lon)
        loc = LocationInfo(latitude=lat_f, longitude=lon_f)
        s = sun(loc.observer, date=event_date)

        offset = lon_f / 15.0  # hours east of UTC implied by longitude

        def _local_hour(dt) -> float:
            h = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
            return (h + offset) % 24.0

        dawn_hour = _local_hour(s["dawn"])
        rise_hour = _local_hour(s["sunrise"])
        set_hour = _local_hour(s["sunset"])
        dusk_hour = _local_hour(s["dusk"])

        hhmm_int = int(hhmm)
        hhmm_float = hhmm_int // 100 + (hhmm_int % 100) / 60.0

        if hhmm_float < dawn_hour:
            return "Night"
        if hhmm_float < rise_hour:
            return "Dawn"
        if hhmm_float <= set_hour:
            return "Daylight"
        if hhmm_float < dusk_hour:
            return "Dusk"
        return "Night"
    except Exception:
        return "Unknown"


def standardize_light_condition(
    light_raw,
    source: str,
    event_date=None,
    lat=None,
    lon=None,
    hhmm=None,
    time_slot=None,
) -> str:
    """
    Returns exactly one of:
        'Daylight' | 'Night' | 'Dusk' | 'Dawn' | 'Unknown'

    source='NTSB' : code map only. Astronomical inference is disabled;
                    event_date/lat/lon/hhmm are accepted but ignored.
    source='ASIAS': c109 code map -> astronomy(hhmm) -> 'Unknown'.
    source='ASRS' : light code map -> time_slot map -> 'Unknown'.
    """
    if source == "NTSB":
        s = _clean(light_raw).upper()
        return {
            "DAYL": "Daylight",
            "NITE": "Night",
            "NDRK": "Night",
            "NBRT": "Night",
            "DUSK": "Dusk",
            "DAWN": "Dawn",
            "NR": "Unknown",
        }.get(s, "Unknown")

    if source == "ASIAS":
        s = _clean(light_raw)
        if s != "" and s.lower() != "unknown":
            mapped = {
                "day": "Daylight",
                "daylight": "Daylight",
                "night": "Night",
                "dusk": "Dusk",
                "dawn": "Dawn",
            }.get(s.lower())
            if mapped is not None:
                return mapped
        if not _is_missing(hhmm):
            return infer_light_from_astronomy(event_date, lat, lon, hhmm)
        return "Unknown"

    if source == "ASRS":
        s = _clean(light_raw)
        if s != "":
            mapped = {
                "daylight": "Daylight",
                "night": "Night",
                "dusk": "Dusk",
                "dawn": "Dawn",
            }.get(s.lower())
            if mapped is not None:
                return mapped
        slot = _clean(time_slot)
        if slot != "":
            return {
                "0001-0600": "Night",
                "0601-1200": "Daylight",
                "1201-1800": "Daylight",
                "1801-2400": "Night",
            }.get(slot, "Unknown")
        return "Unknown"

    return "Unknown"


# ---------------------------------------------------------------------------
# Severity encoders (NTSB and ASIAS align ordinally)
# ---------------------------------------------------------------------------

def encode_ntsb_severity(ev_type: str, inj_tot=None) -> int:
    """
    Derive severity_class from ev_type and total injury count.

    ev_type == 'INC' -> 0  (incident, regardless of inj_tot)
    ev_type == 'ACC':
        inj_tot NaN/None -> 1
        inj_tot == 0     -> 1
        inj_tot == 1     -> 2
        inj_tot 2 or 3   -> 3
        inj_tot >= 4     -> 4
    NaN/None/other ev_type -> -1

    Aligns ordinally with encode_asias_severity (N->0, B->1, S->2, A->4, K->4).
    """
    et = _clean(ev_type).upper()
    if et == "":
        return -1
    if et == "INC":
        return 0
    if et == "ACC":
        n = _to_float(inj_tot)
        if n is None:
            return 1
        if n == 0:
            return 1
        if n == 1:
            return 2
        if n in (2, 3):
            return 3
        if n >= 4:
            return 4
        return 1
    return -1


# ---------------------------------------------------------------------------
# Manufacturer canonicalization (for joining accident data to FAA SDR)
# ---------------------------------------------------------------------------
# Three naming conventions must collapse to one key: SDR uses its own 6-char codes
# (CNDAIR, DOUG, EMB, BOMBDR, DHAV), NTSB uses full names (CANADAIR, DOUGLAS,
# EMBRAER, BOMBARDIER, DE HAVILLAND), ASIAS uses other truncations (BOMBAR, EMBRAE,
# MCDONN, SAAB-S). Aliases map every variant to the SDR code (the join target).
# Ordered prefix-match on the cleaned full string; first hit wins. McDonnell has NO
# SDR record (post-1997 its jets are filed under Boeing) -> kept as 'MCDONNELL' so
# it resolves to unknown by design.
_MAKE_ALIASES = [
    ("MCDONNELL", "MCDONNELL"), ("MCDONN", "MCDONNELL"), ("MCDON", "MCDONNELL"),
    ("EMBRAER", "EMB"), ("EMBRAE", "EMB"), ("EMBRA", "EMB"), ("EMB", "EMB"),
    ("DOUGLAS", "DOUG"), ("DOUGLA", "DOUG"), ("DOUG", "DOUG"),
    ("CANADAIR", "CNDAIR"), ("CNDAIR", "CNDAIR"), ("CANAD", "CNDAIR"),
    ("BOMBARDIER", "BOMBDR"), ("BOMBAR", "BOMBDR"), ("BOMBDR", "BOMBDR"), ("BOMBD", "BOMBDR"),
    ("DE HAVILLAND", "DHAV"), ("DEHAVILLAND", "DHAV"), ("DEHAV", "DHAV"),
    ("DHAV", "DHAV"), ("DHC", "DHAV"),
    ("SAAB", "SAAB"), ("BOEING", "BOEING"), ("AIRBUS", "AIRBUS"),
    ("LOCKHEED", "LKHEED"), ("LKHEED", "LKHEED"),
    ("GULFSTREAM", "GULSTM"), ("GULSTM", "GULSTM"),
]


def normalize_make(value) -> str:
    """Canonical manufacturer key (the SDR code) for joining accident data to SDR.
    Collapses spelling/truncation/punctuation variants across NTSB/ASIAS/SDR; applied
    to BOTH sides so aliasing stays consistent. '' for missing."""
    s = _clean(value).upper()
    s = "".join(ch if (ch.isalnum() or ch == " ") else " " for ch in s)
    s = " ".join(s.split())
    if not s:
        return ""
    for prefix, canon in _MAKE_ALIASES:
        if s.startswith(prefix):
            return canon
    return s.split()[0]


def encode_ntsb_severity_gravity(highest_injury, damage=None) -> int:
    """
    Size-INVARIANT severity from worst-outcome gravity + aircraft damage, instead
    of injury COUNT (which scales with aircraft capacity and isn't comparable
    across a 4-seat trainer and a 180-seat jet).

    ev_highest_injury (FATL/SERS/MINR/NONE) and aircraft damage (DEST/SUBS/MINR/
    NONE) → ordinal that binarizes (>=3) to HIGH = human harm or hull loss
    (fatal OR aircraft destroyed OR serious injury):
        FATL or DEST            -> 4  (high: death / hull loss)
        SERS                    -> 3  (high: serious injury)
        SUBS                    -> 2  (low:  substantial damage, no serious injury)
        MINR (injury or damage) -> 1
        NONE                    -> 0
        both blank/unknown      -> -1 (dropped)

    Note: pure fatal-OR-destroyed is only ~3% of commercial Part-121 events, far
    too rare to learn; serious injury is included on the HIGH side to get a
    learnable, still size-invariant (gravity-based, not injury-count) target.
    """
    inj = _clean(highest_injury).upper()
    dmg = _clean(damage).upper()
    if inj == "" and dmg == "":
        return -1
    if inj == "FATL" or dmg == "DEST":
        return 4
    if inj == "SERS":
        return 3
    if dmg == "SUBS":
        return 2
    if inj == "MINR" or dmg == "MINR":
        return 1
    return 0


def encode_asias_severity(c104):
    """Map ASIAS c104 injury severity. 'N'->0,'B'->1,'S'->2,'A'->4,'K'->4; else None."""
    s = _clean(c104).upper()
    return {"N": 0, "B": 1, "S": 2, "A": 4, "K": 4}.get(s, None)


# Binary severity target (LSTM terminal node + KG D-prior). NOTE: the cleaned data
# only carries injury-COUNT severity_class (no fatality flag), so this is a
# high-severity PROXY, not literally fatal. >=3 (2+ injuries) gives a ~55/45
# balanced, learnable split. To get TRUE fatal/non-fatal, re-derive from raw NTSB
# injury fields (inj_f_count / ev_highest_injury) in data prep.
SEVERITY_HIGH_THRESHOLD = 3


def binarize_severity(value):
    """severity_class ordinal -> high(1)/low(0); None on invalid."""
    try:
        return 1 if int(float(value)) >= SEVERITY_HIGH_THRESHOLD else 0
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Person involved (crew role) -> shared vocabulary
# ---------------------------------------------------------------------------

def map_ntsb_crew_category(crew_category: str) -> str:
    """
    NTSB crew_category code -> shared person_involved vocabulary.

    'PLT'/'FLTI'/'KPLT' -> 'PIC'   (instructor / check pilot act as PIC)
    'CPLT'              -> 'CoPilot'
    'DSTU'/'PRPS'/'OTHR'-> 'Other'
    NaN/None/blank      -> 'Unknown'
    NTSB has no codes for 'Maintenance' or 'ATC'.
    """
    s = _clean(crew_category).upper()
    if s == "":
        return "Unknown"
    return {
        "PLT": "PIC",
        "FLTI": "PIC",
        "KPLT": "PIC",
        "CPLT": "CoPilot",
        "DSTU": "Other",
        "PRPS": "Other",
        "OTHR": "Other",
    }.get(s, "Unknown")


def clean_person_involved(c81) -> str:
    """
    ASIAS c81 -> shared person_involved vocabulary (prefix match).

    startswith 'Pilot-In-Command' -> 'PIC'
    startswith 'Co-Pilot'         -> 'CoPilot'
    startswith 'Maintenance'      -> 'Maintenance'
    startswith 'Controller'/'ATC' -> 'ATC'
    NaN/None/blank                -> 'Unknown'
    else                          -> 'Other'
    """
    s = _clean(c81)
    if s == "":
        return "Unknown"
    if s.startswith("Pilot-In-Command"):
        return "PIC"
    if s.startswith("Co-Pilot"):
        return "CoPilot"
    if s.startswith("Maintenance"):
        return "Maintenance"
    if s.startswith("Controller") or s.startswith("ATC"):
        return "ATC"
    return "Other"


# ---------------------------------------------------------------------------
# Pilot hours bracket (ASIAS c56 and NTSB flight_hours)
# ---------------------------------------------------------------------------

def bracket_pilot_hours(hours) -> str:
    """
    Numeric flight hours -> bracket.

    <500      -> '<500'
    500-1999  -> '500-2000'
    2000-4999 -> '2000-5000'
    >=5000    -> '5000+'
    NaN/None  -> 'Unknown'
    """
    h = _to_float(hours)
    if h is None:
        return "Unknown"
    if h < 500:
        return "<500"
    if h < 2000:
        return "500-2000"
    if h < 5000:
        return "2000-5000"
    return "5000+"


# ---------------------------------------------------------------------------
# Economic QoQ bracket (employment & fuel)
# ---------------------------------------------------------------------------

def bracket_qoq(pct) -> str:
    """
    Quarter-over-quarter percentage change -> bracket.

    < -5.0        -> 'strong_decline'
    -5.0 to -1.0  -> 'mild_decline'
    -1.0 to  1.0  -> 'stable'
     1.0 to  5.0  -> 'mild_growth'
    >  5.0        -> 'strong_growth'
    NaN/None/0.0  -> 'unknown'
    """
    p = _to_float(pct)
    if p is None or p == 0.0:
        return "unknown"
    if p < -5.0:
        return "strong_decline"
    if p < -1.0:
        return "mild_decline"
    if p <= 1.0:
        return "stable"
    if p <= 5.0:
        return "mild_growth"
    return "strong_growth"


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def split_semicolon_field(value) -> list[str]:
    """Split a ';'-separated string into stripped non-empty parts; [] if missing."""
    if _is_missing(value):
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def map_pilot_certificate(c40) -> str:
    """
    LEGACY — retained for reference only. **Never called anywhere in the
    codebase.** pilot_certificate is dropped for cross-source consistency and
    is not stored in any output CSV. Kept so historical references resolve.
    """
    s = _clean(c40)
    if s == "":
        return "Unknown"
    lower = s.lower()
    if "airline transport" in lower or "atp" in lower:
        return "ATP"
    if "commercial" in lower:
        return "Commercial"
    if "private" in lower:
        return "Private"
    if "student" in lower:
        return "Student"
    return "Other"
