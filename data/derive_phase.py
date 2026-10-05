#!/usr/bin/env python
"""
derive_phase.py — phase of flight for every event, read from its own text
=========================================================================
Why this exists. Structural retrieval matched events on weather, light, crew and
four economic brackets. Over half of a match was decided by the calendar month
(the economic brackets) and most of the rest by aircraft make and year (the SDR
maintenance bracket), so structurally "similar" events were mostly events from the
same month. The graph had no node describing what the flight was DOING.

Why from text, and why that settles the cross-source problem. The three sources do
not share a phase-of-flight field: the NTSB corpus's occurrence column is empty in
this build, ASIAS has no phase field, and ASRS records anomaly and result codes,
not phase. Mapping three different native formats onto one vocabulary is exactly
what was tried before and abandoned. Instead, every event's phase is read by the
SAME extractor, with the SAME closed vocabulary, from the narrative that every
source does have. Consistency then comes from one procedure applied everywhere,
not from reconciling three schemas. Coverage is the same for every source; what
remains is extraction error, which is measured (`--audit`), not assumed away.

Phase is knowable at the L1b query level and above (the L1b brief states it), and
not at L1 (the pre-departure brief, by definition, does not). The query side
follows the same rule: a query's phase is read from the query's own text, so an
L1b query gets the phase its L1b brief states, and an L1 query gets none.

Vocabulary (closed, enforced by a JSON-schema enum):
    parked_or_ground_service   at the gate or on the ramp, engines not driving
                               the aircraft: boarding, servicing, maintenance
    pushback_or_towing         pushed or towed by a tug
    taxi                       moving under its own power on the ground
    takeoff                    takeoff roll, rotation, rejected takeoff
    climb                      initial climb and climb to cruise
    cruise                     en route at altitude
    descent                    descent from cruise
    approach                   approach, including a go-around
    landing                    flare, touchdown, rollout
    unknown                    the text does not state or clearly imply it

Outputs (resumable; rerunning only fills what is missing):
    data/event_phase.csv        key, source, phase   for every graph event and every
                                NTSB corpus record (from its full narrative)
    data/test_query_phase.csv   ev_id, l1b_phase, l2_phase   for the test query views

    python data/derive_phase.py                 # everything (~1 hour on the GPU)
    python data/derive_phase.py --limit 20      # smoke test
    python data/derive_phase.py --audit         # agreement and coverage report
"""

import argparse
import os
import sys
import time

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from ollama_json import chat_json, DEFAULT_MODEL   # noqa: E402

PHASES = ["parked_or_ground_service", "pushback_or_towing", "taxi", "takeoff", "climb",
          "cruise", "descent", "approach", "landing", "unknown"]
OUT = os.path.join(_HERE, "event_phase.csv")
OUT_TEST = os.path.join(_HERE, "test_query_phase.csv")
SCHEMA = {"type": "object",
          "properties": {"phase": {"type": "string", "enum": PHASES}},
          "required": ["phase"]}
SYSTEM = ("You classify the phase of flight of an aviation event from a written "
          "account. Respond only with JSON.")
PROMPT = (
    "In which phase of flight was the aircraft when the event described below took "
    "place? Choose exactly one:\n"
    "  parked_or_ground_service - at the gate or on the ramp, not moving under its "
    "own power (boarding, deplaning, servicing, maintenance)\n"
    "  pushback_or_towing - being pushed back or towed\n"
    "  taxi - moving on the ground under its own power\n"
    "  takeoff - takeoff roll, rotation, or a rejected takeoff\n"
    "  climb - initial climb or climb to cruise\n"
    "  cruise - en route at cruising altitude\n"
    "  descent - descending from cruise, before the approach\n"
    "  approach - on approach, including a go-around\n"
    "  landing - flare, touchdown or landing rollout\n"
    "  unknown - the text does not state or clearly imply the phase\n"
    "If the text only describes a planned or scheduled flight and not a moment in "
    "it, answer unknown. Do not guess.\n\nTEXT:\n")
MAX_CHARS = 6000          # phase is stated near the start of every source's account

# key -> text, per source
SOURCES = {
    "NTSB-corpus": (os.path.join(_HERE, "ntsb_clean.csv"), "ev_id", "NTSB",
                    lambda r: r.get("combined_text")),
    "NTSB-kg": (os.path.join(_HERE, "ntsb_kg_subset.csv"), "ev_id", "NTSB",
                lambda r: r.get("combined_text")),
    "ASIAS": (os.path.join(_HERE, "asias_subset.csv"), "accident_id", "ASIAS",
              lambda r: r.get("combined_narrative")),
    "ASRS": (os.path.join(_HERE, "asrs_subset.csv"), "acn", "ASRS",
             lambda r: "\n\n".join(str(x) for x in (r.get("synopsis"), r.get("narrative"))
                                   if isinstance(x, str) and x.strip())),
}


def classify(text: str, model: str) -> str:
    text = str(text or "").strip()
    if not text or text.lower() == "nan":
        return "unknown"
    out = chat_json(SYSTEM, PROMPT + text[:MAX_CHARS], SCHEMA, model=model,
                    temperature=0.0, num_ctx=4096, num_predict=40, seed=0)
    phase = (out or {}).get("phase", "unknown")
    return phase if phase in PHASES else "unknown"


def _save(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False)


def run_events(model, limit):
    done = {}
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT, dtype=str).fillna("")
        done = {(r["key"], r["source"]): r for r in prev.to_dict("records")}
    rows = list(done.values())
    todo = []
    for name, (path, idcol, source, get) in SOURCES.items():
        d = pd.read_csv(path, dtype=str).drop_duplicates(idcol)
        if limit:
            d = d.head(limit)
        for r in d.to_dict("records"):
            key = str(r[idcol])
            if (key, source) not in done:
                todo.append((key, source, name, get(r)))
                done[(key, source)] = True
    print(f"events: {len(rows)} done, {len(todo)} to classify", flush=True)
    t0 = time.time()
    for i, (key, source, name, text) in enumerate(todo, 1):
        rows.append({"key": key, "source": source, "origin": name,
                     "phase": classify(text, model)})
        if i % 50 == 0 or i == len(todo):
            _save(rows, OUT)
            print(f"  {i}/{len(todo)} ({(time.time() - t0) / i:.2f}s each)", flush=True)
    _save(rows, OUT)


def run_test_views(model, limit):
    views = pd.read_csv(os.path.join(_HERE, "test_query_views.csv"), dtype=str).fillna("")
    if limit:
        views = views.head(limit)
    have = {}
    if os.path.exists(OUT_TEST):
        have = {r["ev_id"]: r for r in pd.read_csv(OUT_TEST, dtype=str).fillna("").to_dict("records")}
    rows = []
    todo = [r for r in views.to_dict("records") if r["ev_id"] not in have]
    print(f"test views: {len(have)} done, {len(todo)} to classify", flush=True)
    for i, r in enumerate(todo, 1):
        have[r["ev_id"]] = {"ev_id": r["ev_id"],
                            "l1b_phase": classify(r.get("l1b_text"), model),
                            "l2_phase": classify(r.get("l2_text"), model)}
        if i % 25 == 0 or i == len(todo):
            _save(list(have.values()), OUT_TEST)
            print(f"  {i}/{len(todo)}", flush=True)
    _save(list(have.values()), OUT_TEST)


def audit():
    ev = pd.read_csv(OUT, dtype=str)
    print("phase distribution by source (% of events):")
    print((pd.crosstab(ev.phase, ev.origin, normalize="columns") * 100).round(0)
          .reindex(PHASES).fillna(0).astype(int).to_string())
    if os.path.exists(OUT_TEST):
        t = pd.read_csv(OUT_TEST, dtype=str)
        full = ev[ev.origin == "NTSB-corpus"].set_index("key").phase
        t["full_phase"] = t.ev_id.map(full)
        for col in ("l1b_phase", "l2_phase"):
            known = t[(t[col] != "unknown") & (t.full_phase != "unknown")]
            print(f"{col}: unknown in {(t[col] == 'unknown').mean():.0%} of briefs; "
                  f"agrees with the full-narrative phase in "
                  f"{(known[col] == known.full_phase).mean():.0%} of the {len(known)} "
                  f"briefs where both are known")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--audit", action="store_true")
    a = ap.parse_args()
    if not a.audit:
        run_events(a.model, a.limit)
        run_test_views(a.model, a.limit)
    audit()


if __name__ == "__main__":
    main()
