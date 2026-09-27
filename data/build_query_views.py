#!/usr/bin/env python
"""
build_query_views.py — reduced-information retrieval text for the TEST split
===========================================================================
The retrieval query for a record is its `combined_text`: the NTSB preliminary
narrative, the factual narrative AND the probable-cause statement. At test time
that is the answer key. This script writes what an analyst would actually hold at
earlier moments, for the TEST records only:

    L2   preliminary     THE STANDARD VIEW. Circumstances, the kind of event and its
                         sequence, as a preliminary report states them. No injury
                         or damage wording and no causal attribution.
    L1   pre-departure   Lower bound. What a dispatcher knew before the flight left:
                         operator, aircraft, route, date and time of day, weather,
                         crew. Nothing about where in the flight the event happened.
    L1b  circumstances   Optional diagnostic, written only with --tiers L1b. L1 plus
                         the phase of flight and what the crew were doing, but not
                         WHAT happened.

L2 is the level the evaluation is reported at: it is what someone really holds when
they go looking for similar past events, a few days to two weeks after the
occurrence. L1 is kept as the floor, because it shows what is knowable before the
event (nothing, as it turns out). L1b is no longer part of the standard run. It was
the first attempt at "circumstances only" and it fails the severity leakage audit
(AUC ~0.88 with every outcome word removed), because in Part-121 data the phase of
flight all but determines the kind of event.

Nothing else is rewritten. Train and validation records, the exemplar pool, the
knowledge graph and every FAISS index keep their full narratives — the historical
database is made of closed investigations, and only the new event is thin.
Labels are untouched: they come from the full-narrative extraction, which is what
the investigation eventually established.

Each brief is regenerated (up to --max-attempts) while it trips its tier's
blocklist; any sentence still tripping it after that is dropped. The result is
gated by data/leakage_audit.py, not trusted on the prompt's say-so.

    python data/build_query_views.py                 # all test records, resumable
    python data/build_query_views.py --limit 5       # smoke test

Output: data/test_query_views.csv  (ev_id, l2_text, l1_text, [l1b_text], + attempts/dropped)
"""

import argparse
import os
import re
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from ntsbdataloader import load_and_join, _split, NTSB_CLEAN  # noqa: E402
from ollama_json import chat_json, DEFAULT_MODEL              # noqa: E402

OUT = os.path.join(_HERE, "test_query_views.csv")
TIERS = ("L2", "L1", "L1b")                 # every tier the script can write
DEFAULT_TIERS = ("L2", "L1")                # the standard view and its lower bound
COL = {"L1": "l1", "L1b": "l1b", "L2": "l2"}
_SCHEMA = {"type": "object", "properties": {"brief": {"type": "string"}},
           "required": ["brief"]}

# Outcome wording: forbidden at EVERY tier.
OUTCOME_TERMS = [r"injur", r"fatal", r"\bdied\b", r"\bdeath", r"\bkilled", r"hospital",
                 r"damage", r"destroy", r"substantial", r"evacuat", r"\bminor\b",
                 r"\bserious", r"unhurt", r"uninjured"]
# Causal attribution: forbidden at every tier.
CAUSE_TERMS = [r"\bcaus", r"contribut", r"probable", r"\bfinding", r"investigat",
               r"failed to", r"failure to", r"did not", r"\berror", r"mistake",
               r"improper", r"inadequate", r"neglig"]
# What happened: forbidden at L1 and L1b (L2 may name the occurrence).
OCCURRENCE_TERMS = [r"\bfail", r"collid", r"collision", r"struck", r"\bstrike", r"impact",
                    r"crash", r"accident", r"incident", r"\bfire\b", r"smoke", r"excursion",
                    r"overr[au]n", r"veer", r"\bstall", r"separat", r"emergency",
                    r"divert", r"abort", r"reject", r"malfunction", r"encounter",
                    r"\bfell\b", r"inverted", r"came to rest", r"\bloss of"]
# Where in the flight it happened: forbidden at L1 only.
PHASE_TERMS = [r"\btaxi", r"pushback", r"push back", r"pushed back", r"\bgate\b", r"\bramp\b",
               r"runway", r"taxiway", r"\bclimb", r"cruis", r"descen", r"approach",
               r"landing", r"\blanded", r"touchdown", r"touched down", r"rollout",
               r"takeoff", r"take-off", r"took off", r"lift-?off", r"flight level",
               r"\bFL\s?\d", r"altitude", r"\bfeet\b", r"\bft\b", r"holding", r"go-around",
               r"turbulen", r"seat ?belt", r"\bsign\b", r"cabin service", r"\bgalley",
               r"lavator", r"\baisle", r"\btug\b", r"\btow", r"parked", r"\bphase\b",
               r"en route", r"enroute", r"airborne", r"in flight", r"in-flight"]

BLOCK = {"L1": OUTCOME_TERMS + CAUSE_TERMS + OCCURRENCE_TERMS + PHASE_TERMS,
         "L1b": OUTCOME_TERMS + CAUSE_TERMS + OCCURRENCE_TERMS,
         "L2": OUTCOME_TERMS + CAUSE_TERMS}
_RE = {k: re.compile("|".join(v), re.I) for k, v in BLOCK.items()}
_SENT = re.compile(r"(?<=[.!?])\s+")

SYSTEM = ("You write short factual briefs about commercial flights for an aviation "
          "safety database. You follow content restrictions exactly. Respond only "
          "with JSON.")

PROMPT = {
    "L1": (
        "From the report below, write a 2 to 4 sentence PRE-DEPARTURE brief: only what a "
        "dispatcher would have known before this flight (or ground operation) began. "
        "Include what is available of: operator, flight number, aircraft type, origin "
        "and destination, the date, whether it was day or night, the weather and "
        "visibility, and the crew's composition and experience (flight hours).\n"
        "STRICT RULES: do NOT say where in the flight anything took place. Do not name "
        "a phase of flight, a position, an altitude, a runway, taxiway or gate, or "
        "what the pilots or cabin crew were doing at any moment. Do not mention "
        "turbulence, any other aircraft or vehicle, any event, outcome, injury, damage, "
        "anything anyone did wrong, or any finding or cause. Do not hint at any of it. "
        "It must read as the description of a flight that has not happened yet."),
    "L1b": (
        "From the report below, write a 3 to 5 sentence brief of the CIRCUMSTANCES of "
        "this flight as they stood BEFORE anything went wrong. Include what is "
        "available of: operator, aircraft type, scheduled or intended route, phase of "
        "flight being conducted, time of day and lighting, weather and visibility, "
        "crew composition and experience, and what the flight or ground operation was "
        "doing.\n"
        "STRICT RULES: do NOT say what happened. Do not mention any event, occurrence, "
        "collision, malfunction, emergency, injury, damage, evacuation, outcome, "
        "anything anyone did wrong, or any finding or cause. Do not hint at it. Write "
        "it as a neutral description of an ordinary operation."),
    "L2": (
        "From the report below, write a 4 to 6 sentence brief as a PRELIMINARY "
        "notification would state it. First the circumstances: operator, aircraft "
        "type, route, phase of flight, time of day, weather, crew. Then what kind of "
        "event occurred and the sequence of what happened, stated neutrally.\n"
        "STRICT RULES: do NOT state or imply how many people were hurt or how badly, "
        "or how much the aircraft was damaged. Do NOT say why it happened: no causes, "
        "no contributing factors, no judgement of anyone's actions, no findings."),
}


def _source_text(row) -> str:
    """Report text handed to the rewriter: probable-cause block removed, and only
    the opening of the narrative, which is where circumstances are stated."""
    parts = str(row.get("combined_text") or "").split("\n\n")
    body = "\n\n".join(parts[:-1]) if len(parts) > 1 else parts[0]
    return body[:7000]


def blocklist_hits(text: str, tier: str) -> list:
    return sorted({m.group(0).lower() for m in _RE[tier].finditer(text or "")})


def _drop_sentences(text: str, tier: str):
    sents = _SENT.split(text or "")
    kept = [s for s in sents if not _RE[tier].search(s)]
    return " ".join(kept).strip(), len(sents) - len(kept)


def make_brief(row, tier: str, model: str, max_attempts: int):
    src = _source_text(row)
    ctx = (f"Known structured fields: weather={row.get('visual_condition')}, "
           f"light={row.get('light_conditions')}, aircraft={row.get('acft_make')} "
           f"{row.get('acft_model')}, year={row.get('year')}.")
    note, text, attempts = "", "", 0
    for attempts in range(1, max_attempts + 1):
        out = chat_json(SYSTEM, f"{PROMPT[tier]}{note}\n\n{ctx}\n\nREPORT:\n{src}",
                        _SCHEMA, model=model, temperature=0.0 if attempts == 1 else 0.4,
                        num_predict=350, seed=attempts)
        text = (out or {}).get("brief", "").strip()
        bad = blocklist_hits(text, tier)
        if text and not bad:
            return text, attempts, 0
        note = ("\nYour previous draft broke the rules by using these words: "
                + ", ".join(bad) + ". Rewrite it without them and without synonyms "
                "that carry the same meaning.")
    text, dropped = _drop_sentences(text, tier)
    if len(text) < 40:
        # Every sentence tripped the blocklist. An empty query retrieves nothing and a
        # full-narrative fallback would leak, so fall back to the structured fields,
        # which are tier-L0 information by construction.
        text = (f"A {row.get('acft_make')} {row.get('acft_model')} operated in {row.get('year')}. "
                f"Weather conditions: {row.get('visual_condition')}. "
                f"Light conditions: {row.get('light_conditions')}.")
        dropped = -1                                   # marks the structured fallback
    return text, attempts, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=NTSB_CLEAN)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--split", choices=["test", "val"], default="test",
                    help="'val' exists only for the optional threshold diagnostic; "
                         "the evaluation protocol rewrites the TEST split only.")
    ap.add_argument("--tiers", nargs="+", choices=TIERS, default=list(DEFAULT_TIERS),
                    help="Tiers to write. Default: L2 (the standard view) and L1 (the "
                         "lower bound). L1b is an optional diagnostic.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-attempts", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out_path = a.out or (OUT if a.split == "test" else OUT.replace("test_", "val_"))
    df = load_and_join(a.input)
    _tr, va, te = _split(df)
    part = te if a.split == "test" else va
    if a.limit:
        part = part.head(a.limit)

    rows = {}
    if os.path.exists(out_path):
        prev = pd.read_csv(out_path, dtype=str).fillna("")
        rows = {r["ev_id"]: r for r in prev.to_dict("records")}

    def save():
        pd.DataFrame(list(rows.values())).fillna("").to_csv(out_path, index=False)

    # Resumable per (record, tier): a tier is generated only where its text is empty.
    jobs = [(r, t) for r in part.to_dict("records") for t in a.tiers
            if not str(rows.get(str(r["ev_id"]), {}).get(f"{COL[t]}_text", "")).strip()]
    print(f"{a.split}: {len(part)} records, {len(jobs)} briefs to write "
          f"(tiers {', '.join(a.tiers)})", flush=True)
    for i, (row, tier) in enumerate(jobs, 1):
        ev = str(row["ev_id"])
        text, attempts, dropped = make_brief(row, tier, a.model, a.max_attempts)
        rec = rows.setdefault(ev, {"ev_id": ev})
        rec[f"{COL[tier]}_text"] = text
        rec[f"{COL[tier]}_attempts"] = attempts
        rec[f"{COL[tier]}_dropped"] = dropped
        if i % 10 == 0 or i == len(jobs):
            save()
            print(f"  {i}/{len(jobs)} written", flush=True)
    save()
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
