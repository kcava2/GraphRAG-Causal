#!/usr/bin/env python
"""
adjudicate_violation.py — strict, consensus re-labelling of `unsafe_violation`
=============================================================================
The committed extraction assigns `unsafe_violation` to almost any non-compliance:
passengers ignoring the seatbelt sign, ATC not relaying weather, an operator not
acting on a Service Bulletin. HFACS reserves the tier for a KNOWING deviation from
a rule by someone in an operational role. Two extraction runs agree on this label
at kappa 0.10, which caps anything a model can do with it.

This pass repairs that one tier without touching `hfacs_results.csv`:

    run 0   every record, temperature 0, the strict definition below
    run 1,2 temperature 0.7, on the CANDIDATES (the committed label or run 0 said
            violation) plus a random reliability sample of other records
    final   majority of the runs a record received

When a committed violation is overturned and the act was really an unintentional
one by operational personnel, the adjudicator names the error tier it belongs in
(HFACS: no intent -> error, not violation), so the record does not silently lose
its only unsafe act.

Output `data/violation_adjudication.csv` is applied as an OVERRIDE by
`ntsbdataloader.load_and_join` (set HFACS_VIOLATION_OVERRIDE=0 to ignore it).
The committed extraction stays the comparable baseline.

    python data/adjudicate_violation.py                # resumable
    python data/adjudicate_violation.py --report       # agreement table only
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from ollama_json import chat_json, head_tail, DEFAULT_MODEL  # noqa: E402

NTSB_CLEAN = os.path.join(_HERE, "ntsb_clean.csv")
HFACS_RESULTS = os.path.join(_HERE, "hfacs_results.csv")
OUT = os.path.join(_HERE, "violation_adjudication.csv")
# Work in progress lives here and is renamed to OUT only when every run has
# finished: the dataloader applies OUT as a label override, and a half-written
# file would silently relabel part of the corpus.
PARTIAL = os.path.join(_HERE, "violation_adjudication.partial.csv")

ERROR_TIERS = ["unsafe_decision", "unsafe_skill", "unsafe_perception"]
SCHEMA = {
    "type": "object",
    "properties": {
        "operational_person": {"type": "string"},
        "rule": {"type": "string"},
        "intent_evidence": {"type": "string"},
        "violation": {"type": "boolean"},
        "if_not_violation_error_tier": {"type": "string",
                                        "enum": ERROR_TIERS + ["none"]},
    },
    "required": ["operational_person", "rule", "intent_evidence", "violation",
                 "if_not_violation_error_tier"],
}

SYSTEM = ("You are an HFACS (Human Factors Analysis and Classification System) "
          "analyst. You apply the HFACS distinction between ERRORS and VIOLATIONS "
          "strictly. Respond only with JSON.")

RULES = (
    "Decide whether this event involved an HFACS VIOLATION. Assign it ONLY when ALL "
    "THREE hold:\n"
    "  1. OPERATIONAL ROLE: the person was flight crew, cabin crew, maintenance, air "
    "traffic control, dispatch or ground crew, acting in that role. Passengers NEVER "
    "count. A company, operator or manufacturer not complying with a bulletin, "
    "directive or recommendation does NOT count (that is organizational, not an "
    "unsafe act).\n"
    "  2. IDENTIFIABLE RULE: a specific regulation, procedure, clearance, checklist, "
    "limitation or company policy that was deviated from.\n"
    "  3. KNOWING DEVIATION: the narrative shows the person knew the rule and chose "
    "not to follow it, as a habit (routine violation) or a one-off (exceptional "
    "violation). Forgetting, misjudging, misunderstanding, not noticing, or being "
    "unaware is an ERROR, not a violation, even when a rule ended up broken.\n"
    "If any of the three is missing, answer violation=false. When in doubt, answer "
    "false.\n"
    "Fill the fields in order: who the operational person is (or 'none'), the rule "
    "(or 'none'), a short quote showing they KNEW and CHOSE to deviate (or 'none'), "
    "then the verdict. If the verdict is false but an operational person committed "
    "an unintentional unsafe act involving a rule or procedure, give its error tier "
    "(unsafe_decision = wrong plan or procedure choice; unsafe_skill = slip, lapse or "
    "poor technique; unsafe_perception = misperceived the situation); otherwise "
    "'none'.")


def _committed():
    """ev_id -> (set of committed unsafe tiers, committed violation evidence)."""
    out = {}
    h = pd.read_csv(HFACS_RESULTS, dtype=str)
    for ev, js, st in zip(h.ev_id.astype(str), h.hfacs_json, h.extraction_status):
        if st != "success":
            continue
        try:
            d = json.loads(js or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        tiers = {t for t in ERROR_TIERS + ["unsafe_violation"] if d.get(t)}
        out[ev] = (tiers, d.get("unsafe_violation") or [])
    return out


def _ask(row, prior_evidence, model, temperature, seed):
    text = head_tail(row.get("combined_text"), 16000, 6000)
    parts = [RULES, f"NARRATIVE:\n{text}"]
    fnd = str(row.get("finding_description_agg") or "").strip()
    if fnd and fnd.lower() != "nan":
        parts.append("NTSB CODED FINDINGS:\n" + fnd[:1500])
    if prior_evidence:
        parts.append("A less strict earlier pass flagged these phrases as a violation. "
                     "Judge them against the three conditions; do not defer to them:\n"
                     + json.dumps(prior_evidence)[:1200])
    out = chat_json(SYSTEM, "\n\n".join(parts), SCHEMA, model=model,
                    temperature=temperature, num_ctx=8192, num_predict=300, seed=seed)
    if not out:
        return None
    # The verdict must be backed by its own evidence fields; an unsupported "true"
    # is exactly the looseness this pass exists to remove.
    none = lambda v: str(v or "").strip().lower() in ("", "none", "n/a", "na")
    ok = bool(out.get("violation")) and not (none(out.get("operational_person"))
                                             or none(out.get("rule"))
                                             or none(out.get("intent_evidence")))
    out["violation"] = ok
    return out


def report(df):
    from sklearn.metrics import cohen_kappa_score
    print(f"\nrecords adjudicated: {len(df)}")
    print(f"committed violation rate : {df.committed.astype(int).mean():.3f} "
          f"(n={int(df.committed.astype(int).sum())})")
    print(f"strict final rate        : {df.final.astype(int).mean():.3f} "
          f"(n={int(df.final.astype(int).sum())})")
    print(f"kappa committed vs final : "
          f"{cohen_kappa_score(df.committed.astype(int), df.final.astype(int)):.3f}")
    multi = df[df.run1.notna() & df.run2.notna() & (df.run1 != "") & (df.run2 != "")]
    if len(multi) > 5:
        r = [multi[c].astype(float).astype(int) for c in ("run0", "run1", "run2")]
        ks = [cohen_kappa_score(r[i], r[j]) for i, j in ((0, 1), (0, 2), (1, 2))]
        print(f"run-to-run kappa on the {len(multi)} multiply-sampled records "
              f"(candidate-enriched): mean {np.mean(ks):.3f}  [{', '.join(f'{k:.3f}' for k in ks)}]")
    ct = pd.crosstab(df.committed.astype(int), df.final.astype(int),
                     rownames=["committed"], colnames=["final"])
    print(ct.to_string())
    moved = df[(df.committed.astype(int) == 1) & (df.final.astype(int) == 0)]
    print("overturned violations re-filed as:",
          moved.reclass.value_counts().to_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--reliability-sample", type=int, default=100)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    if a.report:
        report(pd.read_csv(OUT, dtype=str).fillna(""))
        return

    df = pd.read_csv(NTSB_CLEAN, dtype=str)
    if a.limit:
        df = df.head(a.limit)
    committed = _committed()
    rows = {}
    resume = PARTIAL if os.path.exists(PARTIAL) else (OUT if os.path.exists(OUT) else None)
    if resume:
        rows = {r["ev_id"]: r for r in pd.read_csv(resume, dtype=str).fillna("").to_dict("records")}

    def save():
        pd.DataFrame(list(rows.values())).to_csv(PARTIAL, index=False)

    recs = df.to_dict("records")
    # ---- run 0: everything, deterministic ----
    todo = [r for r in recs if str(r["ev_id"]) not in rows]
    print(f"run 0: {len(todo)} of {len(recs)} records to adjudicate", flush=True)
    for i, row in enumerate(todo, 1):
        ev = str(row["ev_id"])
        tiers, evid = committed.get(ev, (set(), []))
        out = _ask(row, evid, a.model, 0.0, 0)
        v = int(bool(out and out["violation"]))
        rows[ev] = {"ev_id": ev, "committed": int("unsafe_violation" in tiers),
                    "run0": v, "run1": "", "run2": "", "final": v,
                    "reclass": (out or {}).get("if_not_violation_error_tier", "none"),
                    "person": (out or {}).get("operational_person", ""),
                    "rule": (out or {}).get("rule", ""),
                    "intent_evidence": (out or {}).get("intent_evidence", "")}
        if i % 25 == 0 or i == len(todo):
            save(); print(f"  run 0: {i}/{len(todo)}", flush=True)

    # ---- runs 1-2: candidates + a reliability sample ----
    cand = [ev for ev, r in rows.items() if int(r["committed"]) or int(float(r["run0"]))]
    others = sorted(set(rows) - set(cand))
    rng = np.random.default_rng(42)
    sample = list(rng.choice(others, size=min(a.reliability_sample, len(others)),
                             replace=False)) if others else []
    by_id = {str(r["ev_id"]): r for r in recs}
    again = [ev for ev in cand + sample if ev in by_id and rows[ev]["run2"] == ""]
    print(f"runs 1-2: {len(cand)} candidates + {len(sample)} reliability sample "
          f"-> {len(again)} to do", flush=True)
    for i, ev in enumerate(again, 1):
        _tiers, evid = committed.get(ev, (set(), []))
        votes = [int(float(rows[ev]["run0"]))]
        for run in (1, 2):
            out = _ask(by_id[ev], evid, a.model, 0.7, run)
            v = int(bool(out and out["violation"]))
            rows[ev][f"run{run}"] = v
            votes.append(v)
        rows[ev]["final"] = int(sum(votes) >= 2)
        if i % 25 == 0 or i == len(again):
            save(); print(f"  runs 1-2: {i}/{len(again)}", flush=True)
    save()
    if not a.limit:
        os.replace(PARTIAL, OUT)                      # publish only a complete pass
    report(pd.DataFrame(list(rows.values())).astype(str))
    print(f"saved {OUT if not a.limit else PARTIAL}")


if __name__ == "__main__":
    main()
