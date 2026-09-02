"""
causal_discovery.py  —  PC algorithm vs the theoretical HFACS DAG
================================================================
Empirically tests the HFACS causal structure against the NTSB training data with
the PC constraint-based algorithm (causal-learn, chi-square independence test),
then compares the discovered edges to a feature-level projection of the
theoretical DAG.

RECONCILED to the CURRENT model (2026-06): `operator` = Preconditions, reduced to
the PRIMARY (first-present) tier code (0 = none); `violation` = the binary Unsafe
head (1 if an unsafe_violation tier was extracted, else 0); `severity_class` is the
binary gravity+damage target (high/low). Economic context is the 4 QoQ brackets
(employment, fuel, operating revenue, load factor). `maintenance_defect` is the SDR
reliability bracket (0=unknown,1=low,2=med,3=high) keyed by make+year — included so
PC can TEST whether maintenance reliability has causal edges to preconditions /
violations / severity BEFORE we decide to wire SDR into the model.

Severity is NTSB-defined, so the matrix is restricted to NTSB rows (a `_source`
column, if present, is filtered to NTSB) — ASIAS severity is masked in the model
and would re-introduce the source confound here.

Outputs:
  results/causal_edge_comparison.csv   per-edge: confirmed | new | missing
  results/causal_metrics.csv           precision / recall / f1 / tp / fp / fn
  figures/causal_adjacency_heatmap.png discovered adjacency (reference outlined)

Dependency: `pip install causal-learn`  (not bundled).

Usage:
  python models/causal_discovery.py --input data/ntsb_subset.csv
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "data"))
from hfacs_extractor import HFACS_SCHEMA                 # noqa: E402
from ntsbdataloader import (PRECOND_TIERS, UNSAFE_VIOLATION_TIER, NTSB_CLEAN,  # noqa: E402
                            HFACS_RESULTS)
from standardize import binarize_severity               # noqa: E402
from models import eval_utils as EU                      # noqa: E402

RESULTS = os.path.join(_ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

FEATURES = ["employment_qoq", "fuel_qoq", "revenue_qoq", "loadfactor_qoq",
            "maintenance_defect",
            "visual_condition", "light_conditions", "time_of_day",
            "person_involved", "pilot_hours_bracket",
            "operator", "violation", "severity_class"]

# Feature-level projection of the model's WIRED structure PLUS the exploratory
# hypotheses we want PC to adjudicate. 'operator' = Preconditions (B), 'violation' =
# binary Unsafe head (C), 'severity_class' = binary Severity (D).
REFERENCE_EDGES = [
    # --- WIRED: economic context + env + operator -> Preconditions (seeds B) ---
    ("employment_qoq", "operator"), ("fuel_qoq", "operator"),
    ("revenue_qoq", "operator"), ("loadfactor_qoq", "operator"),
    ("visual_condition", "operator"), ("light_conditions", "operator"),
    ("time_of_day", "operator"), ("person_involved", "operator"),
    ("pilot_hours_bracket", "operator"),
    # --- WIRED: Preconditions -> Violation (B->C) + env/operator -> Violation ---
    ("operator", "violation"),
    ("visual_condition", "violation"), ("light_conditions", "violation"),
    ("time_of_day", "violation"), ("person_involved", "violation"),
    ("pilot_hours_bracket", "violation"),
    # --- WIRED: Violation -> Severity (C->D) + Preconditions/env -> Severity ---
    ("violation", "severity_class"), ("operator", "severity_class"),
    ("visual_condition", "severity_class"), ("light_conditions", "severity_class"),
    ("time_of_day", "severity_class"),
    # --- EXPLORATORY: prior PC run suggested a DIRECT economic -> severity edge ---
    ("employment_qoq", "severity_class"), ("fuel_qoq", "severity_class"),
    ("revenue_qoq", "severity_class"), ("loadfactor_qoq", "severity_class"),
    # --- EXPLORATORY (SDR test): does maintenance reliability cause anything? ---
    ("maintenance_defect", "operator"), ("maintenance_defect", "violation"),
    ("maintenance_defect", "severity_class"),
]


_SDR_PATH = os.path.join(_ROOT, "data", "sdr_defect_brackets.csv")
_SDR_ORD = {"low": 1, "medium": 2, "high": 3}


def _load_sdr():
    if not os.path.exists(_SDR_PATH):
        print(f"  (no {os.path.basename(_SDR_PATH)} — maintenance_defect will be all 0)")
        return {}
    t = pd.read_csv(_SDR_PATH, dtype=str)
    return {(str(r["make"]).upper(), int(float(r["year"]))): r["bracket"]
            for _, r in t.iterrows()}


def _sdr_code(make, year, lut) -> int:
    """SDR reliability bracket -> ordinal (0 unknown, 1 low, 2 med, 3 high)."""
    from standardize import normalize_make
    try:
        b = lut.get((normalize_make(make), int(float(year))))
    except (ValueError, TypeError, AttributeError):
        b = None
    return _SDR_ORD.get(b, 0)


def build_feature_matrix(input_csv=NTSB_CLEAN, hfacs_csv=HFACS_RESULTS):
    """Discrete variables from the NTSB rows of input_csv ⋈ hfacs_results."""
    df = pd.read_csv(input_csv, dtype=str)
    if "_source" in df.columns:                 # severity is NTSB-defined — NTSB only
        df = df[df["_source"].astype(str).str.upper() == "NTSB"].copy()
    hf = pd.read_csv(hfacs_csv, dtype=str)
    hf = hf[hf["extraction_status"] == "success"][["ev_id", "hfacs_json"]]
    m = df.merge(hf, on="ev_id", how="inner")
    if m.empty:
        raise SystemExit("No NTSB records with successful HFACS extraction — run Stage 2 first.")

    # 'operator' = primary (first-present) precondition TIER code (0=none).
    # 'violation' = binary Unsafe head: 1 if an unsafe_violation tier was extracted.
    def _primary_tier(cls, tiers):
        for i, t in enumerate(tiers):
            if cls.get(t):
                return i + 1
        return 0

    lut = _load_sdr()
    ope, viol = [], []
    for j in m["hfacs_json"]:
        try:
            cls = json.loads(j or "{}")
        except (json.JSONDecodeError, TypeError):
            cls = {}
        ope.append(_primary_tier(cls, PRECOND_TIERS))
        viol.append(1 if cls.get(UNSAFE_VIOLATION_TIER) else 0)

    maint = [_sdr_code(mk, yr, lut) for mk, yr in zip(m["acft_make"], m["year"])]
    sev = m["severity_class"].apply(binarize_severity).fillna(0).astype(int).to_numpy()

    def fac(col):
        return pd.factorize(m[col].fillna("Unknown").astype(str))[0]

    cols = {
        "employment_qoq": fac("employment_bracket"),
        "fuel_qoq": fac("fuel_bracket"),
        "revenue_qoq": fac("revenue_bracket"),
        "loadfactor_qoq": fac("loadfactor_bracket"),
        "maintenance_defect": np.array(maint),
        "visual_condition": fac("visual_condition"),
        "light_conditions": fac("light_conditions"),
        "time_of_day": (m["light_conditions"].astype(str).str.strip() == "Daylight").astype(int).to_numpy(),
        "person_involved": fac("person_involved"),
        "pilot_hours_bracket": fac("pilot_hours_bracket"),
        "operator": np.array(ope),
        "violation": np.array(viol),
        "severity_class": sev,
    }
    X = np.column_stack([cols[f] for f in FEATURES]).astype("float64")
    print(f"Feature matrix: {X.shape[0]} rows x {X.shape[1]} vars")
    for f in FEATURES:
        nun = len(np.unique(cols[f]))
        print(f"  {f:22} {nun} distinct value(s)" + ("  (zero variance)" if nun < 2 else ""))
    return X, FEATURES


def run_pc(X, alpha=0.05):
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import chisq
    return pc(X, alpha=alpha, indep_test=chisq)


def _edges_from_graph(G):
    """Return (directed set i->j, skeleton set frozenset{i,j}) from a causal-learn graph.
    Convention: graph[i,j]=-1 & graph[j,i]=1 -> i->j; both -1 -> i--j (undirected)."""
    n = G.shape[0]
    directed, skeleton = set(), set()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if G[i, j] == -1 and G[j, i] == 1:
                directed.add((i, j)); skeleton.add(frozenset((i, j)))
            elif G[i, j] == -1 and G[j, i] == -1:
                skeleton.add(frozenset((i, j)))
    return directed, skeleton


def compare_against_dag(adj_graph, names):
    directed, skeleton = _edges_from_graph(adj_graph)
    idx = {n: i for i, n in enumerate(names)}
    ref = {(idx[a], idx[b]) for a, b in REFERENCE_EDGES}
    ref_skel = {frozenset(e) for e in ref}

    # an expected edge is "confirmed" if its endpoints are adjacent in the PC skeleton
    confirmed = {e for e in ref if frozenset(e) in skeleton}
    missing = ref - confirmed
    new_edges = {e for e in directed if frozenset(e) not in ref_skel}

    fmt = lambda S: sorted(f"{names[a]} -> {names[b]}" for a, b in S)
    return {"confirmed": fmt(confirmed), "missing": fmt(missing),
            "discovered": fmt(new_edges),
            "tp": len(confirmed), "fp": len(new_edges), "fn": len(missing)}


def compute_causal_metrics(cmp):
    tp, fp, fn = cmp["tp"], cmp["fp"], cmp["fn"]
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def main():
    ap = argparse.ArgumentParser(description="PC causal discovery vs HFACS DAG")
    ap.add_argument("--input", default=NTSB_CLEAN)
    ap.add_argument("--hfacs", default=HFACS_RESULTS)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    EU._utf8()

    X, names = build_feature_matrix(args.input, args.hfacs)
    try:
        cg = run_pc(X, alpha=args.alpha)
    except ImportError:
        raise SystemExit("causal-learn not installed. Run: pip install causal-learn")
    G = cg.G.graph

    cmp = compare_against_dag(G, names)
    metrics = compute_causal_metrics(cmp)
    print(f"\nPC vs theoretical DAG: precision={metrics['precision']:.2f} "
          f"recall={metrics['recall']:.2f} f1={metrics['f1']:.2f} "
          f"(tp={metrics['tp']} fp={metrics['fp']} fn={metrics['fn']})")

    # per-edge comparison CSV
    rows = ([{"edge": e, "status": "confirmed"} for e in cmp["confirmed"]]
            + [{"edge": e, "status": "missing"} for e in cmp["missing"]]
            + [{"edge": e, "status": "new_discovered"} for e in cmp["discovered"]])
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "causal_edge_comparison.csv"), index=False)
    pd.DataFrame([metrics]).to_csv(os.path.join(RESULTS, "causal_metrics.csv"), index=False)

    # adjacency heatmap (directed 0/1) with reference overlay
    n = len(names)
    adj = np.zeros((n, n), int)
    directed, skeleton = _edges_from_graph(G)
    for i, j in directed:
        adj[i, j] = 1
    for e in skeleton:                       # mark undirected both ways (lighter)
        a, b = tuple(e)
        if (a, b) not in directed and (b, a) not in directed:
            adj[a, b] = adj[b, a] = 1
    idx = {nm: i for i, nm in enumerate(names)}
    ref = np.zeros((n, n), int)
    for a, b in REFERENCE_EDGES:
        ref[idx[a], idx[b]] = 1
    EU.plot_adjacency_heatmap(adj, names, reference=ref)

    print(f"\nWrote results/causal_edge_comparison.csv, results/causal_metrics.csv")


if __name__ == "__main__":
    main()
