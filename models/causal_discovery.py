"""
causal_discovery.py  —  PC algorithm vs the theoretical HFACS DAG
================================================================
Empirically tests the HFACS causal structure against the NTSB training data with
the PC constraint-based algorithm (causal-learn, chi-square independence test),
then compares the discovered edges to a feature-level projection of the
theoretical DAG.

RECONCILED to the multi-label design: the HFACS "group" variables
(org_climate, supervisory, operator=Preconditions, unsafe) are reduced from the
multi-label hfacs_json to a single discrete code = the PRIMARY (first listed)
subcategory in that group (0 = none). All other variables are integer-encoded.
`sky_conditions` is constant ('UNK') → zero variance; retained for completeness,
PC results on it are trivial (documented).

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
from ntsbdataloader import (PRECOND_TIERS, UNSAFE_TIERS, NTSB_CLEAN,  # noqa: E402
                            HFACS_RESULTS)
from models import eval_utils as EU                      # noqa: E402

RESULTS = os.path.join(_ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

FEATURES = ["org_climate", "employment_qoq", "visual_condition", "light_conditions",
            "sky_conditions", "time_of_day", "person_involved", "pilot_hours_bracket",
            "supervisory", "operator", "unsafe", "severity_class"]

# Feature-level projection of the theoretical HFACS DAG onto the 12 variables.
REFERENCE_EDGES = [
    ("employment_qoq", "org_climate"),
    ("org_climate", "supervisory"),
    ("supervisory", "operator"),
    ("person_involved", "operator"),
    ("pilot_hours_bracket", "operator"),
    ("visual_condition", "operator"),
    ("light_conditions", "operator"),
    ("time_of_day", "operator"),
    ("operator", "unsafe"),
    ("supervisory", "unsafe"),
    ("unsafe", "severity_class"),
]


def _primary_code(cls: dict, tiers, subs_of) -> int:
    """First listed subcategory across `tiers` -> 1-based code; 0 if none."""
    for t in tiers:
        for s in cls.get(t, []):
            if s in subs_of:
                return subs_of[s] + 1
    return 0


def build_feature_matrix(input_csv=NTSB_CLEAN, hfacs_csv=HFACS_RESULTS):
    """12 discrete variables from ntsb_clean ⋈ hfacs_results (success rows)."""
    df = pd.read_csv(input_csv, dtype=str)
    hf = pd.read_csv(hfacs_csv, dtype=str)
    hf = hf[hf["extraction_status"] == "success"][["ev_id", "hfacs_json"]]
    m = df.merge(hf, on="ev_id", how="inner")
    if m.empty:
        raise SystemExit("No NTSB records with successful HFACS extraction — run Stage 2 first.")

    # group -> {subcategory: index}
    org_idx = {s: i for i, s in enumerate(HFACS_SCHEMA["org_climate"])}
    sup_idx = {s: i for i, s in enumerate(HFACS_SCHEMA["supervisory"])}
    pre_idx = {s: i for i, s in enumerate(s for t in PRECOND_TIERS for s in HFACS_SCHEMA[t])}
    uns_idx = {s: i for i, s in enumerate(s for t in UNSAFE_TIERS for s in HFACS_SCHEMA[t])}

    org, sup, ope, uns = [], [], [], []
    for j in m["hfacs_json"]:
        try:
            cls = json.loads(j or "{}")
        except (json.JSONDecodeError, TypeError):
            cls = {}
        org.append(_primary_code(cls, ["org_climate"], org_idx))
        sup.append(_primary_code(cls, ["supervisory"], sup_idx))
        ope.append(_primary_code(cls, PRECOND_TIERS, pre_idx))
        uns.append(_primary_code(cls, UNSAFE_TIERS, uns_idx))

    def fac(col):
        return pd.factorize(m[col].fillna("Unknown").astype(str))[0]

    cols = {
        "org_climate": np.array(org),
        "employment_qoq": fac("employment_bracket"),
        "visual_condition": fac("visual_condition"),
        "light_conditions": fac("light_conditions"),
        "sky_conditions": fac("sky_conditions"),
        "time_of_day": (m["light_conditions"].astype(str).str.strip() == "Daylight").astype(int).to_numpy(),
        "person_involved": fac("person_involved"),
        "pilot_hours_bracket": fac("pilot_hours_bracket"),
        "supervisory": np.array(sup),
        "operator": np.array(ope),
        "unsafe": np.array(uns),
        "severity_class": pd.to_numeric(m["severity_class"], errors="coerce").fillna(0).astype(int).to_numpy(),
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
