"""
visualize.py  —  interpretability & evaluation figures for GraphRAG-Causal
==========================================================================
Static PNGs written to figures/:

  Offline (clean CSVs + hfacs_results.csv):
    causal_dag_schema.png   the HFACS DAG (conceptual O->A->B->C->D causal model)
    data_quality.png        cross-source shared-feature distributions + Unknown rates
    extraction_coverage.png text-mining quality (status, per-tier coverage, ...)

  Live Neo4j (skipped gracefully if unreachable):
    kg_factor_network.png   factor-level learned causal graph (LEADS_TO weighted)
    kg_tier_causal.png      tier-level O->A->B->C->D flow (aggregated weights)
    kg_event_subgraph.png   one EventNode ego-graph (factors + context)
    kg_stats.png            node/edge counts, weight dist, top causal edges

Usage:
    python visualize.py                # all figures
    python visualize.py --no-kg        # offline only
    python visualize.py --event-id 20000110001199I
"""

import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import networkx as nx

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "data"))
from hfacs_extractor import HFACS_SCHEMA          # noqa: E402
from kg_builder import DAG_EDGES                  # noqa: E402
from ntsbdataloader import (                      # noqa: E402
    ORG_TIERS, PRECOND_TIERS, UNSAFE_TIERS, ORG_SUBS, SUP_SUBS,
    PRECOND_SUBS, UNSAFE_SUBS,
)

DATA = os.path.join(_HERE, "data")
FIG = os.path.join(_HERE, "figures")
os.makedirs(FIG, exist_ok=True)


# ---------------------------------------------------------------------------
# Tier -> HFACS level + colors (single source of truth for layout/colour)
# ---------------------------------------------------------------------------

LEVEL_OF = {}
for t in ORG_TIERS:        LEVEL_OF[t] = 0   # Organizational Influences
LEVEL_OF["supervisory"] = 1                  # Supervision
for t in PRECOND_TIERS:    LEVEL_OF[t] = 2   # Preconditions
for t in UNSAFE_TIERS:     LEVEL_OF[t] = 3   # Unsafe Acts
LEVEL_OF["severity"] = 4                     # Outcome

LEVEL_NAME = {0: "Organizational", 1: "Supervision", 2: "Preconditions",
              3: "Unsafe Acts", 4: "Severity"}
LEVEL_COLOR = {0: "#5A189A", 1: "#3A0CA3", 2: "#4361EE", 3: "#2D6A4F", 4: "#C44E52"}

VALUE_TIER = {v: t for t, subs in HFACS_SCHEMA.items() for v in subs}


def _tier_color(tier):
    return LEVEL_COLOR.get(LEVEL_OF.get(tier, -1), "#999999")


def _layered_pos(node_levels: dict, jitter=0.0):
    """x = level; y = evenly spread within the level. node_levels: node->level."""
    by_level = defaultdict(list)
    for n, lv in node_levels.items():
        by_level[lv].append(n)
    pos = {}
    for lv, nodes in by_level.items():
        nodes = sorted(nodes)
        m = len(nodes)
        ys = np.linspace(1, -1, m) if m > 1 else [0.0]
        for n, y in zip(nodes, ys):
            pos[n] = (lv * 2.4, float(y) + jitter)
    return pos


def _save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# 1. Causal DAG (schema) — DAG_EDGES laid out by HFACS level
# ---------------------------------------------------------------------------

def fig_causal_dag_schema():
    G = nx.DiGraph()
    G.add_nodes_from(LEVEL_OF.keys())
    G.add_edges_from(DAG_EDGES)
    pos = _layered_pos({t: LEVEL_OF[t] for t in G.nodes})

    fig, ax = plt.subplots(figsize=(15, 9))
    for lv, name in LEVEL_NAME.items():
        ax.text(lv * 2.4, 1.25, name, ha="center", va="bottom", fontsize=12,
                fontweight="bold", color=LEVEL_COLOR[lv])
    for a, b in G.edges:
        if a in pos and b in pos:
            ax.add_patch(FancyArrowPatch(pos[a], pos[b], arrowstyle="-|>",
                         mutation_scale=14, color="#888888", lw=1.3,
                         connectionstyle="arc3,rad=0.06", alpha=0.8, zorder=1))
    for n, (x, y) in pos.items():
        ax.scatter([x], [y], s=900, color=_tier_color(n), edgecolors="white",
                   linewidths=1.5, zorder=2)
        ax.text(x, y, n.replace("_", "\n"), ha="center", va="center",
                fontsize=7.5, color="white", fontweight="bold", zorder=3)
    ax.set_title("HFACS Causal DAG (schema) — Organizational → Supervision → "
                 "Preconditions → Unsafe Acts → Severity", fontsize=13, fontweight="bold")
    ax.axis("off"); ax.set_ylim(-1.4, 1.5)
    _save(fig, "causal_dag_schema.png")


# ---------------------------------------------------------------------------
# 2. Data quality — cross-source shared features
# ---------------------------------------------------------------------------

def _counts(df, col, order=None):
    vc = df[col].fillna("Unknown").astype(str).value_counts()
    if order:
        return [int(vc.get(o, 0)) for o in order]
    return vc


def fig_data_quality():
    ntsb = pd.read_csv(os.path.join(DATA, "ntsb_clean.csv"), dtype=str)
    asias = pd.read_csv(os.path.join(DATA, "asias_clean.csv"), dtype=str)
    asrs = pd.read_csv(os.path.join(DATA, "asrs_clean.csv"), dtype=str)
    sources = [("NTSB", ntsb), ("ASIAS", asias), ("ASRS", asrs)]
    scolor = {"NTSB": "#4C72B0", "ASIAS": "#55A868", "ASRS": "#DD8452"}

    feats = {
        "visual_condition": ["VMC", "IMC", "Unknown"],
        "light_conditions": ["Daylight", "Night", "Dusk", "Dawn", "Unknown"],
        "person_involved": ["PIC", "CoPilot", "Maintenance", "ATC", "Other", "Unknown"],
        "pilot_hours_bracket": ["<500", "500-2000", "2000-5000", "5000+", "Unknown"],
    }
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle("Cross-source data quality — shared KG/LSTM features",
                 fontsize=15, fontweight="bold")
    axflat = axes.flatten()

    for ax, (feat, order) in zip(axflat, feats.items()):
        x = np.arange(len(order)); w = 0.26
        for i, (sname, df) in enumerate(sources):
            frac = np.array(_counts(df, feat, order)) / max(len(df), 1) * 100
            ax.bar(x + (i - 1) * w, frac, w, label=sname, color=scolor[sname])
        ax.set_xticks(x); ax.set_xticklabels(order, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("% of records"); ax.set_title(feat); ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    # Unknown rate panel
    ax = axflat[4]
    feats2 = list(feats.keys())
    x = np.arange(len(feats2)); w = 0.26
    for i, (sname, df) in enumerate(sources):
        unk = [100 * (df[f].fillna("Unknown").astype(str) == "Unknown").mean() for f in feats2]
        ax.bar(x + (i - 1) * w, unk, w, label=sname, color=scolor[sname])
    ax.set_xticks(x); ax.set_xticklabels([f.split("_")[0] for f in feats2], fontsize=9)
    ax.set_ylabel("% 'Unknown'"); ax.set_title("Missing/Unknown rate"); ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # severity distribution (NTSB + ASIAS)
    ax = axflat[5]
    sev_levels = [0, 1, 2, 3, 4]; xs = np.arange(len(sev_levels)); w = 0.38
    for i, (sname, df) in enumerate([("NTSB", ntsb), ("ASIAS", asias)]):
        s = pd.to_numeric(df["severity_class"], errors="coerce").dropna().astype(int)
        frac = [100 * (s == lv).mean() for lv in sev_levels]
        ax.bar(xs + (i - 0.5) * w, frac, w, label=sname, color=scolor[sname])
    ax.set_xticks(xs); ax.set_xticklabels([f"sev {l}" for l in sev_levels])
    ax.set_ylabel("% of records"); ax.set_title("Severity class distribution")
    ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    _save(fig, "data_quality.png")


# ---------------------------------------------------------------------------
# 3. Extraction coverage — text-mining quality
# ---------------------------------------------------------------------------

def fig_extraction_coverage():
    path = os.path.join(DATA, "hfacs_results.csv")
    if not os.path.exists(path):
        print("  skip extraction_coverage: hfacs_results.csv not found")
        return
    import json
    df = pd.read_csv(path, dtype=str)
    total = max(len(df), 1)
    ok = df[df["extraction_status"] == "success"]

    tier_cov = Counter(); n_subs = []; n_rel = 0
    for _, r in ok.iterrows():
        try:
            h = json.loads(r["hfacs_json"] or "{}")
        except (json.JSONDecodeError, TypeError):
            h = {}
        cnt = 0
        for t, subs in h.items():
            if subs:
                tier_cov[t] += 1; cnt += len(subs)
        n_subs.append(cnt)
        try:
            n_rel += len(json.loads(r["relationships_json"] or "[]"))
        except (json.JSONDecodeError, TypeError):
            pass

    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    fig.suptitle(f"Text-mining extraction coverage  ({total} records)",
                 fontsize=15, fontweight="bold")

    # status
    ax = axes[0, 0]
    st = df["extraction_status"].value_counts()
    colors = {"success": "#55A868", "empty": "#E9C46A", "parse_error": "#C44E52"}
    bars = ax.bar(st.index, st.values, color=[colors.get(s, "#999") for s in st.index])
    for b, v in zip(bars, st.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v}\n({100*v/total:.0f}%)",
                ha="center", va="bottom", fontsize=10)
    ax.set_title("Parse status"); ax.set_ylabel("records")
    ax.spines[["top", "right"]].set_visible(False)

    # per-tier coverage
    ax = axes[0, 1]
    tiers = list(HFACS_SCHEMA.keys())
    cov = [100 * tier_cov.get(t, 0) / max(len(ok), 1) for t in tiers]
    ax.barh(tiers[::-1], cov[::-1], color=[_tier_color(t) for t in tiers[::-1]])
    ax.set_xlabel("% of successful records with tier"); ax.set_title("Per-tier coverage")
    ax.tick_params(axis="y", labelsize=8); ax.spines[["top", "right"]].set_visible(False)

    # subcategories per record
    ax = axes[1, 0]
    if n_subs:
        ax.hist(n_subs, bins=range(0, max(n_subs) + 2), color="#4C72B0", edgecolor="white")
        ax.axvline(np.mean(n_subs), color="#C44E52", ls="--",
                   label=f"mean {np.mean(n_subs):.1f}")
        ax.legend()
    ax.set_xlabel("subcategories per record"); ax.set_ylabel("records")
    ax.set_title("Extraction density"); ax.spines[["top", "right"]].set_visible(False)

    # summary text
    ax = axes[1, 1]; ax.axis("off")
    succ = int((df["extraction_status"] == "success").sum())
    txt = (f"records: {total}\n"
           f"success: {succ} ({100*succ/total:.0f}%)\n"
           f"tiers covered: {len([t for t in tiers if tier_cov.get(t,0)])}/15\n"
           f"avg subcategories/record: {np.mean(n_subs):.1f}\n"
           f"Task-2 relationships: {n_rel}")
    ax.text(0.05, 0.5, txt, fontsize=14, va="center", family="monospace")
    ax.set_title("Summary")

    plt.tight_layout()
    _save(fig, "extraction_coverage.png")


# ---------------------------------------------------------------------------
# Neo4j helpers
# ---------------------------------------------------------------------------

def _driver():
    from neo4j import GraphDatabase
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    auth = (os.environ.get("NEO4J_USER", "neo4j"),
            os.environ.get("NEO4J_PASSWORD", "neo4j"))
    d = GraphDatabase.driver(uri, auth=auth)
    d.verify_connectivity()
    return d


def _fetch(driver, q, **kw):
    recs, _, _ = driver.execute_query(q, database_=os.environ.get("NEO4J_DATABASE", "neo4j"), **kw)
    return [r.data() for r in recs]


# ---------------------------------------------------------------------------
# 4. Factor-level learned causal graph
# ---------------------------------------------------------------------------

def fig_kg_factor_network(driver):
    leads = _fetch(driver,
                   "MATCH (a:HFACSFactorNode)-[r:LEADS_TO]->(b:HFACSFactorNode) "
                   "RETURN a.value AS a, b.value AS b, r.weight AS w, "
                   "r.evidence IS NOT NULL AS ev")
    if not leads:
        print("  skip kg_factor_network: no LEADS_TO edges"); return
    nodes = set()
    for e in leads:
        nodes.add(e["a"]); nodes.add(e["b"])
    levels = {n: LEVEL_OF.get(VALUE_TIER.get(n, ""), 2) for n in nodes}
    pos = _layered_pos(levels)

    fig, ax = plt.subplots(figsize=(20, 12))
    for lv, name in LEVEL_NAME.items():
        ax.text(lv * 2.4, 1.3, name, ha="center", fontsize=12, fontweight="bold",
                color=LEVEL_COLOR[lv])
    wmax = max(e["w"] or 1 for e in leads)
    for e in leads:
        a, b = e["a"], e["b"]
        if a not in pos or b not in pos:
            continue
        lw = 0.6 + 3.2 * (e["w"] or 1) / wmax
        col = "#C44E52" if e["ev"] else "#BBBBBB"
        ax.add_patch(FancyArrowPatch(pos[a], pos[b], arrowstyle="-|>",
                     mutation_scale=11, color=col, lw=lw, alpha=0.85 if e["ev"] else 0.4,
                     connectionstyle="arc3,rad=0.08", zorder=1))
    for n, (x, y) in pos.items():
        ax.scatter([x], [y], s=420, color=_tier_color(VALUE_TIER.get(n, "")),
                   edgecolors="white", linewidths=1, zorder=2)
        ax.text(x, y - 0.055, n, ha="center", va="top", fontsize=6.2, zorder=3)
    ax.plot([], [], color="#C44E52", lw=2.5, label="LEADS_TO with LLM evidence")
    ax.plot([], [], color="#BBBBBB", lw=1.5, label="structural co-occurrence")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_title("Learned causal graph (factor level) — LEADS_TO edges, "
                 "width ∝ weight", fontsize=14, fontweight="bold")
    ax.axis("off"); ax.set_ylim(-1.5, 1.55)
    _save(fig, "kg_factor_network.png")


# ---------------------------------------------------------------------------
# 5. Tier-level aggregated causal flow
# ---------------------------------------------------------------------------

def fig_kg_tier_causal(driver):
    rows = _fetch(driver,
                  "MATCH (a:HFACSFactorNode)-[r:LEADS_TO]->(b:HFACSFactorNode) "
                  "RETURN a.tier AS ta, b.tier AS tb, r.weight AS w")
    agg = Counter()
    for r in rows:
        if r["ta"] and r["tb"]:
            agg[(r["ta"], r["tb"])] += (r["w"] or 1)
    if not agg:
        print("  skip kg_tier_causal: no tier LEADS_TO"); return

    tiers = [t for t in LEVEL_OF if any(t in k for k in agg)]
    pos = _layered_pos({t: LEVEL_OF[t] for t in LEVEL_OF})
    fig, ax = plt.subplots(figsize=(16, 10))
    for lv, name in LEVEL_NAME.items():
        ax.text(lv * 2.4, 1.3, name, ha="center", fontsize=12, fontweight="bold",
                color=LEVEL_COLOR[lv])
    wmax = max(agg.values())
    for (ta, tb), w in agg.items():
        if ta not in pos or tb not in pos:
            continue
        lw = 1 + 6 * w / wmax
        ax.add_patch(FancyArrowPatch(pos[ta], pos[tb], arrowstyle="-|>",
                     mutation_scale=16, color="#4361EE", lw=lw, alpha=0.55,
                     connectionstyle="arc3,rad=0.1", zorder=1))
        mx, my = (pos[ta][0] + pos[tb][0]) / 2, (pos[ta][1] + pos[tb][1]) / 2
        ax.text(mx, my, str(int(w)), fontsize=8, color="#222", ha="center",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7), zorder=2)
    for t in LEVEL_OF:
        x, y = pos[t]
        ax.scatter([x], [y], s=1100, color=_tier_color(t), edgecolors="white",
                   linewidths=1.5, zorder=3)
        ax.text(x, y, t.replace("_", "\n"), ha="center", va="center", fontsize=7.5,
                color="white", fontweight="bold", zorder=4)
    ax.set_title("Tier-level causal flow — aggregated LEADS_TO weights (O→A→B→C→D)",
                 fontsize=14, fontweight="bold")
    ax.axis("off"); ax.set_ylim(-1.45, 1.5)
    _save(fig, "kg_tier_causal.png")


# ---------------------------------------------------------------------------
# 6. Example event subgraph
# ---------------------------------------------------------------------------

def fig_kg_event_subgraph(driver, event_id=None):
    if event_id is None:
        row = _fetch(driver, "MATCH (e:EventNode)-[:HAS_FACTOR]->() "
                             "RETURN e.event_id AS id, e.source AS src LIMIT 1")
        if not row:
            print("  skip kg_event_subgraph: no events with factors"); return
        event_id, src = row[0]["id"], row[0]["src"]
    else:
        r = _fetch(driver, "MATCH (e:EventNode {event_id:$id}) RETURN e.source AS src", id=event_id)
        src = r[0]["src"] if r else "?"

    facs = _fetch(driver, "MATCH (e:EventNode {event_id:$id})-[:HAS_FACTOR]->(f) "
                          "RETURN f.value AS v, f.tier AS t", id=event_id)
    ctx = _fetch(driver, "MATCH (e:EventNode {event_id:$id})-[rel]->(c) "
                         "WHERE type(rel) STARTS WITH 'HAS_' AND type(rel)<>'HAS_FACTOR' "
                         "RETURN c.feature AS f, coalesce(c.value,c.value_bracket) AS v",
                 id=event_id)
    G = nx.Graph(); center = f"EVENT\n{event_id}"
    G.add_node(center)
    for f in facs[:24]:
        G.add_node(f["v"], kind="factor", tier=f["t"]); G.add_edge(center, f["v"])
    for c in ctx:
        lbl = f"{c['f']}={c['v']}"; G.add_node(lbl, kind="context"); G.add_edge(center, lbl)

    pos = nx.spring_layout(G, seed=42, k=0.9)
    fig, ax = plt.subplots(figsize=(15, 11))
    for n, (x, y) in pos.items():
        if n == center:
            col, s = "#222222", 1800
        elif G.nodes[n].get("kind") == "factor":
            col, s = _tier_color(G.nodes[n].get("tier", "")), 600
        else:
            col, s = "#999999", 500
        ax.scatter([x], [y], s=s, color=col, edgecolors="white", linewidths=1.2, zorder=2)
        ax.text(x, y, n, ha="center", va="center", fontsize=6.5,
                color="white" if n == center else "black", zorder=3,
                fontweight="bold" if n == center else "normal")
    for a, b in G.edges:
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], color="#cccccc", lw=0.8, zorder=1)
    ax.set_title(f"Example event subgraph — {src} {event_id} "
                 f"({len(facs)} HFACS factors, {len(ctx)} context)", fontsize=13, fontweight="bold")
    ax.axis("off")
    _save(fig, "kg_event_subgraph.png")


# ---------------------------------------------------------------------------
# 7. KG stats
# ---------------------------------------------------------------------------

def fig_kg_stats(driver):
    nodes = _fetch(driver, "MATCH (n) RETURN labels(n)[0] AS l, count(*) AS c ORDER BY c DESC")
    edges = _fetch(driver, "MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c ORDER BY c DESC")
    weights = [r["w"] for r in _fetch(driver,
               "MATCH ()-[r:LEADS_TO]->() RETURN r.weight AS w") if r["w"]]
    top = _fetch(driver, "MATCH (a:HFACSFactorNode)-[r:LEADS_TO]->(b:HFACSFactorNode) "
                         "RETURN a.value AS a, b.value AS b, r.weight AS w, "
                         "r.evidence IS NOT NULL AS ev ORDER BY r.weight DESC LIMIT 15")

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle("Knowledge graph — structure overview", fontsize=15, fontweight="bold")

    ax = axes[0, 0]
    ax.barh([n["l"] for n in nodes][::-1], [n["c"] for n in nodes][::-1], color="#5A189A")
    ax.set_title("Node counts by label"); ax.spines[["top", "right"]].set_visible(False)

    ax = axes[0, 1]
    ax.barh([e["t"] for e in edges][::-1], [e["c"] for e in edges][::-1], color="#2D6A4F")
    ax.set_title("Edge counts by type"); ax.spines[["top", "right"]].set_visible(False)
    for i, e in enumerate(edges[::-1]):
        ax.text(e["c"], i, f" {e['c']}", va="center", fontsize=8)

    ax = axes[1, 0]
    if weights:
        ax.hist(weights, bins=range(1, max(weights) + 2), color="#4361EE", edgecolor="white")
    ax.set_xlabel("LEADS_TO weight"); ax.set_ylabel("edges"); ax.set_title("Edge weight distribution")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1, 1]
    labels = [f"{t['a']} → {t['b']}" for t in top][::-1]
    vals = [t["w"] for t in top][::-1]
    cols = ["#C44E52" if t["ev"] else "#BBBBBB" for t in top][::-1]
    ax.barh(labels, vals, color=cols)
    ax.set_title("Top-15 LEADS_TO causal edges (red = has evidence)")
    ax.tick_params(axis="y", labelsize=7); ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    _save(fig, "kg_stats.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-kg", action="store_true", help="offline figures only")
    ap.add_argument("--event-id", default=None)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("Offline figures:")
    fig_causal_dag_schema()
    fig_data_quality()
    fig_extraction_coverage()

    if not args.no_kg:
        print("KG figures (Neo4j):")
        try:
            driver = _driver()
        except Exception as e:
            print(f"  Neo4j unavailable ({e.__class__.__name__}) — skipping KG figures. "
                  "Start Neo4j + set NEO4J_PASSWORD, or use --no-kg.")
            driver = None
        if driver is not None:
            try:
                fig_kg_factor_network(driver)
                fig_kg_tier_causal(driver)
                fig_kg_event_subgraph(driver, args.event_id)
                fig_kg_stats(driver)
            finally:
                driver.close()

    print(f"\nDone. Figures in {FIG}/")


if __name__ == "__main__":
    main()
