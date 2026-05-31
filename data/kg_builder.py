"""
Knowledge Graph Construction (Stage 3)
======================================
Reads ``data/asias_clean.csv`` and ``data/asrs_clean.csv`` (Stage 1), runs the
Stage-2 Task 1 (HFACS classification) + Task 2 (relationship extraction) LLM
passes **inline** on the ASIAS/ASRS narratives, and builds a Neo4j knowledge
graph (MERGE/upsert only). Finally it writes two read-only FAISS indexes
(``asias.faiss`` / ``asrs.faiss``) for semantic retrieval.

**NTSB never enters this stage** — not as nodes, edges, properties, or LLM
input. This file never reads ``ntsb_clean.csv`` or ``hfacs_results.csv`` (the
Stage-2 output is the NTSB training corpus only). After this stage the KG and
both indexes are intended to be read-only.

Graph shape
-----------
    (EventNode {event_id, source})            one per ASIAS/ASRS record
      -[:HAS_FACTOR]->   (HFACSFactorNode {tier, value})          shared
      -[:HAS_CONTEXT]->  (EnvironmentalContextNode {feature,value})  shared
                         (PersonnelContextNode {feature,value})       shared
                         (OrganizationalContextNode {feature,value_bracket})

    (HFACSFactorNode)-[:LEADS_TO {weight,evidence}]->(HFACSFactorNode)
    (HFACSFactorNode)-[:CO_OCCURS_WITH {weight,evidence}]-(HFACSFactorNode)
    (context node)-[:CO_OCCURS_WITH {weight}]->(HFACSFactorNode)

Run model (chunked + phased; LLM extraction is very heavy ~98k calls total):
    python data/kg_builder.py --source both                 # KG + FAISS
    python data/kg_builder.py --source asrs --limit 500 --sleep 1.0
    python data/kg_builder.py --faiss-only                  # just indexes
    python data/kg_builder.py --source asias --limit 2 --dry-run   # no DB

Env: NEO4J_URI (bolt://localhost:7687), NEO4J_USER (neo4j), NEO4J_PASSWORD,
NEO4J_DATABASE (neo4j). Ollama model gemma4. SBERT all-MiniLM-L6-v2.
"""

import argparse
import logging
import os
import time
from collections import Counter
from itertools import combinations

import pandas as pd

# Reuse Stage-2 building blocks verbatim (single source of truth; do not modify
# hfacs_extractor.py). Importing it also pulls in ollama/torch — that is fine.
from hfacs_extractor import (  # noqa: E402
    HFACS_SCHEMA, VALID_SUBS, VALID_RELATIONS,
    SYSTEM_TASK1, SYSTEM_TASK2,
    _call_ollama, _extract_json,
    _validate_classifications, _validate_relationships,
    _resolve_model, _clean, _GEN_OPTIONS,
    DEFAULT_MODEL, SBERT_MODEL,
)

import json  # noqa: E402  (after the package import block, mirrors extractor style)

_HERE = os.path.dirname(os.path.abspath(__file__))
ASIAS_CSV = os.path.join(_HERE, "asias_clean.csv")
ASRS_CSV = os.path.join(_HERE, "asrs_clean.csv")


# ---------------------------------------------------------------------------
# DAG edges — direction of LEADS_TO between HFACS tiers
# ---------------------------------------------------------------------------
# The ('unsafe_*', 'severity') pairs are DORMANT in this stage: severity is an
# EventNode property, not an HFACSFactorNode, so no severity factor pairs are
# ever active. They are kept for DAG consistency with the LSTM's terminal node.

DAG_EDGES = {
    ("resource_mgmt", "supervisory"),
    ("org_climate", "supervisory"),
    ("org_process", "supervisory"),
    ("supervisory", "operator_mental"),
    ("supervisory", "operator_physical"),
    ("supervisory", "operator_limits"),
    ("situational_tech", "operator_mental"),
    ("situational_tech", "unsafe_skill"),
    ("situational_tech", "unsafe_decision"),
    ("personnel_crm", "operator_mental"),
    ("personnel_readiness", "operator_physical"),
    ("operator_mental", "unsafe_skill"),
    ("operator_mental", "unsafe_decision"),
    ("operator_mental", "unsafe_perception"),
    ("operator_mental", "unsafe_violation"),
    ("operator_physical", "unsafe_skill"),
    ("operator_physical", "unsafe_perception"),
    ("operator_limits", "unsafe_skill"),
    ("operator_limits", "unsafe_decision"),
    ("operator_limits", "unsafe_perception"),
    ("unsafe_skill", "severity"),
    ("unsafe_decision", "severity"),
    ("unsafe_perception", "severity"),
    ("unsafe_violation", "severity"),
}


def classify_edge(t1: str, t2: str):
    """
    Decide the edge between two HFACS tiers (pure; unit-testable).

    Returns ('LEADS_TO', 'forward')  if (t1,t2) in DAG_EDGES,
            ('LEADS_TO', 'reversed') if (t2,t1) in DAG_EDGES,
            ('CO_OCCURS_WITH', None) otherwise.
    """
    if (t1, t2) in DAG_EDGES:
        return ("LEADS_TO", "forward")
    if (t2, t1) in DAG_EDGES:
        return ("LEADS_TO", "reversed")
    return ("CO_OCCURS_WITH", None)


# ---------------------------------------------------------------------------
# KG-specific prompts — mirror Stage 2 Task 1 / Task 2 exactly, but take a
# source-specific narrative + structured-context block and an empty few-shot.
# ---------------------------------------------------------------------------

def _kg_task1_prompt(narrative: str, context: str) -> str:
    schema_json = json.dumps(HFACS_SCHEMA, indent=2)
    parts = [f"NARRATIVE:\n{narrative}"]
    if context:
        parts.append(f"STRUCTURED CONTEXT:\n{context}")
    parts.append(f"HFACS SCHEMA:\n{schema_json}")
    parts.append(
        'Classify only tiers and subcategories present in the schema above. '
        'Respond with valid JSON only, in the shape:\n'
        '{"entities": [{"text": "...", "role": "...", "tier": "..."}], '
        '"hfacs_classifications": {"<tier_name>": ["<sub_category>"]}}'
    )
    return "\n\n".join(parts)


def _kg_task2_prompt(narrative: str, entities: list) -> str:
    valid_subs = sorted(VALID_SUBS.keys())
    return (
        f"NARRATIVE:\n{narrative}\n\n"
        f"HFACS ENTITIES:\n{json.dumps(entities)}\n\n"
        f"VALID SUBCATEGORIES (subject/object must be one of these):\n"
        f"{json.dumps(valid_subs)}\n\n"
        'Extract directed causal relationships. relation must be "LEADS_TO" or '
        '"CO_OCCURS_WITH". Respond with valid JSON only, in the shape:\n'
        '{"relationships": [{"subject": "<sub>", "relation": "LEADS_TO", '
        '"object": "<sub>", "evidence": "<narrative phrase>"}]}'
    )


def extract_record(model_name: str, narrative: str, context: str):
    """Run Task 1 + Task 2; return (entities, classifications, relationships, status)."""
    raw1 = _call_ollama(model_name, SYSTEM_TASK1, _kg_task1_prompt(narrative, context))
    parsed1 = _extract_json(raw1)
    if parsed1 is None:
        return [], {}, [], "parse_error"

    entities = parsed1.get("entities", [])
    if not isinstance(entities, list):
        entities = []
    classifications = _validate_classifications(parsed1.get("hfacs_classifications"))
    status = "success" if (entities or classifications) else "empty"

    relationships = []
    if status == "success":
        raw2 = _call_ollama(model_name, SYSTEM_TASK2, _kg_task2_prompt(narrative, entities))
        relationships = _validate_relationships(_extract_json(raw2))
    return entities, classifications, relationships, status


# ---------------------------------------------------------------------------
# Source-specific row helpers
# ---------------------------------------------------------------------------

_ID_COL = {"ASIAS": "accident_id", "ASRS": "acn"}


def _narrative_and_context(row: pd.Series, source: str):
    """Build the LLM narrative + STRUCTURED CONTEXT block for a record."""
    if source == "ASIAS":
        narrative = _clean(row.get("combined_narrative"))
        ctx_cols = [
            ("Visual condition", "visual_condition"),
            ("Light condition", "light_conditions"),
            ("Cause factor", "cause_factor"),
            ("Cause subcategory", "cause_subcategory"),
            ("Weather factor", "weather_factor"),
        ]
    else:  # ASRS
        narrative = "\n\n".join(
            x for x in (_clean(row.get("narrative")), _clean(row.get("synopsis"))) if x
        )
        ctx_cols = [
            ("Visual condition", "visual_condition"),
            ("Light condition", "light_conditions"),
            ("Anomaly", "anomaly"),
            ("Human factors", "human_factors"),
            ("Primary problem", "primary_problem"),
        ]
    lines = [f"- {label}: {_clean(row.get(col))}"
             for label, col in ctx_cols if _clean(row.get(col))]
    return narrative, "\n".join(lines)


def _context_nodes(row: pd.Series, source: str):
    """Active context nodes for a record as (label, key_dict). No sky_conditions."""
    nodes = []
    vc = _clean(row.get("visual_condition"))
    if vc:
        nodes.append(("EnvironmentalContextNode", {"feature": "visual_condition", "value": vc}))
    lc = _clean(row.get("light_conditions"))
    if lc:
        nodes.append(("EnvironmentalContextNode", {"feature": "light_conditions", "value": lc}))
    if source == "ASIAS":
        wf = _clean(row.get("weather_factor"))
        if wf:
            nodes.append(("EnvironmentalContextNode", {"feature": "weather_factor", "value": wf}))
    pi = _clean(row.get("person_involved"))
    if pi:
        nodes.append(("PersonnelContextNode", {"feature": "person_involved", "value": pi}))
    ph = _clean(row.get("pilot_hours_bracket"))
    if ph:
        nodes.append(("PersonnelContextNode", {"feature": "pilot_hours_bracket", "value": ph}))
    eb = _clean(row.get("employment_bracket"))
    if eb:
        nodes.append(("OrganizationalContextNode",
                      {"feature": "employment_pressure", "value_bracket": eb}))
    fb = _clean(row.get("fuel_bracket"))
    if fb:
        nodes.append(("OrganizationalContextNode",
                      {"feature": "fuel_cost_pressure", "value_bracket": fb}))
    return nodes


def _event_date(row: pd.Series):
    y, m = _clean(row.get("year")), _clean(row.get("month"))
    try:
        return f"{int(float(y)):04d}-{int(float(m)):02d}"
    except (ValueError, TypeError):
        return None


def _severity(row: pd.Series, source: str):
    if source != "ASIAS":
        return None
    try:
        return int(float(_clean(row.get("severity_class"))))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Neo4j writer (no-op + tally in --dry-run)
# ---------------------------------------------------------------------------

class KGWriter:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.driver = None
        self.database = os.environ.get("NEO4J_DATABASE", "neo4j")
        self.stats = Counter()
        if not dry_run:
            from neo4j import GraphDatabase
            uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
            user = os.environ.get("NEO4J_USER", "neo4j")
            pwd = os.environ.get("NEO4J_PASSWORD", "neo4j")
            try:
                self.driver = GraphDatabase.driver(uri, auth=(user, pwd))
                self.driver.verify_connectivity()
                logging.info("Connected to Neo4j at %s (db=%s)", uri, self.database)
            except Exception as e:
                raise SystemExit(
                    f"\nERROR: cannot reach Neo4j at {uri} ({e}).\n"
                    "Start Neo4j and set NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD, "
                    "or pass --dry-run to run without a database."
                )

    def close(self):
        if self.driver is not None:
            self.driver.close()

    def _run(self, query: str, **params):
        if self.driver is None:
            return []
        records, _, _ = self.driver.execute_query(
            query, database_=self.database, **params
        )
        return records

    # ---- schema ----------------------------------------------------------
    def ensure_schema(self):
        stmts = [
            "CREATE INDEX evnode_key IF NOT EXISTS FOR (n:EventNode) ON (n.event_id, n.source)",
            "CREATE INDEX hfacs_key IF NOT EXISTS FOR (n:HFACSFactorNode) ON (n.tier, n.value)",
            "CREATE INDEX env_key IF NOT EXISTS FOR (n:EnvironmentalContextNode) ON (n.feature, n.value)",
            "CREATE INDEX pers_key IF NOT EXISTS FOR (n:PersonnelContextNode) ON (n.feature, n.value)",
            "CREATE INDEX org_key IF NOT EXISTS FOR (n:OrganizationalContextNode) ON (n.feature, n.value_bracket)",
        ]
        for s in stmts:
            self._run(s)

    # ---- events ----------------------------------------------------------
    def is_processed(self, event_id: str, source: str) -> bool:
        if self.driver is None:
            return False
        recs = self._run(
            "MATCH (e:EventNode {event_id:$id, source:$src}) RETURN e.processed AS p",
            id=event_id, src=source,
        )
        return bool(recs and recs[0]["p"] is True)

    def merge_event(self, event_id, source, date, severity):
        self.stats["EventNode"] += 1
        self._run(
            "MERGE (e:EventNode {event_id:$id, source:$src}) "
            "ON CREATE SET e.processed=false "
            "SET e.date=$date, e.severity_class=$sev",
            id=event_id, src=source, date=date, sev=severity,
        )

    def mark_processed(self, event_id, source):
        self._run(
            "MATCH (e:EventNode {event_id:$id, source:$src}) SET e.processed=true",
            id=event_id, src=source,
        )

    def set_embedding_index(self, event_id, source, idx):
        self._run(
            "MATCH (e:EventNode {event_id:$id, source:$src}) SET e.embedding_index=$idx",
            id=event_id, src=source, idx=idx,
        )

    # ---- event -> node connections (also MERGE the target node) ----------
    def connect_event_factor(self, event_id, source, tier, value):
        self.stats["HFACSFactorNode"] += 1
        self.stats["HAS_FACTOR"] += 1
        self._run(
            "MATCH (e:EventNode {event_id:$id, source:$src}) "
            "MERGE (f:HFACSFactorNode {tier:$t, value:$v}) "
            "MERGE (e)-[:HAS_FACTOR]->(f)",
            id=event_id, src=source, t=tier, v=value,
        )

    def connect_event_context(self, event_id, source, label, keys):
        self.stats[label] += 1
        self.stats["HAS_CONTEXT"] += 1
        keystr = ", ".join(f"{k}:${k}" for k in keys)
        self._run(
            f"MATCH (e:EventNode {{event_id:$id, source:$src}}) "
            f"MERGE (c:{label} {{{keystr}}}) "
            f"MERGE (e)-[:HAS_CONTEXT]->(c)",
            id=event_id, src=source, **keys,
        )

    # ---- edges -----------------------------------------------------------
    def merge_factor_edge(self, t1, v1, t2, v2, relation, evidence=None):
        assert relation in ("LEADS_TO", "CO_OCCURS_WITH")
        self.stats[relation] += 1
        ev_clause = "SET r.evidence=$evidence" if evidence else ""
        self._run(
            f"MERGE (a:HFACSFactorNode {{tier:$t1, value:$v1}}) "
            f"MERGE (b:HFACSFactorNode {{tier:$t2, value:$v2}}) "
            f"MERGE (a)-[r:{relation}]->(b) "
            f"ON CREATE SET r.weight=1 "
            f"ON MATCH SET r.weight=coalesce(r.weight,0)+1 "
            f"{ev_clause}",
            t1=t1, v1=v1, t2=t2, v2=v2, evidence=evidence,
        )

    def merge_context_factor_edge(self, label, keys, tier, value):
        self.stats["CO_OCCURS_WITH"] += 1
        keystr = ", ".join(f"{k}:${k}" for k in keys)
        self._run(
            f"MATCH (c:{label} {{{keystr}}}) "
            f"MATCH (f:HFACSFactorNode {{tier:$t, value:$v}}) "
            f"MERGE (c)-[r:CO_OCCURS_WITH]->(f) "
            f"ON CREATE SET r.weight=1 "
            f"ON MATCH SET r.weight=coalesce(r.weight,0)+1",
            t=tier, v=value, **keys,
        )


# ---------------------------------------------------------------------------
# Per-record pipeline
# ---------------------------------------------------------------------------

def process_record(writer: KGWriter, model_name: str, source: str, row: pd.Series) -> str:
    event_id = _clean(row.get(_ID_COL[source]))
    if not event_id:
        return "skipped"
    if writer.is_processed(event_id, source):
        return "skipped"

    narrative, context = _narrative_and_context(row, source)
    entities, classifications, relationships, status = extract_record(
        model_name, narrative, context
    )

    writer.merge_event(event_id, source, _event_date(row), _severity(row, source))

    ctx_nodes = _context_nodes(row, source)
    for label, keys in ctx_nodes:
        writer.connect_event_context(event_id, source, label, keys)

    active = [(t, v) for t, subs in classifications.items() for v in subs]
    for t, v in active:
        writer.connect_event_factor(event_id, source, t, v)

    # Structural factor-factor edges (DAG-directed or co-occurrence).
    for (t1, v1), (t2, v2) in combinations(active, 2):
        rel, order = classify_edge(t1, t2)
        if rel == "LEADS_TO":
            if order == "forward":
                writer.merge_factor_edge(t1, v1, t2, v2, "LEADS_TO")
            else:
                writer.merge_factor_edge(t2, v2, t1, v1, "LEADS_TO")
        else:
            (a, b) = sorted([(t1, v1), (t2, v2)])
            writer.merge_factor_edge(a[0], a[1], b[0], b[1], "CO_OCCURS_WITH")

    # Context -> factor co-occurrence.
    for label, keys in ctx_nodes:
        for t, v in active:
            writer.merge_context_factor_edge(label, keys, t, v)

    # LLM-extracted relationship edges (with evidence).
    for r in relationships:
        s, o, rel = r["subject"], r["object"], r["relation"]
        ts, to = VALID_SUBS[s], VALID_SUBS[o]
        if rel == "LEADS_TO":
            writer.merge_factor_edge(ts, s, to, o, "LEADS_TO", evidence=r.get("evidence"))
        else:
            (a, b) = sorted([(ts, s), (to, o)])
            writer.merge_factor_edge(a[0], a[1], b[0], b[1], "CO_OCCURS_WITH",
                                     evidence=r.get("evidence"))

    writer.mark_processed(event_id, source)
    return status


def build_kg(writer: KGWriter, model_name: str, source: str,
             limit=None, sleep=0.0):
    path = ASIAS_CSV if source == "ASIAS" else ASRS_CSV
    df = pd.read_csv(path, dtype=str)
    if limit:
        df = df.head(limit)
    logging.info("%s: building KG from %d records", source, len(df))

    from tqdm import tqdm
    status_counts = Counter()
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"KG[{source}]"):
        status_counts[process_record(writer, model_name, source, row)] += 1
        if sleep:
            time.sleep(sleep)
    logging.info("%s status: %s", source, dict(status_counts))


# ---------------------------------------------------------------------------
# FAISS phase
# ---------------------------------------------------------------------------

def _faiss_text(row: pd.Series, source: str) -> str:
    if source == "ASIAS":
        return _clean(row.get("combined_narrative"))
    return "\n\n".join(
        x for x in (_clean(row.get("narrative")), _clean(row.get("synopsis"))) if x
    )


def build_faiss(writer: KGWriter, source: str, limit=None):
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    path = ASIAS_CSV if source == "ASIAS" else ASRS_CSV
    df = pd.read_csv(path, dtype=str)
    if limit:
        df = df.head(limit)
    texts = [_faiss_text(row, source) for _, row in df.iterrows()]
    ids = [_clean(row.get(_ID_COL[source])) for _, row in df.iterrows()]

    logging.info("%s: embedding %d narratives with %s", source, len(texts), SBERT_MODEL)
    model = SentenceTransformer(SBERT_MODEL)
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    emb = np.asarray(emb, dtype="float32")

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    prefix = source.lower()
    faiss_path = os.path.join(_HERE, f"{prefix}.faiss")
    idmap_path = os.path.join(_HERE, f"{prefix}_id_map.csv")
    faiss.write_index(index, faiss_path)
    pd.DataFrame({"embedding_index": range(len(ids)), "event_id": ids}).to_csv(
        idmap_path, index=False
    )
    logging.info("%s: wrote %s (ntotal=%d, dim=%d) and %s",
                 source, faiss_path, index.ntotal, emb.shape[1], idmap_path)

    # Stamp embedding_index back onto EventNodes (only if a live DB).
    if writer is not None and writer.driver is not None:
        for i, ev in enumerate(ids):
            if ev:
                writer.set_embedding_index(ev, source, i)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build the ASIAS/ASRS Neo4j KG + FAISS indexes")
    parser.add_argument("--source", choices=["asias", "asrs", "both"], default="both")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--faiss-only", action="store_true",
                        help="Skip LLM/KG; only (re)build the FAISS indexes.")
    parser.add_argument("--skip-faiss", action="store_true",
                        help="Build the KG but skip FAISS index construction.")
    parser.add_argument("--dry-run", action="store_true",
                        help="No Neo4j writes; run extraction + edge logic + "
                             "FAISS and print a node/edge tally.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    _GEN_OPTIONS["num_ctx"] = args.num_ctx

    sources = ["ASIAS", "ASRS"] if args.source == "both" else [args.source.upper()]
    writer = KGWriter(dry_run=args.dry_run)

    try:
        if not args.faiss_only:
            model_name = _resolve_model(args.model)
            logging.info("Using model: %s (num_ctx=%d, sleep=%.1fs, dry_run=%s)",
                         model_name, args.num_ctx, args.sleep, args.dry_run)
            writer.ensure_schema()
            for src in sources:
                build_kg(writer, model_name, src, limit=args.limit, sleep=args.sleep)

        if not args.skip_faiss:
            for src in sources:
                build_faiss(writer, src, limit=args.limit)

        print("\n--- KG build tally (merge ops; MERGE dedups in the DB) ---")
        for k in ("EventNode", "HFACSFactorNode", "EnvironmentalContextNode",
                  "PersonnelContextNode", "OrganizationalContextNode",
                  "HAS_FACTOR", "HAS_CONTEXT", "LEADS_TO", "CO_OCCURS_WITH"):
            print(f"  {k:<26} {writer.stats.get(k, 0)}")
        if args.dry_run:
            print("\n(--dry-run: no data written to Neo4j)")
    finally:
        writer.close()


if __name__ == "__main__":
    main()
