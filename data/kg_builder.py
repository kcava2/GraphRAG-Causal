"""
Knowledge Graph Construction (Stage 3)
======================================
Reads ``data/asias_clean.csv`` and ``data/asrs_clean.csv`` (Stage 1), runs the
Stage-2 Task 1 (HFACS classification) + Task 2 (relationship extraction) LLM
passes **inline** on the ASIAS/ASRS/NTSB narratives, and builds a Neo4j knowledge
graph (MERGE/upsert only). Finally it writes per-source read-only FAISS indexes
(``asias.faiss`` / ``asrs.faiss`` / ``ntsb_kg.faiss``) for semantic retrieval.

**NTSB-in-KG is a DISJOINT in-distribution slice** — records NOT in
``ntsb_subset.csv`` (i.e. never used for LSTM train/val/test), passed via
``--ntsb-csv``. This lets the precondition prior be sourced in-distribution
without leakage. Extraction targets Preconditions + Unsafe Acts tiers only
(organizational/supervisory are no longer mined). The KG/indexes are read-only
afterward.

Graph shape
-----------
    (EventNode {event_id, source})            one per ASIAS/ASRS record
      -[:HAS_FACTOR]->          (HFACSFactorNode {tier, value})          shared
      -[:HAS_ENV_CONTEXT]->     (EnvironmentalContextNode {feature,value})  shared
      -[:HAS_PERSONNEL_CONTEXT]->(PersonnelContextNode {feature,value})     shared
      -[:HAS_ORG_CONTEXT]->     (OrganizationalContextNode {feature,value_bracket})

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
    HFACS_SCHEMA, VALID_SUBS, VALID_RELATIONS, EXTRACT_TIERS,
    SYSTEM_TASK1, SYSTEM_TASK2,
    task1_schema, task2_schema,
    _call_ollama, _extract_json,
    _validate_classifications, _validate_relationships,
    _resolve_model, _clean, _GEN_OPTIONS,
    DEFAULT_MODEL,
)
from ntsbdataloader import RETRIEVAL_MODEL  # noqa: E402  (FAISS must match retrieval)
from standardize import strip_outcome  # noqa: E402  (index the same text retrieval queries)

import json  # noqa: E402  (after the package import block, mirrors extractor style)

# EventNode -> context-node edge type, by context label. Granular types let the
# Stage-5 retriever's Task-3 Cypher target them directly (HAS_ENV_CONTEXT, ...).
CONTEXT_EDGE = {
    "EnvironmentalContextNode": "HAS_ENV_CONTEXT",
    "PersonnelContextNode": "HAS_PERSONNEL_CONTEXT",
    "OrganizationalContextNode": "HAS_ORG_CONTEXT",
    "TechnologicalContextNode": "HAS_TECH_CONTEXT",   # SDR maintenance-reliability
    "OperationalContextNode": "HAS_OPS_CONTEXT",      # phase of flight (derive_phase.py)
}

_HERE = os.path.dirname(os.path.abspath(__file__))
ASIAS_CSV = os.path.join(_HERE, "asias_clean.csv")
ASRS_CSV = os.path.join(_HERE, "asrs_clean.csv")
# NTSB-KG: a DISJOINT in-distribution slice (records NOT in ntsb_subset.csv, i.e.
# never used for LSTM train/test). Built so the precondition prior can be sourced
# in-distribution without leakage. Default points at the selected slice.
NTSB_CSV = os.path.join(_HERE, "ntsb_kg_subset.csv")


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

# Grammar vocabularies for structured outputs. EXTRACT_TIERS (not the full
# 15-tier HFACS_SCHEMA) because `_validate_classifications` defaults to the
# extract tiers — org/supervisory/situational_phys were already dropped after
# generation, so constraining the grammar to them changes nothing downstream
# and stops the model spending tokens on tiers that get discarded.
KG_TASK1_FORMAT = task1_schema(EXTRACT_TIERS)
KG_TASK2_FORMAT = task2_schema(EXTRACT_TIERS)


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
    """Relationships between TIERS — mirrors Stage 2 `_build_task2_prompt`.

    This previously asked for SUBCATEGORIES, but `_validate_relationships`
    only accepts subject/object that are tier names, so every relationship
    the model returned was discarded and no LLM evidence edge ever reached
    the KG.
    """
    return (
        f"NARRATIVE:\n{narrative}\n\n"
        f"HFACS ENTITIES:\n{json.dumps(entities)}\n\n"
        f"VALID HFACS TIERS (subject/object must be one of these tier names):\n"
        f"{json.dumps(EXTRACT_TIERS)}\n\n"
        'Extract directed causal relationships between TIERS. relation must '
        'be "LEADS_TO" or "CO_OCCURS_WITH". Respond with valid JSON only, '
        'in the shape:\n'
        '{"relationships": [{"subject": "<tier>", "relation": "LEADS_TO", '
        '"object": "<tier>", "evidence": "<narrative phrase>"}]}'
    )


def extract_record(model_name: str, narrative: str, context: str):
    """Run Task 1 + Task 2; return (entities, classifications, relationships, status)."""
    raw1 = _call_ollama(model_name, SYSTEM_TASK1, _kg_task1_prompt(narrative, context),
                        schema=KG_TASK1_FORMAT)
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
        raw2 = _call_ollama(model_name, SYSTEM_TASK2, _kg_task2_prompt(narrative, entities),
                            schema=KG_TASK2_FORMAT)
        relationships = _validate_relationships(_extract_json(raw2))
    return entities, classifications, relationships, status


# ---------------------------------------------------------------------------
# Source-specific row helpers
# ---------------------------------------------------------------------------

_ID_COL = {"ASIAS": "accident_id", "ASRS": "acn", "NTSB": "ev_id"}
_DEFAULT_CSV = {"ASIAS": ASIAS_CSV, "ASRS": ASRS_CSV, "NTSB": NTSB_CSV}


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
    elif source == "NTSB":
        narrative = _clean(row.get("combined_text"))
        ctx_cols = [
            ("Visual condition", "visual_condition"),
            ("Light condition", "light_conditions"),
            ("NTSB finding path", "finding_description_agg"),
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


# SDR maintenance-reliability lookup {(make_upper, year): bracket}, lazy-loaded from
# data/sdr_defect_brackets.csv (built by sdr_defect_rate.py). Absent file -> no SDR
# context (the rest of the KG is unaffected).
_SDR_BRACKETS = None
_SDR_PATH = os.path.join(_HERE, "sdr_defect_brackets.csv")


def _sdr_brackets():
    global _SDR_BRACKETS
    if _SDR_BRACKETS is None:
        if os.path.exists(_SDR_PATH):
            t = pd.read_csv(_SDR_PATH, dtype=str)
            _SDR_BRACKETS = {(str(r["make"]).upper(), int(float(r["year"]))): r["bracket"]
                             for _, r in t.iterrows()}
        else:
            _SDR_BRACKETS = {}
    return _SDR_BRACKETS


def _sdr_bracket(row: pd.Series, source: str):
    """maintenance_defect_rate bracket for the event's manufacturer + year, or None.
    Keyed by event year so it never reflects post-accident maintenance history."""
    table = _sdr_brackets()
    if not table:
        return None
    from standardize import normalize_make
    make = normalize_make(row.get("acft_make") or row.get("manufacturer"))
    if not make:
        return None
    try:
        year = int(float(_clean(row.get("year"))))
    except (ValueError, TypeError):
        return None
    return table.get((make, year))


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
    # Economic pressure context (QoQ brackets). 'unknown' (neutral/missing, e.g.
    # pre-2002 load factor or a 0% delta) is OMITTED so retrieval never matches on
    # the absence of a signal.
    for feat, col in (("employment_pressure", "employment_bracket"),
                      ("fuel_cost_pressure", "fuel_bracket"),
                      ("revenue_pressure", "revenue_bracket"),
                      ("utilization_pressure", "loadfactor_bracket")):
        b = _clean(row.get(col))
        if b and b != "unknown":
            nodes.append(("OrganizationalContextNode",
                          {"feature": feat, "value_bracket": b}))
    sdr = _sdr_bracket(row, source)
    if sdr:
        nodes.append(("TechnologicalContextNode",
                      {"feature": "maintenance_defect_rate", "value_bracket": sdr}))
    return nodes


def _event_date(row: pd.Series):
    y, m = _clean(row.get("year")), _clean(row.get("month"))
    try:
        return f"{int(float(y)):04d}-{int(float(m)):02d}"
    except (ValueError, TypeError):
        return None


def _asrs_severity(result):
    """Severity ordinal for an ASRS report, derived from its `result` field.

    ASRS records no injury counts, but `result` lists what the event led to, and two
    of its entries are outcome statements. That is enough to place most reports on
    the same gravity scale the other sources use:

        no result recorded                        -> None  (unknown)
        'Physical Injury / Incapacitation'        -> None  (an injury, but minor vs
                                                            serious is not stated,
                                                            and that is exactly the
                                                            low/high boundary)
        'Aircraft Damaged'                        -> 1     (damage of unstated
                                                            extent; low either way)
        anything else                             -> 0     (no harm reported)

    Every value this returns is on the LOW side of the high/low split. That is the
    honest content of an incident-report database, not a gap to be filled.
    """
    res = _clean(result).lower()
    if not res:
        return None
    if "physical injury" in res:
        return None
    if "aircraft damaged" in res:
        return 1
    return 0


def _severity(row: pd.Series, source: str):
    # NTSB + ASIAS carry severity_class; ASRS severity is derived from `result`.
    # Stored on the EventNode so retrieved neighbours carry their outcome.
    if source == "ASRS":
        return _asrs_severity(row.get("result"))
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
            "CREATE INDEX tech_key IF NOT EXISTS FOR (n:TechnologicalContextNode) ON (n.feature, n.value_bracket)",
            "CREATE INDEX ops_key IF NOT EXISTS FOR (n:OperationalContextNode) ON (n.feature, n.value)",
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
        edge = CONTEXT_EDGE[label]
        self.stats[label] += 1
        self.stats[edge] += 1
        keystr = ", ".join(f"{k}:${k}" for k in keys)
        self._run(
            f"MATCH (e:EventNode {{event_id:$id, source:$src}}) "
            f"MERGE (c:{label} {{{keystr}}}) "
            f"MERGE (e)-[:{edge}]->(c)",
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

    def cooccur_context_with_event_factors(self, event_id, source, label, keys):
        """Co-occur a (newly added) context node with ALL of an event's existing
        HFACS factors in one query — used by the context-only updater, which has no
        re-mined factor list. No-op if the event has no factors yet."""
        keystr = ", ".join(f"{k}:${k}" for k in keys)
        self._run(
            f"MATCH (e:EventNode {{event_id:$id, source:$src}})-[:HAS_FACTOR]->(f:HFACSFactorNode) "
            f"MERGE (c:{label} {{{keystr}}}) "
            f"MERGE (c)-[r:CO_OCCURS_WITH]->(f) "
            f"ON CREATE SET r.weight=1 ON MATCH SET r.weight=coalesce(r.weight,0)+1",
            id=event_id, src=source, **keys,
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

    # LLM-extracted relationship edges (with evidence). Subject/object are TIER
    # names (that is all `_validate_relationships` accepts), while factor nodes
    # are keyed by (tier, value) — so each relationship is landed on the values
    # actually extracted for this event. A tier with no extracted value has no
    # node to attach to and is skipped.
    #
    # This path was previously dead: the prompt asked for subcategories, the
    # validator dropped them all, and the `VALID_SUBS[subject]` lookup below
    # would have raised KeyError on any tier that did survive.
    values_by_tier: dict[str, list[str]] = {}
    for t, v in active:
        values_by_tier.setdefault(t, []).append(v)

    for r in relationships:
        subj, obj, rel = r["subject"], r["object"], r["relation"]
        evidence = r.get("evidence")
        for v1 in values_by_tier.get(subj, []):
            for v2 in values_by_tier.get(obj, []):
                if (subj, v1) == (obj, v2):
                    continue
                if rel == "LEADS_TO":
                    writer.merge_factor_edge(subj, v1, obj, v2, "LEADS_TO",
                                             evidence=evidence)
                else:
                    (a, b) = sorted([(subj, v1), (obj, v2)])
                    writer.merge_factor_edge(a[0], a[1], b[0], b[1],
                                             "CO_OCCURS_WITH", evidence=evidence)

    writer.mark_processed(event_id, source)
    return status


def update_context(writer: KGWriter, source: str, limit=None, path=None):
    """Attach structured CONTEXT nodes to EXISTING EventNodes WITHOUT re-mining
    HFACS — adds the new economic context (operating revenue, load factor) and any
    SDR maintenance context to an already-built KG. Idempotent (MERGE); a no-op for
    events not already in the KG."""
    path = path or _DEFAULT_CSV[source]
    df = pd.read_csv(path, dtype=str)
    if limit:
        df = df.head(limit)
    from tqdm import tqdm
    n = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"ctx[{source}]"):
        event_id = _clean(row.get(_ID_COL[source]))
        if not event_id or not writer.is_processed(event_id, source):
            continue                                  # only touch events in the KG
        for label, keys in _context_nodes(row, source):
            writer.connect_event_context(event_id, source, label, keys)
            writer.cooccur_context_with_event_factors(event_id, source, label, keys)
        n += 1
    logging.info("%s: updated context on %d existing events", source, n)


def build_kg(writer: KGWriter, model_name: str, source: str,
             limit=None, sleep=0.0, path=None):
    path = path or _DEFAULT_CSV[source]
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
    if source == "NTSB":
        return _clean(row.get("combined_text"))
    return "\n\n".join(
        x for x in (_clean(row.get("narrative")), _clean(row.get("synopsis"))) if x
    )


def build_faiss(writer: KGWriter, source: str, limit=None, path=None):
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    path = path or _DEFAULT_CSV[source]
    df = pd.read_csv(path, dtype=str)
    if limit:
        df = df.head(limit)
    texts = [_faiss_text(row, source) for _, row in df.iterrows()]
    ids = [_clean(row.get(_ID_COL[source])) for _, row in df.iterrows()]

    logging.info("%s: embedding %d narratives with %s", source, len(texts), RETRIEVAL_MODEL)
    model = SentenceTransformer(RETRIEVAL_MODEL)
    texts = [strip_outcome(t) for t in texts]   # index must match query text
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    emb = np.asarray(emb, dtype="float32")

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    # NTSB-KG index is 'ntsb_kg' to avoid colliding with ntsbdataloader's
    # train-only few-shot 'ntsb.faiss'.
    prefix = "ntsb_kg" if source == "NTSB" else source.lower()
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
# Severity refresh + NTSB training events in the graph (no LLM)
# ---------------------------------------------------------------------------

def update_severity(writer: KGWriter, source: str, path=None):
    """Re-derive severity_class for EXISTING EventNodes of one source and store it.
    No extraction, no new nodes. Used to give ASRS events a severity after the
    fact; safe to repeat."""
    path = path or _DEFAULT_CSV[source]
    df = pd.read_csv(path, dtype=str)
    n_set = n_known = 0
    for _, row in df.iterrows():
        event_id = _clean(row.get(_ID_COL[source]))
        if not event_id:
            continue
        sev = _severity(row, source)
        writer._run(
            "MATCH (e:EventNode {event_id:$id, source:$src}) "
            "SET e.severity_class=$sev, e.severity_basis=$basis",
            id=event_id, src=source, sev=sev,
            basis=("derived from ASRS result field" if source == "ASRS" else "recorded"))
        n_set += 1
        n_known += sev is not None
    logging.info("%s: severity refreshed on %d events (%d known)", source, n_set, n_known)


def update_phase(writer: KGWriter, path=None):
    """Attach a phase-of-flight node to every graph event that has one.

    Phases come from data/derive_phase.py, which reads them from each event's
    narrative with one extractor and one closed vocabulary for every source, so
    the three databases' different native formats never have to be reconciled.
    'unknown' gets no node, so an event without a phase simply cannot match on it.
    Only events already in the graph are touched (MATCH), so the validation and
    test entries in the phase file never enter the graph. Old phase edges are
    removed first, so this is safe to repeat. No LLM call.
    """
    path = path or os.path.join(_HERE, "event_phase.csv")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found. Run:  python data/derive_phase.py")
    ph = pd.read_csv(path, dtype=str).fillna("unknown")
    ph = ph[ph["phase"] != "unknown"]
    writer.ensure_schema()
    writer._run("MATCH (:EventNode)-[r:HAS_OPS_CONTEXT]->() DELETE r")
    rows = [{"id": k, "src": s, "v": v} for k, s, v in zip(ph["key"], ph["source"], ph["phase"])]
    n = 0
    for i in range(0, len(rows), 500):
        res = writer._run(
            "UNWIND $rows AS r "
            "MATCH (e:EventNode {event_id: r.id, source: r.src}) "
            "MERGE (c:OperationalContextNode {feature:'phase_of_flight', value: r.v}) "
            "MERGE (e)-[:HAS_OPS_CONTEXT]->(c) RETURN count(e) AS n",
            rows=rows[i:i + 500])
        n += res[0]["n"] if res else 0
    by = writer._run(
        "MATCH (e:EventNode) OPTIONAL MATCH (e)-[:HAS_OPS_CONTEXT]->(c) "
        "RETURN CASE WHEN e.origin = 'ntsb_train' THEN 'NTSB-train' ELSE e.source END AS s, "
        "count(e) AS n, count(c) AS k")
    logging.info("phase of flight attached to %d graph events; coverage by source: %s", n,
                 {r["s"]: f"{r['k']}/{r['n']}" for r in by})


NTSB_TRAIN_ORIGIN = "ntsb_train"
NTSB_TRAIN_FAISS = os.path.join(_HERE, "ntsb_train_kg.faiss")
NTSB_TRAIN_IDMAP = os.path.join(_HERE, "ntsb_train_kg_id_map.csv")
# Factor nodes are keyed (tier, value) and the KG prompt fills `value` with a schema
# subcategory. The Stage-2 extraction records the TIER plus a free-text evidence
# phrase, not a subcategory, so its factors attach to one tier-level node per tier.
TIER_LEVEL_VALUE = "Unspecified"


def ingest_ntsb_train(writer: KGWriter):
    """Put the NTSB TRAINING events into the knowledge graph, as graph events.

    Why. Retrieval now reads the graph and nothing else. The graph held only
    incident-database events (ASIAS, ASRS) and 100 older NTSB events from a
    different population, so it had almost no outcome information for the events
    being predicted: ASIAS is entirely low severity and ASRS records none. The
    training events are the past investigated accidents a real knowledge base
    would contain, and they are where the severity information is.

    What is written, per training event: the EventNode (source 'NTSB',
    origin 'ntsb_train', its recorded severity ordinal), its context nodes, one
    HAS_FACTOR edge per mined tier, and its extracted causal links. The labels are
    the ones the model is trained on: hfacs_results.csv with the strict violation
    adjudication applied. NO LLM call is made.

    What is deliberately NOT written: factor-factor and context-factor
    co-occurrence edges. Their weights are counters, so re-running this step would
    double them; nothing in retrieval reads them.

    Leakage. Only the TRAIN split goes in. Validation and test events are never
    written, and the retriever refuses to run if it finds one in the graph. A
    training event never retrieves itself (self-exclusion by event id).

    Idempotent: previously ingested training events are removed first. Re-run it
    whenever the split or the labels change; the retriever checks a signature and
    stops with an instruction if the graph is out of date.
    """
    import json
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer
    import ntsbdataloader as N
    from rag_retriever import embed_many

    df = N.load_and_join()
    train, _val, _test = N._split(df)
    raw_sev = pd.read_csv(N.NTSB_CLEAN, dtype=str).set_index("ev_id")["severity_class"]
    hf = pd.read_csv(N.HFACS_RESULTS, dtype=str)
    hf = hf[hf["extraction_status"] == "success"].drop_duplicates("ev_id", keep="last")
    hf_json = dict(zip(hf["ev_id"].astype(str), hf["hfacs_json"].fillna("{}")))
    rel_json = dict(zip(hf["ev_id"].astype(str), hf["relationships_json"].fillna("[]")))

    if writer.driver is not None:
        clash = writer._run(
            "MATCH (e:EventNode {source:'NTSB'}) WHERE e.origin IS NULL "
            "AND e.event_id IN $ids RETURN count(e) AS n",
            ids=train["ev_id"].astype(str).tolist())
        if clash and clash[0]["n"]:
            raise SystemExit(f"{clash[0]['n']} training events already exist in the graph "
                             "as NTSB-KG events. Refusing to overwrite them.")
        old = writer._run("MATCH (e:EventNode {origin:$o}) DETACH DELETE e RETURN count(e) AS n",
                          o=NTSB_TRAIN_ORIGIN)
        logging.info("NTSB-train: removed %d previously ingested events",
                     old[0]["n"] if old else 0)
    writer.ensure_schema()

    from tqdm import tqdm
    ids, texts = [], []
    for _, row in tqdm(train.iterrows(), total=len(train), desc="KG[NTSB-train]"):
        ev = str(row["ev_id"])
        try:
            sev = int(float(raw_sev.get(ev)))
        except (TypeError, ValueError):
            sev = None
        writer.merge_event(ev, "NTSB", _event_date(row), sev)

        links = []
        try:
            for r in json.loads(rel_json.get(ev, "[]")):
                if isinstance(r, dict) and r.get("relation") == "LEADS_TO" \
                        and r.get("subject") in EXTRACT_TIERS and r.get("object") in EXTRACT_TIERS:
                    links.append(f"{r['subject']}>{r['object']}")
        except (json.JSONDecodeError, TypeError):
            pass
        writer._run(
            "MATCH (e:EventNode {event_id:$id, source:'NTSB'}) "
            "SET e.origin=$o, e.causal_links=$links, e.severity_basis='recorded'",
            id=ev, o=NTSB_TRAIN_ORIGIN, links=sorted(set(links)))

        for label, keys in _context_nodes(row, "NTSB"):
            writer.connect_event_context(ev, "NTSB", label, keys)

        try:
            mined = json.loads(hf_json.get(ev, "{}")) or {}
        except (json.JSONDecodeError, TypeError):
            mined = {}
        # Preconditions at the raw-tier level from the extraction; unsafe acts from
        # the dataloader, which has the strict violation adjudication applied.
        tiers = {t for t in N.PRECOND_TIERS if mined.get(t)} | set(row["_uns"])
        for t in sorted(tiers):
            writer.connect_event_factor(ev, "NTSB", t, TIER_LEVEL_VALUE)
        writer.mark_processed(ev, "NTSB")
        ids.append(ev)
        texts.append(strip_outcome(_clean(row.get("combined_text"))))

    # Index over the same text the retriever queries with.
    emb = np.ascontiguousarray(embed_many(SentenceTransformer(RETRIEVAL_MODEL), texts),
                               dtype="float32")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, NTSB_TRAIN_FAISS)
    pd.DataFrame({"embedding_index": range(len(ids)), "event_id": ids}).to_csv(
        NTSB_TRAIN_IDMAP, index=False)
    for i, ev in enumerate(ids):
        writer.set_embedding_index(ev, "NTSB", i)

    sig = N.ntsb_train_signature(train)
    writer._run("MERGE (m:KGMeta {key:$k}) SET m.signature=$sig, m.n=$n, m.updated=$ts",
                k=NTSB_TRAIN_ORIGIN, sig=sig, n=len(ids),
                ts=time.strftime("%Y-%m-%d %H:%M:%S"))
    logging.info("NTSB-train: %d training events in the graph (signature %s); wrote %s "
                 "(ntotal=%d, dim=%d)", len(ids), sig[:12], NTSB_TRAIN_FAISS,
                 index.ntotal, emb.shape[1])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build the ASIAS/ASRS/NTSB Neo4j KG + FAISS indexes")
    parser.add_argument("--source", choices=["asias", "asrs", "ntsb", "both", "all"],
                        default="all")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--num-predict", type=int, default=None,
                        help="Cap generation length (Ollama num_predict) to bound "
                             "slow outliers; ~512 is safe for the bounded JSON.")
    parser.add_argument("--asias-csv", default=None,
                        help="Override ASIAS input CSV (e.g. data/asias_subset.csv).")
    parser.add_argument("--asrs-csv", default=None,
                        help="Override ASRS input CSV (e.g. data/asrs_subset.csv).")
    parser.add_argument("--ntsb-csv", default=None,
                        help="Override NTSB-KG input CSV (the disjoint slice, e.g. "
                             "data/ntsb_kg_subset.csv).")
    parser.add_argument("--faiss-only", action="store_true",
                        help="Skip LLM/KG; only (re)build the FAISS indexes.")
    parser.add_argument("--skip-faiss", action="store_true",
                        help="Build the KG but skip FAISS index construction.")
    parser.add_argument("--update-context", action="store_true",
                        help="Attach NEW structured context (operating revenue, load "
                             "factor, SDR) to existing EventNodes without re-mining "
                             "HFACS or rebuilding FAISS. Cheap; needs the KG already built.")
    parser.add_argument("--dry-run", action="store_true",
                        help="No Neo4j writes; run extraction + edge logic + "
                             "FAISS and print a node/edge tally.")
    parser.add_argument("--update-severity", action="store_true",
                        help="Re-derive and store severity on existing EventNodes of "
                             "the chosen --source (gives ASRS events a severity). No LLM.")
    parser.add_argument("--update-phase", action="store_true",
                        help="Attach phase-of-flight nodes (from data/event_phase.csv, "
                             "built by data/derive_phase.py) to the events in the graph. "
                             "No LLM. Run after --ingest-ntsb-train.")
    parser.add_argument("--ingest-ntsb-train", action="store_true",
                        help="Write the NTSB TRAINING events into the graph from the "
                             "committed labels (no LLM) and build their FAISS index. "
                             "Validation and test events are never written.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    _GEN_OPTIONS["num_ctx"] = args.num_ctx
    if args.num_predict is not None:
        _GEN_OPTIONS["num_predict"] = args.num_predict

    csv_for = {"ASIAS": args.asias_csv, "ASRS": args.asrs_csv, "NTSB": args.ntsb_csv}
    if args.source == "all":
        sources = ["ASIAS", "ASRS", "NTSB"]
    elif args.source == "both":
        sources = ["ASIAS", "ASRS"]
    else:
        sources = [args.source.upper()]
    writer = KGWriter(dry_run=args.dry_run)

    try:
        if args.update_context:
            writer.ensure_schema()
            for src in sources:
                update_context(writer, src, limit=args.limit, path=csv_for[src])
            return

        if args.update_severity or args.ingest_ntsb_train or args.update_phase:
            if args.update_severity:
                for src in sources:
                    update_severity(writer, src, path=csv_for[src])
            if args.ingest_ntsb_train:
                ingest_ntsb_train(writer)
            if args.update_phase:
                update_phase(writer)
            return

        if not args.faiss_only:
            model_name = _resolve_model(args.model)
            logging.info("Using model: %s (num_ctx=%d, sleep=%.1fs, dry_run=%s)",
                         model_name, args.num_ctx, args.sleep, args.dry_run)
            writer.ensure_schema()
            for src in sources:
                build_kg(writer, model_name, src, limit=args.limit,
                         sleep=args.sleep, path=csv_for[src])

        if not args.skip_faiss:
            for src in sources:
                build_faiss(writer, src, limit=args.limit, path=csv_for[src])

        print("\n--- KG build tally (merge ops; MERGE dedups in the DB) ---")
        for k in ("EventNode", "HFACSFactorNode", "EnvironmentalContextNode",
                  "PersonnelContextNode", "OrganizationalContextNode",
                  "HAS_FACTOR", "HAS_ENV_CONTEXT", "HAS_PERSONNEL_CONTEXT",
                  "HAS_ORG_CONTEXT", "LEADS_TO", "CO_OCCURS_WITH"):
            print(f"  {k:<26} {writer.stats.get(k, 0)}")
        if args.dry_run:
            print("\n(--dry-run: no data written to Neo4j)")
    finally:
        writer.close()


if __name__ == "__main__":
    main()
