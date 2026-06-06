"""
rag_retriever.py  (Stage 5)
===========================
Hybrid retrieval for the RAG-augmented LSTM conditions (C2-C4) and inference.
Combines:

    Mode 1  FAISS semantic search over asias.faiss + asrs.faiss (ASIAS weighted
            0.6, ASRS 0.4) on the NTSB record's combined_text.
    Mode 2  Gemma-generated Cypher structural search over the Neo4j KG, using the
            record's standardized structured fields as parameters.

The two score sets are min-max normalized to [0,1] and combined 0.5/0.5; the
top-k EventNodes' HFACSFactorNode mappings, weighted by combined score, build
soft prior distributions over the causal-chain target classes. Priors are
appended to LSTM inputs by NTSBSequenceDataset following the HFACS DAG:

    organizational_prior -> step_o   (Step O)
    supervisory_prior    -> step_a   (Step A)
    precondition_prior   -> step_b   (Step B)

unsafe_prior is constructed for completeness but is not appended to any step.

**Read-only**: this module never writes to Neo4j, asias.faiss, asrs.faiss, or
ntsb.faiss. It only runs MATCH/OPTIONAL MATCH Cypher. On ANY retrieval failure
(Neo4j down, FAISS error, invalid Cypher) it silently returns uniform priors.

Run order: Stage 3 must complete (KG + ASIAS/ASRS FAISS indexes) for real
retrieval; absent those, the retriever still constructs and returns uniform priors.
"""

import logging
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from hfacs_extractor import DEFAULT_MODEL, _call_ollama, _clean  # noqa: E402
from ntsbdataloader import (  # noqa: E402  (label spaces — single source of truth)
    ORG_SUBS, SUP_SUBS, PRECOND_SUBS, UNSAFE_SUBS, N_O, N_A, N_B, N_C,
)

ASIAS_FAISS = os.path.join(_HERE, "asias.faiss")
ASIAS_IDMAP = os.path.join(_HERE, "asias_id_map.csv")
ASRS_FAISS = os.path.join(_HERE, "asrs.faiss")
ASRS_IDMAP = os.path.join(_HERE, "asrs_id_map.csv")
SBERT_MODEL = "all-MiniLM-L6-v2"

ASIAS_WEIGHT, ASRS_WEIGHT = 0.6, 0.4
TOP_K = 5

# subcategory value -> (prior_name, index) — values are unique across tiers.
_GROUPS = {"organizational_prior": ORG_SUBS, "supervisory_prior": SUP_SUBS,
           "precondition_prior": PRECOND_SUBS, "unsafe_prior": UNSAFE_SUBS}
_PRIOR_SIZE = {"organizational_prior": N_O, "supervisory_prior": N_A,
               "precondition_prior": N_B, "unsafe_prior": N_C}
VALUE_TO_GROUP = {v: (name, i) for name, subs in _GROUPS.items()
                  for i, v in enumerate(subs)}


SYSTEM_TASK3 = (
    "You generate ONE read-only Neo4j Cypher query that finds EventNodes similar "
    "to a described accident. Respond with the query only — no markdown, no prose.\n\n"
    "Available node types:\n"
    "  EventNode, HFACSFactorNode, EnvironmentalContextNode,\n"
    "  PersonnelContextNode, OrganizationalContextNode\n"
    "Available edge types:\n"
    "  HAS_FACTOR, HAS_ENV_CONTEXT, HAS_PERSONNEL_CONTEXT,\n"
    "  HAS_ORG_CONTEXT, LEADS_TO, CO_OCCURS_WITH\n\n"
    "Requirements:\n"
    "  - Start the query with MATCH (e:EventNode).\n"
    "  - RETURN e.event_id, e.source, e.embedding_index, and a computed score.\n"
    "  - Use OPTIONAL MATCH for non-required context nodes.\n"
    "  - LIMIT results to $k.\n"
    "  - Use only read clauses (MATCH, OPTIONAL MATCH, WHERE, WITH, RETURN, "
    "ORDER BY, LIMIT). Never MERGE, CREATE, SET, or DELETE.\n"
    "  - Output a single valid Cypher query string, no markdown fencing."
)

# Structured params handed to Task 3 (and bound when executing the query).
CYPHER_PARAMS = ["visual_condition", "light_conditions", "employment_bracket",
                 "fuel_bracket", "person_involved", "pilot_hours_bracket"]

_FORBIDDEN = re.compile(r"\b(MERGE|CREATE|SET|DELETE|REMOVE|DROP|CALL)\b", re.I)


def _minmax(scores: dict) -> dict:
    """Min-max normalize dict values to [0,1]; if all equal, map to 1.0."""
    if not scores:
        return {}
    vals = np.array(list(scores.values()), dtype="float64")
    lo, hi = vals.min(), vals.max()
    if hi - lo < 1e-12:
        return {k: 1.0 for k in scores}
    return {k: float((v - lo) / (hi - lo)) for k, v in scores.items()}


def _uniform_priors() -> dict:
    return {name: np.full(_PRIOR_SIZE[name], 1.0 / _PRIOR_SIZE[name], dtype="float32")
            for name in _GROUPS}


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class RAGRetriever:
    """See module docstring. Construct via build_retriever()."""

    def __init__(self, strategy: str = "hybrid", k: int = TOP_K,
                 model: str = DEFAULT_MODEL):
        self.strategy = strategy            # 'hybrid' | 'faiss' | 'cypher'
        self.k = k
        self.model = model
        self._sbert = None
        self._faiss = {}                    # source -> (index, [event_id,...])
        self.driver = None
        self.database = os.environ.get("NEO4J_DATABASE", "neo4j")
        self._load_faiss()
        self._connect_neo4j()

    # ---- setup (best-effort; never raises) ----
    def _load_faiss(self):
        try:
            import faiss
            import pandas as pd
            for src, fp, mp in (("ASIAS", ASIAS_FAISS, ASIAS_IDMAP),
                                ("ASRS", ASRS_FAISS, ASRS_IDMAP)):
                if os.path.exists(fp) and os.path.exists(mp):
                    ids = pd.read_csv(mp, dtype=str)["event_id"].tolist()
                    self._faiss[src] = (faiss.read_index(fp), ids)
            if not self._faiss:
                logging.warning("RAG: no ASIAS/ASRS FAISS indexes found "
                                "(Stage 3 not run) — semantic mode disabled.")
        except Exception as e:
            logging.warning("RAG: FAISS load failed (%s) — semantic mode disabled.", e)

    def _connect_neo4j(self):
        try:
            from neo4j import GraphDatabase
            uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
            auth = (os.environ.get("NEO4J_USER", "neo4j"),
                    os.environ.get("NEO4J_PASSWORD", "neo4j"))
            drv = GraphDatabase.driver(uri, auth=auth)
            drv.verify_connectivity()
            self.driver = drv
            logging.info("RAG: connected to Neo4j at %s.", uri)
        except Exception as e:
            logging.warning("RAG: Neo4j unavailable (%s) — Cypher mode disabled.", e)

    def _ensure_sbert(self):
        if self._sbert is None:
            from sentence_transformers import SentenceTransformer
            self._sbert = SentenceTransformer(SBERT_MODEL)
        return self._sbert

    def close(self):
        if self.driver is not None:
            self.driver.close()

    # ---- Mode 1: FAISS semantic ----
    def _faiss_scores(self, text: str) -> dict:
        if not self._faiss or not _clean(text):
            return {}
        model = self._ensure_sbert()
        emb = np.asarray(model.encode([text], normalize_embeddings=True), dtype="float32")
        merged = {}
        for src, weight in (("ASIAS", ASIAS_WEIGHT), ("ASRS", ASRS_WEIGHT)):
            if src not in self._faiss:
                continue
            index, ids = self._faiss[src]
            kk = min(self.k, index.ntotal)
            if kk == 0:
                continue
            sims, idx = index.search(emb, kk)
            for s, i in zip(sims[0], idx[0]):
                if 0 <= i < len(ids):
                    merged[(ids[i], src)] = float(s) * weight
        # top-k by weighted score
        return dict(sorted(merged.items(), key=lambda kv: kv[1], reverse=True)[:self.k])

    # ---- Mode 2: Gemma Cypher structural ----
    def _generate_cypher(self, record: dict) -> str | None:
        if self.driver is None:
            return None
        params = "\n".join(f"  ${p} = {record.get(p, '')!r}" for p in CYPHER_PARAMS)
        user = ("Find EventNodes whose context/factors resemble this accident. "
                "Bind these query parameters:\n" + params + "\n  $k = result limit\n\n"
                "Return the Cypher query only.")
        raw = _call_ollama(self.model, SYSTEM_TASK3, user)
        if not raw:
            return None
        q = raw.strip().strip("`").strip()
        if q.lower().startswith("cypher"):
            q = q[6:].strip()
        # Validation: must start with MATCH and contain no write clauses.
        if not q.upper().lstrip().startswith("MATCH") or _FORBIDDEN.search(q):
            logging.warning("RAG: discarding invalid Cypher (failed MATCH/read-only check).")
            return None
        return q

    def _cypher_scores(self, record: dict) -> dict:
        q = self._generate_cypher(record)
        if q is None:
            return {}
        try:
            params = {p: record.get(p, "") for p in CYPHER_PARAMS}
            params["k"] = self.k
            recs, _, _ = self.driver.execute_query(q, database_=self.database, **params)
            out = {}
            for r in recs:
                d = r.data()
                eid, src = d.get("e.event_id"), d.get("e.source")
                score = d.get("score", 1.0)
                if eid is not None and src is not None:
                    out[(str(eid), str(src))] = float(score if score is not None else 1.0)
            return out
        except Exception as e:
            logging.warning("RAG: Cypher execution failed (%s) — skipping.", e)
            return {}

    # ---- combine + factor lookup ----
    def _combine(self, faiss_scores: dict, cypher_scores: dict) -> dict:
        f = _minmax(faiss_scores)
        c = _minmax(cypher_scores)
        keys = set(f) | set(c)
        combined = {k: 0.5 * f.get(k, 0.0) + 0.5 * c.get(k, 0.0) for k in keys}
        return dict(sorted(combined.items(), key=lambda kv: kv[1], reverse=True)[:self.k])

    def _fetch_factors(self, event_id: str, source: str):
        if self.driver is None:
            return []
        try:
            recs, _, _ = self.driver.execute_query(
                "MATCH (e:EventNode {event_id:$id, source:$src})-[:HAS_FACTOR]->"
                "(f:HFACSFactorNode) RETURN f.value AS value",
                id=event_id, src=source, database_=self.database)
            return [r["value"] for r in recs if r.get("value")]
        except Exception:
            return []

    # ---- public API ----
    def retrieve(self, ntsb_record_dict: dict, encoders=None) -> dict:
        """
        Per-record soft priors over the causal-chain targets. Returns
        {'organizational_prior', 'supervisory_prior', 'precondition_prior',
         'unsafe_prior'} (each float32, sums to 1.0). Uniform on any failure.
        """
        try:
            text = ntsb_record_dict.get("combined_text", "")
            faiss_scores = self._faiss_scores(text) if self.strategy in ("hybrid", "faiss") else {}
            cypher_scores = self._cypher_scores(ntsb_record_dict) if self.strategy in ("hybrid", "cypher") else {}
            combined = self._combine(faiss_scores, cypher_scores)
            if not combined:
                return _uniform_priors()

            acc = {name: np.zeros(_PRIOR_SIZE[name], dtype="float64") for name in _GROUPS}
            for (eid, src), weight in combined.items():
                for value in self._fetch_factors(eid, src):
                    g = VALUE_TO_GROUP.get(value)
                    if g is not None:
                        acc[g[0]][g[1]] += weight

            out = {}
            for name, vec in acc.items():
                total = vec.sum()
                out[name] = (vec / total if total > 0
                             else np.full(_PRIOR_SIZE[name], 1.0 / _PRIOR_SIZE[name])
                             ).astype("float32")
            return out
        except Exception as e:                       # never break training
            logging.warning("RAG: retrieve failed (%s) — uniform priors.", e)
            return _uniform_priors()

    def get_ntsb_fewshot_examples(self, narrative_text: str, n: int = 5) -> str:
        """Delegate to the Stage-2 train-only few-shot retriever (read-only)."""
        from hfacs_extractor import get_ntsb_fewshot_examples as _fewshot
        return _fewshot(narrative_text, n=n)


def build_retriever(strategy: str = "hybrid", **kwargs) -> RAGRetriever:
    """Factory used by models/lstm/train.py for C2-C4."""
    return RAGRetriever(strategy=strategy, **kwargs)
