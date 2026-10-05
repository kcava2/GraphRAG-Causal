"""
rag_retriever.py  (Stage 5)
===========================
Hybrid retrieval for the RAG-augmented LSTM conditions (C2-C4) and inference.
Combines:

    Mode 1  FAISS semantic search over asias.faiss + asrs.faiss (ASIAS weighted
            0.6, ASRS 0.4) on the NTSB record's combined_text.
    Mode 2  Deterministic, schema-grounded Cypher structural search over the Neo4j
            KG: scores each event by how many {feature,value} context nodes it
            shares with the record. No LLM (was LLM text-to-Cypher; small models
            produced invalid/hallucinated Cypher, so it was replaced).

The two score sets are min-max normalized to [0,1] and combined 0.5/0.5; the
top-k EventNodes' HFACSFactorNode mappings (and stored severity), weighted by
combined score, build soft prior distributions over the causal-chain targets,
appended to step_b by NTSBSequenceDataset and routed to each head:

    precondition_prior -> B (Preconditions)
    unsafe_prior       -> C (Unsafe Acts)
    severity_prior     -> D (Severity; binary high/low, from EventNode severity —
                            ASRS contributes none as it has no injury data)

organizational/supervisory priors are still computed but unused (org/sup are no
longer mined or predicted).

**Read-only**: this module never writes to Neo4j, asias.faiss, asrs.faiss, or
ntsb.faiss. It only runs MATCH/OPTIONAL MATCH Cypher. On ANY retrieval failure
(Neo4j down, FAISS error, invalid Cypher) it silently returns uniform priors.

Run order: Stage 3 must complete (KG + ASIAS/ASRS FAISS indexes) for real
retrieval; absent those, the retriever still constructs and returns uniform priors.
"""

import logging
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from hfacs_extractor import DEFAULT_MODEL, _clean  # noqa: E402
from ntsbdataloader import (  # noqa: E402  (label spaces — single source of truth)
    ORG_SUBS, SUP_SUBS, PRECOND_SUBS, PRECOND_GROUP_INDEX, UNSAFE_SUBS,
    UNSAFE_VIOLATION_TIER, N_O, N_A, N_B, N_C, RETRIEVAL_MODEL,
)
from standardize import binarize_severity, strip_outcome  # noqa: E402

# Embedding caches shared by every retriever in the process. The encoder runs on
# CPU in this environment, and one run builds several retrievers (C2, C4, C5, then
# evaluation) over the same narratives; without these each would re-encode the
# whole corpus. Keyed by encoder name so a changed RETRIEVAL_MODEL cannot collide.
_QUERY_EMB = {}   # (model, stripped text) -> [1, dim]
_EMB_DIR = os.path.join(_HERE, ".emb_cache")


def _emb_path(text: str) -> str:
    import hashlib
    h = hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()
    return os.path.join(_EMB_DIR, RETRIEVAL_MODEL.replace("/", "_"), h[:2], h + ".npy")


def embed_many(sbert, texts) -> np.ndarray:
    """[n, dim] normalised vectors for already-stripped `texts`: memory, then the
    on-disk cache, then one batched encode for whatever is left.

    The disk cache is what keeps training and the per-view evaluations (separate
    processes) from each re-encoding the corpus on CPU. A vector is a pure function
    of (encoder, text), so the cache can never go stale; delete data/.emb_cache to
    reclaim the space.
    """
    texts = [str(t) for t in texts]
    todo = []
    for t in dict.fromkeys(texts):
        if (RETRIEVAL_MODEL, t) in _QUERY_EMB:
            continue
        fp = _emb_path(t)
        if os.path.exists(fp):
            try:
                _QUERY_EMB[(RETRIEVAL_MODEL, t)] = np.load(fp)[None, :]
                continue
            except Exception:
                pass
        todo.append(t)
    if todo:
        embs = np.asarray(sbert.encode(todo, normalize_embeddings=True, batch_size=32,
                                       show_progress_bar=False), dtype="float32")
        for t, e in zip(todo, embs):
            _QUERY_EMB[(RETRIEVAL_MODEL, t)] = e[None, :]
            try:
                fp = _emb_path(t)
                os.makedirs(os.path.dirname(fp), exist_ok=True)
                np.save(fp, e)
            except OSError:
                pass                                   # cache is best-effort
    return np.concatenate([_QUERY_EMB[(RETRIEVAL_MODEL, t)] for t in texts], axis=0)

SEVERITY_N = 2   # binary severity prior (high/low) for the D head
UNSAFE_N = N_C   # unsafe prior over the four unsafe-act tiers (C is 4-tier multi-label)
_UNSAFE_IDX = {t: i for i, t in enumerate(UNSAFE_SUBS)}

# The structural query references only existing properties now, but silence any
# residual Neo4j notifications so the terminal isn't flooded during retrieval.
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

# SDR maintenance-reliability lookup {(make_upper, year): bracket} — lets the
# structural query match a record's aircraft defect-rate against the KG's
# TechnologicalContextNodes. Absent file -> SDR matching simply never fires.
_SDR_PATH = os.path.join(_HERE, "sdr_defect_brackets.csv")
_SDR_BRACKETS = None


def _sdr_bracket(make: str, year: str):
    global _SDR_BRACKETS
    if _SDR_BRACKETS is None:
        try:
            import pandas as pd
            t = pd.read_csv(_SDR_PATH, dtype=str)
            _SDR_BRACKETS = {(str(r["make"]).upper(), int(float(r["year"]))): r["bracket"]
                             for _, r in t.iterrows()}
        except Exception:
            _SDR_BRACKETS = {}
    if not _SDR_BRACKETS or not _clean(make):
        return ""
    from standardize import normalize_make
    try:
        return _SDR_BRACKETS.get((normalize_make(make), int(float(year))), "")
    except (ValueError, TypeError):
        return ""


ASIAS_FAISS = os.path.join(_HERE, "asias.faiss")
ASIAS_IDMAP = os.path.join(_HERE, "asias_id_map.csv")
ASRS_FAISS = os.path.join(_HERE, "asrs.faiss")
ASRS_IDMAP = os.path.join(_HERE, "asrs_id_map.csv")
# In-distribution NTSB KG slice (disjoint from LSTM train/test). Lets the prior be
# sourced from the same population the LSTM predicts (counters domain shift).
NTSB_FAISS = os.path.join(_HERE, "ntsb_kg.faiss")
NTSB_IDMAP = os.path.join(_HERE, "ntsb_kg_id_map.csv")
# NTSB TRAINING events written into the graph by `kg_builder.py --ingest-ntsb-train`.
NTSB_TRAIN_FAISS = os.path.join(_HERE, "ntsb_train_kg.faiss")
NTSB_TRAIN_IDMAP = os.path.join(_HERE, "ntsb_train_kg_id_map.csv")
NTSB_TRAIN_ORIGIN = "ntsb_train"
# Retrieval encoder: single source of truth is ntsbdataloader.RETRIEVAL_MODEL
# (imported below). Do not redefine it here.

# Per-source FAISS weights. Default gives the in-distribution NTSB source the most
# say; set two of three to 0 for the single-source ablations (C5/C6/C7).
ASIAS_WEIGHT, ASRS_WEIGHT, NTSB_WEIGHT = 0.34, 0.33, 0.33
TOP_K = 5

# Factor value -> (prior_name, index). B's precondition prior is over the 3 HFACS
# GROUPS, but KG factor nodes are stored at the raw-tier level, so each precondition
# tier maps up to its group index (PRECOND_GROUP_INDEX). The unsafe head is now
# BINARY (violation vs error), so its prior is built like severity — a 2-class
# outcome, not a tier accumulation (see retrieve).
_GROUPS = {"organizational_prior": ORG_SUBS, "supervisory_prior": SUP_SUBS,
           "precondition_prior": PRECOND_SUBS}
_PRIOR_SIZE = {"organizational_prior": N_O, "supervisory_prior": N_A,
               "precondition_prior": N_B}
VALUE_TO_GROUP = {v: (name, i) for name, subs in _GROUPS.items()
                  for i, v in enumerate(subs)}
VALUE_TO_GROUP.update({tier: ("precondition_prior", gidx)        # raw tier -> group idx
                       for tier, gidx in PRECOND_GROUP_INDEX.items()})


# Structured params bound into the structural query (match the record's discrete
# fields against the KG's {feature,value} / {feature,value_bracket} context nodes).
CYPHER_PARAMS = ["visual_condition", "light_conditions", "employment_bracket",
                 "fuel_bracket", "revenue_bracket", "loadfactor_bracket",
                 "person_involved", "pilot_hours_bracket"]

# Deterministic, read-only structural retrieval. Scores each EventNode by the
# number of structured context fields it shares with the record. Replaces the old
# LLM-generated Cypher (which hallucinated node labels/properties on small models).
# The {feature,value} schema and HAS_*_CONTEXT edges mirror kg_builder exactly.
_STRUCTURAL_CYPHER = """
MATCH (e:EventNode)
OPTIONAL MATCH (e)-[:HAS_ENV_CONTEXT]->(env:EnvironmentalContextNode)
WHERE (env.feature = 'visual_condition' AND env.value = $visual_condition)
   OR (env.feature = 'light_conditions' AND env.value = $light_conditions)
OPTIONAL MATCH (e)-[:HAS_PERSONNEL_CONTEXT]->(pc:PersonnelContextNode)
WHERE (pc.feature = 'person_involved' AND pc.value = $person_involved)
   OR (pc.feature = 'pilot_hours_bracket' AND pc.value = $pilot_hours_bracket)
OPTIONAL MATCH (e)-[:HAS_ORG_CONTEXT]->(oc:OrganizationalContextNode)
WHERE (oc.feature = 'employment_pressure' AND oc.value_bracket = $employment_bracket)
   OR (oc.feature = 'fuel_cost_pressure' AND oc.value_bracket = $fuel_bracket)
   OR (oc.feature = 'revenue_pressure' AND oc.value_bracket = $revenue_bracket)
   OR (oc.feature = 'utilization_pressure' AND oc.value_bracket = $loadfactor_bracket)
OPTIONAL MATCH (e)-[:HAS_TECH_CONTEXT]->(tc:TechnologicalContextNode)
WHERE tc.feature = 'maintenance_defect_rate' AND tc.value_bracket = $maintenance_defect_bracket
WITH e, count(DISTINCT env) + count(DISTINCT pc) + count(DISTINCT oc)
        + count(DISTINCT tc) AS score
WHERE score > 0
RETURN e.event_id AS event_id, e.source AS source,
       e.embedding_index AS embedding_index, score
ORDER BY score DESC
LIMIT $k
"""


# ---------------------------------------------------------------------------
# Exemplar retrieval reads the GRAPH and nothing else
# ---------------------------------------------------------------------------
# query field -> the context-node feature it is compared against
_CTX_FEATURE = {
    "visual_condition": "visual_condition",
    "light_conditions": "light_conditions",
    "person_involved": "person_involved",
    "pilot_hours_bracket": "pilot_hours_bracket",
    "employment_bracket": "employment_pressure",
    "fuel_bracket": "fuel_cost_pressure",
    "revenue_bracket": "revenue_pressure",
    "loadfactor_bracket": "utilization_pressure",
    "maintenance_defect_bracket": "maintenance_defect_rate",
    "phase_of_flight": "phase_of_flight",
}
_NO_MATCH = "__no_match__"         # bound in place of an unusable query value

# Structural similarity is scored by GROUP, not by field. Until 2026-10-04 every
# field counted on its own, IDF-weighted, so the four economic brackets (all set by
# the calendar month) plus the maintenance bracket (set by aircraft make and year)
# carried about 70% of every match: "structurally similar" meant "same month, same
# make". Grouping lets each kind of context count once, however many fields it has.
#
# Weights. Measured on TRAINING events only (pairs of training events: does sharing
# a value make two events more likely to share a label?), none of the original
# fields carries label information: every lift is within +/-0.03 and the most common
# fields are slightly negative. Phase of flight is the one field that does. The
# weights therefore put phase first and keep the HFACS context groups the design
# calls for (environment, crew, organisational pressure, technology) as smaller,
# equal contributions, so they shape the ranking among events of the same phase
# rather than deciding it. Within a group, fields are IDF-weighted as before.
# A group the query has no usable value for is left out and the rest renormalised,
# so an L1 query (no phase, by definition) is scored on the remaining groups.
STRUCT_GROUPS = {
    "operation":    (0.50, ("phase_of_flight",)),
    "environment":  (0.125, ("visual_condition", "light_conditions")),
    "crew":         (0.125, ("person_involved", "pilot_hours_bracket")),
    "organisation": (0.125, ("employment_pressure", "fuel_cost_pressure",
                             "revenue_pressure", "utilization_pressure")),
    "technology":   (0.125, ("maintenance_defect_rate",)),
}
_GROUP_OF = {f: g for g, (_w, fs) in STRUCT_GROUPS.items() for f in fs}

# For one query: every graph event that shares at least one context node with it,
# and WHICH features matched. Unlike _STRUCTURAL_CYPHER this returns no count and
# has no LIMIT: the count ties heavily (a handful of distinct values), so the
# matched features are weighted by rarity in Python and ranked there.
_GRAPH_MATCH_CYPHER = """
MATCH (e:EventNode)
OPTIONAL MATCH (e)-[:HAS_ENV_CONTEXT]->(env:EnvironmentalContextNode)
WHERE (env.feature = 'visual_condition' AND env.value = $visual_condition)
   OR (env.feature = 'light_conditions' AND env.value = $light_conditions)
WITH e, collect(DISTINCT env.feature) AS f1
OPTIONAL MATCH (e)-[:HAS_PERSONNEL_CONTEXT]->(pc:PersonnelContextNode)
WHERE (pc.feature = 'person_involved' AND pc.value = $person_involved)
   OR (pc.feature = 'pilot_hours_bracket' AND pc.value = $pilot_hours_bracket)
WITH e, f1, collect(DISTINCT pc.feature) AS c2
WITH e, f1 + c2 AS f2
OPTIONAL MATCH (e)-[:HAS_ORG_CONTEXT]->(oc:OrganizationalContextNode)
WHERE (oc.feature = 'employment_pressure' AND oc.value_bracket = $employment_bracket)
   OR (oc.feature = 'fuel_cost_pressure' AND oc.value_bracket = $fuel_bracket)
   OR (oc.feature = 'revenue_pressure' AND oc.value_bracket = $revenue_bracket)
   OR (oc.feature = 'utilization_pressure' AND oc.value_bracket = $loadfactor_bracket)
WITH e, f2, collect(DISTINCT oc.feature) AS c3
WITH e, f2 + c3 AS f3
OPTIONAL MATCH (e)-[:HAS_TECH_CONTEXT]->(tc:TechnologicalContextNode)
WHERE tc.feature = 'maintenance_defect_rate' AND tc.value_bracket = $maintenance_defect_bracket
WITH e, f3, collect(DISTINCT tc.feature) AS c4
WITH e, f3 + c4 AS f4
OPTIONAL MATCH (e)-[:HAS_OPS_CONTEXT]->(op:OperationalContextNode)
WHERE op.feature = 'phase_of_flight' AND op.value = $phase_of_flight
WITH e, f4, collect(DISTINCT op.feature) AS c5
WITH e, f4 + c5 AS matched
WHERE size(matched) > 0
RETURN e.event_id AS event_id, e.source AS source, matched
"""

_GRAPH_IDF_CYPHER = """
MATCH (e:EventNode)-[:HAS_ENV_CONTEXT|HAS_PERSONNEL_CONTEXT|HAS_ORG_CONTEXT|HAS_TECH_CONTEXT|HAS_OPS_CONTEXT]->(c)
RETURN c.feature AS feature, coalesce(c.value, c.value_bracket) AS value,
       count(DISTINCT e) AS n
"""


# Structural matches depend only on the query's context values and the graph, so
# they are shared by every retriever in the process (C3, C4 and C5 ask the same
# questions). The graph is never written during training or evaluation.
_STRUCT_CACHE = {}


class GraphUnavailable(RuntimeError):
    """Exemplar retrieval was asked for but the knowledge graph cannot serve it."""


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
    d = {name: np.full(_PRIOR_SIZE[name], 1.0 / _PRIOR_SIZE[name], dtype="float32")
         for name in _GROUPS}
    d["unsafe_prior"] = np.full(UNSAFE_N, 1.0 / UNSAFE_N, dtype="float32")
    d["severity_prior"] = np.full(SEVERITY_N, 1.0 / SEVERITY_N, dtype="float32")
    return d


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class RAGRetriever:
    """See module docstring. Construct via build_retriever()."""

    def __init__(self, strategy: str = "hybrid", k: int = TOP_K,
                 model: str = DEFAULT_MODEL,
                 asias_weight: float = ASIAS_WEIGHT, asrs_weight: float = ASRS_WEIGHT,
                 ntsb_weight: float = NTSB_WEIGHT, factor_priors: bool = True):
        self.strategy = strategy            # 'hybrid' | 'faiss' | 'cypher'
        self.k = k
        self.model = model
        # When False (condition C8): the LLM-mined HFACS factor priors (precondition,
        # unsafe) are held UNIFORM — retrieval + the structured severity-outcome prior
        # still inform prediction. Isolates RAG retrieval from text mining.
        self.factor_priors = factor_priors
        # Per-source FAISS weights — knobs for the single-source ablations
        # (C5 ASIAS-only, C6 ASRS-only, C7 NTSB-only). Set the others to 0.
        self.asias_weight = asias_weight
        self.asrs_weight = asrs_weight
        self.ntsb_weight = ntsb_weight
        self._sbert = None
        self._faiss = {}                    # source -> (index, [event_id,...])
        self._lofo = None                   # legacy prior path only (set_source_df)
        self._universe_cache = None         # every retrievable graph event + its vector
        self._idf_cache = None              # rarity of each context value in the graph
        self._struct_cache = _STRUCT_CACHE  # query context -> {event key: share}
        self._attr_cache = None
        self.driver = None
        self.database = os.environ.get("NEO4J_DATABASE", "neo4j")
        self._load_faiss()
        self._connect_neo4j()

    def set_source_df(self, source_df):
        """Attach the NTSB in-distribution retrieval source (the train split) as a
        leave-one-out FAISS source. Only built when ntsb_weight > 0. Replaces the
        old stale on-disk ntsb_kg.faiss; self-excludes each query's ev_id."""
        if self.ntsb_weight and self.ntsb_weight > 0 and source_df is not None:
            self._lofo = LOFORetriever(source_df, k=self.k)

    # ---- setup (best-effort; never raises) ----
    def _load_faiss(self):
        try:
            import faiss
            import pandas as pd
            # NTSB is NOT loaded from disk anymore — it's the in-distribution LOFO
            # source built from the train split via set_source_df().
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
            if self._lofo is not None and getattr(self._lofo, "_sbert", None) is not None:
                self._sbert = self._lofo._sbert          # same encoder: load it once
            else:
                from sentence_transformers import SentenceTransformer
                self._sbert = SentenceTransformer(RETRIEVAL_MODEL)
        return self._sbert

    def warm(self, texts):
        """Batch-encode query narratives ahead of the per-record lookups.

        Lookups encode one narrative at a time, which dominates the cost of
        building a dataset. Encoding a split in one batched pass is an order of
        magnitude faster; the per-record path then finds its vector in the cache.
        """
        todo = sorted({strip_outcome(str(t)) for t in texts} - {""})
        if not todo or not (self._faiss or self._lofo is not None):
            return
        embed_many(self._ensure_sbert(), todo)

    def _embed(self, text: str):
        key = (RETRIEVAL_MODEL, strip_outcome(str(text)))
        e = _QUERY_EMB.get(key)
        if e is None:
            e = embed_many(self._ensure_sbert(), [key[1]])
        return e

    def close(self):
        if self.driver is not None:
            self.driver.close()

    # ---- Mode 1: FAISS semantic ----
    def _faiss_scores(self, text: str, exclude_id: str = "", k: int | None = None) -> dict:
        """Source-weighted cosine per candidate. `k` overrides the retriever depth
        so an exemplar count above TOP_K is actually honoured."""
        if not _clean(text):
            return {}
        k = int(k or self.k)
        merged = {}
        if not (self._faiss or (self._lofo is not None and self.ntsb_weight > 0)):
            return {}
        emb = self._embed(text)             # one vector, shared by every source
        # ASIAS / ASRS — on-disk FAISS indexes (different corpora; no self-overlap).
        if self._faiss:
            for src, weight in (("ASIAS", self.asias_weight), ("ASRS", self.asrs_weight)):
                if src not in self._faiss or weight <= 0:
                    continue
                index, ids = self._faiss[src]
                kk = min(k, index.ntotal)
                if kk == 0:
                    continue
                sims, idx = index.search(emb, kk)
                for s, i in zip(sims[0], idx[0]):
                    if 0 <= i < len(ids):
                        merged[(ids[i], src)] = float(s) * weight
        # NTSB — in-distribution LOFO source, self-excluding the query's ev_id.
        if self._lofo is not None and self.ntsb_weight > 0:
            # Same encoder, same text: reuse the query vector instead of encoding
            # the narrative a second time.
            for eid, s in self._lofo.neighbors(text, exclude_id, k, emb=emb):
                merged[(eid, "NTSB")] = s * self.ntsb_weight
        # top-k by weighted score
        return dict(sorted(merged.items(), key=lambda kv: kv[1], reverse=True)[:k])

    # ---- Mode 2: deterministic schema-grounded Cypher structural search ----
    def _cypher_scores(self, record: dict, k: int | None = None) -> dict:
        """LEGACY prior path only — exemplar retrieval uses `_structural_shares`.

        Structural retrieval: score candidates by shared STRUCTURED CONTEXT.

        Two sources, merged:

        1. **In-distribution NTSB train records** via `LOFORetriever.context_neighbors`,
           IDF-weighted. This is the important half. Before it existed, structural
           retrieval was crippled three ways at once: the Cypher below excludes NTSB
           outright, so it could only ever return ASIAS/ASRS (measured: 179 ASIAS,
           21 ASRS, 0 NTSB); its score is a small integer count of shared context
           nodes, giving only 5 distinct values with 3.2 of 5 returned candidates
           tied at the top, so ranking among them was arbitrary; and it never touches
           HFACSFactorNode or LEADS_TO, so despite the name it matches attributes,
           not causal structure.
        2. **KG events** via the fixed Cypher query, as before.

        Note on "graph isomorphism on causal patterns" (spec 2.2 Strategy B): that
        cannot be implemented as written. Matching a query's causal pattern requires
        the query's HFACS factors, which ARE the prediction targets — using them
        would leak the labels. Context matching is the leak-free substitute, and it
        has the useful property of touching only PRE-NARRATIVE fields.
        """
        # Each sub-source is min-max normalized to [0,1] SEPARATELY and only then
        # weighted. They are on incomparable scales — the Cypher score is a small
        # integer count of shared context nodes (observed range 5-9), the
        # in-distribution score is an IDF-weighted sum. Normalizing the merged dict
        # instead put every in-distribution match at the bottom of the ranking.
        k = int(k or self.k)
        ctx_scores = {}
        if self._lofo is not None and self.ntsb_weight > 0:
            for eid, sc in self._lofo.context_neighbors(
                    record, str(record.get("ev_id", "")), k):
                ctx_scores[(eid, "NTSB")] = float(sc)
        # In-distribution matches occupy [0.5, 1.0]; KG matches [0, 0.5). The bands
        # are disjoint ON PURPOSE. Weighting alone did not work: min-max maps each
        # source's best to 1.0, so asias_weight (0.34) edged out ntsb_weight (0.33),
        # and the Cypher side's integer scores tie 3-of-5 at the top and all inflate
        # to 1.0 — between them the in-distribution matches were pushed out of every
        # slot. A train record with the right label distribution is worth more than
        # any ASIAS match here, so the ordering is made explicit rather than left to
        # near-equal weights.
        out = {key: 0.5 + 0.5 * v for key, v in _minmax(ctx_scores).items()}
        if self.driver is None:
            return out
        kg_scores = {}
        try:
            params = {p: record.get(p, "") for p in CYPHER_PARAMS}
            # SDR maintenance bracket is computed (make+year), not a record column.
            params["maintenance_defect_bracket"] = _sdr_bracket(
                record.get("acft_make", ""), record.get("year", ""))
            params["k"] = k
            recs, _, _ = self.driver.execute_query(
                _STRUCTURAL_CYPHER, database_=self.database, **params)
            for r in recs:
                d = r.data()
                eid, src, score = d.get("event_id"), d.get("source"), d.get("score")
                # NTSB structural matches are skipped: NTSB now comes from the LOFO
                # source, and the on-disk Neo4j NTSB-KG slice is stale/leaky.
                if eid is not None and src is not None and str(src) != "NTSB":
                    kg_scores[(str(eid), str(src))] = float(score or 0.0)
            w = {"ASIAS": self.asias_weight, "ASRS": self.asrs_weight}
            # The Cypher score counts matched context nodes: at most 2 env + 2
            # personnel + 4 organizational + 1 technological.
            for key, v in _minmax(kg_scores).items():
                out[key] = 0.49 * v * w.get(key[1], 0.33) / max(self.asias_weight,
                                                                self.asrs_weight, 1e-9)
            return out
        except Exception as e:
            logging.warning("RAG: structural Cypher failed (%s) — in-distribution "
                            "matches (if any) are kept.", e)
            return out
            return {}

    # ---- combine + factor lookup ----
    def _combine(self, faiss_scores: dict, cypher_scores: dict) -> dict:
        f = _minmax(faiss_scores)
        c = _minmax(cypher_scores)
        keys = set(f) | set(c)
        combined = {k: 0.5 * f.get(k, 0.0) + 0.5 * c.get(k, 0.0) for k in keys}
        return dict(sorted(combined.items(), key=lambda kv: kv[1], reverse=True)[:self.k])

    # ---- graph-only exemplar retrieval -------------------------------------
    def require_graph(self):
        """Stop, loudly, if the graph cannot be read.

        Exemplar retrieval used to degrade in silence: with Neo4j down or the
        password unset it fell back to an in-memory pool of training records and
        the run looked healthy. There is no such pool any more. No graph, no
        retrieval, and that is an error rather than a warning.
        """
        if self.driver is None:
            raise GraphUnavailable(
                "Cannot reach the Neo4j knowledge graph, and exemplar retrieval reads "
                "nothing else.\n  1. Start the database in Neo4j Desktop.\n"
                "  2. In THIS terminal:  $env:NEO4J_PASSWORD = \"<password>\"\n"
                "  3. Check:  python run_all.py --preflight-only --start-at 5 --stop-after 8")

    def _universe(self) -> dict:
        """Every retrievable graph event with its narrative vector.

        The vectors come from the FAISS indexes built alongside the graph
        (ASIAS, ASRS, the NTSB-KG slice, and the NTSB training events); an event is
        kept only if it actually exists as a node in Neo4j. A source whose weight
        is 0 is left out, which is how a single-source ablation is run.
        """
        if self._universe_cache is not None:
            return self._universe_cache
        self.require_graph()
        import faiss
        attrs = self.event_attributes()
        wanted = (("ASIAS", ASIAS_FAISS, ASIAS_IDMAP, self.asias_weight),
                  ("ASRS", ASRS_FAISS, ASRS_IDMAP, self.asrs_weight),
                  ("NTSB", NTSB_FAISS, NTSB_IDMAP, self.ntsb_weight),
                  ("NTSB", NTSB_TRAIN_FAISS, NTSB_TRAIN_IDMAP, self.ntsb_weight))
        keys, vecs, seen = [], [], set()
        for src, fp, mp, weight in wanted:
            if weight <= 0 or not (os.path.exists(fp) and os.path.exists(mp)):
                continue
            index = faiss.read_index(fp)
            ids = pd.read_csv(mp, dtype=str)["event_id"].tolist()
            allv = index.reconstruct_n(0, index.ntotal)
            for i, eid in enumerate(ids[:index.ntotal]):
                key = (eid, src)
                if key in attrs and key not in seen:      # a real node, once
                    seen.add(key); keys.append(key); vecs.append(allv[i])
        if not keys:
            raise GraphUnavailable(
                "The graph is reachable but no event in it has a narrative vector. "
                "Build the indexes:  python run_all.py --start-at 8 --stop-after 8")
        E = np.ascontiguousarray(np.stack(vecs), dtype="float32")
        sb = self._ensure_sbert()
        dim = getattr(sb, "get_embedding_dimension", None) or sb.get_sentence_embedding_dimension
        if E.shape[1] != dim():
            raise GraphUnavailable(
                f"FAISS vectors are {E.shape[1]}-dimensional but the retrieval encoder "
                f"({RETRIEVAL_MODEL}) is not. Rebuild the indexes (run_all.py stage 8).")
        self._universe_cache = {"keys": keys, "E": E, "pos": {k: i for i, k in enumerate(keys)}}
        by_src = {}
        for eid, src in keys:
            o = "NTSB-train" if attrs[(eid, src)].get("origin") == NTSB_TRAIN_ORIGIN else src
            by_src[o] = by_src.get(o, 0) + 1
        logging.info("RAG: %d retrievable graph events %s", len(keys), by_src)
        n_phase = self.driver.execute_query(
            "MATCH (e:EventNode)-[:HAS_OPS_CONTEXT]->(:OperationalContextNode "
            "{feature:'phase_of_flight'}) RETURN count(DISTINCT e) AS n",
            database_=self.database)[0][0]["n"]
        if n_phase == 0 and self.strategy in ("cypher", "hybrid"):
            logging.warning("RAG: no graph event has a phase-of-flight node, so structural "
                            "retrieval is matching on weather, crew and calendar context "
                            "only. Run:  python data/kg_builder.py --update-phase")
        self._universe_cache["n_with_phase"] = n_phase
        self._universe_cache["by_source"] = by_src
        return self._universe_cache

    def _idf(self) -> dict:
        """{(feature, value): log(N / events carrying it)} over the whole graph, so
        matching a rare context value counts for more than matching 'VMC'."""
        if self._idf_cache is None:
            self.require_graph()
            recs, _, _ = self.driver.execute_query(_GRAPH_IDF_CYPHER, database_=self.database)
            n_ev, _, _ = self.driver.execute_query(
                "MATCH (e:EventNode) RETURN count(e) AS n", database_=self.database)
            total = max(int(n_ev[0]["n"]), 1)
            self._idf_cache = {(r["feature"], str(r["value"])): float(np.log(total / max(r["n"], 1)))
                               for r in recs if r["feature"]}
        return self._idf_cache

    def _structural_shares(self, record: dict) -> dict:
        """{event key: structural similarity in [0, 1]}, from one Cypher query.

        Similarity = sum over context GROUPS of (group weight x share of that
        group's query context the event matches), divided by the total weight of
        the groups the query has a usable value for. Within a group each field is
        weighted by how rare its value is in the graph (IDF). See STRUCT_GROUPS.
        """
        self.require_graph()
        q = {f: str(record.get(f, "") or "") for f in CYPHER_PARAMS + ["phase_of_flight"]}
        q["maintenance_defect_bracket"] = _sdr_bracket(record.get("acft_make", ""),
                                                       record.get("year", "")) or ""
        usable = {f: v for f, v in q.items() if v and v.lower() not in ("nan", "unknown", "none")}
        cache_key = tuple(sorted(usable.items()))
        if cache_key in self._struct_cache:
            return self._struct_cache[cache_key]
        idf = self._idf()
        feat_w = {_CTX_FEATURE[f]: idf.get((_CTX_FEATURE[f], v), 0.0) for f, v in usable.items()}
        group_tot = {}
        for f, w in feat_w.items():
            g = _GROUP_OF.get(f)
            if g is None:                                 # field not used for matching
                continue
            group_tot[g] = group_tot.get(g, 0.0) + w
        group_tot = {g: t for g, t in group_tot.items() if t > 0}
        weight_sum = sum(STRUCT_GROUPS[g][0] for g in group_tot)
        out = {}
        if weight_sum > 0:
            params = {f: usable.get(f, _NO_MATCH) for f in _CTX_FEATURE}
            recs, _, _ = self.driver.execute_query(_GRAPH_MATCH_CYPHER,
                                                   database_=self.database, **params)
            for r in recs:
                got = {}
                for f in set(r["matched"]):
                    g = _GROUP_OF.get(f)
                    if g in group_tot:
                        got[g] = got.get(g, 0.0) + feat_w.get(f, 0.0)
                score = sum(STRUCT_GROUPS[g][0] * got[g] / group_tot[g] for g in got) / weight_sum
                if score > 0:
                    out[(str(r["event_id"]), str(r["source"]))] = float(min(score, 1.0))
        self._struct_cache[cache_key] = out
        return out

    def ranked_neighbors(self, record: dict, k: int | None = None,
                         fetch: int | None = None) -> list:
        """Top neighbours as [((event_id, source), score), ...], highest first.

        EVERY candidate is an event in the knowledge graph. There is no other pool.

            faiss   -> cosine between the query narrative and the event's narrative
            cypher  -> share of the query's structured context the event matches
            hybrid  -> the mean of the two, for every event (each event has both)

        All sources are scored on the same scale and compete on similarity alone:
        no per-source weights, no bands, no quotas. The query's own event is
        excluded, so a training event never retrieves itself; validation and test
        events are not in the graph to begin with.

        Scores are absolute, never rescaled per query, so "0.8" means the same
        thing for every record and no neighbour is forced to zero.

        Raises GraphUnavailable when the graph cannot be read. It does not return
        an empty list and carry on.
        """
        k = int(k or self.k)
        depth = int(fetch or k)
        uni = self._universe()
        keys, n = uni["keys"], len(uni["keys"])
        text = record.get("combined_text", "")
        sem = struct = None
        if self.strategy in ("faiss", "hybrid") and _clean(text):
            sem = uni["E"] @ self._embed(text)[0]
        if self.strategy in ("cypher", "hybrid"):
            shares = self._structural_shares(record)
            struct = np.zeros(n, dtype="float32")
            for key, share in shares.items():
                i = uni["pos"].get(key)
                if i is not None:
                    struct[i] = share
        if self.strategy == "faiss":
            score = sem
        elif self.strategy == "cypher":
            score = struct
        else:
            score = 0.5 * (sem if sem is not None else 0.0) + 0.5 * struct
        if score is None:
            return []                                     # no narrative to search with
        score = np.asarray(score, dtype="float64").copy()
        me = uni["pos"].get((str(record.get("ev_id", "")), "NTSB"))
        if me is not None:
            score[me] = -np.inf                           # never retrieve yourself
        # Structural scores tie in large blocks. Break ties with a draw that is
        # fixed per query, so the order is reproducible but favours no source.
        import zlib
        tie = np.random.RandomState(zlib.crc32(str(record.get("ev_id", "")).encode())).rand(n)
        order = np.lexsort((tie, -score))
        return [(keys[i], float(score[i])) for i in order[:depth] if score[i] > 0]

    def event_attributes(self) -> dict:
        """Every graph event's labels, context and outcome in one query
        -> {(event_id, source): {...}}. Cached for the life of the retriever.

        Exemplars need each neighbour's features as well as its labels, and one
        query per neighbour would be thousands of round-trips, so the lot is pulled
        once and looked up locally.

        `origin` is 'ntsb_train' for the NTSB training events written by
        `kg_builder.py --ingest-ntsb-train`; their labels come from the same
        extraction as the prediction targets. Every other event was labelled by the
        KG builder's own prompt.

        `causal` is the event's extracted causal chain as (cause tier, effect tier)
        pairs. Training events store theirs on the node. For the rest it is read
        from evidence-bearing LEADS_TO edges: edges without evidence come from
        classify_edge's deterministic DAG mapping and are a pure function of the
        factor set, so they would add nothing the exemplar does not already have.
        """
        if self._attr_cache is not None:
            return self._attr_cache
        self.require_graph()
        recs, _, _ = self.driver.execute_query(
            "MATCH (e:EventNode) "
            "OPTIONAL MATCH (e)-[:HAS_FACTOR]->(f:HFACSFactorNode) "
            "WITH e, collect(DISTINCT f.tier) AS tiers "
            "OPTIONAL MATCH (e)-[:HAS_ENV_CONTEXT]->(env:EnvironmentalContextNode) "
            "WITH e, tiers, collect(DISTINCT [env.feature, env.value]) AS env "
            "OPTIONAL MATCH (e)-[:HAS_PERSONNEL_CONTEXT]->(pc:PersonnelContextNode) "
            "WITH e, tiers, env, collect(DISTINCT [pc.feature, pc.value]) AS pers "
            "OPTIONAL MATCH (e)-[:HAS_FACTOR]->(a:HFACSFactorNode) "
            "                 -[l:LEADS_TO]->(b:HFACSFactorNode) "
            "                 <-[:HAS_FACTOR]-(e) "
            "WHERE l.evidence IS NOT NULL "
            "RETURN e.event_id AS eid, e.source AS src, e.severity_class AS sev, "
            "       e.origin AS origin, e.causal_links AS links, tiers, env, pers, "
            "       collect(DISTINCT [a.tier, b.tier]) AS causal",
            database_=self.database)
        out = {}
        for r in recs:
            ctx = {}
            for pair in (r["env"] or []) + (r["pers"] or []):
                if isinstance(pair, list) and len(pair) == 2 and pair[0]:
                    ctx[pair[0]] = pair[1]
            if r["origin"] == NTSB_TRAIN_ORIGIN:
                causal = [tuple(x.split(">", 1)) for x in (r["links"] or []) if ">" in x]
            else:
                causal = [(e[0], e[1]) for e in (r["causal"] or [])
                          if isinstance(e, list) and len(e) == 2 and e[0] and e[1]]
            out[(r["eid"], r["src"])] = {
                "tiers": [t for t in (r["tiers"] or []) if t],
                "severity": r["sev"],
                "context": ctx,
                "causal": causal,
                "origin": r["origin"],
            }
        logging.info("RAG: cached attributes for %d KG events.", len(out))
        self._attr_cache = out
        return out

    def graph_meta(self, key: str = NTSB_TRAIN_ORIGIN) -> dict:
        """The bookkeeping node the ingestion step leaves behind ({} if absent)."""
        self.require_graph()
        recs, _, _ = self.driver.execute_query(
            "MATCH (m:KGMeta {key:$k}) RETURN m.signature AS signature, m.n AS n, "
            "m.updated AS updated", k=key, database_=self.database)
        return dict(recs[0]) if recs else {}

    def _fetch_factors(self, event_id: str, source: str):
        """Distinct HFACS TIERS of the event — the tier is the prior's atomic unit
        (subcategories were only prompt examples; VALUE_TO_GROUP maps tier->group)."""
        if source == "NTSB" and self._lofo is not None:     # in-distribution LOFO source
            return self._lofo.factors(event_id)
        if self.driver is None:
            return []
        try:
            recs, _, _ = self.driver.execute_query(
                "MATCH (e:EventNode {event_id:$id, source:$src})-[:HAS_FACTOR]->"
                "(f:HFACSFactorNode) RETURN DISTINCT f.tier AS tier",
                id=event_id, src=source, database_=self.database)
            return [r["tier"] for r in recs if r.get("tier")]
        except Exception:
            return []

    def _fetch_severity(self, event_id: str, source: str):
        """Binarized severity outcome stored on the EventNode; None if absent
        (e.g. ASRS, which has no injury data) so it doesn't bias the prior."""
        if source == "NTSB" and self._lofo is not None:     # in-distribution LOFO source
            return self._lofo.severity(event_id)
        if self.driver is None:
            return None
        try:
            recs, _, _ = self.driver.execute_query(
                "MATCH (e:EventNode {event_id:$id, source:$src}) "
                "RETURN e.severity_class AS s",
                id=event_id, src=source, database_=self.database)
            for r in recs:
                if r.get("s") is not None:
                    return binarize_severity(r["s"])
        except Exception:
            pass
        return None

    # ---- public API ----
    def retrieve(self, ntsb_record_dict: dict, encoders=None) -> dict:
        """
        Per-record soft priors over the causal-chain targets. Returns
        {'organizational_prior', 'supervisory_prior', 'precondition_prior',
         'unsafe_prior'} (each float32, sums to 1.0). Uniform on any failure.
        """
        try:
            text = ntsb_record_dict.get("combined_text", "")
            exclude_id = ntsb_record_dict.get("ev_id", "")          # LOFO self-exclusion
            faiss_scores = (self._faiss_scores(text, exclude_id)
                            if self.strategy in ("hybrid", "faiss") else {})
            cypher_scores = self._cypher_scores(ntsb_record_dict) if self.strategy in ("hybrid", "cypher") else {}
            combined = self._combine(faiss_scores, cypher_scores)
            if not combined:
                return _uniform_priors()

            acc = {name: np.zeros(_PRIOR_SIZE[name], dtype="float64") for name in _GROUPS}
            uns_acc = np.zeros(UNSAFE_N, dtype="float64")
            sev_acc = np.zeros(SEVERITY_N, dtype="float64")
            for (eid, src), weight in combined.items():
                if self.factor_priors:                            # text-mined B/C priors
                    factors = self._fetch_factors(eid, src)
                    for value in factors:                         # precondition prior -> B
                        g = VALUE_TO_GROUP.get(value)
                        if g is not None:
                            acc[g[0]][g[1]] += weight
                    for value in factors:                         # unsafe-tier prior -> C
                        ui = _UNSAFE_IDX.get(value)
                        if ui is not None:
                            uns_acc[ui] += weight
                s = self._fetch_severity(eid, src)                # structured D prior (kept)
                if s is not None and 0 <= s < SEVERITY_N:
                    sev_acc[s] += weight

            out = {}
            for name, vec in acc.items():
                total = vec.sum()
                out[name] = (vec / total if total > 0
                             else np.full(_PRIOR_SIZE[name], 1.0 / _PRIOR_SIZE[name])
                             ).astype("float32")
            ut = uns_acc.sum()
            out["unsafe_prior"] = (uns_acc / ut if ut > 0
                                   else np.full(UNSAFE_N, 1.0 / UNSAFE_N)).astype("float32")
            st = sev_acc.sum()
            out["severity_prior"] = (sev_acc / st if st > 0
                                     else np.full(SEVERITY_N, 1.0 / SEVERITY_N)
                                     ).astype("float32")
            return out
        except Exception as e:                       # never break training
            logging.warning("RAG: retrieve failed (%s) — uniform priors.", e)
            return _uniform_priors()

    def get_ntsb_fewshot_examples(self, narrative_text: str, n: int = 5) -> str:
        """Delegate to the Stage-2 train-only few-shot retriever (read-only)."""
        from hfacs_extractor import get_ntsb_fewshot_examples as _fewshot
        return _fewshot(narrative_text, n=n)


def build_retriever(strategy: str = "hybrid", model: str = None,
                    k: int = TOP_K, asias_weight: float = ASIAS_WEIGHT,
                    asrs_weight: float = ASRS_WEIGHT, ntsb_weight: float = NTSB_WEIGHT,
                    **kwargs) -> RAGRetriever:
    """Factory used by models/lstm/train.py and eval.py for the RAG conditions."""
    kw = dict(strategy=strategy, k=k, asias_weight=asias_weight,
              asrs_weight=asrs_weight, ntsb_weight=ntsb_weight)
    if model:
        kw["model"] = model
    return RAGRetriever(**kw, **kwargs)


# ---------------------------------------------------------------------------
# Leave-one-fold-out (in-distribution) retriever
# ---------------------------------------------------------------------------

class LOFORetriever:
    """IN-DISTRIBUTION retrieval whose source is the NTSB TRAINING split itself.

    For a query it pools the HFACS factors + severity of its nearest TRAIN
    neighbours, EXCLUDING the query's own ev_id (no self-retrieval -> no leakage).
    A training record never sees its own label; a held-out test record isn't in the
    source at all. Produces the same prior dict as RAGRetriever, so the dataloader
    and model are unchanged. Pure SBERT + FAISS over the train narratives (no Neo4j);
    factors/severity come straight from the already-parsed train dataframe.
    """

    def __init__(self, source_df, k: int = TOP_K, factor_priors: bool = True):
        self.k = k
        self.factor_priors = factor_priors
        self.ids = source_df["ev_id"].astype(str).tolist()
        self._pos = {e: i for i, e in enumerate(self.ids)}       # ev_id -> row index
        self._pre = list(source_df["_pre"])     # set of precond GROUP names per record
        self._uns = list(source_df["_uns"])     # set of unsafe TIER names per record
        self._sev = (pd.to_numeric(source_df["severity_class"], errors="coerce")
                     .fillna(0).astype(int).tolist())          # already binarized 0/1
        self._gidx = {g: i for i, g in enumerate(PRECOND_SUBS)}  # group -> column
        self._texts = [strip_outcome(t) for t in            # retrieval text only:
                       source_df["combined_text"].astype(str).fillna("").tolist()]
        self._sbert = None
        self._index = None
        self._emb = None                                    # [n, dim] pool vectors
        self._build_index()
        self._build_context_table(source_df)

    # ---- in-distribution STRUCTURAL retrieval (pre-narrative fields only) ------
    def _build_context_table(self, source_df):
        """Per-record structured context + IDF weights, for `context_neighbors`."""
        self._ctx = [{p: str(source_df.iloc[i].get(p, "") or "") for p in CYPHER_PARAMS}
                     for i in range(len(source_df))]
        # IDF over each (feature, value): matching a RARE value is far more
        # informative than matching "visual_condition=VMC", which nearly everything
        # shares. Without this the score is a small integer count and most
        # candidates tie, which is precisely what crippled the Cypher version.
        n = max(len(self._ctx), 1)
        counts = {}
        for row in self._ctx:
            for f, v in row.items():
                counts[(f, v)] = counts.get((f, v), 0) + 1
        self._idf = {k: float(np.log(n / c)) for k, c in counts.items()}

    def context_neighbors(self, record: dict, exclude_id: str, k: int) -> list:
        """Top-k (ev_id, score) by IDF-weighted agreement on STRUCTURED context.

        Uses only fields knowable before the narrative exists — weather, light,
        crew, economic brackets — so a condition built on this retrieves without
        touching outcome-revealing text. Self-excluding, train split only.
        """
        if not self._ctx:
            return []
        q = {p: str(record.get(p, "") or "") for p in CYPHER_PARAMS}
        out = []
        for i, row in enumerate(self._ctx):
            if self.ids[i] == str(exclude_id):
                continue                                       # self-exclusion
            s = 0.0
            for f, v in q.items():
                if v and v.lower() not in ("", "nan", "unknown") and row.get(f) == v:
                    s += self._idf.get((f, v), 0.0)
            if s > 0:
                out.append((self.ids[i], s))
        out.sort(key=lambda kv: kv[1], reverse=True)
        return out[:k]

    # ---- source API (used when LOFO is the NTSB source inside RAGRetriever) ----
    def neighbors(self, text: str, exclude_id: str, k: int, emb=None) -> list:
        """Top-k (ev_id, similarity) from the train split, self-excluding exclude_id.
        `emb` is an already-computed query vector from the same encoder."""
        if self._index is None or not _clean(text):
            return []
        if emb is None:
            emb = np.asarray(self._sbert.encode([strip_outcome(text)],
                                                normalize_embeddings=True), dtype="float32")
        sims, idx = self._index.search(emb, min(k + 1, len(self.ids)))
        out = []
        for s, i in zip(sims[0], idx[0]):
            if i < 0 or i >= len(self.ids) or self.ids[i] == str(exclude_id):
                continue
            out.append((self.ids[i], float(s)))
            if len(out) >= k:
                break
        return out

    def factors(self, ev_id: str) -> list:
        """Precondition GROUP names + raw unsafe tiers of a train record (for priors)."""
        i = self._pos.get(str(ev_id))
        return (list(self._pre[i]) + list(self._uns[i])) if i is not None else []

    def severity(self, ev_id: str):
        """Binarized severity (0/1) of a train record; None if unknown."""
        i = self._pos.get(str(ev_id))
        return self._sev[i] if i is not None else None

    def _build_index(self):
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
            self._sbert = SentenceTransformer(RETRIEVAL_MODEL)
            # Cached per text, so the pool vectors double as the query vectors of
            # the same train records when their own exemplars are looked up.
            emb = np.ascontiguousarray(embed_many(self._sbert, self._texts))
            self._emb = emb
            self._index = faiss.IndexFlatIP(emb.shape[1])
            self._index.add(emb)
            logging.info("LOFO: indexed %d in-distribution train records.", len(self.ids))
        except Exception as e:
            logging.warning("LOFO: index build failed (%s) — uniform priors.", e)
            self._index = None

    def retrieve(self, record: dict, encoders=None) -> dict:
        if self._index is None:
            return _uniform_priors()
        try:
            text = record.get("combined_text", "")
            if not _clean(text):
                return _uniform_priors()
            exclude = str(record.get("ev_id", ""))
            emb = np.asarray(self._sbert.encode([strip_outcome(text)],
                                                normalize_embeddings=True),
                             dtype="float32")
            sims, idx = self._index.search(emb, min(self.k + 1, len(self.ids)))

            pre = np.zeros(N_B, dtype="float64")
            uns = np.zeros(UNSAFE_N, dtype="float64")
            sev = np.zeros(SEVERITY_N, dtype="float64")
            used = 0
            for s, i in zip(sims[0], idx[0]):
                if i < 0 or i >= len(self.ids) or self.ids[i] == exclude:
                    continue                                   # self-exclusion (LOFO)
                w = float(s)
                if self.factor_priors:
                    for g in self._pre[i]:                     # precond groups -> B
                        if g in self._gidx:
                            pre[self._gidx[g]] += w
                    for t in self._uns[i]:                     # unsafe tiers -> C
                        if t in _UNSAFE_IDX:
                            uns[_UNSAFE_IDX[t]] += w
                if 0 <= self._sev[i] < SEVERITY_N:
                    sev[self._sev[i]] += w                     # severity outcome -> D
                used += 1
                if used >= self.k:
                    break

            out = _uniform_priors()
            if pre.sum() > 0:
                out["precondition_prior"] = (pre / pre.sum()).astype("float32")
            if uns.sum() > 0:
                out["unsafe_prior"] = (uns / uns.sum()).astype("float32")
            if sev.sum() > 0:
                out["severity_prior"] = (sev / sev.sum()).astype("float32")
            return out
        except Exception as e:
            logging.warning("LOFO: retrieve failed (%s) — uniform priors.", e)
            return _uniform_priors()

    def close(self):
        pass


def build_lofo_retriever(source_df, k: int = TOP_K, factor_priors: bool = True):
    """Factory: in-distribution LOFO retriever over the train split (df_train)."""
    return LOFORetriever(source_df, k=k, factor_priors=factor_priors)
