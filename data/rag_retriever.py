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
    ORG_SUBS, SUP_SUBS, PRECOND_SUBS, PRECOND_GROUP_INDEX,
    UNSAFE_VIOLATION_TIER, N_O, N_A, N_B, N_C, RETRIEVAL_MODEL,
)
from standardize import binarize_severity, strip_outcome  # noqa: E402

SEVERITY_N = 2   # binary severity prior (high/low) for the D head
UNSAFE_N = N_C   # binary unsafe prior [P(error), P(violation)] for the (now binary) C head

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
        self._lofo = None                   # NTSB source = in-distribution LOFO (set_source_df)
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
            from sentence_transformers import SentenceTransformer
            self._sbert = SentenceTransformer(RETRIEVAL_MODEL)
        return self._sbert

    def close(self):
        if self.driver is not None:
            self.driver.close()

    # ---- Mode 1: FAISS semantic ----
    def _faiss_scores(self, text: str, exclude_id: str = "") -> dict:
        if not _clean(text):
            return {}
        merged = {}
        # ASIAS / ASRS — on-disk FAISS indexes (different corpora; no self-overlap).
        if self._faiss:
            model = self._ensure_sbert()
            emb = np.asarray(model.encode([strip_outcome(text)],
                                          normalize_embeddings=True), dtype="float32")
            for src, weight in (("ASIAS", self.asias_weight), ("ASRS", self.asrs_weight)):
                if src not in self._faiss or weight <= 0:
                    continue
                index, ids = self._faiss[src]
                kk = min(self.k, index.ntotal)
                if kk == 0:
                    continue
                sims, idx = index.search(emb, kk)
                for s, i in zip(sims[0], idx[0]):
                    if 0 <= i < len(ids):
                        merged[(ids[i], src)] = float(s) * weight
        # NTSB — in-distribution LOFO source, self-excluding the query's ev_id.
        if self._lofo is not None and self.ntsb_weight > 0:
            for eid, s in self._lofo.neighbors(text, exclude_id, self.k):
                merged[(eid, "NTSB")] = s * self.ntsb_weight
        # top-k by weighted score
        return dict(sorted(merged.items(), key=lambda kv: kv[1], reverse=True)[:self.k])

    # ---- Mode 2: deterministic schema-grounded Cypher structural search ----
    def _cypher_scores(self, record: dict) -> dict:
        """Structural retrieval: score candidates by shared STRUCTURED CONTEXT.

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
        ctx_scores = {}
        if self._lofo is not None and self.ntsb_weight > 0:
            for eid, sc in self._lofo.context_neighbors(
                    record, str(record.get("ev_id", "")), self.k):
                ctx_scores[(eid, "NTSB")] = float(sc)
        # In-distribution matches occupy [0.5, 1.0]; KG matches [0, 0.5). The bands
        # are disjoint ON PURPOSE. Weighting alone did not work: min-max maps each
        # source's best to 1.0, so asias_weight (0.34) edged out ntsb_weight (0.33),
        # and the Cypher side's integer scores tie 3-of-5 at the top and all inflate
        # to 1.0 — between them the in-distribution matches were pushed out of every
        # slot. A train record with the right label distribution is worth more than
        # any ASIAS match here, so the ordering is made explicit rather than left to
        # near-equal weights.
        out = {k: 0.5 + 0.5 * v for k, v in _minmax(ctx_scores).items()}
        if self.driver is None:
            return out
        kg_scores = {}
        try:
            params = {p: record.get(p, "") for p in CYPHER_PARAMS}
            # SDR maintenance bracket is computed (make+year), not a record column.
            params["maintenance_defect_bracket"] = _sdr_bracket(
                record.get("acft_make", ""), record.get("year", ""))
            params["k"] = self.k
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
            for k, v in _minmax(kg_scores).items():
                out[k] = 0.49 * v * w.get(k[1], 0.33) / max(self.asias_weight,
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

    def ranked_neighbors(self, record: dict, k: int | None = None) -> list:
        """Top-k neighbours as [((event_id, source), score), ...], highest first.

        Same scoring path `retrieve()` uses — semantic FAISS and/or structural
        Cypher per `self.strategy`, min-max normalized and combined 50/50 — but
        stopped BEFORE the per-neighbour labels are pooled into priors. This is
        what few-shot needs: the neighbours themselves, not their average.

        Returns [] on any failure, which the caller must treat as "no exemplars"
        (a zero-filled, masked-out block) rather than as an error.
        """
        try:
            text = record.get("combined_text", "")
            exclude_id = record.get("ev_id", "")          # LOFO self-exclusion
            faiss_scores = (self._faiss_scores(text, exclude_id)
                            if self.strategy in ("hybrid", "faiss") else {})
            cypher_scores = (self._cypher_scores(record)
                             if self.strategy in ("hybrid", "cypher") else {})
            combined = self._combine(faiss_scores, cypher_scores)
            items = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)
            return items[:(k or self.k)]
        except Exception as e:
            logging.warning("RAG: ranked_neighbors failed (%s) — no exemplars.", e)
            return []

    def event_attributes(self) -> dict:
        """Every KG event's labels and context in ONE query -> {(event_id, source): {...}}.

        Few-shot needs each neighbour's FEATURES as well as its labels, and doing
        that per neighbour would be ~5k round-trips per epoch-zero pass. The graph
        holds ~2k events, so it is far cheaper to pull the lot once and look up
        locally. NTSB events are included but are normally served by the LOFO
        source instead (see GraphFewShotSource).
        """
        if self.driver is None:
            return {}
        try:
            recs, _, _ = self.driver.execute_query(
                "MATCH (e:EventNode) "
                "OPTIONAL MATCH (e)-[:HAS_FACTOR]->(f:HFACSFactorNode) "
                "OPTIONAL MATCH (e)-[:HAS_ENV_CONTEXT]->(env:EnvironmentalContextNode) "
                "OPTIONAL MATCH (e)-[:HAS_PERSONNEL_CONTEXT]->(pc:PersonnelContextNode) "
                # Only EVIDENCE-bearing LEADS_TO: those are the LLM-extracted causal
                # links. Edges without evidence come from classify_edge's
                # deterministic DAG mapping and are a pure function of the factor
                # set, so they would add nothing the exemplar does not already have.
                "OPTIONAL MATCH (e)-[:HAS_FACTOR]->(a:HFACSFactorNode) "
                "                 -[l:LEADS_TO]->(b:HFACSFactorNode) "
                "                 <-[:HAS_FACTOR]-(e) "
                "WHERE l.evidence IS NOT NULL "
                "RETURN e.event_id AS eid, e.source AS src, "
                "       e.severity_class AS sev, "
                "       collect(DISTINCT f.tier) AS tiers, "
                "       collect(DISTINCT [env.feature, env.value]) AS env, "
                "       collect(DISTINCT [pc.feature, pc.value]) AS pers, "
                "       collect(DISTINCT [a.tier, b.tier]) AS causal",
                database_=self.database)
            out = {}
            for r in recs:
                ctx = {}
                for pair in (r["env"] or []) + (r["pers"] or []):
                    if isinstance(pair, list) and len(pair) == 2 and pair[0]:
                        ctx[pair[0]] = pair[1]
                causal = [(e[0], e[1]) for e in (r["causal"] or [])
                          if isinstance(e, list) and len(e) == 2 and e[0] and e[1]]
                out[(r["eid"], r["src"])] = {
                    "tiers": [t for t in (r["tiers"] or []) if t],
                    "severity": r["sev"],
                    "context": ctx,
                    "causal": causal,
                }
            logging.info("RAG: cached attributes for %d KG events.", len(out))
            return out
        except Exception as e:
            logging.warning("RAG: event_attributes failed (%s).", e)
            return {}

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

    def _fetch_violation(self, factors: list):
        """C is binary (violation vs error). From an event's HFACS tiers: 1 if it
        has an unsafe_violation, 0 if it has tiers but no violation, None if no
        factor info (so it doesn't bias the prior)."""
        if not factors:
            return None
        return 1 if UNSAFE_VIOLATION_TIER in factors else 0

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
                    v = self._fetch_violation(factors)            # violation prior -> C
                    if v is not None:
                        uns_acc[v] += weight
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
    def neighbors(self, text: str, exclude_id: str, k: int) -> list:
        """Top-k (ev_id, similarity) from the train split, self-excluding exclude_id."""
        if self._index is None or not _clean(text):
            return []
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
            emb = np.asarray(self._sbert.encode(self._texts, normalize_embeddings=True),
                             dtype="float32")
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
                    v = 1 if UNSAFE_VIOLATION_TIER in self._uns[i] else 0
                    uns[v] += w                                # violation outcome -> C
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
