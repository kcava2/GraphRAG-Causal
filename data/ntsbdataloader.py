"""
ntsbdataloader.py  (Stage 4)
============================
Builds the NTSB-only training pipeline for the multi-label causal-chain LSTM:

    [org/sup context] -> B = Preconditions -> C = Unsafe Acts -> D = Severity

Organizational and Supervisory influences are NOT text-mined or predicted. The
upper HFACS tier is represented by **structured economic context** (employment +
fuel cost) on a non-predicted root node (``step_ctx``) that seeds the chain —
preserving the HFACS edge (organizational pressure -> preconditions) without the
data-starved org/supervisory heads.

B/C are **multi-label** (a record may have several co-occurring HFACS
subcategories within a step, and factors co-occur across the chain). Their label
spaces are fixed by ``HFACS_SCHEMA`` (imported from hfacs_extractor), so the
multi-hot targets come straight from ``hfacs_results.csv``. D (severity) is a
single-class ordinal target from ``ntsb_clean.csv`` (classes 0+1 merged).

ASIAS/ASRS never enter this stage. The KG/FAISS indexes from Stage 3 are reached
only via an optional ``retriever`` (Stage 5) that appends RAG priors.

Side effect of ``get_dataloaders``: writes ``ntsb.faiss`` + ``ntsb_faiss_ids.json``
from the **training split** ``combined_text`` (read-only afterward).

No SMOTE, no synthetic data.
"""

import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:                 # allow sibling import whether run as a
    sys.path.insert(0, _HERE)             # script (cwd=data/) or as data.ntsbdataloader
from hfacs_extractor import HFACS_SCHEMA, EXTRACT_TIERS  # single source of truth
from standardize import SEVERITY_HIGH_THRESHOLD, strip_outcome  # noqa

NTSB_CLEAN = os.path.join(_HERE, "ntsb_clean.csv")
HFACS_RESULTS = os.path.join(_HERE, "hfacs_results.csv")
FAISS_INDEX = os.path.join(_HERE, "ntsb.faiss")
FAISS_IDMAP = os.path.join(_HERE, "ntsb_faiss_ids.json")
# Two DIFFERENT sentence encoders, deliberately.
#
#   SBERT_MODEL      Stage-2 extraction few-shot (data/ntsb.faiss). Pinned to
#                    all-MiniLM-L6-v2 because that index is what the committed
#                    extraction was produced with; changing it would silently alter
#                    the prompts and break comparability with hfacs_results.csv.
#
#   RETRIEVAL_MODEL  Stage-5 retrieval — the LOFO pool, the query encoding, and the
#                    asias/asrs/ntsb_kg FAISS indexes. Free to improve, because
#                    nothing downstream of it is already committed to disk except
#                    those indexes, which are rebuilt together with it.
#
# Chosen by measuring the retrieval vote directly on the test split (k=5, no model).
# AUC is the metric that matters, because the vote enters the model as a FEATURE —
# what counts is whether it RANKS records correctly, not whether it clears 0.5:
#
#     encoder             B auc   C auc   D auc      B balacc  C balacc  D balacc
#     all-MiniLM-L6-v2    0.671   0.708   0.905        0.589     0.531     0.827
#     all-mpnet-base-v2   0.763   0.773   0.965        0.631     0.495     0.926  <- chosen
#     bge-base-en-v1.5    0.736   0.647   0.932        0.598     0.497     0.870
#     e5-base-v2          0.705   0.501   0.953        0.580     0.495     0.892
#
# mpnet wins on all three heads by AUC. Note C: its thresholded balanced accuracy
# looks WORSE than MiniLM's (0.495 vs 0.531) while its AUC is better (0.773 vs
# 0.708). C is only 9% positive, so a 0.5 cut on the vote almost never fires and
# measures the threshold rather than the ranking. Judging encoders on the
# thresholded number would have picked the weaker one.
#
# **Changing RETRIEVAL_MODEL requires rebuilding the FAISS indexes**
# (`python data/kg_builder.py --faiss-only`): query vectors from a 768-dim encoder
# cannot be searched against a 384-dim index.
SBERT_MODEL = "all-MiniLM-L6-v2"
RETRIEVAL_MODEL = os.environ.get("RETRIEVAL_MODEL", "all-mpnet-base-v2")


# ---------------------------------------------------------------------------
# Fixed multi-label group label spaces (order is stable -> column order)
# ---------------------------------------------------------------------------

ORG_TIERS = ["org_climate", "resource_mgmt", "org_process"]
SUP_TIERS = ["supervisory"]
# situational_phys (Weather/Lighting/Terrain) excluded: ENVIRONMENTAL context,
# supplied as structured inputs (visual/light), not text-mined/predicted.
PRECOND_TIERS = ["operator_mental", "operator_physical", "operator_limits",
                 "situational_tech", "personnel_crm", "personnel_readiness"]
UNSAFE_TIERS = ["unsafe_skill", "unsafe_decision", "unsafe_perception",
                "unsafe_violation"]

# B (Preconditions) is predicted at the COARSER HFACS-GROUP level, not the 6 raw
# tiers: the rare tiers (operator_limits 4%, personnel_readiness 1% = 20 records)
# pinned the 6-way head at chance, exactly like unsafe_perception did for C. The 6
# tiers collapse to 3 learnable groups (each 22-50% prevalent). Multi-label kept.
PRECOND_GROUPS = {
    "precond_operator":    ["operator_mental", "operator_physical", "operator_limits"],
    "precond_personnel":   ["personnel_crm", "personnel_readiness"],
    "precond_situational": ["situational_tech"],
}
# tier -> group column index, for the retriever's precondition prior (KG factor
# nodes are stored at the raw-tier level, so they're mapped up to the group here).
PRECOND_GROUP_INDEX = {t: i for i, tiers in enumerate(PRECOND_GROUPS.values()) for t in tiers}

ORG_SUBS = ORG_TIERS                # y_O space (3; org not predicted)
SUP_SUBS = SUP_TIERS                # y_A space (1; sup not predicted)
PRECOND_SUBS = list(PRECOND_GROUPS) # y_B space (3 precondition GROUPS, multi-label)
UNSAFE_SUBS = UNSAFE_TIERS          # y_C space (4 unsafe-act TIERS, multi-label)

# C (Unsafe Acts) is a FOUR-TIER MULTI-LABEL head: skill / decision / perception /
# violation, one sigmoid each. It was collapsed to a binary violation-vs-error
# target while unsafe_perception sat at ~6% and pinned a four-way head at chance.
# On the current extraction that no longer holds (decision 71%, skill 31%,
# perception 14%, violation 9%), and the binary target was the worst of the four:
# 15 positives in the test split and a label two extraction runs agree on at
# kappa 0.10. Multi-label rather than single-label because 23% of records carry
# two or more unsafe-act tiers; a softmax would have to throw one away.
UNSAFE_VIOLATION_TIER = "unsafe_violation"
N_O, N_A, N_B = len(ORG_SUBS), len(SUP_SUBS), len(PRECOND_SUBS)
N_C = len(UNSAFE_SUBS)              # 4 unsafe-act tiers, multi-label

# Strict, consensus re-labelling of the violation tier (data/adjudicate_violation.py).
# Applied as an override on top of hfacs_results.csv when the file exists, so the
# committed extraction stays intact. HFACS_VIOLATION_OVERRIDE=0 switches it off.
VIOLATION_OVERRIDE = os.path.join(_HERE, "violation_adjudication.csv")

# step_ctx = organizational/supervisory CONTEXT, sourced from structured economic
# data (no text mining), QoQ-only (no absolute levels). invest_type is EXCLUDED:
# it encodes accident-vs-incident, which directly leaks the severity target.
ECON_DIM = 8                   # emp_qoq, fuel_qoq, revenue_qoq, loadfactor_qoq,
                               # + emp/fuel/revenue/loadfactor brackets
STEP_CTX_DIM = ECON_DIM

# step_b base layout (indices the model slices) — keep in sync with the model.
# Environmental/person features; sky_conditions dropped (100% Unknown = dead).
ENV_SLICE = slice(0, 3)        # visual, light, time_of_day (env -> C, env -> D)
OPER_SLICE = slice(3, 5)       # person_involved, pilot_hours (operator -> C)
STEP_B_BASE = 5                # visual, light, tod, person, pilot_hours


# ---------------------------------------------------------------------------
# Loading / HFACS join
# ---------------------------------------------------------------------------

def _multihot(active: set, vocab: list) -> np.ndarray:
    return np.array([1.0 if s in active else 0.0 for s in vocab], dtype="float32")


def _apply_violation_override(ev_ids, unsafe_sets, path: str = None, verbose: bool = True):
    """Replace the committed `unsafe_violation` label with the strict adjudicated one.

    The committed extraction hands out `unsafe_violation` for almost any
    non-compliance, including by passengers and by operators as organisations;
    `adjudicate_violation.py` re-decides that single tier under the HFACS
    definition (operational role + identifiable rule + knowing deviation) by
    majority vote. When a violation is overturned and that leaves the record with
    no unsafe act at all, the adjudicator's error tier is used instead — HFACS
    files an unintentional rule breach as an error, not as nothing.

    No-op when the file is absent or HFACS_VIOLATION_OVERRIDE=0.
    """
    path = path or VIOLATION_OVERRIDE
    if os.environ.get("HFACS_VIOLATION_OVERRIDE", "1") == "0" or not os.path.exists(path):
        return list(unsafe_sets)
    adj = pd.read_csv(path, dtype=str).fillna("")
    final = {e: int(float(f or 0)) for e, f in zip(adj["ev_id"], adj["final"])}
    reclass = dict(zip(adj["ev_id"], adj.get("reclass", pd.Series([""] * len(adj)))))
    out, removed, added, refiled = [], 0, 0, 0
    for ev, s in zip(ev_ids.astype(str), unsafe_sets):
        s = set(s)
        if ev in final:
            had = UNSAFE_VIOLATION_TIER in s
            if final[ev] and not had:
                s.add(UNSAFE_VIOLATION_TIER); added += 1
            elif had and not final[ev]:
                s.discard(UNSAFE_VIOLATION_TIER); removed += 1
                if not s and reclass.get(ev) in UNSAFE_TIERS:
                    s.add(reclass[ev]); refiled += 1
        out.append(s)
    if verbose:
        print(f"  Violation override: {removed} removed, {added} added, {refiled} "
              f"re-filed as an error tier ({os.path.basename(path)}).")
    return out


def load_and_join(filepath: str = NTSB_CLEAN,
                  hfacs_path: str = HFACS_RESULTS) -> pd.DataFrame:
    """
    Load ntsb_clean.csv and LEFT-JOIN the HFACS extraction on ev_id. Parses
    hfacs_json into per-group active-subcategory sets. Records missing from
    hfacs_results.csv (or non-success) get empty sets -> all-zero multi-hot.
    Drops rows whose severity_class is missing/invalid.
    """
    df = pd.read_csv(filepath, dtype=str)

    hf = {}
    if os.path.exists(hfacs_path):
        h = pd.read_csv(hfacs_path, dtype=str)
        for _, r in h.iterrows():
            if str(r.get("extraction_status")) != "success":
                continue
            try:
                hf[str(r["ev_id"])] = json.loads(r.get("hfacs_json") or "{}")
            except (json.JSONDecodeError, TypeError):
                continue

    def _sets(ev_id):
        # Which labels are present. Preconditions are returned at the GROUP level (a
        # group is present if any of its tiers was extracted); org/sup/unsafe at tier.
        cls = hf.get(str(ev_id), {})
        grab = lambda tiers: {t for t in tiers if cls.get(t)}
        pre_groups = {g for g, tiers in PRECOND_GROUPS.items()
                      if any(cls.get(t) for t in tiers)}
        return (grab(ORG_TIERS), grab(SUP_TIERS), pre_groups, grab(UNSAFE_TIERS))

    sets = [_sets(e) for e in df["ev_id"]]
    df["_org"] = [s[0] for s in sets]
    df["_sup"] = [s[1] for s in sets]
    df["_pre"] = [s[2] for s in sets]
    df["_uns"] = _apply_violation_override(df["ev_id"], [s[3] for s in sets])

    sev = pd.to_numeric(df["severity_class"], errors="coerce")
    df = df[sev.notna()].reset_index(drop=True)
    # Binarize severity -> high(1)/low(0). The cleaned data has injury-COUNT
    # severity only (no fatality flag), so this is a high-severity proxy; >=3
    # (2+ injuries) is balanced ~55/45 and far more learnable than the old 4-class.
    df["severity_class"] = (pd.to_numeric(df["severity_class"], errors="coerce")
                            >= SEVERITY_HIGH_THRESHOLD).astype(int).astype(str)
    return df


# ---------------------------------------------------------------------------
# Encoders (fit on the TRAINING split only)
# ---------------------------------------------------------------------------

def _num(series, fill=0.0):
    return pd.to_numeric(series, errors="coerce").fillna(fill).astype("float32")


def _time_of_day(light_series) -> np.ndarray:
    """Binary proxy from light_conditions: Daylight -> 1, else 0 (LocalEventTime absent)."""
    return (light_series.astype(str).str.strip() == "Daylight").astype("float32").to_numpy()


class NTSBEncoders:
    """
    LabelEncoders fit on the training split only (no leakage). Multi-label
    O/A/B/C label spaces are fixed by the schema (no fitting needed).
    """

    def __init__(self, df_train: pd.DataFrame):
        self.enc_visual = LabelEncoder().fit(df_train["visual_condition"].astype(str))
        self.enc_light = LabelEncoder().fit(df_train["light_conditions"].astype(str))
        self.enc_sky = LabelEncoder().fit(df_train["sky_conditions"].astype(str))
        self.enc_person = LabelEncoder().fit(df_train["person_involved"].astype(str))
        self.enc_pilot_hours = LabelEncoder().fit(df_train["pilot_hours_bracket"].astype(str))
        self.enc_emp_bracket = LabelEncoder().fit(df_train["employment_bracket"].astype(str))
        self.enc_fuel_bracket = LabelEncoder().fit(df_train["fuel_bracket"].astype(str))
        self.enc_rev_bracket = LabelEncoder().fit(df_train["revenue_bracket"].astype(str))
        self.enc_lf_bracket = LabelEncoder().fit(df_train["loadfactor_bracket"].astype(str))
        # Fit on STRING-cast severity so it matches _safe_transform's str compare
        # (Bug fix: fitting on ints made every value "unseen" -> collapsed to 0).
        self.enc_severity = LabelEncoder().fit(
            pd.to_numeric(df_train["severity_class"], errors="coerce").astype(int).astype(str))

    @property
    def n_O(self): return N_O

    @property
    def n_A(self): return N_A

    @property
    def n_B(self): return N_B

    @property
    def n_C(self): return N_C

    @property
    def n_severity(self): return len(self.enc_severity.classes_)

    @staticmethod
    def _safe_transform(enc: LabelEncoder, values) -> np.ndarray:
        """Transform, mapping unseen labels (not in train) to class 0."""
        known = set(enc.classes_)
        vals = [v if v in known else enc.classes_[0] for v in values.astype(str)]
        return enc.transform(vals).astype("float32")


# ---------------------------------------------------------------------------
# Few-shot exemplar source (spec 2.3 "prompt augmentation")
# ---------------------------------------------------------------------------

def encode_step_b_base(df: pd.DataFrame, e: "NTSBEncoders") -> np.ndarray:
    """The 5 base step_b features. Shared by the dataset and the few-shot source
    so a query and an exemplar are always encoded identically."""
    return np.column_stack([
        e._safe_transform(e.enc_visual, df["visual_condition"]),
        e._safe_transform(e.enc_light, df["light_conditions"]),
        _time_of_day(df["light_conditions"]),
        e._safe_transform(e.enc_person, df["person_involved"]),
        e._safe_transform(e.enc_pilot_hours, df["pilot_hours_bracket"]),
    ]).astype("float32")


# ---------------------------------------------------------------------------
# Causal-chain encoding for exemplars
# ---------------------------------------------------------------------------
# Each exemplar carries the CAUSAL CHAIN the extractor found in that event, as a
# role vector over the 10 mined tiers: for each tier, was it the SOURCE of a
# LEADS_TO edge, and was it the TARGET of one.
#
# Why roles from the EXTRACTED edges, and not the graph's LEADS_TO:
# `kg_builder.classify_edge` derives graph LEADS_TO deterministically from
# DAG_EDGES given which factors co-occur. That makes those edges a pure function
# of the tier set the exemplar already encodes in y_B/y_C — reading them back
# would add exactly zero information, which is why nothing reading them has cost
# nothing. The LLM-extracted relationships are independent evidence: the model
# chose specific directed links with supporting quotes, and they frequently do NOT
# follow the DAG (the three most common are unsafe_decision -> unsafe_skill,
# unsafe_perception -> unsafe_decision, unsafe_decision -> unsafe_violation, none
# of which is a DAG edge). Those carry real signal about chain ORDER, beyond which
# tiers are present.
CAUSAL_TIERS = list(EXTRACT_TIERS)
CAUSAL_DIM = 2 * len(CAUSAL_TIERS)        # [is-source x10 | is-target x10]
_CAUSAL_IDX = {t: i for i, t in enumerate(CAUSAL_TIERS)}


def causal_roles(edges) -> np.ndarray:
    """Directed tier pairs -> role vector. `edges` is an iterable of (src, dst)."""
    v = np.zeros(CAUSAL_DIM, dtype="float32")
    for src, dst in edges or []:
        i, j = _CAUSAL_IDX.get(src), _CAUSAL_IDX.get(dst)
        if i is not None:
            v[i] = 1.0                                    # tier acted as a cause
        if j is not None:
            v[len(CAUSAL_TIERS) + j] = 1.0                # tier acted as an effect
    return v


# Exemplar row layout — ONE definition, read by the sources here and by the encoder:
#
#   [ base (5) | y_B (3) | y_C (4) | y_D one-hot (2) | causal roles (20)
#     | has_factor_labels (1) | has_severity (1) | similarity (1) ]
#
# The two `has_*` flags say whether a label block is KNOWN for this neighbour. They
# exist because "no label" and "label is zero" used to be the same bytes: an ASRS
# event with no injury data, or a KG event whose HFACS factors were mined by a
# different prompt, showed up as an all-zero block and was averaged into the
# neighbour vote as a confident negative. With the flag the vote for a block is
# taken over the neighbours that actually carry it.
#
# `similarity` is the retrieval score on an ABSOLUTE scale (cosine, or the matched
# share of structured context). It used to be min-max normalised across the k
# neighbours, which forced the weakest one to exactly 0.0 — silently dropping it
# from the vote — and turned the column into a rank index rather than a similarity.
FS_BASE = slice(0, STEP_B_BASE)
FS_B = slice(FS_BASE.stop, FS_BASE.stop + N_B)
FS_C = slice(FS_B.stop, FS_B.stop + N_C)
FS_D = slice(FS_C.stop, FS_C.stop + 2)
FS_CAUSAL = slice(FS_D.stop, FS_D.stop + CAUSAL_DIM)
FS_HAS_BC = FS_CAUSAL.stop
FS_HAS_D = FS_HAS_BC + 1
FS_SIM = FS_HAS_D + 1
FEWSHOT_DIM = FS_SIM + 1


def _exemplar_matrix(df: pd.DataFrame, encoders: "NTSBEncoders",
                     raw_mode: bool = False) -> np.ndarray:
    """[n, FEWSHOT_DIM - 1] exemplar rows for NTSB records (similarity is per query).

    Shared by both exemplar sources so train-split rows are built identically.
    `raw_mode` (condition C5) removes the LLM-mined B/C labels and marks them
    unknown, leaving structured severity as the only label carried.
    """
    df = df.reset_index(drop=True)
    mat = np.zeros((len(df), FEWSHOT_DIM - 1), dtype="float32")
    mat[:, FS_BASE] = encode_step_b_base(df, encoders)
    mat[:, FS_B] = np.stack([_multihot(s, PRECOND_SUBS) for s in df["_pre"]])
    mat[:, FS_C] = np.stack([_multihot(s, UNSAFE_SUBS) for s in df["_uns"]])
    y_D = pd.to_numeric(df["severity_class"], errors="coerce").fillna(0).astype(int).to_numpy()
    mat[:, FS_D] = np.eye(2, dtype="float32")[np.clip(y_D, 0, 1)]
    mat[:, FS_CAUSAL] = _train_causal_roles(df)
    mat[:, FS_HAS_BC] = 1.0
    mat[:, FS_HAS_D] = 1.0
    if raw_mode:
        mat[:, FS_B] = 0.0
        mat[:, FS_C] = 0.0
        # The causal roles are LLM-mined too, and a role vector over the ten tiers
        # says which tiers are present. Leaving it in made C5 a partial ablation.
        mat[:, FS_CAUSAL] = 0.0
        mat[:, FS_HAS_BC] = 0.0
    return mat


class FewShotSource:
    """Retrieves labelled EXEMPLARS — (features, labels) pairs — for a query.

    This is the neural analogue of few-shot prompting: rather than collapsing the
    neighbours into an averaged prior (what `_retrieve_priors` does), each
    neighbour is kept intact as an input/output pair, and the model reads the
    whole set. The averaged prior discards which features produced which label;
    an exemplar keeps that association, which is the entire point of a shot.

    **Leakage discipline.** The source is the NTSB TRAIN split only, and a query
    never retrieves itself (`exclude_id`). A val/test record is not in the source
    at all, and a train record cannot see its own label. This mirrors
    `LOFORetriever` exactly — exemplars are drawn from the same pool, and for the
    same reason. ASIAS/ASRS are deliberately NOT exemplar sources: their label
    space differs (ASIAS severity is ignore_index) and mixing them would teach the
    model from labels it is never scored on.
    """

    def __init__(self, source_df: pd.DataFrame, encoders: "NTSBEncoders"):
        df = source_df.reset_index(drop=True)
        self.ids = df["ev_id"].astype(str).tolist()
        self._texts = [strip_outcome(t) for t in           # retrieval text only
                       df["combined_text"].astype(str).fillna("").tolist()]

        # Columns follow the FS_* layout; similarity is filled per query.
        self.matrix = _exemplar_matrix(df, encoders)

        self._sbert = None
        self._index = None
        self._build_index()

    def _build_index(self):
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
            self._sbert = SentenceTransformer(RETRIEVAL_MODEL)
            emb = np.asarray(self._sbert.encode(self._texts, normalize_embeddings=True),
                             dtype="float32")
            self._index = faiss.IndexFlatIP(emb.shape[1])
            self._index.add(emb)
            print(f"  Few-shot source: indexed {len(self.ids)} train exemplars.")
        except Exception as exc:
            print(f"  Few-shot source: index build failed ({exc}) — exemplars disabled.")
            self._index = None

    def lookup(self, record: dict, k: int):
        """-> (exemplars [k, FEWSHOT_DIM], mask [k]). Zero-padded when short."""
        text = str(record.get("combined_text", ""))
        exclude_id = str(record.get("ev_id", ""))
        out = np.zeros((k, FEWSHOT_DIM), dtype="float32")
        mask = np.zeros(k, dtype="float32")
        if self._index is None or not text.strip():
            return out, mask
        try:
            emb = np.asarray(self._sbert.encode([strip_outcome(text)],
                                                normalize_embeddings=True),
                             dtype="float32")
            sims, idx = self._index.search(emb, min(k + 1, len(self.ids)))
            used = 0
            for s, i in zip(sims[0], idx[0]):
                if i < 0 or i >= len(self.ids) or self.ids[i] == exclude_id:
                    continue                                  # self-exclusion (LOFO)
                out[used, :-1] = self.matrix[i]
                out[used, -1] = float(s)                      # similarity as a feature
                mask[used] = 1.0
                used += 1
                if used >= k:
                    break
        except Exception as exc:
            # Narrow, and loud. A bare pass here previously swallowed a column-count
            # mismatch and returned an all-zero mask, which looks exactly like
            # "nothing retrieved" — the failure mode is invisible at the call site.
            logging.warning("FewShotSource.lookup failed (%s) — no exemplars for %s",
                            exc, exclude_id)
        return out, mask


def _train_causal_roles(df: pd.DataFrame) -> np.ndarray:
    """[n, CAUSAL_DIM] role vectors from hfacs_results.csv relationships_json."""
    rel = {}
    try:
        r = pd.read_csv(HFACS_RESULTS, dtype=str)
        r = r[r["extraction_status"] == "success"]
        for ev, js in zip(r["ev_id"].astype(str), r["relationships_json"].fillna("[]")):
            try:
                items = json.loads(js)
            except (json.JSONDecodeError, TypeError):
                continue
            rel[ev] = [(e.get("subject"), e.get("object")) for e in items
                       if isinstance(e, dict) and e.get("relation") == "LEADS_TO"]
    except Exception:
        pass
    return np.stack([causal_roles(rel.get(str(ev), [])) for ev in df["ev_id"]])


class GraphFewShotSource:
    """Few-shot exemplars retrieved from the KNOWLEDGE GRAPH (spec 2.2 A/B/C).

    Same exemplar layout as `FewShotSource`, but the neighbours come from
    `RAGRetriever` under a chosen strategy, so the retrieval STRATEGY becomes the
    experimental variable:

        faiss   -> Strategy A, semantic  (narrative similarity)
        cypher  -> Strategy B, structural (shared context nodes)
        hybrid  -> Strategy C, 50/50 combination

    Each neighbour is one exemplar row: its context features from the graph, its
    HFACS tiers as y_B/y_C, its stored severity as y_D, and its retrieval score.
    All event attributes are pulled once (`event_attributes`) and looked up
    locally — per-neighbour queries would be thousands of round-trips.

    Missing data is encoded, not dropped. ASRS carries no injury data, so its
    severity one-hot is left all-zeros — distinct from a real [1,0]/[0,1] — and
    a neighbour missing a context feature gets that feature's "Unknown" code.

    `raw_mode=True` zeroes the y_B/y_C columns and keeps only severity: the
    exemplar equivalent of `--no-factor-priors`, i.e. retrieval with the
    LLM-mined content removed (condition C5).
    """

    def __init__(self, retriever, raw_mode: bool = False,
                 kg_factor_labels: bool = False):
        self.retriever = retriever
        self.raw_mode = raw_mode
        # KG events (ASIAS/ASRS) had their HFACS factors mined by kg_builder under
        # a different prompt, and their base rates do not match the NTSB targets
        # (see attach_encoders). By default those B/C labels are therefore marked
        # UNKNOWN: the neighbour still contributes its context, causal roles,
        # severity and similarity, but does not vote on B or C. Pass True to let
        # them vote (the behaviour before this flag existed).
        self.kg_factor_labels = kg_factor_labels
        self._rows = {}

    def attach_encoders(self, encoders: "NTSBEncoders", source_df: pd.DataFrame = None):
        """Build the exemplar rows. Deferred because the encoders are fit on the
        train split inside get_dataloaders, after this source is constructed.

        When `source_df` (the TRAIN split) is given, it is also registered on the
        retriever as the in-distribution LOFO source and its records become
        exemplars in their own right. This matters more than anything else here:
        KG exemplars are mined from ASIAS/ASRS by `kg_builder` with a different
        prompt, so their label distribution does not match the NTSB targets —

            precond_operator     KG 16.2%   vs   NTSB target 71.7%
            precond_situational  KG 43.5%   vs   NTSB target 20.6%

        For head D that mismatch is survivable, because severity is a structured
        field computed the same way for every source. For B and C, whose labels are
        LLM-mined, KG exemplars are actively misleading. Train-split exemplars carry
        exactly the target distribution, and LOFO self-exclusion keeps them honest:
        a record never retrieves itself, and val/test records are not in the pool.
        """
        if self._rows:
            return                                             # already built
        attrs = self.retriever.event_attributes() if self.retriever is not None else {}
        for key, a in attrs.items():
            self._rows[key] = self._build_row(a, encoders)
        n_kg = len(self._rows)

        n_train = 0
        if source_df is not None and self.retriever is not None:
            if hasattr(self.retriever, "set_source_df"):
                self.retriever.set_source_df(source_df)        # activates LOFO
            n_train = self._add_train_rows(source_df, encoders)

        n_sev = sum(1 for r in self._rows.values() if r[FS_HAS_D] > 0)
        n_lab = sum(1 for r in self._rows.values() if r[FS_HAS_BC] > 0)
        print(f"  Graph few-shot source: {n_kg} KG events + {n_train} in-distribution "
              f"train records ({n_sev} with severity, {n_lab} voting on B/C)"
              f"{' [raw mode: no factor labels]' if self.raw_mode else ''}.")

    def _add_train_rows(self, df: pd.DataFrame, e: "NTSBEncoders") -> int:
        """Exemplar rows for the NTSB train split, keyed to match LOFO's output.

        `LOFORetriever` returns neighbours as (ev_id, "NTSB"), so these are keyed
        the same way. The 100 NTSB-KG events already in `_rows` come from the
        disjoint `ntsb_kg_subset`, so there is no id collision.
        """
        df = df.reset_index(drop=True)
        # Same builder as FewShotSource, so both sources encode a train record
        # identically. raw_mode (C5) strips every LLM-mined column.
        mat = _exemplar_matrix(df, e, raw_mode=self.raw_mode)
        for i, ev in enumerate(df["ev_id"].astype(str)):
            self._rows[(ev, "NTSB")] = mat[i]
        return len(df)

    def _build_row(self, a: dict, e: "NTSBEncoders") -> np.ndarray:
        from standardize import binarize_severity
        row = np.zeros(FEWSHOT_DIM - 1, dtype="float32")      # sim filled per query
        ctx = a.get("context", {})
        one = lambda enc, v: float(e._safe_transform(enc, pd.Series([str(v)]))[0])
        light = ctx.get("light_conditions", "Unknown")
        row[0] = one(e.enc_visual, ctx.get("visual_condition", "Unknown"))
        row[1] = one(e.enc_light, light)
        row[2] = float(_time_of_day(pd.Series([str(light)]))[0])
        row[3] = one(e.enc_person, ctx.get("person_involved", "Unknown"))
        row[4] = one(e.enc_pilot_hours, ctx.get("pilot_hours_bracket", "Unknown"))

        tiers = a.get("tiers", []) or []
        if not self.raw_mode:                                  # LLM-mined content
            if tiers and self.kg_factor_labels:
                for t in tiers:
                    gi = PRECOND_GROUP_INDEX.get(t)
                    if gi is not None:
                        row[FS_B.start + gi] = 1.0
                    if t in UNSAFE_SUBS:
                        row[FS_C.start + UNSAFE_SUBS.index(t)] = 1.0
                row[FS_HAS_BC] = 1.0                           # labels known
            row[FS_CAUSAL] = causal_roles(a.get("causal"))     # extracted chain

        sev = a.get("severity")                                # structured outcome
        if sev is not None:
            try:
                b = binarize_severity(sev)
                if b is not None and 0 <= int(b) < 2:
                    row[FS_D.start + int(b)] = 1.0
                    row[FS_HAS_D] = 1.0
            except Exception:
                pass                                           # flag stays 0 = unknown
        return row

    def warm(self, texts):
        """Batch-encode query narratives before the per-record lookups."""
        if self.retriever is not None and hasattr(self.retriever, "warm") \
                and self.retriever.strategy in ("faiss", "hybrid"):
            self.retriever.warm(texts)

    def lookup(self, record: dict, k: int):
        """-> (exemplars [k, FEWSHOT_DIM], mask [k]) from the graph."""
        out = np.zeros((k, FEWSHOT_DIM), dtype="float32")
        mask = np.zeros(k, dtype="float32")
        if self.retriever is None:
            return out, mask
        used = 0
        # Ask for more candidates than slots. A neighbour with no exemplar row (a KG
        # event when Neo4j is down, say) used to consume a slot and leave the record
        # short of exemplars without any message.
        for key, score in self.retriever.ranked_neighbors(record, k=k, fetch=3 * k):
            row = self._rows.get(key)
            if row is None:
                continue
            out[used, :-1] = row
            out[used, -1] = float(score)
            mask[used] = 1.0
            used += 1
            if used >= k:
                break
        return out, mask


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NTSBSequenceDataset(Dataset):
    """
    Feeds the five causal LSTM steps. RAG priors (when ``retriever`` is given)
    are appended at the END of each step vector so the base indices the model
    slices (ENV_SLICE, supervisory block in step_b) stay valid.

    Organizational/Supervisory influences are NOT text-mined or predicted; the
    upper HFACS tier is represented by structured economic context (step_ctx),
    which seeds the chain  context -> B(Preconditions) -> C(Unsafe) -> D(Severity).

    __getitem__ -> (step_ctx, step_b, y_B, y_C, y_D, fewshot, fewshot_mask)
        step_ctx : [emp_qoq, fuel_qoq, revenue_qoq, loadfactor_qoq,
                    emp_bracket, fuel_bracket, revenue_bracket, loadfactor_bracket]
                   (organizational/economic pressure, QoQ-only)
        step_b   : [visual, light, time_of_day, person, pilot_hours]
                   (+ precond_prior | unsafe_prior | severity_prior  when RAG)
        y_B      : multi-hot float vector (Preconditions, BCE target)
        y_C      : multi-hot float vector over the 4 unsafe-act tiers (BCE target)
        y_D      : binary severity class index (long; high/low) — NTSB only; ASIAS
                   rows carry -100 (ignore_index) so they don't train/eval D
        fewshot  : [k, FEWSHOT_DIM] retrieved labelled exemplars, empty when
                   fewshot_k=0 (see FewShotSource)
        fewshot_mask : [k] 1.0 for a real exemplar, 0.0 for padding
    """

    def __init__(self, df: pd.DataFrame, encoders: NTSBEncoders, retriever=None,
                 fewshot_source: "FewShotSource | None" = None, fewshot_k: int = 0):
        e = encoders
        df = df.reset_index(drop=True)

        # ---- targets: Preconditions (B, multi-label) + Unsafe Acts (C, binary) ----
        # Organizational + Supervisory are no longer text-mined or predicted; the
        # upper HFACS tier is the structured economic context (step_ctx) below.
        self.y_B = torch.tensor(
            np.stack([_multihot(s, PRECOND_SUBS) for s in df["_pre"]]), dtype=torch.float32)
        # C is a multi-hot float vector over the four unsafe-act tiers (BCE target).
        self.y_C = torch.tensor(
            np.stack([_multihot(s, UNSAFE_SUBS) for s in df["_uns"]]), dtype=torch.float32)

        # D (severity) is trained/evaluated on NTSB rows ONLY. ASIAS severity is
        # gravity-based, ~all low, and trivially separable from NTSB (sky='UNK',
        # empty crew_age, etc.) — combining it lets the model predict source≈severity
        # instead of real severity. Non-NTSB rows get the CE ignore_index (-100) so
        # they still train B/C (their narratives) but contribute nothing to D.
        y_D = e._safe_transform(
            e.enc_severity,
            pd.to_numeric(df["severity_class"], errors="coerce").astype(int)).astype("int64")
        if "_source" in df.columns:
            is_ntsb = df["_source"].astype(str).str.upper().eq("NTSB").to_numpy()
            y_D = np.where(is_ntsb, y_D, -100)
        self.y_D = torch.tensor(y_D, dtype=torch.long)

        # ---- step_ctx: organizational/supervisory context (structured economic) ----
        # QoQ-only (no absolute levels): employment, fuel, operating revenue, load
        # factor — each as a continuous delta + its discretized bracket.
        step_ctx = np.column_stack([
            _num(df["employment_qoq_pct"]),
            _num(df["fuel_cost_qoq_pct"]),
            _num(df["operating_revenue_qoq_pct"]),
            _num(df["load_factor_qoq_pct"]),
            e._safe_transform(e.enc_emp_bracket, df["employment_bracket"]),
            e._safe_transform(e.enc_fuel_bracket, df["fuel_bracket"]),
            e._safe_transform(e.enc_rev_bracket, df["revenue_bracket"]),
            e._safe_transform(e.enc_lf_bracket, df["loadfactor_bracket"]),
        ]).astype("float32")

        # ---- step_b: environmental/person features (sky_conditions dropped) ----
        step_b = encode_step_b_base(df, e)

        # ---- optional RAG priors appended to step_b: precond | unsafe | severity ----
        # Kept as a separate attribute too, so the ensemble (spec 2.3) can read the
        # retrieval-only prediction without re-running retrieval.
        self.rag_priors = None
        if retriever is not None:
            pre_p, uns_p, sev_p = self._retrieve_priors(retriever, df, e)
            step_b = np.concatenate([step_b, pre_p, uns_p, sev_p], axis=1).astype("float32")
            self.rag_priors = {
                "precondition": torch.tensor(pre_p, dtype=torch.float32),
                "unsafe": torch.tensor(uns_p, dtype=torch.float32),
                "severity": torch.tensor(sev_p, dtype=torch.float32),
            }

        self.step_ctx = torch.tensor(step_ctx, dtype=torch.float32)
        self.step_b = torch.tensor(step_b, dtype=torch.float32)

        # ---- optional few-shot exemplars (spec 2.3 "prompt augmentation") ----
        # Shape [n, k, FEWSHOT_DIM]; k=0 yields an empty tensor so the tuple arity
        # of __getitem__ never changes and downstream code stays uniform.
        if fewshot_source is not None and fewshot_k > 0:
            ex = np.zeros((len(df), fewshot_k, FEWSHOT_DIM), dtype="float32")
            mk = np.zeros((len(df), fewshot_k), dtype="float32")
            # Full record dicts: the graph-backed source needs the Cypher params
            # (context brackets, make/year) as well as the narrative and ev_id.
            fs_recs = df[[c for c in self._RETRIEVE_COLS
                          if c in df.columns]].astype(str).to_dict("records")
            if hasattr(fewshot_source, "warm"):          # batch-encode the queries once
                fewshot_source.warm([r.get("combined_text", "") for r in fs_recs])
            for i, rec in enumerate(fs_recs):
                ex[i], mk[i] = fewshot_source.lookup(rec, fewshot_k)
            n_with = int((mk.sum(1) > 0).sum())
            print(f"  Few-shot exemplars: {n_with}/{len(df)} records got >=1 "
                  f"(k={fewshot_k}).")
            self.fewshot = torch.tensor(ex, dtype=torch.float32)
            self.fewshot_mask = torch.tensor(mk, dtype=torch.float32)
        else:
            self.fewshot = torch.zeros(len(df), 0, FEWSHOT_DIM, dtype=torch.float32)
            self.fewshot_mask = torch.zeros(len(df), 0, dtype=torch.float32)

    # Columns the Stage-5 retriever needs (narrative + Cypher structural params).
    _RETRIEVE_COLS = ["ev_id", "combined_text", "visual_condition", "light_conditions",
                      "employment_bracket", "fuel_bracket", "revenue_bracket",
                      "loadfactor_bracket", "person_involved", "pilot_hours_bracket",
                      "acft_make", "year"]   # ev_id -> LOFO self-exclusion; make/year -> SDR

    def _retrieve_priors(self, retriever, df, encoders):
        """RAG priors over Preconditions (n_B), Unsafe Acts (n_C), and Severity
        (n_severity); uniform on any failure. Returns (precond, unsafe, severity)
        so the model can feed each into B, C, and D respectively.

        Reports how many records got a NON-uniform prior per head — the honest
        measure of whether RAG carries signal (a flat prior is invisible).
        """
        n = len(df)
        sev_n = encoders.n_severity
        pre = np.full((n, N_B), 1.0 / N_B, dtype="float32")
        uns = np.full((n, N_C), 1.0 / N_C, dtype="float32")
        sev = np.full((n, sev_n), 1.0 / sev_n, dtype="float32")
        nonuni = {"pre": 0, "uns": 0, "sev": 0}
        records = df[self._RETRIEVE_COLS].astype(str).to_dict("records")
        for i, rec in enumerate(records):
            try:
                p = retriever.retrieve(rec, encoders)
                for key, arr, exp, tag in (("precondition_prior", pre, N_B, "pre"),
                                           ("unsafe_prior", uns, N_C, "uns"),
                                           ("severity_prior", sev, sev_n, "sev")):
                    v = np.asarray(p.get(key), dtype="float32")
                    if v.shape == (exp,):
                        arr[i] = v
                        if float(np.ptp(v)) > 1e-9:      # not flat -> real signal
                            nonuni[tag] += 1
            except Exception:
                pass  # keep uniform priors on failure
        if n:
            print(f"  RAG priors non-uniform: precond {nonuni['pre']}/{n}, "
                  f"unsafe {nonuni['uns']}/{n}, severity {nonuni['sev']}/{n}")
        return pre, uns, sev

    def __len__(self):
        return len(self.y_D)

    def __getitem__(self, idx):
        # Few-shot tensors are appended LAST so existing star-unpacking consumers
        # (e.g. eval.infer_probs' `s_ctx, s_b, *_`) keep working unchanged.
        return (self.step_ctx[idx], self.step_b[idx],
                self.y_B[idx], self.y_C[idx], self.y_D[idx],
                self.fewshot[idx], self.fewshot_mask[idx])


# ---------------------------------------------------------------------------
# Retrieval gate — does the exemplar block carry anything, before any training?
# ---------------------------------------------------------------------------

def exemplar_vote(fewshot: np.ndarray, mask: np.ndarray, tau: float = 0.1) -> dict:
    """Similarity-weighted neighbour vote per head, straight from an exemplar block.

    Mirrors `FewShotEncoder`'s vote: softmax(similarity / tau) over the real
    neighbours, and each label block averaged only over neighbours whose labels
    are KNOWN (the has_* flags). Returns {'B': [n,N_B], 'C': [n,N_C], 'D': [n]}.
    """
    sim = fewshot[..., FS_SIM]
    logit = np.where(mask > 0, sim / tau, -1e9)
    w = np.exp(logit - logit.max(1, keepdims=True)) * (mask > 0)
    w = w / np.clip(w.sum(1, keepdims=True), 1e-9, None)

    def block(sl, flag):
        wf = w * fewshot[..., flag]
        return (fewshot[..., sl] * wf[..., None]).sum(1) / np.clip(
            wf.sum(1, keepdims=True), 1e-9, None)
    d = block(FS_D, FS_HAS_D)
    return {"B": block(FS_B, FS_HAS_BC), "C": block(FS_C, FS_HAS_BC), "D": d[:, 1]}


def retrieval_gate(dataset: "NTSBSequenceDataset", name: str = "val") -> dict:
    """Print and return the exemplar vote's AUC per label on `dataset`.

    This is the honest measure of whether retrieval can help a head at all. The
    vote is a fixed function of the exemplar block — no weights, no training — so
    if it cannot rank a label, no architecture reading the same block will. It
    is also where a silent retrieval failure shows up: coverage can read 100%
    while every neighbour is uninformative.
    """
    from sklearn.metrics import roc_auc_score
    if dataset.fewshot.shape[1] == 0:
        return {}
    v = exemplar_vote(dataset.fewshot.numpy(), dataset.fewshot_mask.numpy())
    out = {}

    def auc(y, p):
        return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")
    for j, g in enumerate(PRECOND_SUBS):
        out[f"B:{g.replace('precond_', '')}"] = auc(dataset.y_B[:, j].numpy(), v["B"][:, j])
    for j, t in enumerate(UNSAFE_SUBS):
        out[f"C:{t.replace('unsafe_', '')}"] = auc(dataset.y_C[:, j].numpy(), v["C"][:, j])
    yd = dataset.y_D.numpy()
    m = yd != -100
    out["D:severity"] = auc(yd[m], v["D"][m])
    sim = dataset.fewshot[..., FS_SIM].numpy()
    msk = dataset.fewshot_mask.numpy() > 0
    out["_mean_similarity"] = float(sim[msk].mean()) if msk.any() else float("nan")
    out["_mean_exemplars"] = float(msk.sum(1).mean())
    print(f"  Retrieval gate [{name}] vote AUC: "
          + "  ".join(f"{k} {a:.3f}" for k, a in out.items() if not k.startswith("_"))
          + f"  | mean similarity {out['_mean_similarity']:.3f}, "
            f"exemplars/record {out['_mean_exemplars']:.1f}")
    return out


# ---------------------------------------------------------------------------
# FAISS index (training split only) — built once, read-only afterward
# ---------------------------------------------------------------------------

def _build_ntsb_faiss(df_train: pd.DataFrame):
    """Embed train combined_text and write ntsb.faiss + ntsb_faiss_ids.json."""
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("WARNING: faiss / sentence-transformers unavailable — "
              "skipping ntsb.faiss build.")
        return
    texts = df_train["combined_text"].astype(str).fillna("").tolist()
    ids = df_train["ev_id"].astype(str).tolist()
    model = SentenceTransformer(SBERT_MODEL)
    emb = np.asarray(model.encode(texts, normalize_embeddings=True), dtype="float32")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, FAISS_INDEX)
    with open(FAISS_IDMAP, "w", encoding="utf-8") as f:
        json.dump(ids, f)
    print(f"Built {FAISS_INDEX} (ntotal={index.ntotal}, dim={emb.shape[1]}) "
          f"and {os.path.basename(FAISS_IDMAP)}")


# ---------------------------------------------------------------------------
# Split + get_dataloaders
# ---------------------------------------------------------------------------

def _split(df, seed=42, test_split=0.2, val_split=0.1):
    """70/10/10 split via seeded torch.randperm. Single source of truth."""
    n = len(df)
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=rng).tolist()
    n_test, n_val = int(n * test_split), int(n * val_split)
    n_train = n - n_test - n_val
    return (df.iloc[perm[:n_train]].reset_index(drop=True),
            df.iloc[perm[n_train:n_train + n_val]].reset_index(drop=True),
            df.iloc[perm[n_train + n_val:]].reset_index(drop=True))


def build_faiss_only(filepath=NTSB_CLEAN, seed=42, test_split=0.2, val_split=0.1):
    """Build ntsb.faiss + ntsb_faiss_ids.json from the train split only — run
    BEFORE Stage-2 extraction so few-shot retrieval can fire. No training."""
    df = load_and_join(filepath)
    df_train, _, _ = _split(df, seed, test_split, val_split)
    print(f"Building ntsb.faiss from {len(df_train)} train records of {filepath}")
    _build_ntsb_faiss(df_train)


def get_dataloaders(filepath: str = NTSB_CLEAN, test_split=0.2, val_split=0.1,
                    batch_size=32, seed=42, retriever=None, build_faiss=True,
                    limit=None, fewshot_k: int = 0, fewshot_source=None):
    """
    Returns (train_loader, val_loader, test_loader, encoders).

    70/10/10 split (seed=42, mirrors ntsb_train_ids). Encoders fit on the train
    split only; ntsb.faiss built from the train split only. No SMOTE.

    The retriever (if any) attaches to train + val; the test set stays prior-free
    (eval.py rebuilds it with the retriever). If the retriever exposes
    set_source_df, it is given df_train so its IN-DISTRIBUTION NTSB source (the
    leave-one-out LOFO source, active when ntsb_weight>0) is built from this split.
    """
    df = load_and_join(filepath)
    if limit:                                    # smoke-test subset
        df = df.head(limit).reset_index(drop=True)
    df_train, df_val, df_test = _split(df, seed, test_split, val_split)

    encoders = NTSBEncoders(df_train)
    if build_faiss:
        _build_ntsb_faiss(df_train)

    if retriever is not None and hasattr(retriever, "set_source_df"):
        retriever.set_source_df(df_train)        # in-distribution NTSB-LOFO source

    # Few-shot exemplar source. Caller may supply one (e.g. GraphFewShotSource for
    # a retrieval-strategy condition); otherwise fall back to the train-split
    # semantic source. Either way it is built once and shared, so val/test query
    # the same pool the model trained against.
    if fewshot_k > 0 and fewshot_source is None:
        fewshot_source = FewShotSource(df_train, encoders)
    elif fewshot_k == 0:
        fewshot_source = None
    if fewshot_source is not None and hasattr(fewshot_source, "attach_encoders"):
        # df_train is passed so the source can register the in-distribution LOFO
        # pool and use train records as exemplars (see attach_encoders).
        fewshot_source.attach_encoders(encoders, df_train)

    mk = lambda d, r: NTSBSequenceDataset(d, encoders, retriever=r,
                                          fewshot_source=fewshot_source,
                                          fewshot_k=fewshot_k)
    train_set = mk(df_train, retriever)
    val_set = mk(df_val, retriever)
    test_set = mk(df_test, None)                 # eval.py re-attaches the retriever

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, encoders


def main():
    import argparse
    ap = argparse.ArgumentParser(description="NTSB dataloader / FAISS index builder")
    ap.add_argument("--build-faiss-only", action="store_true",
                    help="Build ntsb.faiss from the train split of --input and exit "
                         "(run before Stage-2 extraction so few-shot can fire).")
    ap.add_argument("--input", default=NTSB_CLEAN)
    args = ap.parse_args()

    if args.build_faiss_only:
        build_faiss_only(args.input)
        return

    tr, va, te, enc = get_dataloaders(filepath=args.input, build_faiss=False)
    s_ctx, s_b, yB, yC, yD, _fs, _fsm = next(iter(tr))
    print("step_ctx:", s_ctx.shape, "step_b:", s_b.shape)
    print("y_B/y_C/y_D:", yB.shape, yC.shape, yD.shape)
    print("n_B/n_C(tiers)/n_severity:", enc.n_B, enc.n_C, enc.n_severity)


if __name__ == "__main__":
    main()
