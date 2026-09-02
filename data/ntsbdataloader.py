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
from hfacs_extractor import HFACS_SCHEMA  # single source of truth for the schema
from standardize import SEVERITY_HIGH_THRESHOLD  # binary severity threshold

NTSB_CLEAN = os.path.join(_HERE, "ntsb_clean.csv")
HFACS_RESULTS = os.path.join(_HERE, "hfacs_results.csv")
FAISS_INDEX = os.path.join(_HERE, "ntsb.faiss")
FAISS_IDMAP = os.path.join(_HERE, "ntsb_faiss_ids.json")
SBERT_MODEL = "all-MiniLM-L6-v2"


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
UNSAFE_SUBS = UNSAFE_TIERS          # retained for imports; C is binary now (see N_C)

# C (Unsafe Acts) is a BINARY SINGLE-LABEL head — violation vs error-only — not a
# 4-way multi-label. The 4 unsafe tiers are collapsed: class 1 if an
# unsafe_violation was extracted, else class 0 (errors only / none). This trades
# granularity (the rare perception tier sat at ~5% prevalence and pinned C at
# chance) for a learnable target. Full 4-tier multi-label C is future work.
UNSAFE_VIOLATION_TIER = "unsafe_violation"
N_O, N_A, N_B = len(ORG_SUBS), len(SUP_SUBS), len(PRECOND_SUBS)
N_C = 2                             # C classes: 0 = error/none, 1 = violation

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
    df["_uns"] = [s[3] for s in sets]

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

    __getitem__ -> (step_ctx, step_b, y_B, y_C, y_D)
        step_ctx : [emp_qoq, fuel_qoq, revenue_qoq, loadfactor_qoq,
                    emp_bracket, fuel_bracket, revenue_bracket, loadfactor_bracket]
                   (organizational/economic pressure, QoQ-only)
        step_b   : [visual, light, time_of_day, person, pilot_hours]
                   (+ precond_prior | unsafe_prior | severity_prior  when RAG)
        y_B      : multi-hot float vector (Preconditions, BCE target)
        y_C      : binary class index (long; 1 = violation, 0 = error/none)
        y_D      : binary severity class index (long; high/low) — NTSB only; ASIAS
                   rows carry -100 (ignore_index) so they don't train/eval D
    """

    def __init__(self, df: pd.DataFrame, encoders: NTSBEncoders, retriever=None):
        e = encoders
        df = df.reset_index(drop=True)

        # ---- targets: Preconditions (B, multi-label) + Unsafe Acts (C, binary) ----
        # Organizational + Supervisory are no longer text-mined or predicted; the
        # upper HFACS tier is the structured economic context (step_ctx) below.
        self.y_B = torch.tensor(
            np.stack([_multihot(s, PRECOND_SUBS) for s in df["_pre"]]), dtype=torch.float32)
        # C is a binary single-label class index: 1 if a violation was extracted.
        self.y_C = torch.tensor(
            np.array([1 if UNSAFE_VIOLATION_TIER in s else 0 for s in df["_uns"]],
                     dtype="int64"), dtype=torch.long)

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
        step_b = np.column_stack([
            e._safe_transform(e.enc_visual, df["visual_condition"]),
            e._safe_transform(e.enc_light, df["light_conditions"]),
            _time_of_day(df["light_conditions"]),
            e._safe_transform(e.enc_person, df["person_involved"]),
            e._safe_transform(e.enc_pilot_hours, df["pilot_hours_bracket"]),
        ]).astype("float32")

        # ---- optional RAG priors appended to step_b: precond | unsafe | severity ----
        if retriever is not None:
            pre_p, uns_p, sev_p = self._retrieve_priors(retriever, df, e)
            step_b = np.concatenate([step_b, pre_p, uns_p, sev_p], axis=1).astype("float32")

        self.step_ctx = torch.tensor(step_ctx, dtype=torch.float32)
        self.step_b = torch.tensor(step_b, dtype=torch.float32)

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
        return (self.step_ctx[idx], self.step_b[idx],
                self.y_B[idx], self.y_C[idx], self.y_D[idx])


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
                    limit=None):
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

    train_set = NTSBSequenceDataset(df_train, encoders, retriever=retriever)
    val_set = NTSBSequenceDataset(df_val, encoders, retriever=retriever)
    test_set = NTSBSequenceDataset(df_test, encoders, retriever=None)  # eval.py re-attaches

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
    s_ctx, s_b, yB, yC, yD = next(iter(tr))
    print("step_ctx:", s_ctx.shape, "step_b:", s_b.shape)
    print("y_B/y_C/y_D:", yB.shape, yC.shape, yD.shape)
    print("n_B/n_C(classes)/n_severity:", enc.n_B, enc.n_C, enc.n_severity)


if __name__ == "__main__":
    main()
