"""
ntsbdataloader.py  (Stage 4)
============================
Builds the NTSB-only training pipeline for the multi-label causal-chain LSTM:

    O = Organizational Influences -> A = Supervisory -> B = Preconditions
        -> C = Unsafe Acts -> D = Severity

O/A/B/C are **multi-label** (a record may have several co-occurring HFACS
subcategories within a step, and factors co-occur across the chain). Their label
spaces are fixed by ``HFACS_SCHEMA`` (imported from hfacs_extractor), so the
multi-hot targets come straight from ``hfacs_results.csv`` — no lossy
single-class reduction. D (severity) is a single-class ordinal target from
``ntsb_clean.csv``.

Economic pressure (employment + fuel cost) drives the Organizational step as
direct inputs — raw floats and LabelEncoded brackets, analogous to how the
environmental features feed the Preconditions step.

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

NTSB_CLEAN = os.path.join(_HERE, "ntsb_clean.csv")
HFACS_RESULTS = os.path.join(_HERE, "hfacs_results.csv")
FAISS_INDEX = os.path.join(_HERE, "ntsb.faiss")
FAISS_IDMAP = os.path.join(_HERE, "ntsb_faiss_ids.json")
SBERT_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Fixed multi-label group label spaces (order is stable -> column order)
# ---------------------------------------------------------------------------

ORG_TIERS = ["org_climate", "resource_mgmt", "org_process"]
ORG_SUBS = [s for t in ORG_TIERS for s in HFACS_SCHEMA[t]]          # y_O (12)
SUP_SUBS = list(HFACS_SCHEMA["supervisory"])                        # y_A (4)
PRECOND_TIERS = ["operator_mental", "operator_physical", "operator_limits",
                 "situational_phys", "situational_tech",
                 "personnel_crm", "personnel_readiness"]
PRECOND_SUBS = [s for t in PRECOND_TIERS for s in HFACS_SCHEMA[t]]  # y_B (30)
UNSAFE_TIERS = ["unsafe_skill", "unsafe_decision", "unsafe_perception",
                "unsafe_violation"]
UNSAFE_SUBS = [s for t in UNSAFE_TIERS for s in HFACS_SCHEMA[t]]    # y_C (19)

N_O, N_A, N_B, N_C = len(ORG_SUBS), len(SUP_SUBS), len(PRECOND_SUBS), len(UNSAFE_SUBS)

# step_b base layout (indices the model slices) — keep in sync with the model.
ENV_SLICE = slice(0, 3)        # visual, light, sky
SUP_START = 6                  # supervisory multi-hot block starts here
STEP_B_BASE = 6 + N_A          # 6 scalar features + supervisory multi-hot (n_A)


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
        cls = hf.get(str(ev_id), {})
        grab = lambda tiers: {s for t in tiers for s in cls.get(t, [])
                              if isinstance(s, str)}
        return (grab(ORG_TIERS), grab(["supervisory"]),
                grab(PRECOND_TIERS), grab(UNSAFE_TIERS))

    sets = [_sets(e) for e in df["ev_id"]]
    df["_org"] = [s[0] for s in sets]
    df["_sup"] = [s[1] for s in sets]
    df["_pre"] = [s[2] for s in sets]
    df["_uns"] = [s[3] for s in sets]

    sev = pd.to_numeric(df["severity_class"], errors="coerce")
    df = df[sev.notna()].reset_index(drop=True)
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
        self.enc_severity = LabelEncoder().fit(
            pd.to_numeric(df_train["severity_class"], errors="coerce").astype(int))

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

    __getitem__ -> (step_o, step_a, step_b, y_O, y_A, y_B, y_C, y_D)
        step_o : [emp_qoq, fuel_qoq, industry_total, fuel_cost_per_gallon
                  | emp_bracket_enc, fuel_bracket_enc | invest_type_binary]  (+ org_prior)
        step_a : [org influences multi-hot y_O (n_O)]                        (+ sup_prior)
        step_b : [visual, light, sky, time_of_day, person, pilot_hours
                  | supervisory multi-hot y_A (n_A)]                         (+ precond_prior)
        y_O/y_A/y_B/y_C : multi-hot float vectors (BCE targets)
        y_D : severity class index (long)
    """

    def __init__(self, df: pd.DataFrame, encoders: NTSBEncoders, retriever=None):
        e = encoders
        df = df.reset_index(drop=True)

        # ---- multi-hot targets straight from the joined HFACS sets ----
        self.y_O = torch.tensor(
            np.stack([_multihot(s, ORG_SUBS) for s in df["_org"]]), dtype=torch.float32)
        self.y_A = torch.tensor(
            np.stack([_multihot(s, SUP_SUBS) for s in df["_sup"]]), dtype=torch.float32)
        self.y_B = torch.tensor(
            np.stack([_multihot(s, PRECOND_SUBS) for s in df["_pre"]]), dtype=torch.float32)
        self.y_C = torch.tensor(
            np.stack([_multihot(s, UNSAFE_SUBS) for s in df["_uns"]]), dtype=torch.float32)
        y_D = e._safe_transform(
            e.enc_severity, pd.to_numeric(df["severity_class"], errors="coerce").astype(int))
        self.y_D = torch.tensor(y_D, dtype=torch.long)

        # ---- step_o: economic pressure (raw floats + bracket categoricals) ----
        step_o = np.column_stack([
            _num(df["employment_qoq_pct"]),
            _num(df["fuel_cost_qoq_pct"]),
            _num(df["industry_total"]),
            _num(df["fuel_cost_per_gallon"]),
            e._safe_transform(e.enc_emp_bracket, df["employment_bracket"]),
            e._safe_transform(e.enc_fuel_bracket, df["fuel_bracket"]),
            _num(df["invest_type_binary"]),
        ]).astype("float32")

        # ---- step_a: organizational influences (teacher-forced y_O) ----
        step_a = self.y_O.numpy().astype("float32")

        # ---- step_b: environmental/person features + supervisory (teacher-forced y_A) ----
        step_b_scalars = np.column_stack([
            e._safe_transform(e.enc_visual, df["visual_condition"]),
            e._safe_transform(e.enc_light, df["light_conditions"]),
            e._safe_transform(e.enc_sky, df["sky_conditions"]),
            _time_of_day(df["light_conditions"]),
            e._safe_transform(e.enc_person, df["person_involved"]),
            e._safe_transform(e.enc_pilot_hours, df["pilot_hours_bracket"]),
        ]).astype("float32")
        step_b = np.concatenate([step_b_scalars, self.y_A.numpy()], axis=1).astype("float32")

        # ---- optional RAG priors appended at the END ----
        if retriever is not None:
            org_p, sup_p, pre_p = self._retrieve_priors(retriever, df, e)
            step_o = np.concatenate([step_o, org_p], axis=1).astype("float32")
            step_a = np.concatenate([step_a, sup_p], axis=1).astype("float32")
            step_b = np.concatenate([step_b, pre_p], axis=1).astype("float32")

        self.step_o = torch.tensor(step_o, dtype=torch.float32)
        self.step_a = torch.tensor(step_a, dtype=torch.float32)
        self.step_b = torch.tensor(step_b, dtype=torch.float32)

    # Columns the Stage-5 retriever needs (narrative + Cypher structural params).
    _RETRIEVE_COLS = ["combined_text", "visual_condition", "light_conditions",
                      "employment_bracket", "fuel_bracket", "person_involved",
                      "pilot_hours_bracket"]

    def _retrieve_priors(self, retriever, df, encoders):
        """RAG priors over O (n_O), A (n_A), B (n_B); uniform on any failure."""
        n = len(df)
        org = np.full((n, N_O), 1.0 / N_O, dtype="float32")
        sup = np.full((n, N_A), 1.0 / N_A, dtype="float32")
        pre = np.full((n, N_B), 1.0 / N_B, dtype="float32")
        records = df[self._RETRIEVE_COLS].astype(str).to_dict("records")
        for i, rec in enumerate(records):
            try:
                p = retriever.retrieve(rec, encoders)
                for key, arr, exp in (("organizational_prior", org, N_O),
                                      ("supervisory_prior", sup, N_A),
                                      ("precondition_prior", pre, N_B)):
                    v = np.asarray(p.get(key), dtype="float32")
                    if v.shape == (exp,):
                        arr[i] = v
            except Exception:
                pass  # keep uniform priors on failure
        return org, sup, pre

    def __len__(self):
        return len(self.y_D)

    def __getitem__(self, idx):
        return (self.step_o[idx], self.step_a[idx], self.step_b[idx],
                self.y_O[idx], self.y_A[idx], self.y_B[idx], self.y_C[idx], self.y_D[idx])


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
    split only; ntsb.faiss built from the train split only. No SMOTE. The
    retriever (if any) attaches to train + val only — the test set stays a
    clean, prior-free baseline.
    """
    df = load_and_join(filepath)
    if limit:                                    # smoke-test subset
        df = df.head(limit).reset_index(drop=True)
    df_train, df_val, df_test = _split(df, seed, test_split, val_split)

    encoders = NTSBEncoders(df_train)
    if build_faiss:
        _build_ntsb_faiss(df_train)

    train_set = NTSBSequenceDataset(df_train, encoders, retriever=retriever)
    val_set = NTSBSequenceDataset(df_val, encoders, retriever=retriever)
    test_set = NTSBSequenceDataset(df_test, encoders, retriever=None)  # clean baseline

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
    s_o, s_a, s_b, yO, yA, yB, yC, yD = next(iter(tr))
    print("step_o:", s_o.shape, "step_a:", s_a.shape, "step_b:", s_b.shape)
    print("y_O/y_A/y_B/y_C/y_D:", yO.shape, yA.shape, yB.shape, yC.shape, yD.shape)
    print("n_O/n_A/n_B/n_C/n_severity:", enc.n_O, enc.n_A, enc.n_B, enc.n_C, enc.n_severity)


if __name__ == "__main__":
    main()
