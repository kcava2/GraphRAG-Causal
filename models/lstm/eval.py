"""
eval.py  —  Stage-6 cross-condition evaluation of the HFACS causal LSTM
=======================================================================
Evaluates trained checkpoints across up to 8 conditions on the held-out TEST
split and emits comparison images + results/eval_summary.csv.

Current architecture (3 predicted heads):
  * Organizational/Supervisory influences are a structured economic CONTEXT input
    (not text-mined, not predicted). The predicted chain is
    B (Preconditions) -> C (Unsafe Acts) -> D (Severity).
  * train.evaluate() returns 6 values. B/C are MULTI-LABEL (sigmoid + per-head
    tuned thresholds) -> F1 (micro), accuracy (label-wise), support. D (severity)
    is single-class -> F1 (macro), accuracy (+ balanced-acc, kappa, confusion).
  * Each step reports F1, accuracy, and generalization error (train - test on the
    same metric, over a capped train subsample with the checkpoint's thresholds).
  * get_dataloaders builds the test set prior-free, so a RAG checkpoint is
    evaluated on a test loader built WITH the matching retriever here.

Conditions (each needs results/c{n}.pt; evaluate whichever exist) — a source
ablation over the KG. The in-distribution NTSB-KG source is dropped: its FAISS
index is stale and overlaps the training corpus (leakage), so retrieval uses ASIAS
+ ASRS only:
  C1 no-RAG · C4 ASIAS+ASRS · C5 ASIAS-only · C6 ASRS-only · C8 raw-RAG (no factors)

Usage:
  python models/lstm/eval.py --input data/ntsb_subset.csv --model qwen2.5:3b
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (f1_score, accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix,
                             roc_curve, roc_auc_score)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..", "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "data"))
import ntsbdataloader as N                              # noqa: E402
from ntsbdataloader import (NTSBSequenceDataset, NTSBEncoders, load_and_join,   # noqa: E402
                            _split, PRECOND_SUBS, UNSAFE_SUBS, N_B, N_C,
                            ECON_DIM, STEP_B_BASE, FewShotSource)
from models.lstm.train import make_model, evaluate  # noqa: E402
from models import eval_utils as EU                     # noqa: E402

RESULTS = os.path.join(_ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

# Source ablation over the 3 retrieval sources: ASIAS + ASRS (on-disk FAISS/Neo4j)
# and NTSB = the in-distribution LOFO source (train split, self-excluding ev_id; the
# ntsb_weight component, built via set_source_df from df_train).
CONDITIONS = [
    dict(name="C1", ckpt="c1.pt", strategy=None, kw={}),
    dict(name="C4", ckpt="c4.pt", strategy="hybrid",                        # all 3 sources
         kw=dict(asias_weight=0.34, asrs_weight=0.33, ntsb_weight=0.33)),   # (incl. LOFO)
    dict(name="C5", ckpt="c5.pt", strategy="hybrid",
         kw=dict(asias_weight=1.0, asrs_weight=0.0, ntsb_weight=0.0)),       # ASIAS only
    dict(name="C6", ckpt="c6.pt", strategy="hybrid",
         kw=dict(asias_weight=0.0, asrs_weight=1.0, ntsb_weight=0.0)),       # ASRS only
    dict(name="C7", ckpt="c7.pt", strategy="hybrid",
         kw=dict(asias_weight=0.0, asrs_weight=0.0, ntsb_weight=1.0)),       # NTSB-LOFO only
    dict(name="C8", ckpt="c8.pt", strategy="hybrid",                        # all 3, no text-
         kw=dict(asias_weight=0.34, asrs_weight=0.33, ntsb_weight=0.33,      # mined factor
                 factor_priors=False)),                                     # priors (sev only)
    # Spec 2.3 prompt augmentation. C9 isolates exemplars (no priors at all);
    # C10 combines them with full hybrid retrieval.
    dict(name="C9", ckpt="c9.pt", strategy=None, kw={}),                    # few-shot only
    dict(name="C10", ckpt="c10.pt", strategy="hybrid",                      # few-shot + priors
         kw=dict(asias_weight=0.34, asrs_weight=0.33, ntsb_weight=0.33)),
]


# ---------------------------------------------------------------------------
# Inference / metrics
# ---------------------------------------------------------------------------

_FEWSHOT_SRC = None


def _fewshot_source(df_train, encoders):
    """Build the exemplar source once and reuse it across conditions (each build
    re-encodes the whole train split with SBERT)."""
    global _FEWSHOT_SRC
    if _FEWSHOT_SRC is None:
        _FEWSHOT_SRC = FewShotSource(df_train, encoders)
    return _FEWSHOT_SRC


def _loader(df, encoders, retriever, batch, fewshot_source=None, fewshot_k=0):
    """Build an eval loader. `fewshot_*` must be supplied for checkpoints trained
    with exemplars (config['fewshot_dim'] > 0), or the model silently receives
    empty exemplars and its context root is fed zeros."""
    ds = NTSBSequenceDataset(df, encoders, retriever=retriever,
                             fewshot_source=fewshot_source, fewshot_k=fewshot_k)
    return DataLoader(ds, batch_size=batch, shuffle=False)


def infer_probs(model, loader, device):
    """One pass collecting probabilities + stacked inputs. B is multi-label
    (sigmoid); C (violation vs error) and D (severity) are single-label (softmax)."""
    model.eval()
    pB, pC, pD = [], [], []
    inputs = []
    with torch.no_grad():
        for s_ctx, s_b, _yb, _yc, _yd, fs, fsm in loader:
            s_ctx, s_b = s_ctx.to(device), s_b.to(device)
            fs, fsm = fs.to(device), fsm.to(device)
            lB, lC, lD = model(s_ctx, s_b, fs, fsm)
            pB.append(torch.sigmoid(lB).cpu().numpy())
            pC.append(torch.softmax(lC, 1).cpu().numpy())
            pD.append(torch.softmax(lD, 1).cpu().numpy())
            inputs.append(torch.cat([s_ctx, s_b], dim=1).cpu().numpy())
    cat = lambda xs: np.concatenate(xs, 0) if xs else np.empty((0,))
    return cat(pB), cat(pC), cat(pD), cat(inputs)


def _f1_micro(y, p):
    return f1_score(y, p, average="micro", zero_division=0) if len(y) else 0.0


def _f1_macro_ml(y, p):
    return f1_score(y, p, average="macro", zero_division=0) if len(y) else 0.0


def _valid(a, p):
    """Drop ignore_index (-100) positions — masked ASIAS rows for the NTSB-only D."""
    a, p = np.asarray(a), np.asarray(p)
    m = a != -100
    return a[m], p[m]


def _f1_macro_sev(a, p, n):
    a, p = _valid(a, p)
    labels = list(range(n))
    return f1_score(a, p, average="macro", labels=labels, zero_division=0) if len(a) else 0.0


def _macro_balacc(y, p):
    """Macro per-tier balanced accuracy: mean over tiers (with >=1 positive) of
    (sensitivity+specificity)/2. An all-zero head scores ~0.5 (chance), unlike
    Hamming accuracy which is misleadingly high on sparse labels."""
    if y.size == 0:
        return 0.0
    accs = [balanced_accuracy_score(y[:, j], p[:, j])
            for j in range(y.shape[1]) if y[:, j].sum() > 0]
    return float(np.mean(accs)) if accs else 0.0


def ml_metrics(prefix, y, p):
    """Multi-label head: F1 (micro, headline), accuracy (label-wise), support.
    macro-F1 and macro balanced-accuracy are also kept in the CSV for reference."""
    return {
        f"{prefix}_F1": _f1_micro(y, p),
        f"{prefix}_accuracy": float((y == p).mean()) if y.size else 0.0,
        f"{prefix}_macroF1": _f1_macro_ml(y, p),
        f"{prefix}_balacc": _macro_balacc(y, p),
        f"{prefix}_support": int(y.sum()),
    }


def class_metrics(prefix, a, p, n):
    """Single-label head (binary C = violation/error, or severity D): macro-F1,
    accuracy, balanced-acc, kappa. Drops ignore_index (-100) positions."""
    a, p = _valid(a, p)
    labels = list(range(n))
    return {
        f"{prefix}_F1": f1_score(a, p, average="macro", labels=labels, zero_division=0) if len(a) else 0.0,
        f"{prefix}_accuracy": accuracy_score(a, p) if len(a) else 0.0,
        f"{prefix}_balanced_acc": balanced_accuracy_score(a, p) if len(a) else 0.0,
        f"{prefix}_kappa": cohen_kappa_score(a, p) if len(set(a.tolist())) > 1 else 0.0,
        f"{prefix}_support": int((a == 1).sum()),
    }


def severity_metrics(aD, pD, n_D):
    """Severity head: class metrics + confusion (NTSB-only; masked rows dropped)."""
    av, pv = _valid(aD, pD)
    labels = list(range(n_D))
    cm = confusion_matrix(av, pv, labels=labels) if len(av) else np.zeros((n_D, n_D), int)
    return class_metrics("D", aD, pD, n_D), (cm, labels)


def chain_completion_rate(tB, tC, tD, pB, pC, pD):
    """Full B->C->D exact match, over NTSB rows only (D defined there). B is
    multi-hot; C (violation/error) and D (severity) are single-label indices."""
    tD = np.asarray(tD)
    m = tD != -100
    if m.sum() == 0:
        return 0.0
    ok = ((tB == pB).all(1) & (np.asarray(tC) == np.asarray(pC)) & (tD == np.asarray(pD)))
    return float(ok[m].mean())


def _roc(y_true_2d, prob_2d):
    yt, pp = y_true_2d.ravel(), prob_2d.ravel()
    if yt.min() == yt.max():            # one class only -> ROC undefined
        return None
    fpr, tpr, _ = roc_curve(yt, pp)
    return fpr, tpr, roc_auc_score(yt, pp)


def mcnemar_test(base_correct, rag_correct):
    """Per-label McNemar on flattened (record x label) correctness for one head.

    Exact-match McNemar is uninformative when no record is fully correct (always
    p=1.0); per-label correctness gives many discordant pairs and a real test.
    Uses the asymptotic (chi-square) form, appropriate for large n.
    """
    from statsmodels.stats.contingency_tables import mcnemar
    both = int((base_correct & rag_correct).sum())
    b10 = int((base_correct & ~rag_correct).sum())   # base right, RAG wrong
    b01 = int((~base_correct & rag_correct).sum())   # base wrong, RAG right (RAG helps)
    neither = int((~base_correct & ~rag_correct).sum())
    n_disc = b01 + b10
    if n_disc == 0:                                  # no discordant pairs -> no difference
        return {"p_value": 1.0, "b01": b01, "b10": b10, "significant": False}
    # exact binomial for few discordant pairs (e.g. the small D head), chi-square
    # with continuity correction otherwise.
    exact = n_disc < 25
    res = mcnemar([[both, b10], [b01, neither]], exact=exact, correction=not exact)
    pv = float(res.pvalue)
    return {"p_value": pv, "b01": b01, "b10": b10, "significant": bool(pv < 0.05)}


# ---------------------------------------------------------------------------
# Feature names + sensitivity + SHAP (C1 & C4)
# ---------------------------------------------------------------------------

def feature_names(cfg):
    """Names for the flattened [step_ctx | step_b] input vector."""
    sc, sb = cfg["step_ctx_dim"], cfg["step_b_dim"]
    names = [f"ctx:{x}" for x in ["emp_qoq", "fuel_qoq", "revenue_qoq", "loadfactor_qoq",
                                  "emp_bracket", "fuel_bracket", "revenue_bracket",
                                  "loadfactor_bracket"]]
    names += [f"b:{x}" for x in ["visual", "light", "tod", "person", "pilot_hours"]]
    extra_b = sb - STEP_B_BASE                       # RAG priors: precond | unsafe | severity
    if extra_b > 0:
        names += [f"b:precPrior[{PRECOND_SUBS[i]}]" for i in range(N_B)]
        names += [f"b:unsafePrior[{x}]" for x in ("error", "violation")[:N_C]]
        names += [f"b:sevPrior[{i}]" for i in range(extra_b - N_B - N_C)]
    return names[:sc + sb]


def _scalar(model, flat, sctx_dim, device):
    """Target = mean over batch of max softmax severity probability."""
    X = torch.tensor(np.asarray(flat, dtype="float32"), device=device)
    if X.ndim == 1:
        X = X.unsqueeze(0)
    s_ctx, s_b = X[:, :sctx_dim], X[:, sctx_dim:]
    with torch.no_grad():
        *_, lD = model(s_ctx, s_b)
        return torch.softmax(lD, 1).max(1).values.cpu().numpy()


def sensitivity(model, flatX, sctx_dim, device, n_repeat=3, seed=0):
    """Perturbation importance: shuffle each feature column, measure |Δ target|."""
    rng = np.random.default_rng(seed)
    base = _scalar(model, flatX, sctx_dim, device).mean()
    D = flatX.shape[1]
    imp = np.zeros(D)
    for d in range(D):
        deltas = []
        for _ in range(n_repeat):
            Xp = flatX.copy()
            Xp[:, d] = flatX[rng.permutation(len(flatX)), d]
            deltas.append(abs(_scalar(model, Xp, sctx_dim, device).mean() - base))
        imp[d] = np.mean(deltas)
    return imp


def shap_importance(model, flatX, sctx_dim, device, nbg=20, nsample=20):
    """mean|SHAP| via KernelExplainer on the severity target; fallback=sensitivity."""
    try:
        import shap
        bg = flatX[:min(nbg, len(flatX))]
        f = lambda X: _scalar(model, X, sctx_dim, device)
        expl = shap.KernelExplainer(f, bg)
        sv = expl.shap_values(flatX[:min(nsample, len(flatX))], nsamples=100, silent=True)
        return np.abs(np.asarray(sv)).mean(axis=0)
    except Exception as e:
        print(f"  SHAP unavailable ({e.__class__.__name__}); using permutation importance.")
        return sensitivity(model, flatX[:min(nsample, len(flatX))], sctx_dim, device)


# ---------------------------------------------------------------------------
# Per-condition + main
# ---------------------------------------------------------------------------

def _retriever(strategy, model, kw):
    if not strategy:
        return None
    try:
        from rag_retriever import build_retriever
        return build_retriever(strategy=strategy, model=model, **kw)
    except Exception as e:
        print(f"  retriever unavailable ({e.__class__.__name__}) — uniform priors.")
        return None


def main():
    ap = argparse.ArgumentParser(description="Stage-6 cross-condition LSTM evaluation")
    ap.add_argument("--input", default=N.NTSB_CLEAN)
    ap.add_argument("--model", default=None, help="Ollama model for C2-C8 Cypher.")
    ap.add_argument("--results-dir", default=RESULTS)
    ap.add_argument("--train-sample", type=int, default=128,
                    help="Train rows used for the generalization gap (bounds retrieval).")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    EU._utf8()

    df = load_and_join(args.input)
    df_train, _, df_test = _split(df)
    encoders = NTSBEncoders(df_train)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Test records: {len(df_test)} | train(for gap): {min(args.train_sample, len(df_train))}")

    summary, confusions, roc_data, sens, shap_d, fnames = [], {}, {}, {}, {}, {}
    head_correct = {}   # cond -> {head: per-item correctness bool array} for McNemar

    for cond in CONDITIONS:
        path = os.path.join(args.results_dir, cond["ckpt"])
        if not os.path.exists(path):
            print(f"{cond['name']}: no checkpoint ({cond['ckpt']}) — skipping.")
            continue
        ck = torch.load(path, weights_only=False)
        cfg = ck["config"]
        if "step_ctx_dim" not in cfg:
            print(f"{cond['name']}: pre-redesign checkpoint — retrain with current "
                  f"train.py; skipping.")
            continue
        print(f"\n=== {cond['name']} ({path}) ===")
        n_D = cfg["n_D"]
        thr = ck.get("thresholds")
        retr = _retriever(cond["strategy"], args.model, cond["kw"])
        if retr is not None and hasattr(retr, "set_source_df"):
            retr.set_source_df(df_train)         # build the in-distribution NTSB-LOFO source
        model = make_model(cfg).to(device)
        model.load_state_dict(ck["state_dict"]); model.eval()

        # Checkpoints trained with exemplars must be evaluated with them; the
        # source is the train split, exactly as during training.
        fs_k = int(cfg.get("fewshot_k", 0))
        fs_src = _fewshot_source(df_train, encoders) if fs_k else None

        test_loader = _loader(df_test, encoders, retr, args.batch_size,
                              fewshot_source=fs_src, fewshot_k=fs_k)
        aB, aC, aD, pB, pC, pD = evaluate(model, test_loader, device, thr)

        n_C = cfg["n_C"]
        row = {"condition": cond["name"], "n_test": len(aD)}
        row.update(ml_metrics("B", aB, pB))                 # B multi-label
        row.update(class_metrics("C", aC, pC, n_C))         # C binary single-label
        sev, cm = severity_metrics(aD, pD, n_D)             # D NTSB-only (masked)
        row.update(sev)
        row["chain_completion_rate"] = chain_completion_rate(aB, aC, aD, pB, pC, pD)

        # generalization error per step (train − test, same thresholds + metric)
        tr_sub = df_train.head(args.train_sample)
        tr_loader = _loader(tr_sub, encoders, retr, args.batch_size,
                            fewshot_source=fs_src, fewshot_k=fs_k)
        taB, taC, taD, tpB, tpC, tpD = evaluate(model, tr_loader, device, thr)
        row["B_generror"] = float(_f1_micro(taB, tpB) - row["B_F1"])
        row["C_generror"] = float(_f1_macro_sev(taC, tpC, n_C) - row["C_F1"])
        row["D_generror"] = float(_f1_macro_sev(taD, tpD, n_D) - row["D_F1"])

        # per-head correctness for McNemar: B per-LABEL (record×group flattened, many
        # discordant pairs), C and D per-RECORD (single-label); D on NTSB rows only.
        mDok = np.asarray(aD) != -100
        head_correct[cond["name"]] = {
            "B": (aB == pB).reshape(-1),
            "C": (np.asarray(aC) == np.asarray(pC)),
            "D": (np.asarray(aD)[mDok] == np.asarray(pD)[mDok]),
        }
        confusions[cond["name"]] = cm
        summary.append(row)

        # ROC (probabilities). C/D are single-label -> use P(positive class).
        prB, prC, prD, flatX = infer_probs(model, test_loader, device)
        mD = np.asarray(aD) != -100
        onehotD = np.eye(n_D)[np.asarray(aD)[mD]] if mD.any() else np.zeros((0, n_D))
        roc_data[cond["name"]] = {
            "B": _roc(aB, prB),
            "C": _roc(np.asarray(aC), prC[:, 1]) if prC.shape[1] > 1 else None,
            "D": _roc(onehotD, prD[mD])}

        # sensitivity + SHAP for C1 and C4 only
        if cond["name"] in ("C1", "C4") and len(flatX):
            fnames[cond["name"]] = feature_names(cfg)
            sens[cond["name"]] = sensitivity(model, flatX, cfg["step_ctx_dim"], device)
            shap_d[cond["name"]] = shap_importance(model, flatX, cfg["step_ctx_dim"], device)

        if retr is not None:
            retr.close()

    if not summary:
        print("No usable checkpoints in results/. Train conditions first "
              "(train.py --save-path).")
        return

    # McNemar: C1 (no-RAG baseline) vs EACH condition, for EACH head. b01 = C1 wrong /
    # condition right (the condition HELPS that head); b10 = the reverse (HURTS). Lets
    # you pick the best condition PER HEAD instead of one config for all three.
    HEADS3 = ["B", "C", "D"]
    HEAD_NAME = {"B": "B Preconditions", "C": "C Violation", "D": "D Severity"}
    mcn_rows = []
    if "C1" in head_correct:
        base = head_correct["C1"]
        for name in [c["name"] for c in CONDITIONS if c["name"] != "C1"]:
            if name not in head_correct:
                continue
            for h in HEADS3:
                r = mcnemar_test(base[h], head_correct[name][h])
                mcn_rows.append({"head": h, "condition": name, "vs": "C1",
                                 "b01_helps": r["b01"], "b10_hurts": r["b10"],
                                 "net": r["b01"] - r["b10"], "p_value": r["p_value"],
                                 "significant": r["significant"]})

    mcn = None
    if mcn_rows:
        mdf = pd.DataFrame(mcn_rows)
        mdf.to_csv(os.path.join(args.results_dir, "mcnemar_by_head.csv"), index=False)
        print("\nMcNemar vs C1 (no-RAG) per head — recommended condition per head:")
        for h in HEADS3:
            sub = mdf[mdf["head"] == h]
            wins = sub[sub["significant"] & (sub["net"] > 0)]
            if len(wins):
                b = wins.sort_values("net", ascending=False).iloc[0]
                rec = f"{b['condition']} (net +{int(b['net'])}, p={b['p_value']:.3g})"
            else:
                hurt = sub[sub["significant"] & (sub["net"] < 0)]["condition"].tolist()
                rec = "C1 / no-RAG" + (f"  [RAG significantly HURTS: {', '.join(hurt)}]" if hurt else "")
            print(f"  {HEAD_NAME[h]:18}: {rec}")
        # legacy single-row file (C1 vs C4 on C) for back-compat
        c4c = mdf[(mdf["condition"] == "C4") & (mdf["head"] == "C")]
        if len(c4c):
            r = c4c.iloc[0]
            mcn = {"p_value": float(r["p_value"]), "b01": int(r["b01_helps"]),
                   "b10": int(r["b10_hurts"]), "significant": bool(r["significant"])}

    # save summary CSV
    pd.DataFrame(summary).to_csv(os.path.join(args.results_dir, "eval_summary.csv"), index=False)
    if mcn:
        pd.DataFrame([mcn]).to_csv(os.path.join(args.results_dir, "mcnemar_c1_c4.csv"), index=False)
    print(f"\nWrote {os.path.join(args.results_dir, 'eval_summary.csv')} + mcnemar_by_head.csv")

    # figures
    print("\nFigures:")
    EU.plot_step_metrics(summary)            # F1 / accuracy / generalization error per step
    EU.plot_chain(summary)                   # chain-completion rate per condition
    EU.plot_severity(summary, confusions)
    EU.plot_roc(roc_data)
    for cn in ("C1", "C4"):
        if cn in sens:
            EU.plot_sensitivity({cn: sens[cn]}, fnames[cn], name=f"eval_sensitivity_{cn}.png")
        if cn in shap_d:
            EU.plot_shap({cn: shap_d[cn]}, fnames[cn], name=f"eval_shap_{cn}.png")
    cols = ["n_test",
            "B_F1", "B_accuracy", "B_generror",
            "C_F1", "C_accuracy", "C_generror",
            "D_F1", "D_accuracy", "D_generror",
            "chain_completion_rate"]
    EU.plot_summary_table(summary, cols)
    print("\nDone.")


if __name__ == "__main__":
    main()
