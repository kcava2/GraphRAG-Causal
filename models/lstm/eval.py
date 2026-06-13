"""
eval.py  —  Stage-6 cross-condition evaluation of the HFACS causal LSTM
=======================================================================
Evaluates trained checkpoints across up to 8 conditions on the held-out TEST
split and emits comparison images + results/eval_summary.csv.

RECONCILED to the current architecture (supersedes the old spec):
  * train.evaluate() returns 10 values (5 heads O/A/B/C/D). O/A/B/C are
    MULTI-LABEL (sigmoid) → micro/macro-F1, precision, recall, exact-match.
    D (severity) is single-class → accuracy, balanced-acc, macro-F1, kappa.
  * get_dataloaders builds the test set prior-free, so a C2-C8 checkpoint (which
    expects prior-appended dims) is evaluated on a test loader built WITH the
    matching retriever here (RAG priors computed at test time).

Conditions (each needs results/c{n}.pt; evaluate whichever exist):
  C1 no-RAG · C2 FAISS · C3 Cypher · C4 hybrid ·
  C5 hybrid ASIAS-only (1.0/0.0) · C6 hybrid ASRS-only (0.0/1.0) ·
  C7 FAISS k=10 · C8 hybrid k=1   (C5-C8 = retrieval ablations)

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
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix,
                             roc_curve, roc_auc_score)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..", "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "data"))
import ntsbdataloader as N                              # noqa: E402
from ntsbdataloader import (NTSBSequenceDataset, NTSBEncoders, load_and_join,   # noqa: E402
                            _split, ORG_SUBS, SUP_SUBS, PRECOND_SUBS, N_O, N_A)
from models.lstm.train import HFACSCausalLSTM, evaluate  # noqa: E402
from models import eval_utils as EU                     # noqa: E402

RESULTS = os.path.join(_ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

# Condition table. C5-C8 are retrieval ablations exercising the rag_retriever
# knobs (per-source FAISS weights, retrieval depth k).
CONDITIONS = [
    dict(name="C1", ckpt="c1.pt", strategy=None, kw={}),
    dict(name="C2", ckpt="c2.pt", strategy="faiss", kw={}),
    dict(name="C3", ckpt="c3.pt", strategy="cypher", kw={}),
    dict(name="C4", ckpt="c4.pt", strategy="hybrid", kw={}),
    dict(name="C5", ckpt="c5.pt", strategy="hybrid", kw=dict(asias_weight=1.0, asrs_weight=0.0)),
    dict(name="C6", ckpt="c6.pt", strategy="hybrid", kw=dict(asias_weight=0.0, asrs_weight=1.0)),
    dict(name="C7", ckpt="c7.pt", strategy="faiss", kw=dict(k=10)),
    dict(name="C8", ckpt="c8.pt", strategy="hybrid", kw=dict(k=1)),
]


# ---------------------------------------------------------------------------
# Inference / metrics
# ---------------------------------------------------------------------------

def _loader(df, encoders, retriever, batch):
    ds = NTSBSequenceDataset(df, encoders, retriever=retriever)
    return DataLoader(ds, batch_size=batch, shuffle=False)


def infer_probs(model, loader, device):
    """One pass collecting sigmoid/softmax probabilities + stacked inputs."""
    model.eval()
    pO, pA, pB, pC, pD = [], [], [], [], []
    inputs = []
    with torch.no_grad():
        for s_o, s_a, s_b, *_ in loader:
            s_o, s_a, s_b = s_o.to(device), s_a.to(device), s_b.to(device)
            lO, lA, lB, lC, lD = model(s_o, s_a, s_b)
            for lg, store in ((lO, pO), (lA, pA), (lB, pB), (lC, pC)):
                store.append(torch.sigmoid(lg).cpu().numpy())
            pD.append(torch.softmax(lD, 1).cpu().numpy())
            inputs.append(torch.cat([s_o, s_a, s_b], dim=1).cpu().numpy())
    cat = lambda xs: np.concatenate(xs, 0) if xs else np.empty((0,))
    return (cat(pO), cat(pA), cat(pB), cat(pC), cat(pD), cat(inputs))


def head_ml_metrics(prefix, y, p):
    kw = dict(zero_division=0)
    return {
        f"{prefix}_microF1": f1_score(y, p, average="micro", **kw),
        f"{prefix}_macroF1": f1_score(y, p, average="macro", **kw),
        f"{prefix}_P": precision_score(y, p, average="micro", **kw),
        f"{prefix}_R": recall_score(y, p, average="micro", **kw),
        f"{prefix}_exact": float((y == p).all(axis=1).mean()) if len(y) else 0.0,
        f"{prefix}_support": int(y.sum()),
    }


def severity_metrics(aD, pD, n_D):
    labels = list(range(n_D))
    cm = confusion_matrix(aD, pD, labels=labels) if len(aD) else np.zeros((n_D, n_D), int)
    return {
        "D_accuracy": accuracy_score(aD, pD) if len(aD) else 0.0,
        "D_balanced_acc": balanced_accuracy_score(aD, pD) if len(aD) else 0.0,
        "D_macroF1": f1_score(aD, pD, average="macro", labels=labels, zero_division=0) if len(aD) else 0.0,
        "D_kappa": cohen_kappa_score(aD, pD) if len(set(aD.tolist())) > 1 else 0.0,
    }, (cm, labels)


def chain_completion_rate(tO, tA, tB, tC, tD, pO, pA, pB, pC, pD):
    if len(tD) == 0:
        return 0.0
    ok = ((tO == pO).all(1) & (tA == pA).all(1) & (tB == pB).all(1)
          & (tC == pC).all(1) & (tD == pD))
    return float(ok.mean())


def _roc(y_true_2d, prob_2d):
    yt, pp = y_true_2d.ravel(), prob_2d.ravel()
    if yt.min() == yt.max():            # one class only -> ROC undefined
        return None
    fpr, tpr, _ = roc_curve(yt, pp)
    return fpr, tpr, roc_auc_score(yt, pp)


def mcnemar_test(base_correct, rag_correct):
    from statsmodels.stats.contingency_tables import mcnemar
    both = int((base_correct & rag_correct).sum())
    b10 = int((base_correct & ~rag_correct).sum())   # base right, RAG wrong
    b01 = int((~base_correct & rag_correct).sum())   # base wrong, RAG right (RAG helps)
    neither = int((~base_correct & ~rag_correct).sum())
    res = mcnemar([[both, b10], [b01, neither]], exact=True)
    return {"p_value": float(res.pvalue), "b01": b01, "b10": b10,
            "significant": bool(res.pvalue < 0.05)}


# ---------------------------------------------------------------------------
# Feature names + sensitivity + SHAP (C1 & C4)
# ---------------------------------------------------------------------------

def feature_names(cfg):
    n_o, n_a, n_b = cfg["step_o_dim"], cfg["step_a_dim"], cfg["step_b_dim"]
    names = [f"o:{x}" for x in ["emp_qoq", "fuel_qoq", "industry_total", "fuel_cpg",
                                "emp_bracket", "fuel_bracket", "invest_type"]]
    names += [f"o:orgPrior[{ORG_SUBS[i]}]" for i in range(n_o - 7)]
    names += [f"a:org[{ORG_SUBS[i]}]" for i in range(N_O)]
    names += [f"a:supPrior[{SUP_SUBS[i]}]" for i in range(n_a - N_O)]
    names += [f"b:{x}" for x in ["visual", "light", "sky", "tod", "person", "pilot_hours"]]
    names += [f"b:sup[{SUP_SUBS[i]}]" for i in range(N_A)]
    names += [f"b:precPrior[{PRECOND_SUBS[i]}]" for i in range(n_b - (6 + N_A))]
    return names


def _scalar(model, flat, dims, device):
    """Target = mean over batch of max softmax severity probability."""
    n_o, n_a = dims
    X = torch.tensor(np.asarray(flat, dtype="float32"), device=device)
    if X.ndim == 1:
        X = X.unsqueeze(0)
    s_o, s_a, s_b = X[:, :n_o], X[:, n_o:n_o + n_a], X[:, n_o + n_a:]
    with torch.no_grad():
        *_, lD = model(s_o, s_a, s_b)
        return torch.softmax(lD, 1).max(1).values.cpu().numpy()


def sensitivity(model, flatX, dims, device, n_repeat=3, seed=0):
    """Perturbation importance: shuffle each feature column, measure |Δ target|."""
    rng = np.random.default_rng(seed)
    base = _scalar(model, flatX, dims, device).mean()
    D = flatX.shape[1]
    imp = np.zeros(D)
    for d in range(D):
        deltas = []
        for _ in range(n_repeat):
            Xp = flatX.copy()
            Xp[:, d] = flatX[rng.permutation(len(flatX)), d]
            deltas.append(abs(_scalar(model, Xp, dims, device).mean() - base))
        imp[d] = np.mean(deltas)
    return imp


def shap_importance(model, flatX, dims, device, nbg=20, nsample=20):
    """mean|SHAP| via KernelExplainer on the severity target; fallback=sensitivity."""
    try:
        import shap
        bg = flatX[:min(nbg, len(flatX))]
        f = lambda X: _scalar(model, X, dims, device)
        expl = shap.KernelExplainer(f, bg)
        sv = expl.shap_values(flatX[:min(nsample, len(flatX))], nsamples=100, silent=True)
        sv = np.asarray(sv)
        return np.abs(sv).mean(axis=0)
    except Exception as e:
        print(f"  SHAP unavailable ({e.__class__.__name__}); using permutation importance.")
        return sensitivity(model, flatX[:min(nsample, len(flatX))], dims, device)


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
    c_correct = {}

    for cond in CONDITIONS:
        path = os.path.join(args.results_dir, cond["ckpt"])
        if not os.path.exists(path):
            print(f"{cond['name']}: no checkpoint ({cond['ckpt']}) — skipping.")
            continue
        print(f"\n=== {cond['name']} ({path}) ===")
        ck = torch.load(path, weights_only=False)
        cfg = ck["config"]; n_D = cfg["n_D"]
        retr = _retriever(cond["strategy"], args.model, cond["kw"])
        model = HFACSCausalLSTM(**cfg).to(device)
        model.load_state_dict(ck["state_dict"]); model.eval()

        test_loader = _loader(df_test, encoders, retr, args.batch_size)
        aO, aA, aB, aC, aD, pO, pA, pB, pC, pD = evaluate(model, test_loader, device)

        row = {"condition": cond["name"], "n_test": len(aD)}
        for pre, y, p in (("O", aO, pO), ("A", aA, pA), ("B", aB, pB), ("C", aC, pC)):
            row.update(head_ml_metrics(pre, y, p))
        sev, cm = severity_metrics(aD, pD, n_D)
        row.update(sev)
        row["chain_completion_rate"] = chain_completion_rate(aO, aA, aB, aC, aD, pO, pA, pB, pC, pD)

        # generalization gap on C (capped train subsample)
        tr_sub = df_train.head(args.train_sample)
        tr_loader = _loader(tr_sub, encoders, retr, args.batch_size)
        _, _, trC, _, _, _, _, prC, _, _ = evaluate(model, tr_loader, device)
        train_f1C = f1_score(trC, prC, average="micro", zero_division=0)
        row["generalization_error_C"] = float(train_f1C - row["C_microF1"])

        c_correct[cond["name"]] = (aC == pC).all(axis=1)
        confusions[cond["name"]] = cm
        summary.append(row)

        # ROC (probabilities)
        prO, prA, prB, prC2, prD, flatX = infer_probs(model, test_loader, device)
        onehotD = np.eye(n_D)[aD] if len(aD) else np.zeros((0, n_D))
        roc_data[cond["name"]] = {
            "O": _roc(aO, prO), "A": _roc(aA, prA), "B": _roc(aB, prB),
            "C": _roc(aC, prC2), "D": _roc(onehotD, prD)}

        # sensitivity + SHAP for C1 and C4 only
        if cond["name"] in ("C1", "C4") and len(flatX):
            dims = (cfg["step_o_dim"], cfg["step_a_dim"])
            fnames[cond["name"]] = feature_names(cfg)
            sens[cond["name"]] = sensitivity(model, flatX, dims, device)
            shap_d[cond["name"]] = shap_importance(model, flatX, dims, device)

        if retr is not None:
            retr.close()

    if not summary:
        print("No checkpoints found in results/. Train conditions first (train.py --save-path).")
        return

    # McNemar C1 vs C4 on Unsafe Acts
    mcn = None
    if "C1" in c_correct and "C4" in c_correct:
        mcn = mcnemar_test(c_correct["C1"], c_correct["C4"])
        print(f"\nMcNemar C1 vs C4 on Unsafe Acts (C): "
              f"p={mcn['p_value']:.4f} b01(RAG helps)={mcn['b01']} "
              f"b10(RAG hurts)={mcn['b10']} significant={mcn['significant']}")

    # save summary CSV
    pd.DataFrame(summary).to_csv(os.path.join(args.results_dir, "eval_summary.csv"), index=False)
    if mcn:
        pd.DataFrame([mcn]).to_csv(os.path.join(args.results_dir, "mcnemar_c1_c4.csv"), index=False)
    print(f"\nWrote {os.path.join(args.results_dir, 'eval_summary.csv')}")

    # figures
    print("\nFigures:")
    EU.plot_metric_grouped(summary, "microF1")
    EU.plot_chain_and_gengap(summary)
    EU.plot_severity(summary, confusions)
    EU.plot_roc(roc_data)
    for cn in ("C1", "C4"):
        if cn in sens:
            EU.plot_sensitivity({cn: sens[cn]}, fnames[cn], name=f"eval_sensitivity_{cn}.png")
        if cn in shap_d:
            EU.plot_shap({cn: shap_d[cn]}, fnames[cn], name=f"eval_shap_{cn}.png")
    cols = ["n_test", "O_microF1", "A_microF1", "B_microF1", "C_microF1",
            "D_accuracy", "D_macroF1", "chain_completion_rate", "generalization_error_C"]
    EU.plot_summary_table(summary, cols)
    print("\nDone.")


if __name__ == "__main__":
    main()
