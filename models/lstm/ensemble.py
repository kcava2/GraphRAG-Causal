"""
ensemble.py  (Stage 6)
======================
Spec 2.3 — **ensemble augmentation: RAG predictions as an additional model.**

Treats retrieval as a standalone predictor rather than as a feature source, and
blends it with the trained causal model:

    p_final = (1 - alpha) * p_model + alpha * p_rag

`alpha` is tuned per head on the VALIDATION split and applied unchanged to test,
so the test split never influences the blend weight.

The RAG predictor is a similarity-weighted vote over the retrieved train
neighbours — the same exemplars `FewShotSource` supplies for prompt augmentation,
reused here rather than retrieved a second time. Its columns are the neighbours'
own labels:

    pB = sum_k w_k * y_B(k) / sum_k w_k        per-group probability (multi-label)
    pC = sum_k w_k * onehot(y_C(k)) / sum w_k  class distribution
    pD = sum_k w_k * onehot(y_D(k)) / sum w_k  class distribution

with w_k the (clamped) FAISS similarity of neighbour k. Leakage discipline is
inherited from `FewShotSource`: train split only, query self-excluded.

This condition is motivated directly by the C8 result — dropping the LLM factor
priors left head D unchanged, which implies a k-NN vote on narrative similarity
is doing the work. An explicit ensemble measures that instead of inferring it.

CLI:
    python models/lstm/ensemble.py --checkpoint results/c7.pt --fewshot-k 5
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from data.ntsbdataloader import (  # noqa: E402
    NTSBSequenceDataset, NTSBEncoders, FewShotSource, load_and_join, _split,
    STEP_B_BASE, N_B, N_C, NTSB_CLEAN)
from models.lstm.train import make_model  # noqa: E402
from models.lstm.eval import ml_metrics, class_metrics  # noqa: E402

# Column layout of a FewShotSource exemplar row.
_B_SLICE = slice(STEP_B_BASE, STEP_B_BASE + N_B)
_C_SLICE = slice(STEP_B_BASE + N_B, STEP_B_BASE + N_B + N_C)
_D_SLICE = slice(STEP_B_BASE + N_B + N_C, STEP_B_BASE + N_B + N_C + 2)


# ---------------------------------------------------------------------------
# The RAG predictor
# ---------------------------------------------------------------------------

def rag_probs(fewshot: np.ndarray, mask: np.ndarray):
    """Similarity-weighted neighbour vote -> (pB [n,N_B], pC [n,N_C], pD [n,2]).

    Records with no retrieved neighbour fall back to an uninformative prediction
    (0.5 for each multi-label group, uniform over classes) so they neither help
    nor hurt the blend.
    """
    n = fewshot.shape[0]
    pB = np.full((n, N_B), 0.5, dtype="float64")
    pC = np.full((n, N_C), 1.0 / N_C, dtype="float64")
    pD = np.full((n, 2), 0.5, dtype="float64")
    if fewshot.ndim != 3 or fewshot.shape[1] == 0:
        return pB, pC, pD

    sim = np.clip(fewshot[:, :, -1], 0.0, None)          # FAISS IP can go negative
    w = sim * mask                                       # [n, k]
    total = w.sum(axis=1)
    ok = total > 1e-9
    if not ok.any():
        return pB, pC, pD

    wn = np.zeros_like(w)
    wn[ok] = w[ok] / total[ok, None]
    einsum = lambda sl: np.einsum("nk,nkd->nd", wn, fewshot[:, :, sl])
    pB[ok] = einsum(_B_SLICE)[ok]
    pC[ok] = einsum(_C_SLICE)[ok]
    pD[ok] = einsum(_D_SLICE)[ok]
    return pB, pC, pD


def collect(model, loader, device):
    """-> (model probs, rag probs, targets), each a dict keyed B/C/D."""
    model.eval()
    mB, mC, mD, fsA, fsmA, tB, tC, tD = [], [], [], [], [], [], [], []
    with torch.no_grad():
        for s_ctx, s_b, yB, yC, yD, fs, fsm in loader:
            s_ctx, s_b = s_ctx.to(device), s_b.to(device)
            fs_d, fsm_d = fs.to(device), fsm.to(device)
            lB, lC, lD = model(s_ctx, s_b, fs_d, fsm_d)
            mB.append(torch.sigmoid(lB).cpu().numpy())
            mC.append(torch.softmax(lC, 1).cpu().numpy())
            mD.append(torch.softmax(lD, 1).cpu().numpy())
            fsA.append(fs.numpy())
            fsmA.append(fsm.numpy())
            tB.append(yB.numpy()); tC.append(yC.numpy()); tD.append(yD.numpy())

    cat = lambda xs: np.concatenate(xs, 0)
    fs_all, fsm_all = cat(fsA), cat(fsmA)
    rB, rC, rD = rag_probs(fs_all, fsm_all)
    return ({"B": cat(mB), "C": cat(mC), "D": cat(mD)},
            {"B": rB, "C": rC, "D": rD},
            {"B": cat(tB), "C": cat(tC), "D": cat(tD)})


# ---------------------------------------------------------------------------
# Blending
# ---------------------------------------------------------------------------

def blend(model_p, rag_p, alpha):
    return (1.0 - alpha) * model_p + alpha * rag_p


def _score(head, probs, target, thresholds):
    """Head-appropriate scalar used only for tuning alpha (higher is better)."""
    if head == "B":
        pred = (probs >= thresholds).astype(int)
        return ml_metrics("B", target, pred)["B_F1"]
    valid = target != -100
    if valid.sum() == 0:
        return 0.0
    pred = probs.argmax(1)
    n = probs.shape[1]
    return class_metrics(head, target[valid], pred[valid], n)[f"{head}_F1"]


def tune_alpha(head, model_p, rag_p, target, thresholds, grid=None):
    """Best blend weight for one head, chosen on the validation split only.

    Ties resolve to the SMALLER alpha, so the causal model is preferred when
    retrieval adds nothing measurable — the conservative reading.
    """
    grid = np.linspace(0.0, 1.0, 21) if grid is None else grid
    best, best_s = 0.0, -np.inf
    for a in grid:
        s = _score(head, blend(model_p, rag_p, a), target, thresholds)
        if s > best_s + 1e-12:
            best, best_s = float(a), s
    return best, best_s


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Ensemble a trained causal model with a retrieval-only "
                    "predictor; blend weight tuned on validation.")
    ap.add_argument("--checkpoint", required=True, help="Trained model, e.g. results/c7.pt")
    ap.add_argument("--input", default=NTSB_CLEAN)
    ap.add_argument("--fewshot-k", type=int, default=5,
                    help="Neighbours for the RAG predictor. Independent of whether "
                         "the checkpoint itself was trained with exemplars.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default=os.path.join("results", "ensemble_summary.csv"))
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.checkpoint, weights_only=False)
    cfg = ck["config"]
    model = make_model(cfg).to(device)
    model.load_state_dict(ck["state_dict"])

    thr = ck.get("thresholds")
    thr = np.asarray(thr, dtype="float64") if thr is not None else np.full(N_B, 0.5)

    df = load_and_join(a.input)
    df_train, df_val, df_test = _split(df)
    encoders = NTSBEncoders(df_train)

    # The checkpoint may or may not have been trained with exemplars; the RAG
    # predictor needs them either way, so the source is always built.
    src = FewShotSource(df_train, encoders)
    k_model = a.fewshot_k if cfg.get("fewshot_dim", 0) > 0 else 0
    k_rag = a.fewshot_k

    def loader(d):
        ds = NTSBSequenceDataset(d, encoders, retriever=None,
                                 fewshot_source=src, fewshot_k=max(k_model, k_rag))
        return DataLoader(ds, batch_size=a.batch_size, shuffle=False)

    print(f"Checkpoint: {a.checkpoint}  (arch={cfg.get('arch','lstm')}, "
          f"fewshot_dim={cfg.get('fewshot_dim', 0)})")
    val_m, val_r, val_t = collect(model, loader(df_val), device)
    test_m, test_r, test_t = collect(model, loader(df_test), device)

    rows = []
    for head in ("B", "C", "D"):
        alpha, _ = tune_alpha(head, val_m[head], val_r[head], val_t[head], thr)
        variants = {
            "model_only": test_m[head],
            "rag_only": test_r[head],
            f"ensemble(a={alpha:.2f})": blend(test_m[head], test_r[head], alpha),
        }
        for name, probs in variants.items():
            rows.append({"head": head, "variant": name, "alpha": alpha,
                         "score": _score(head, probs, test_t[head], thr)})

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out.to_csv(a.out, index=False)

    print(f"\n{'head':<6}{'variant':<24}{'test F1':>10}")
    print("-" * 40)
    for _, r in out.iterrows():
        print(f"{r['head']:<6}{r['variant']:<24}{r['score']:>10.3f}")
    print(f"\nWrote {a.out}")
    print("\nalpha is tuned on VALIDATION only. alpha near 1.0 means retrieval "
          "carries the head on its own; near 0.0 means the causal model does.")


if __name__ == "__main__":
    main()
