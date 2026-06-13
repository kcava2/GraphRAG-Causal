"""
eval_lstm.py  —  held-out test-set metrics for the trained HFACS causal LSTM
============================================================================
Loads models/lstm/hfacs_lstm.pt and evaluates the TEST split, then writes a
figure (figures/lstm_eval.png) plus a short console summary:
  - O/A/B/C (multi-label): micro/macro F1, precision, recall, exact-match, support
  - D (severity, single-class): accuracy, balanced accuracy, macro-F1, confusion

Auto-detects C1 vs C4 from the checkpoint dims and builds a retriever for C4.
Pass the SAME --input you trained on so the encoders/severity classes match.

Usage:
  python eval_lstm.py --input data/ntsb_subset.csv
  python eval_lstm.py --input data/ntsb_subset.csv --model qwen2.5:3b   # if C4
"""

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             accuracy_score, balanced_accuracy_score,
                             confusion_matrix)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "data"))
sys.path.insert(0, _HERE)
import ntsbdataloader as N                              # noqa: E402
from models.lstm.train import HFACSCausalLSTM, evaluate  # noqa: E402

FIG = os.path.join(_HERE, "figures")
os.makedirs(FIG, exist_ok=True)


def _ml_metrics(y_true, y_pred):
    kw = dict(zero_division=0)
    return {
        "microF1": f1_score(y_true, y_pred, average="micro", **kw),
        "macroF1": f1_score(y_true, y_pred, average="macro", **kw),
        "precision": precision_score(y_true, y_pred, average="micro", **kw),
        "recall": recall_score(y_true, y_pred, average="micro", **kw),
        "exact": float((y_true == y_pred).all(axis=1).mean()) if len(y_true) else 0.0,
        "support": int(y_true.sum()),
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate the trained HFACS causal LSTM")
    ap.add_argument("--input", default=N.NTSB_CLEAN,
                    help="Same NTSB CSV used for training (e.g. data/ntsb_subset.csv).")
    ap.add_argument("--checkpoint", default=os.path.join(_HERE, "models", "lstm", "hfacs_lstm.pt"))
    ap.add_argument("--model", default=None, help="Ollama model for the C4 retriever.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", default=os.path.join(FIG, "lstm_eval.png"))
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ckpt = torch.load(args.checkpoint, weights_only=False)
    cfg = ckpt["config"]
    is_c4 = cfg["step_a_dim"] > N.N_O
    cond = "C4 (RAG priors)" if is_c4 else "C1 (no RAG)"
    print(f"Checkpoint: {args.checkpoint}\nCondition: {cond}")

    retriever = None
    if is_c4:
        from rag_retriever import build_retriever
        kw = {"model": args.model} if args.model else {}
        retriever = build_retriever(strategy="hybrid", **kw)

    _, _, test_loader, enc = N.get_dataloaders(
        filepath=args.input, batch_size=args.batch_size,
        retriever=retriever, build_faiss=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HFACSCausalLSTM(**cfg).to(device)
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    aO, aA, aB, aC, aD, pO, pA, pB, pC, pD = evaluate(model, test_loader, device)
    n = len(aD)

    heads = ["O Organizational", "A Supervisory", "B Preconditions", "C Unsafe Acts"]
    mets = [_ml_metrics(*p) for p in ((aO, pO), (aA, pA), (aB, pB), (aC, pC))]

    n_D = cfg["n_D"]
    sev_labels = list(range(n_D))
    sev_acc = accuracy_score(aD, pD) if n else 0.0
    sev_bal = balanced_accuracy_score(aD, pD) if n else 0.0
    sev_mf1 = f1_score(aD, pD, average="macro", labels=sev_labels, zero_division=0) if n else 0.0
    cm = confusion_matrix(aD, pD, labels=sev_labels) if n else np.zeros((n_D, n_D), int)
    n_sev_present = len(set(aD.tolist())) if n else 0

    # ---- console ----
    print(f"\nTest records: {n}")
    for h, m in zip(heads, mets):
        print(f"  {h:16} microF1={m['microF1']:.3f} macroF1={m['macroF1']:.3f} "
              f"P={m['precision']:.3f} R={m['recall']:.3f} exact={m['exact']:.3f} "
              f"(pos={m['support']})")
    print(f"  Severity         acc={sev_acc:.3f} bal_acc={sev_bal:.3f} macroF1={sev_mf1:.3f} "
          f"({n_sev_present} class(es) present in test)")

    # ---- figure ----
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    fig.suptitle(f"HFACS Causal LSTM — test-set evaluation  ({cond}, {n} records)",
                 fontsize=15, fontweight="bold")
    short = ["O", "A", "B", "C"]

    # (0,0) grouped metric bars
    ax = axes[0, 0]
    keys = ["microF1", "macroF1", "precision", "recall"]
    colors = ["#4C72B0", "#55A868", "#DD8452", "#C44E52"]
    x = np.arange(len(short)); w = 0.2
    for i, k in enumerate(keys):
        ax.bar(x + (i - 1.5) * w, [m[k] for m in mets], w, label=k, color=colors[i])
    ax.set_xticks(x); ax.set_xticklabels(short); ax.set_ylim(0, 1)
    ax.set_ylabel("score"); ax.set_title("Multi-label metrics per causal step")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    for i, m in enumerate(mets):
        ax.text(i, m["microF1"] + 0.02, f"{m['microF1']:.2f}", ha="center", fontsize=8)

    # (0,1) support per head
    ax = axes[0, 1]
    sup = [m["support"] for m in mets]
    bars = ax.bar(short, sup, color="#5A189A")
    for b, v in zip(bars, sup):
        ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_title("Positive labels present in test (head support)")
    ax.set_ylabel("# true-positive subcategory labels")
    ax.grid(axis="y", alpha=0.3)

    # (1,0) severity confusion
    ax = axes[1, 0]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(sev_labels); ax.set_yticks(sev_labels)
    ax.set_xlabel("predicted severity class"); ax.set_ylabel("true severity class")
    ax.set_title("Severity (D) confusion matrix")
    vmax = cm.max() if cm.size else 1
    for i in range(n_D):
        for j in range(n_D):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=10,
                    color="white" if cm[i, j] > vmax * 0.6 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (1,1) summary + caveats
    ax = axes[1, 1]; ax.axis("off")
    caveats = []
    for h, m in zip(heads, mets):
        if m["support"] == 0:
            caveats.append(f"- {h.split()[0]} head: 0 positives in test (metric trivial)")
    if n_sev_present < 2:
        caveats.append("- Severity: only ONE class present in test — accuracy is trivial; "
                       "macroF1 (over all classes) is the honest number")
    txt = (f"Condition: {cond}\nTest records: {n}\n\n"
           f"Severity:  accuracy={sev_acc:.3f}\n"
           f"           balanced_acc={sev_bal:.3f}\n"
           f"           macroF1={sev_mf1:.3f}\n\n"
           "Caveats:\n" + ("\n".join(caveats) if caveats else "- none"))
    ax.text(0.02, 0.98, txt, va="top", fontsize=12, family="monospace")
    ax.set_title("Summary")

    plt.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure: {args.out}")

    if retriever is not None:
        retriever.close()


if __name__ == "__main__":
    main()
