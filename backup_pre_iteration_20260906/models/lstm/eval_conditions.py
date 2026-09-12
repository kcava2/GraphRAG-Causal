#!/usr/bin/env python
"""
eval_conditions.py — evaluate the five retrieval-strategy conditions
====================================================================
Reads results/c1.pt .. results/c5.pt (written by run_conditions.py), scores each
on the held-out TEST split, runs McNemar against C1, and writes tables + figures.

    C1 no RAG · C2 semantic · C3 structural · C4 hybrid · C5 raw RAG

Outputs
    results/conditions_metrics.csv     per-head metrics, every condition
    results/conditions_mcnemar.csv     McNemar vs C1, per head
    figures/cond_metrics_table.png     metrics as a rendered table
    figures/cond_mcnemar_table.png     statistical tests as a rendered table
    figures/cond_performance.png       grouped bars, per head
    figures/cond_kappa.png             kappa vs the majority baseline

**Read balanced accuracy and kappa, not F1.** Head B's groups are now ~72%
positive, so a constant all-ones predictor scores ~0.72 micro-F1 while carrying no
information. Every table therefore includes a MAJORITY row (predict the training
majority for every record) as the floor any condition must clear.

    python models/lstm/eval_conditions.py
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..", "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "data"))

from ntsbdataloader import (NTSBSequenceDataset, NTSBEncoders, load_and_join,  # noqa: E402
                            _split, GraphFewShotSource, PRECOND_SUBS, NTSB_CLEAN)
from models.lstm.train import make_model  # noqa: E402
from models.lstm.eval import mcnemar_test  # noqa: E402
from sklearn.metrics import (f1_score, accuracy_score, balanced_accuracy_score,  # noqa: E402
                             cohen_kappa_score)

RESULTS = os.path.join(_ROOT, "results")
FIGURES = os.path.join(_ROOT, "figures")
ORDER = ["C1", "C2", "C3", "C4", "C5"]
LABELS = {"C1": "C1 no RAG", "C2": "C2 semantic", "C3": "C3 structural",
          "C4": "C4 hybrid", "C5": "C5 raw RAG"}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(ckpt_path, df_test, df_train, encoders, batch, device):
    """-> (preds, targets) dicts keyed B/C/D, plus the condition's config."""
    ck = torch.load(ckpt_path, weights_only=False)
    cfg = ck["config"]
    model = make_model(cfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    # Rebuild the SAME exemplar source the condition trained with, or the model
    # silently receives zeros and the condition is evaluated as if it were C1.
    k = int(cfg.get("fewshot_k", 0))
    source, retr = None, None
    if k and cfg.get("strategy"):
        from rag_retriever import build_retriever
        retr = build_retriever(strategy=cfg["strategy"])
        source = GraphFewShotSource(retr, raw_mode=bool(cfg.get("raw_mode")))
        source.attach_encoders(encoders)

    ds = NTSBSequenceDataset(df_test, encoders, retriever=None,
                             fewshot_source=source, fewshot_k=k)
    loader = DataLoader(ds, batch_size=batch, shuffle=False)

    thr = ck.get("thresholds", {})
    thrB = np.asarray(thr.get("B", np.full(cfg["n_B"], 0.5)), dtype="float64")

    pB, pC, pD, tB, tC, tD = [], [], [], [], [], []
    with torch.no_grad():
        for s_ctx, s_b, yB, yC, yD, fs, fsm in loader:
            lB, lC, lD = model(s_ctx.to(device), s_b.to(device),
                               fs.to(device), fsm.to(device))
            pB.append((torch.sigmoid(lB).cpu().numpy() >= thrB).astype(int))
            pC.append(torch.softmax(lC, 1).argmax(1).cpu().numpy())
            pD.append(torch.softmax(lD, 1).argmax(1).cpu().numpy())
            tB.append(yB.numpy()); tC.append(yC.numpy()); tD.append(yD.numpy())
    if retr is not None:
        try:
            retr.close()
        except Exception:
            pass
    cat = np.concatenate
    return ({"B": cat(pB), "C": cat(pC), "D": cat(pD)},
            {"B": cat(tB).astype(int), "C": cat(tC), "D": cat(tD)}, cfg)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def head_metrics(head, pred, targ):
    """Per-head metrics. B is multi-label; C and D are single-label."""
    if head == "B":
        bal = [balanced_accuracy_score(targ[:, j], pred[:, j])
               for j in range(targ.shape[1]) if targ[:, j].sum() > 0]
        kap = [cohen_kappa_score(targ[:, j], pred[:, j])
               for j in range(targ.shape[1]) if len(set(targ[:, j])) > 1]
        return {"F1": f1_score(targ, pred, average="micro", zero_division=0),
                "balanced_acc": float(np.mean(bal)) if bal else 0.0,
                "kappa": float(np.mean(kap)) if kap else 0.0,
                "accuracy": float((targ == pred).mean()),
                "support": int(targ.sum())}
    m = targ != -100                                    # drop ignore_index rows
    t, p = targ[m], pred[m]
    if len(t) == 0:
        return {"F1": 0.0, "balanced_acc": 0.0, "kappa": 0.0, "accuracy": 0.0, "support": 0}
    return {"F1": f1_score(t, p, average="macro", zero_division=0),
            "balanced_acc": balanced_accuracy_score(t, p),
            "kappa": cohen_kappa_score(t, p) if len(set(t)) > 1 else 0.0,
            "accuracy": accuracy_score(t, p),
            "support": int((t == 1).sum())}


def majority_rows(targ_train, targ_test):
    """The floor: predict the training majority for every test record."""
    out = {}
    for head in ("B", "C", "D"):
        t_tr, t_te = targ_train[head], targ_test[head]
        if head == "B":
            maj = (t_tr.mean(0) >= 0.5).astype(int)
            pred = np.tile(maj, (len(t_te), 1))
        else:
            m = t_tr != -100
            maj = int(np.bincount(t_tr[m]).argmax()) if m.sum() else 0
            pred = np.full_like(t_te, maj)
        out[head] = head_metrics(head, pred, t_te)
    return out


def correctness(head, pred, targ):
    """Flattened per-item correctness for McNemar."""
    if head == "B":
        return (pred == targ).ravel()
    m = targ != -100
    return (pred[m] == targ[m])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _table_fig(df, title, path, note=None, highlight=None):
    h = 0.42 * len(df) + (2.0 if note else 1.4)
    fig, ax = plt.subplots(figsize=(min(2.0 + 1.35 * len(df.columns), 18), h))
    ax.axis("off")
    tbl = ax.table(cellText=df.values.astype(str), colLabels=df.columns,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.45)
    for j in range(len(df.columns)):
        tbl[0, j].set_facecolor("#40466e")
        tbl[0, j].get_text().set_color("white")
        tbl[0, j].get_text().set_fontweight("bold")
    for i in range(len(df)):
        base = "#f2f2f2" if i % 2 else "#ffffff"
        if highlight and df.iloc[i, 0] in highlight:
            base = "#ffe9c7"
        for j in range(len(df.columns)):
            tbl[i + 1, j].set_facecolor(base)
    ax.set_title(title, fontweight="bold", pad=16)
    if note:
        fig.text(0.5, 0.02, note, ha="center", fontsize=8, style="italic", wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {path}")


def plot_performance(summary):
    heads, metrics = ["B", "C", "D"], ["balanced_acc", "kappa", "F1"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    conds = [c for c in ORDER if c in summary["condition"].values]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(conds)))
    for ax, head in zip(axes, heads):
        sub = summary[summary["head"] == head].set_index("condition")
        x = np.arange(len(metrics)); w = 0.8 / max(len(conds), 1)
        for i, c in enumerate(conds):
            if c not in sub.index:
                continue
            ax.bar(x + i * w, [sub.loc[c, m] for m in metrics], w,
                   label=LABELS.get(c, c), color=colors[i])
        if "MAJORITY" in sub.index:
            for xi, m in enumerate(x):
                ax.hlines(sub.loc["MAJORITY", metrics[xi]], m - 0.05, m + 0.85,
                          color="crimson", ls="--", lw=1.4,
                          label="majority baseline" if xi == 0 else None)
        ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels(metrics)
        ax.set_title(f"Head {head}"); ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("score")
    axes[-1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Performance by condition (dashed = majority-class floor)",
                 fontweight="bold")
    fig.tight_layout()
    p = os.path.join(FIGURES, "cond_performance.png")
    fig.savefig(p, dpi=170); plt.close(fig); print(f"  saved {p}")


def plot_kappa(summary):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    conds = [c for c in ORDER if c in summary["condition"].values]
    x = np.arange(len(conds)); w = 0.26
    for i, head in enumerate(["B", "C", "D"]):
        sub = summary[summary["head"] == head].set_index("condition")
        ax.bar(x + i * w, [sub.loc[c, "kappa"] if c in sub.index else 0 for c in conds],
               w, label=f"head {head}")
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x + w); ax.set_xticklabels([LABELS.get(c, c) for c in conds], fontsize=9)
    ax.set_ylabel("Cohen's kappa"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Agreement above chance (kappa = 0 means no information)",
                 fontweight="bold")
    fig.tight_layout()
    p = os.path.join(FIGURES, "cond_kappa.png")
    fig.savefig(p, dpi=170); plt.close(fig); print(f"  saved {p}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=NTSB_CLEAN)
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args()

    os.makedirs(FIGURES, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_and_join(a.input)
    df_train, _df_val, df_test = _split(df)
    encoders = NTSBEncoders(df_train)
    print(f"Test records: {len(df_test)} | device: {device}\n")

    rows, correct, targ_ref = [], {}, None
    for name in ORDER:
        path = os.path.join(RESULTS, f"{name.lower()}.pt")
        if not os.path.exists(path):
            print(f"{name}: no checkpoint — skipping.")
            continue
        # Reject checkpoints not written by run_conditions.py. Older files (a
        # different architecture, prior-based inputs, or labels from a previous
        # extraction) would otherwise load and be scored as if they belonged to
        # this comparison.
        meta = torch.load(path, weights_only=False).get("config", {})
        if "condition" not in meta:
            print(f"{name}: {os.path.basename(path)} predates run_conditions.py "
                  f"(no 'condition' in config) — skipping. Re-train with "
                  f"`python run_conditions.py --only {name}`.")
            continue
        if meta["condition"] != name:
            print(f"{name}: checkpoint says condition={meta['condition']} — skipping.")
            continue

        print(f"{name}: evaluating…")
        pred, targ, cfg = predict(path, df_test, df_train, encoders, a.batch_size, device)
        targ_ref = targ
        correct[name] = {h: correctness(h, pred[h], targ[h]) for h in ("B", "C", "D")}
        for head in ("B", "C", "D"):
            rows.append({"condition": name, "head": head,
                         "strategy": cfg.get("strategy") or "none",
                         **head_metrics(head, pred[head], targ[head])})

    if not rows:
        print("No checkpoints found. Run: python run_conditions.py")
        return 1

    # majority floor, from the TRAIN split's majority applied to test
    tr_ds = NTSBSequenceDataset(df_train, encoders)
    tr_t = {"B": tr_ds.y_B.numpy().astype(int), "C": tr_ds.y_C.numpy(), "D": tr_ds.y_D.numpy()}
    for head, m in majority_rows(tr_t, targ_ref).items():
        rows.append({"condition": "MAJORITY", "head": head, "strategy": "-", **m})

    summary = pd.DataFrame(rows)
    csv = os.path.join(RESULTS, "conditions_metrics.csv")
    summary.to_csv(csv, index=False); print(f"\n  saved {csv}")

    # ---- McNemar vs C1 ----
    mc = []
    if "C1" in correct:
        for name in [c for c in ORDER if c in correct and c != "C1"]:
            for head in ("B", "C", "D"):
                r = mcnemar_test(correct["C1"][head], correct[name][head])
                mc.append({"condition": name, "head": head,
                           "helps (RAG right, C1 wrong)": r["b01"],
                           "hurts (C1 right, RAG wrong)": r["b10"],
                           "net": r["b01"] - r["b10"],
                           "p_value": round(r["p_value"], 5),
                           "significant": "yes" if r["significant"] else "no"})
    mcdf = pd.DataFrame(mc)
    if not mcdf.empty:
        p = os.path.join(RESULTS, "conditions_mcnemar.csv")
        mcdf.to_csv(p, index=False); print(f"  saved {p}")

    # ---- figures ----
    disp = summary.copy()
    for c in ("F1", "balanced_acc", "kappa", "accuracy"):
        disp[c] = disp[c].map(lambda v: f"{v:.3f}")
    _table_fig(disp, "Evaluation metrics by condition and head",
               os.path.join(FIGURES, "cond_metrics_table.png"),
               note="Head B is ~72% positive — read balanced_acc and kappa, not F1. "
                    "MAJORITY is the constant-prediction floor.",
               highlight={"MAJORITY"})
    if not mcdf.empty:
        _table_fig(mcdf, "McNemar tests vs C1 (no RAG)",
                   os.path.join(FIGURES, "cond_mcnemar_table.png"),
                   note="net = helps - hurts. p < 0.05 marks a significant "
                        "difference in per-item correctness against the no-RAG baseline.")
    plot_performance(summary)
    plot_kappa(summary[summary["condition"] != "MAJORITY"])

    print("\n" + "=" * 72)
    print(summary.to_string(index=False))
    if not mcdf.empty:
        print("\nMcNemar vs C1:")
        print(mcdf.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
