#!/usr/bin/env python
"""
eval_conditions.py — evaluate the five retrieval-strategy conditions
====================================================================
Reads results/seeds/c{n}_s{seed}.pt (from run_conditions.py), scores every seed of
every condition on the held-out TEST split, and reports mean +/- sd.

    C1 no RAG · C2 semantic · C3 structural · C4 hybrid · C5 raw RAG

Two statistical tests, answering different questions:

  * **Paired t-test across seeds** — does this condition beat C1 on average, given
    run-to-run variance? Seeds are the replicates. This is the headline test: a
    single run cannot distinguish these conditions, because unseeded repeats of
    C1 alone varied by ~0.04 kappa on head D.
  * **McNemar (seed 0)** — on that one run, which individual test records did the
    condition get right that C1 got wrong? A per-item view, not a per-run one.

Outputs
    results/conditions_metrics.csv      per-seed metrics (raw)
    results/conditions_summary.csv      mean +/- sd per condition and head
    results/conditions_stats.csv        paired t-tests vs C1
    results/conditions_mcnemar.csv      McNemar vs C1, seed 0
    figures/cond_metrics_table.png      summary as a rendered table
    figures/cond_stats_table.png        significance tests as a rendered table
    figures/cond_performance.png        grouped bars with sd error bars
    figures/cond_kappa.png              kappa vs the majority floor

**Read balanced accuracy and kappa, not F1.** Head B's groups are ~72% positive, so
a constant all-ones predictor scores ~0.79 micro-F1 while carrying no information.
A MAJORITY row (predict the training majority) is included as the floor.

    python models/lstm/eval_conditions.py
"""

import argparse
import glob
import os
import re
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
                            _split, GraphFewShotSource, NTSB_CLEAN)
from models.lstm.train import make_model  # noqa: E402
from models.lstm.eval import mcnemar_test  # noqa: E402
from sklearn.metrics import (f1_score, accuracy_score, balanced_accuracy_score,  # noqa: E402
                             cohen_kappa_score)

RESULTS = os.path.join(_ROOT, "results")
SEED_DIR = os.path.join(RESULTS, "seeds")
FIGURES = os.path.join(_ROOT, "figures")
ORDER = ["C1", "C2", "C3", "C4", "C5"]
LABELS = {"C1": "C1 no RAG", "C2": "C2 semantic", "C3": "C3 structural",
          "C4": "C4 hybrid", "C5": "C5 raw RAG"}
HEADS = ("B", "C", "D")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

_SOURCE_CACHE = {}


def _source_for(cfg, encoders, df_train):
    """Exemplar source matching a checkpoint's condition, cached across seeds.

    `df_train` MUST be passed. Training builds its exemplars from the KG *and* the
    in-distribution LOFO pool; omitting it here would evaluate the model on
    KG-only exemplars — a different input distribution from the one it was trained
    on, which silently destroys the predictions rather than raising anything.
    """
    strategy, raw = cfg.get("strategy"), bool(cfg.get("raw_mode"))
    if not strategy:
        return None
    key = (strategy, raw)
    if key not in _SOURCE_CACHE:
        from rag_retriever import build_retriever
        src = GraphFewShotSource(build_retriever(strategy=strategy), raw_mode=raw)
        src.attach_encoders(encoders, df_train)
        _SOURCE_CACHE[key] = src
    return _SOURCE_CACHE[key]


def predict(path, df_test, df_train, encoders, batch, device):
    ck = torch.load(path, weights_only=False)
    cfg = ck["config"]
    model = make_model(cfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    k = int(cfg.get("fewshot_k", 0))
    source = _source_for(cfg, encoders, df_train) if k else None
    ds = NTSBSequenceDataset(df_test, encoders, retriever=None,
                             fewshot_source=source, fewshot_k=k)
    loader = DataLoader(ds, batch_size=batch, shuffle=False)

    thrB = np.asarray(ck.get("thresholds", {}).get("B", np.full(cfg["n_B"], 0.5)),
                      dtype="float64")
    pB, pC, pD, tB, tC, tD = [], [], [], [], [], []
    with torch.no_grad():
        for s_ctx, s_b, yB, yC, yD, fs, fsm in loader:
            lB, lC, lD = model(s_ctx.to(device), s_b.to(device),
                               fs.to(device), fsm.to(device))
            pB.append((torch.sigmoid(lB).cpu().numpy() >= thrB).astype(int))
            pC.append(torch.softmax(lC, 1).argmax(1).cpu().numpy())
            pD.append(torch.softmax(lD, 1).argmax(1).cpu().numpy())
            tB.append(yB.numpy()); tC.append(yC.numpy()); tD.append(yD.numpy())
    cat = np.concatenate
    return ({"B": cat(pB), "C": cat(pC), "D": cat(pD)},
            {"B": cat(tB).astype(int), "C": cat(tC), "D": cat(tD)}, cfg)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def head_metrics(head, pred, targ):
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
    m = targ != -100
    t, p = targ[m], pred[m]
    if len(t) == 0:
        return dict.fromkeys(("F1", "balanced_acc", "kappa", "accuracy", "support"), 0)
    return {"F1": f1_score(t, p, average="macro", zero_division=0),
            "balanced_acc": balanced_accuracy_score(t, p),
            "kappa": cohen_kappa_score(t, p) if len(set(t)) > 1 else 0.0,
            "accuracy": accuracy_score(t, p),
            "support": int((t == 1).sum())}


def majority_rows(t_train, t_test):
    out = {}
    for head in HEADS:
        tr, te = t_train[head], t_test[head]
        if head == "B":
            pred = np.tile((tr.mean(0) >= 0.5).astype(int), (len(te), 1))
        else:
            m = tr != -100
            pred = np.full_like(te, int(np.bincount(tr[m]).argmax()) if m.sum() else 0)
        out[head] = head_metrics(head, pred, te)
    return out


def correctness(head, pred, targ):
    if head == "B":
        return (pred == targ).ravel()
    m = targ != -100
    return pred[m] == targ[m]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _table_fig(df, title, path, note=None, highlight=None):
    h = 0.42 * len(df) + (2.2 if note else 1.4)
    fig, ax = plt.subplots(figsize=(min(2.0 + 1.5 * len(df.columns), 19), h))
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
        if highlight and str(df.iloc[i, 0]) in highlight:
            base = "#ffe9c7"
        for j in range(len(df.columns)):
            tbl[i + 1, j].set_facecolor(base)
    ax.set_title(title, fontweight="bold", pad=16)
    if note:
        fig.text(0.5, 0.02, note, ha="center", fontsize=8, style="italic", wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {os.path.relpath(path, _ROOT)}")


def plot_performance(summary, floor):
    metrics = ["balanced_acc", "kappa", "F1"]
    conds = [c for c in ORDER if c in set(summary["condition"])]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(conds)))
    for ax, head in zip(axes, HEADS):
        sub = summary[summary["head"] == head].set_index("condition")
        x = np.arange(len(metrics)); w = 0.8 / max(len(conds), 1)
        for i, c in enumerate(conds):
            if c not in sub.index:
                continue
            means = [sub.loc[c, f"{m}_mean"] for m in metrics]
            errs = [sub.loc[c, f"{m}_sd"] for m in metrics]
            ax.bar(x + i * w, means, w, yerr=errs, capsize=2.5,
                   label=LABELS.get(c, c), color=colors[i],
                   error_kw=dict(lw=0.9, alpha=0.7))
        for xi, m in enumerate(metrics):
            ax.hlines(floor[head][m], xi - 0.05, xi + 0.85, color="crimson",
                      ls="--", lw=1.4, label="majority floor" if xi == 0 else None)
        ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels(metrics)
        ax.set_title(f"Head {head}"); ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("score")
    axes[-1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Performance by condition — mean over seeds, error bars = sd "
                 "(dashed = majority floor)", fontweight="bold")
    fig.tight_layout()
    p = os.path.join(FIGURES, "cond_performance.png")
    fig.savefig(p, dpi=170); plt.close(fig); print(f"  saved {os.path.relpath(p, _ROOT)}")


def plot_kappa(summary):
    conds = [c for c in ORDER if c in set(summary["condition"])]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(conds)); w = 0.26
    for i, head in enumerate(HEADS):
        sub = summary[summary["head"] == head].set_index("condition")
        ax.bar(x + i * w,
               [sub.loc[c, "kappa_mean"] if c in sub.index else 0 for c in conds], w,
               yerr=[sub.loc[c, "kappa_sd"] if c in sub.index else 0 for c in conds],
               capsize=3, label=f"head {head}")
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x + w); ax.set_xticklabels([LABELS.get(c, c) for c in conds], fontsize=9)
    ax.set_ylabel("Cohen's kappa"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Agreement above chance (kappa = 0 means no information)",
                 fontweight="bold")
    fig.tight_layout()
    p = os.path.join(FIGURES, "cond_kappa.png")
    fig.savefig(p, dpi=170); plt.close(fig); print(f"  saved {os.path.relpath(p, _ROOT)}")


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
    print(f"Test records: {len(df_test)} | device: {device}")

    rows, correct, targ_ref = [], {}, None
    for name in ORDER:
        paths = sorted(glob.glob(os.path.join(SEED_DIR, f"{name.lower()}_s*.pt")))
        if not paths:
            legacy = os.path.join(RESULTS, f"{name.lower()}.pt")
            paths = [legacy] if os.path.exists(legacy) else []
        if not paths:
            print(f"{name}: no checkpoints — skipping.")
            continue
        for path in paths:
            meta = torch.load(path, weights_only=False).get("config", {})
            if meta.get("condition") != name:
                print(f"{name}: {os.path.basename(path)} is not this condition — skipping.")
                continue
            seed = int(meta.get("seed", re.search(r"_s(\d+)", path).group(1)
                                if re.search(r"_s(\d+)", path) else 0))
            pred, targ, cfg = predict(path, df_test, df_train, encoders,
                                      a.batch_size, device)
            targ_ref = targ
            if seed == 0:
                correct[name] = {h: correctness(h, pred[h], targ[h]) for h in HEADS}
            for head in HEADS:
                rows.append({"condition": name, "seed": seed, "head": head,
                             "strategy": cfg.get("strategy") or "none",
                             **head_metrics(head, pred[head], targ[head])})
        print(f"{name}: evaluated {len(paths)} seed(s)")

    if not rows:
        print("No checkpoints found. Run: python run_conditions.py")
        return 1

    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(RESULTS, "conditions_metrics.csv"), index=False)

    # ---- summary: mean +/- sd over seeds ----
    metric_cols = ["F1", "balanced_acc", "kappa", "accuracy"]
    g = raw.groupby(["condition", "head"], as_index=False).agg(
        {**{m: ["mean", "std"] for m in metric_cols}, "support": "first", "seed": "count"})
    g.columns = ["condition", "head"] + [f"{m}_{s}" for m in metric_cols
                                         for s in ("mean", "sd")] + ["support", "n_seeds"]
    g = g.fillna(0.0)
    g.to_csv(os.path.join(RESULTS, "conditions_summary.csv"), index=False)

    tr_ds = NTSBSequenceDataset(df_train, encoders)
    floor = majority_rows({"B": tr_ds.y_B.numpy().astype(int),
                           "C": tr_ds.y_C.numpy(), "D": tr_ds.y_D.numpy()}, targ_ref)

    # ---- paired t-test across seeds vs C1 ----
    from scipy import stats as st
    tests = []
    for head in HEADS:
        base = raw[(raw["condition"] == "C1") & (raw["head"] == head)].sort_values("seed")
        for name in [c for c in ORDER if c != "C1" and c in set(raw["condition"])]:
            cur = raw[(raw["condition"] == name) & (raw["head"] == head)].sort_values("seed")
            n = min(len(base), len(cur))
            if n < 2:
                continue
            for metric in ("kappa", "balanced_acc", "F1"):
                b, c = base[metric].values[:n], cur[metric].values[:n]
                diff = float(np.mean(c - b))
                if np.allclose(c - b, 0):
                    t, pv = 0.0, 1.0
                else:
                    t, pv = st.ttest_rel(c, b)
                tests.append({"condition": name, "head": head, "metric": metric,
                              "C1_mean": round(float(np.mean(b)), 4),
                              "cond_mean": round(float(np.mean(c)), 4),
                              "delta": round(diff, 4), "n_seeds": n,
                              "p_value": round(float(pv), 4),
                              "significant": "yes" if pv < 0.05 else "no"})
    stats_df = pd.DataFrame(tests)
    if not stats_df.empty:
        stats_df.to_csv(os.path.join(RESULTS, "conditions_stats.csv"), index=False)

    # ---- McNemar on seed 0 ----
    mc = []
    if "C1" in correct:
        for name in [c for c in ORDER if c in correct and c != "C1"]:
            for head in HEADS:
                r = mcnemar_test(correct["C1"][head], correct[name][head])
                mc.append({"condition": name, "head": head,
                           "helps": r["b01"], "hurts": r["b10"],
                           "net": r["b01"] - r["b10"],
                           "p_value": round(r["p_value"], 5),
                           "significant": "yes" if r["significant"] else "no"})
    mcdf = pd.DataFrame(mc)
    if not mcdf.empty:
        mcdf.to_csv(os.path.join(RESULTS, "conditions_mcnemar.csv"), index=False)

    # ---- figures ----
    disp = g.copy()
    for m in metric_cols:
        disp[m] = disp.apply(lambda r, m=m: f"{r[f'{m}_mean']:.3f} ± {r[f'{m}_sd']:.3f}",
                             axis=1)
    floor_rows = pd.DataFrame([{"condition": "MAJORITY", "head": h,
                                **{m: f"{floor[h][m]:.3f}" for m in metric_cols},
                                "support": floor[h]["support"], "n_seeds": "-"}
                               for h in HEADS])
    disp = pd.concat([disp[["condition", "head"] + metric_cols + ["support", "n_seeds"]],
                      floor_rows], ignore_index=True)
    _table_fig(disp, "Evaluation metrics — mean ± sd over seeds",
               os.path.join(FIGURES, "cond_metrics_table.png"),
               note="Head B is ~72% positive: read balanced_acc and kappa, not F1. "
                    "MAJORITY is the constant-prediction floor.",
               highlight={"MAJORITY"})
    if not stats_df.empty:
        _table_fig(stats_df, "Paired t-tests vs C1 across seeds",
                   os.path.join(FIGURES, "cond_stats_table.png"),
                   note="Seeds are replicates. delta = condition - C1. This is the "
                        "headline test; a single run cannot separate these conditions.")
    plot_performance(g, floor)
    plot_kappa(g)

    print("\n" + "=" * 78)
    print(disp.to_string(index=False))
    if not stats_df.empty:
        print("\nPaired t-tests vs C1 (seeds as replicates):")
        print(stats_df.to_string(index=False))
    if not mcdf.empty:
        print("\nMcNemar vs C1 (seed 0):")
        print(mcdf.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
