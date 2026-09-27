#!/usr/bin/env python
"""
eval_conditions.py — evaluate the five retrieval-strategy conditions
====================================================================
Reads results/seeds/c{n}_s{seed}.pt (from run_conditions.py), scores every seed of
every condition on the held-out TEST split, and reports mean +/- sd.

    C1 no RAG · C2 semantic · C3 structural · C4 hybrid · C5 raw RAG

Heads: B preconditions (3 groups, multi-label), C unsafe acts (4 tiers,
multi-label), D severity (binary). B and C are scored per label and macro-averaged;
the per-label rows are written separately so a weak or unreliable tier (violation)
can neither hide inside the average nor drag it.

**Query views — what the TEST record is allowed to know (`--query-view`).**
Retrieval looks a record up by its narrative, and an NTSB narrative contains the
outcome and the probable-cause verdict. The view swaps the text used to look up
the TEST records only; training text, the exemplar pool, the knowledge graph and
all indexes are untouched, and so are the labels:

    L2    preliminary brief: circumstances, the kind   THE STANDARD VIEW (default).
          of event and its sequence; no injury or       What an analyst holds when a
          damage wording, no cause                      preliminary report is out.
                                                        Scores B and C. Severity is
                                                        already known at this stage,
                                                        so D is not scored unless
                                                        --include-severity is given.
    L1    pre-departure brief                           lower bound: nothing about
                                                        the event is known yet
    full  the complete narrative incl. probable cause   upper bound: retrospective
                                                        classification, not prediction
    L1b   circumstances incl. phase of flight           optional diagnostic, not part
                                                        of the standard run

Briefs come from data/build_query_views.py and are gated by data/leakage_audit.py.
No retraining is involved — the same checkpoints are scored under each view. C1 and
C3 never read text, so their numbers must be IDENTICAL across views; if they move,
the swap has touched something it should not have.

Statistics, answering different questions:

  * **Paired t-test across seeds** — does this condition beat C1 given run-to-run
    TRAINING noise? Seeds are the replicates.
  * **Bootstrap interval over test records** — how much would the number move with a
    different draw of test events? Seeds cannot answer that: they vary the model,
    never the 202 events. For the rare tiers this is the interval that matters.
  * **McNemar (seed 0)** — which individual records a condition gets right that C1
    got wrong.

Outputs. The suffix always names the view, so a file can never be mistaken for
another view's: "_L2" (standard), "_L1", "_L1b"; the full-narrative run keeps the
historical unsuffixed names.
    results/conditions_metrics{sfx}.csv    per-seed, per-head metrics
    results/conditions_tiers{sfx}.csv      per-seed, per-LABEL metrics (B groups, C tiers)
    results/conditions_summary{sfx}.csv    mean +/- sd per condition and head
    results/conditions_ci{sfx}.csv         bootstrap 95% intervals (seed-mean statistic)
    results/conditions_stats{sfx}.csv      paired t-tests vs C1
    results/conditions_mcnemar{sfx}.csv    McNemar vs C1, seed 0
    results/conditions_gate{sfx}.csv       exemplar-vote AUC on the test queries
    figures/cond_*{sfx}.png

**Read balanced accuracy, kappa and AUC, not F1.** B's groups are ~72% positive, so
an all-ones predictor scores ~0.79 micro-F1 carrying no information. AUC is
threshold-free, which makes it the cleanest way to compare retrieval set-ups.

    python models/lstm/eval_conditions.py                      # L2, the standard view
    python models/lstm/eval_conditions.py --query-view L1      # lower bound
    python models/lstm/eval_conditions.py --query-view full    # upper bound
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
                            _split, GraphFewShotSource, NTSB_CLEAN, PRECOND_SUBS,
                            UNSAFE_SUBS, UNSAFE_VIOLATION_TIER, retrieval_gate)
from models.lstm.train import make_model  # noqa: E402
from models.lstm.eval import mcnemar_test  # noqa: E402
from sklearn.metrics import (f1_score, accuracy_score, balanced_accuracy_score,  # noqa: E402
                             cohen_kappa_score, roc_auc_score, average_precision_score)

RESULTS = os.path.join(_ROOT, "results")
SEED_DIR = os.path.join(RESULTS, "seeds")
FIGURES = os.path.join(_ROOT, "figures")
QUERY_VIEWS = os.path.join(_ROOT, "data", "test_query_views.csv")
ORDER = ["C1", "C2", "C3", "C4", "C5"]
LABELS = {"C1": "C1 no RAG", "C2": "C2 semantic", "C3": "C3 structural",
          "C4": "C4 hybrid", "C5": "C5 raw RAG"}
HEADS = ("B", "C", "D")
HEAD_LABELS = {"B": [g.replace("precond_", "") for g in PRECOND_SUBS],
               "C": [t.replace("unsafe_", "") for t in UNSAFE_SUBS]}
_VIOL = UNSAFE_SUBS.index(UNSAFE_VIOLATION_TIER)
# "C-noviol": head C macro-averaged over the three ERROR tiers only. Reported beside
# C because the violation tier has ~15 test positives and the least reliable label.
EXTRA_HEADS = ("C-noviol",)
VIEW_COL = {"L1": "l1_text", "L1b": "l1b_text", "L2": "l2_text"}


# ---------------------------------------------------------------------------
# Query views
# ---------------------------------------------------------------------------

def apply_query_view(df_test: pd.DataFrame, view: str, path: str) -> pd.DataFrame:
    """Swap the TEST records' retrieval text for their reduced-information brief.

    A record without a brief raises. Falling back to the full narrative would put
    the outcome and the probable cause back into a run that claims not to have them.
    """
    if view == "full":
        return df_test
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found. Run: python data/build_query_views.py")
    v = pd.read_csv(path, dtype=str).fillna("")
    text = dict(zip(v["ev_id"].astype(str), v[VIEW_COL[view]]))
    ids = df_test["ev_id"].astype(str)
    missing = [e for e in ids if not text.get(e, "").strip()]
    if missing:
        raise SystemExit(f"{len(missing)} test records have no {view} brief "
                         f"(first: {missing[:3]}). Re-run data/build_query_views.py.")
    out = df_test.copy()
    out["combined_text"] = [text[e] for e in ids]
    print(f"Query view {view}: retrieval text replaced for {len(out)} TEST records "
          f"(mean {out['combined_text'].str.len().mean():.0f} chars, was "
          f"{df_test['combined_text'].astype(str).str.len().mean():.0f}).")
    return out


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

_SOURCE_CACHE = {}
_DATASET_CACHE = {}


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
    k, kgl = int(cfg.get("fewshot_k", 0)), bool(cfg.get("kg_factor_labels"))
    key = (strategy, raw, k, kgl)
    if key not in _SOURCE_CACHE:
        from rag_retriever import build_retriever
        src = GraphFewShotSource(build_retriever(strategy=strategy, k=k),
                                 raw_mode=raw, kg_factor_labels=kgl)
        src.attach_encoders(encoders, df_train)
        _SOURCE_CACHE[key] = src
    return _SOURCE_CACHE[key]


def _dataset_for(cfg, df_test, df_train, encoders):
    """Test dataset for a condition — retrieval does not depend on the seed, so it
    is built once per condition rather than once per checkpoint."""
    k = int(cfg.get("fewshot_k", 0))
    key = (cfg.get("strategy"), bool(cfg.get("raw_mode")), k,
           bool(cfg.get("kg_factor_labels")))
    if key not in _DATASET_CACHE:
        source = _source_for(cfg, encoders, df_train) if k else None
        _DATASET_CACHE[key] = NTSBSequenceDataset(df_test, encoders, retriever=None,
                                                  fewshot_source=source, fewshot_k=k)
    return _DATASET_CACHE[key]


def predict(path, df_test, df_train, encoders, batch, device):
    """-> (pred, prob, targ, cfg, dataset). `prob` is P(positive) per label."""
    ck = torch.load(path, weights_only=False)
    cfg = ck["config"]
    if not cfg.get("c_multilabel"):
        raise SystemExit(f"{os.path.basename(path)} was trained with the binary head C. "
                         f"Retrain with: python run_conditions.py")
    model = make_model(cfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    ds = _dataset_for(cfg, df_test, df_train, encoders)
    loader = DataLoader(ds, batch_size=batch, shuffle=False)
    thr = ck.get("thresholds", {})
    thrB = np.asarray(thr.get("B", np.full(cfg["n_B"], 0.5)), dtype="float64")
    thrC = np.asarray(thr.get("C", np.full(cfg["n_C"], 0.5)), dtype="float64")
    qB, qC, qD, tB, tC, tD = [], [], [], [], [], []
    with torch.no_grad():
        for s_ctx, s_b, yB, yC, yD, fs, fsm in loader:
            lB, lC, lD = model(s_ctx.to(device), s_b.to(device),
                               fs.to(device), fsm.to(device))
            qB.append(torch.sigmoid(lB).cpu().numpy())
            qC.append(torch.sigmoid(lC).cpu().numpy())
            qD.append(torch.softmax(lD, 1)[:, 1].cpu().numpy())
            tB.append(yB.numpy()); tC.append(yC.numpy()); tD.append(yD.numpy())
    cat = np.concatenate
    prob = {"B": cat(qB), "C": cat(qC), "D": cat(qD)}
    pred = {"B": (prob["B"] >= thrB).astype(int), "C": (prob["C"] >= thrC).astype(int),
            "D": (prob["D"] >= 0.5).astype(int)}
    targ = {"B": cat(tB).astype(int), "C": cat(tC).astype(int), "D": cat(tD)}
    return pred, prob, targ, cfg, ds


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _label_metrics(t, p, q):
    """One binary label: thresholded + threshold-free metrics."""
    both = 0 < t.sum() < len(t)
    return {"F1": f1_score(t, p, zero_division=0),
            "balanced_acc": balanced_accuracy_score(t, p) if both else np.nan,
            "kappa": cohen_kappa_score(t, p) if both else np.nan,
            "AUC": roc_auc_score(t, q) if both else np.nan,
            "AP": average_precision_score(t, q) if both else np.nan,
            "accuracy": accuracy_score(t, p), "support": int(t.sum())}


def label_rows(head, pred, prob, targ):
    """Per-label metrics for a multi-label head -> list of dicts."""
    return [{"label": name, **_label_metrics(targ[:, j], pred[:, j], prob[:, j])}
            for j, name in enumerate(HEAD_LABELS[head])]


def head_metrics(head, pred, prob, targ, cols=None):
    """Head-level metrics. Multi-label heads: macro over labels (F1 stays micro so
    it remains comparable with earlier tables). `cols` restricts the labels."""
    if head in ("B", "C"):
        cols = list(range(targ.shape[1])) if cols is None else cols
        per = [_label_metrics(targ[:, j], pred[:, j], prob[:, j]) for j in cols]
        mean = lambda k: float(np.nanmean([m[k] for m in per])) if per else 0.0
        return {"F1": f1_score(targ[:, cols], pred[:, cols], average="micro",
                               zero_division=0),
                "balanced_acc": mean("balanced_acc"), "kappa": mean("kappa"),
                "AUC": mean("AUC"), "AP": mean("AP"),
                "accuracy": float((targ[:, cols] == pred[:, cols]).mean()),
                "support": int(targ[:, cols].sum())}
    m = targ != -100
    t, p, q = targ[m], pred[m], prob[m]
    if len(t) == 0:
        return dict.fromkeys(("F1", "balanced_acc", "kappa", "AUC", "AP",
                              "accuracy", "support"), 0)
    out = _label_metrics(t, p, q)
    out["F1"] = f1_score(t, p, average="macro", zero_division=0)
    return out


def _no_violation_cols():
    return [j for j in range(len(UNSAFE_SUBS)) if j != _VIOL]


def all_head_metrics(pred, prob, targ, heads):
    out = {h: head_metrics(h, pred[h], prob[h], targ[h]) for h in heads}
    out["C-noviol"] = head_metrics("C", pred["C"], prob["C"], targ["C"],
                                   cols=_no_violation_cols())
    return out


def majority_rows(t_train, t_test, heads):
    pred, prob = {}, {}
    for head in HEADS:
        tr, te = t_train[head], t_test[head]
        if head in ("B", "C"):
            rate = tr.mean(0)
            pred[head] = np.tile((rate >= 0.5).astype(int), (len(te), 1))
            prob[head] = np.tile(rate, (len(te), 1)).astype("float64")
        else:
            m = tr != -100
            maj = int(np.bincount(tr[m]).argmax()) if m.sum() else 0
            pred[head] = np.full_like(te, maj)
            prob[head] = np.full(len(te), float(tr[m].mean()) if m.sum() else 0.5)
    out = {}
    for h in list(heads) + list(EXTRA_HEADS):
        base = "C" if h == "C-noviol" else h
        cols = _no_violation_cols() if h == "C-noviol" else None
        r = head_metrics(base, pred[base], prob[base], t_test[base], cols=cols)
        r["AUC"] = 0.5                       # a constant score ranks nothing
        out[h] = r
    return out


def correctness(head, pred, targ):
    if head in ("B", "C"):
        return (pred == targ).ravel()
    m = targ != -100
    return pred[m] == targ[m]


# ---- fast metrics for the bootstrap (sklearn is too slow x thousands) ----

def _auc_fast(t, q):
    pos = t == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(q, kind="mergesort")
    ranks = np.empty(len(q)); ranks[order] = np.arange(1, len(q) + 1)
    sq = q[order]                                   # average ranks over ties
    i = 0
    while i < len(sq):
        j = i
        while j + 1 < len(sq) and sq[j + 1] == sq[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return (ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def _balacc_kappa_fast(t, p):
    tp = int(((t == 1) & (p == 1)).sum()); tn = int(((t == 0) & (p == 0)).sum())
    fp = int(((t == 0) & (p == 1)).sum()); fn = int(((t == 1) & (p == 0)).sum())
    if tp + fn == 0 or tn + fp == 0:
        return np.nan, np.nan
    bal = 0.5 * (tp / (tp + fn) + tn / (tn + fp))
    n = tp + tn + fp + fn
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (n * n)
    return bal, ((po - pe) / (1 - pe) if pe < 1 else 0.0)


def _head_stat_fast(head, pred, prob, targ, idx, cols=None):
    """(bal-acc, kappa, AUC) for one head on the resampled records `idx`."""
    if head in ("B", "C"):
        cols = list(range(targ.shape[1])) if cols is None else cols
        vals = []
        for j in cols:
            t, p, q = targ[idx, j], pred[idx, j], prob[idx, j]
            vals.append((*_balacc_kappa_fast(t, p), _auc_fast(t, q)))
        return tuple(np.nanmean(np.array(vals, dtype="float64"), 0))
    t, p, q = targ[idx], pred[idx], prob[idx]
    m = t != -100
    return (*_balacc_kappa_fast(t[m], p[m]), _auc_fast(t[m], q[m]))


def bootstrap_ci(store, heads, n_boot, seed=0):
    """95% intervals over TEST RECORDS for the seed-mean statistic.

    Each replicate resamples the test records once and averages the metric over
    the condition's seeds on that resample — so the interval reflects which events
    happened to be in the test split, which is the noise seeds cannot show.
    """
    rng = np.random.default_rng(seed)
    n = len(next(iter(store.values()))[0][2]["D"])
    draws = [rng.integers(0, n, n) for _ in range(n_boot)]
    rows = []
    for name, runs in store.items():
        for h in list(heads) + list(EXTRA_HEADS):
            base = "C" if h == "C-noviol" else h
            cols = _no_violation_cols() if h == "C-noviol" else None
            reps = np.array([[_head_stat_fast(base, pr[base], pb[base], tg[base], idx, cols)
                              for (pr, pb, tg) in runs] for idx in draws])  # [boot, seed, 3]
            m = np.nanmean(reps, 1)
            for k, metric in enumerate(("balanced_acc", "kappa", "AUC")):
                lo, hi = np.nanpercentile(m[:, k], [2.5, 97.5])
                rows.append({"condition": name, "head": h, "metric": metric,
                             "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4)})
        # the violation tier on its own: the label everyone will ask about
        reps = np.array([[_head_stat_fast("C", pr["C"], pb["C"], tg["C"], idx, [_VIOL])
                          for (pr, pb, tg) in runs] for idx in draws])
        m = np.nanmean(reps, 1)
        for k, metric in enumerate(("balanced_acc", "kappa", "AUC")):
            lo, hi = np.nanpercentile(m[:, k], [2.5, 97.5])
            rows.append({"condition": name, "head": "C:violation", "metric": metric,
                         "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4)})
    return pd.DataFrame(rows)


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


def plot_performance(summary, floor, heads, sfx, view):
    metrics = ["balanced_acc", "kappa", "AUC"]
    conds = [c for c in ORDER if c in set(summary["condition"])]
    fig, axes = plt.subplots(1, len(heads), figsize=(5.4 * len(heads), 4.8), squeeze=False)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(conds)))
    for ax, head in zip(axes[0], heads):
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
    axes[0][0].set_ylabel("score")
    axes[0][-1].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Performance by condition [query view: {view}] — mean over seeds, "
                 "error bars = sd (dashed = majority floor)", fontweight="bold")
    fig.tight_layout()
    p = os.path.join(FIGURES, f"cond_performance{sfx}.png")
    fig.savefig(p, dpi=170); plt.close(fig); print(f"  saved {os.path.relpath(p, _ROOT)}")


def plot_kappa(summary, heads, sfx, view):
    conds = [c for c in ORDER if c in set(summary["condition"])]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(conds)); w = 0.8 / len(heads)
    for i, head in enumerate(heads):
        sub = summary[summary["head"] == head].set_index("condition")
        ax.bar(x + i * w,
               [sub.loc[c, "kappa_mean"] if c in sub.index else 0 for c in conds], w,
               yerr=[sub.loc[c, "kappa_sd"] if c in sub.index else 0 for c in conds],
               capsize=3, label=f"head {head}")
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x + w * (len(heads) - 1) / 2)
    ax.set_xticklabels([LABELS.get(c, c) for c in conds], fontsize=9)
    ax.set_ylabel("Cohen's kappa"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"Agreement above chance [query view: {view}] "
                 "(kappa = 0 means no information)", fontweight="bold")
    fig.tight_layout()
    p = os.path.join(FIGURES, f"cond_kappa{sfx}.png")
    fig.savefig(p, dpi=170); plt.close(fig); print(f"  saved {os.path.relpath(p, _ROOT)}")


def plot_tiers(tiers, sfx, view):
    """Head C by tier: AUC per condition — where the head's signal actually sits."""
    sub = tiers[tiers["head"] == "C"]
    if sub.empty:
        return
    g = sub.groupby(["condition", "label"])["AUC"].agg(["mean", "std"]).reset_index()
    conds = [c for c in ORDER if c in set(g["condition"])]
    labels = HEAD_LABELS["C"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(labels)); w = 0.8 / max(len(conds), 1)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(conds)))
    for i, c in enumerate(conds):
        gi = g[g["condition"] == c].set_index("label")
        ax.bar(x + i * w, [gi.loc[l, "mean"] if l in gi.index else 0 for l in labels], w,
               yerr=[gi.loc[l, "std"] if l in gi.index else 0 for l in labels],
               capsize=2.5, color=colors[i], label=LABELS.get(c, c))
    ax.axhline(0.5, color="crimson", ls="--", lw=1.4, label="chance")
    sup = sub.groupby("label")["support"].first()
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels([f"{l}\n(n+={int(sup.get(l, 0))})" for l in labels])
    ax.set_ylim(0.3, 1.0); ax.set_ylabel("AUC"); ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    ax.set_title(f"Head C by unsafe-act tier [query view: {view}]", fontweight="bold")
    fig.tight_layout()
    p = os.path.join(FIGURES, f"cond_headC_tiers{sfx}.png")
    fig.savefig(p, dpi=170); plt.close(fig); print(f"  saved {os.path.relpath(p, _ROOT)}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=NTSB_CLEAN)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--query-view", choices=["L2", "L1", "full", "L1b"], default="L2",
                    help="Text used to look up the TEST records. L2 (default, the "
                         "standard view) = preliminary brief: circumstances plus the "
                         "kind of event, no outcome and no cause. L1 = pre-departure "
                         "brief (lower bound). full = complete narrative incl. probable "
                         "cause (upper bound, retrospective). L1b = circumstances incl. "
                         "phase of flight (optional diagnostic).")
    ap.add_argument("--include-severity", action="store_true",
                    help="Also score head D under L2. Off by default: injuries and "
                         "damage are already known when a preliminary report exists, "
                         "and the L2 text alone predicts severity at AUC ~0.95, so a D "
                         "score there measures reading the event type, not prediction.")
    ap.add_argument("--views-file", default=QUERY_VIEWS)
    ap.add_argument("--bootstrap", type=int, default=1000,
                    help="Bootstrap replicates over test records (0 disables).")
    a = ap.parse_args()

    view = a.query_view
    sfx = "" if view == "full" else f"_{view}"
    # Severity is known by the time a preliminary report exists, so under L2 head D
    # is scored only on request, and then flagged wherever it is shown.
    d_flagged = view == "L2" and a.include_severity
    heads = ("B", "C") if (view == "L2" and not a.include_severity) else HEADS
    if d_flagged:
        print("NOTE: scoring head D under L2 on request. The outcome is already known "
              "at this stage; read D here as event-type matching, not prediction.")
    os.makedirs(FIGURES, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_and_join(a.input)
    df_train, _df_val, df_test = _split(df)
    encoders = NTSBEncoders(df_train)
    df_test = apply_query_view(df_test, view, a.views_file)
    print(f"Test records: {len(df_test)} | device: {device} | query view: {view}")

    rows, tier_rows, gate_rows, correct, store, targ_ref = [], [], [], {}, {}, None
    for name in ORDER:
        paths = sorted(glob.glob(os.path.join(SEED_DIR, f"{name.lower()}_s*.pt")))
        if not paths:
            print(f"{name}: no checkpoints in results/seeds — skipping.")
            continue
        for path in paths:
            meta = torch.load(path, weights_only=False).get("config", {})
            if meta.get("condition") != name:
                print(f"{name}: {os.path.basename(path)} is not this condition — skipping.")
                continue
            mt = re.search(r"_s(\d+)", path)
            seed = int(meta.get("seed", mt.group(1) if mt else 0))
            pred, prob, targ, cfg, ds = predict(path, df_test, df_train, encoders,
                                                a.batch_size, device)
            targ_ref = targ
            store.setdefault(name, []).append((pred, prob, targ))
            if seed == 0:
                correct[name] = {h: correctness(h, pred[h], targ[h]) for h in heads}
                if int(cfg.get("fewshot_k", 0)):
                    g = retrieval_gate(ds, f"test/{name}/{view}")
                    gate_rows.append({"condition": name, "query_view": view, **g})
            for head, met in all_head_metrics(pred, prob, targ, heads).items():
                if head == "C-noviol" and "C" not in heads:
                    continue
                rows.append({"condition": name, "seed": seed, "head": head,
                             "strategy": cfg.get("strategy") or "none",
                             "fewshot_k": int(cfg.get("fewshot_k", 0)), **met})
            for head in ("B", "C"):
                for r in label_rows(head, pred[head], prob[head], targ[head]):
                    tier_rows.append({"condition": name, "seed": seed, "head": head, **r})
        print(f"{name}: evaluated {len(store.get(name, []))} seed(s)")

    if not rows:
        print("No checkpoints found. Run: python run_conditions.py")
        return 1

    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(RESULTS, f"conditions_metrics{sfx}.csv"), index=False)
    tiers = pd.DataFrame(tier_rows)
    tiers.to_csv(os.path.join(RESULTS, f"conditions_tiers{sfx}.csv"), index=False)
    if gate_rows:
        pd.DataFrame(gate_rows).round(4).to_csv(
            os.path.join(RESULTS, f"conditions_gate{sfx}.csv"), index=False)

    # ---- summary: mean +/- sd over seeds ----
    metric_cols = ["balanced_acc", "kappa", "AUC", "AP", "F1", "accuracy"]
    g = raw.groupby(["condition", "head"], as_index=False).agg(
        {**{m: ["mean", "std"] for m in metric_cols}, "support": "first", "seed": "count"})
    g.columns = ["condition", "head"] + [f"{m}_{s}" for m in metric_cols
                                         for s in ("mean", "sd")] + ["support", "n_seeds"]
    g = g.fillna(0.0)
    g.to_csv(os.path.join(RESULTS, f"conditions_summary{sfx}.csv"), index=False)

    tr_ds = NTSBSequenceDataset(df_train, encoders)
    floor = majority_rows({"B": tr_ds.y_B.numpy().astype(int),
                           "C": tr_ds.y_C.numpy().astype(int),
                           "D": tr_ds.y_D.numpy()}, targ_ref, heads)

    # ---- bootstrap intervals over test records ----
    ci = pd.DataFrame()
    if a.bootstrap > 0:
        print(f"Bootstrapping {a.bootstrap} resamples of the test records ...")
        ci = bootstrap_ci(store, heads, a.bootstrap)
        ci.to_csv(os.path.join(RESULTS, f"conditions_ci{sfx}.csv"), index=False)

    # ---- paired t-test across seeds vs C1 ----
    from scipy import stats as st
    tests = []
    for head in [h for h in list(heads) + list(EXTRA_HEADS) if h in set(raw["head"])]:
        base = raw[(raw["condition"] == "C1") & (raw["head"] == head)].sort_values("seed")
        for name in [c for c in ORDER if c != "C1" and c in set(raw["condition"])]:
            cur = raw[(raw["condition"] == name) & (raw["head"] == head)].sort_values("seed")
            n = min(len(base), len(cur))
            if n < 2:
                continue
            for metric in ("kappa", "balanced_acc", "AUC"):
                b, c = base[metric].values[:n], cur[metric].values[:n]
                diff = float(np.mean(c - b))
                if np.allclose(c - b, 0):
                    pv = 1.0
                else:
                    _t, pv = st.ttest_rel(c, b)
                tests.append({"condition": name, "head": head, "metric": metric,
                              "C1_mean": round(float(np.mean(b)), 4),
                              "cond_mean": round(float(np.mean(c)), 4),
                              "delta": round(diff, 4), "n_seeds": n,
                              "p_value": round(float(pv), 4),
                              "significant": "yes" if pv < 0.05 else "no"})
    stats_df = pd.DataFrame(tests)
    if not stats_df.empty:
        stats_df.to_csv(os.path.join(RESULTS, f"conditions_stats{sfx}.csv"), index=False)

    # ---- McNemar on seed 0 ----
    mc = []
    if "C1" in correct:
        for name in [c for c in ORDER if c in correct and c != "C1"]:
            for head in heads:
                r = mcnemar_test(correct["C1"][head], correct[name][head])
                mc.append({"condition": name, "head": head,
                           "helps": r["b01"], "hurts": r["b10"],
                           "net": r["b01"] - r["b10"],
                           "p_value": round(r["p_value"], 5),
                           "significant": "yes" if r["significant"] else "no"})
    mcdf = pd.DataFrame(mc)
    if not mcdf.empty:
        mcdf.to_csv(os.path.join(RESULTS, f"conditions_mcnemar{sfx}.csv"), index=False)

    # ---- figures ----
    shown = ["balanced_acc", "kappa", "AUC", "F1"]
    disp = g.copy()
    for m in shown:
        disp[m] = disp.apply(lambda r, m=m: f"{r[f'{m}_mean']:.3f} ± {r[f'{m}_sd']:.3f}",
                             axis=1)
    floor_rows = pd.DataFrame([{"condition": "MAJORITY", "head": h,
                                **{m: f"{floor[h][m]:.3f}" for m in shown},
                                "support": floor[h]["support"], "n_seeds": "-"}
                               for h in floor if h in set(raw["head"])])
    disp = pd.concat([disp[["condition", "head"] + shown + ["support", "n_seeds"]],
                      floor_rows], ignore_index=True)
    _table_fig(disp, f"Evaluation metrics [query view: {view}] — mean ± sd over seeds",
               os.path.join(FIGURES, f"cond_metrics_table{sfx}.png"),
               note="Head B is ~72% positive: read balanced_acc, kappa and AUC, not F1. "
                    "C-noviol = head C without the violation tier. "
                    "MAJORITY is the constant-prediction floor."
                    + (" Head D is shown on request only: severity is already known "
                       "at L2, so it is not a prediction." if d_flagged else ""),
               highlight={"MAJORITY"})
    if not stats_df.empty:
        _table_fig(stats_df, f"Paired t-tests vs C1 across seeds [query view: {view}]",
                   os.path.join(FIGURES, f"cond_stats_table{sfx}.png"),
                   note="Seeds are replicates. delta = condition - C1. Training noise only; "
                        "see conditions_ci for test-sampling intervals.")
    plot_performance(g, floor, heads, sfx, view)
    plot_kappa(g, heads, sfx, view)
    plot_tiers(tiers, sfx, view)

    pd.set_option("display.width", 220)
    print("\n" + "=" * 78)
    print(disp.to_string(index=False))
    tier_mean = (tiers.groupby(["head", "label", "condition"])[["balanced_acc", "kappa", "AUC"]]
                 .mean().round(3).unstack("condition"))
    print("\nPer-label means over seeds:")
    print(tier_mean.to_string())
    if not ci.empty:
        print("\nBootstrap 95% intervals over test records (seed-mean statistic), AUC:")
        print(ci[ci.metric == "AUC"].pivot(index="head", columns="condition",
                                           values=["ci_low", "ci_high"]).round(3).to_string())
    if not stats_df.empty:
        print("\nPaired t-tests vs C1 (seeds as replicates):")
        print(stats_df.to_string(index=False))
    if not mcdf.empty:
        print("\nMcNemar vs C1 (seed 0):")
        print(mcdf.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
