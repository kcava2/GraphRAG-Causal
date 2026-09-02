"""
eval_utils.py  —  shared plotting utilities for Stage-6 evaluation
==================================================================
Used by models/lstm/eval.py and models/causal_discovery.py. All functions write
a PNG to figures/ and return the path. Static matplotlib only (no new deps).

The evaluation operates on the 3-head causal LSTM (org/supervisory influences are
a structured context input, not predicted):
  B/C are multi-label HFACS heads (sigmoid); D is single-class severity.
Single-label metrics (balanced-acc, kappa, confusion) apply to D only.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(_HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)

HEADS = ["B", "C"]                          # multi-label heads (D handled apart)
STEPS = ["B", "C", "D"]                     # all predicted steps
HEAD_LABEL = {"B": "Preconditions", "C": "Unsafe Acts", "D": "Severity"}
# stable colour per condition
COND_COLORS = ["#4C72B0", "#55A868", "#DD8452", "#C44E52", "#8172B3",
               "#937860", "#DA8BC3", "#8C8C8C"]


def _utf8():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {os.path.normpath(path)}")
    return path


def _cond_color(i):
    return COND_COLORS[i % len(COND_COLORS)]


# ---------------------------------------------------------------------------
# Condition comparison: per-head F1
# ---------------------------------------------------------------------------

def plot_metric_grouped(summary, metric="microF1", name="eval_f1_by_condition.png"):
    """summary: list of dicts with keys 'condition' and f'{head}_{metric}'."""
    conds = [s["condition"] for s in summary]
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(HEADS)); w = 0.8 / max(len(conds), 1)
    for i, s in enumerate(summary):
        vals = [s.get(f"{h}_{metric}", 0.0) for h in HEADS]
        ax.bar(x + (i - (len(conds) - 1) / 2) * w, vals, w,
               label=s["condition"], color=_cond_color(i))
    ax.set_xticks(x)
    ax.set_xticklabels([HEAD_LABEL[h] for h in HEADS])
    ax.set_ylabel(metric); ax.set_ylim(0, 1)
    ax.set_title(f"Per-step {metric} by condition (multi-label heads)",
                 fontsize=14, fontweight="bold")
    ax.legend(ncol=min(len(conds), 4), fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, name)


def plot_step_metrics(summary, name="eval_steps.png"):
    """Per-step macro-F1, balanced accuracy, and generalization error, overlaid.

    Reads '{S}_macroF1', '{S}_balacc', '{S}_generror' for S in B/C/D. Macro-F1
    averages F1 over tiers (all equal); balanced accuracy = macro per-tier
    (sensitivity+specificity)/2, so a collapsed/all-zero head scores ~0.5 (chance),
    not a misleadingly high Hamming accuracy.
    """
    conds = [s["condition"] for s in summary]
    panels = [("_F1", "F1 (micro B/C · macro D)", (0, 1)),
              ("_accuracy", "Accuracy (label-wise B/C · class D)", (0, 1)),
              ("_generror", "Generalization error (train − test F1)", None)]
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    x = np.arange(len(STEPS)); w = 0.8 / max(len(conds), 1)
    for ax, (suffix, title, ylim) in zip(axes, panels):
        for i, s in enumerate(summary):
            vals = [s.get(f"{st}{suffix}", 0.0) for st in STEPS]
            ax.bar(x + (i - (len(conds) - 1) / 2) * w, vals, w,
                   label=s["condition"], color=_cond_color(i))
        ax.set_xticks(x)
        ax.set_xticklabels([HEAD_LABEL[st] for st in STEPS], rotation=20, ha="right")
        ax.set_title(title)
        if ylim:
            ax.set_ylim(*ylim)
        else:
            ax.axhline(0, color="black", lw=0.8)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(ncol=min(len(conds), 4), fontsize=8)
    fig.suptitle("Per-step performance by condition", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return _save(fig, name)


def plot_chain(summary, name="eval_chain_completion.png"):
    """Chain-completion rate (all 4 heads exact) per condition."""
    conds = [s["condition"] for s in summary]
    chain = [s.get("chain_completion_rate", 0.0) for s in summary]
    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(conds)), 6))
    b = ax.bar(conds, chain, color=[_cond_color(i) for i in range(len(conds))])
    for bar, v in zip(b, chain):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_title("Chain-completion rate (A·B·C·D all exact)", fontsize=13, fontweight="bold")
    ax.set_ylabel("rate"); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return _save(fig, name)


def plot_severity(summary, confusions, name="eval_severity.png"):
    """confusions: dict condition -> (cm ndarray, labels list)."""
    conds = [s["condition"] for s in summary]
    ncm = len(confusions)
    fig = plt.figure(figsize=(18, 5 + 3 * ((ncm + 2) // 3)))
    gs = fig.add_gridspec(1 + (ncm + 2) // 3, 3)

    ax = fig.add_subplot(gs[0, :])
    x = np.arange(len(conds)); w = 0.25
    for j, met in enumerate(["D_accuracy", "D_balanced_acc", "D_F1"]):
        ax.bar(x + (j - 1) * w, [s.get(met, 0.0) for s in summary], w,
               label=met.replace("D_", ""))
    ax.set_xticks(x); ax.set_xticklabels(conds); ax.set_ylim(0, 1)
    ax.set_title("Severity (D) metrics by condition"); ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    for idx, (cond, (cm, labels)) in enumerate(confusions.items()):
        r, c = 1 + idx // 3, idx % 3
        axc = fig.add_subplot(gs[r, c])
        im = axc.imshow(cm, cmap="Blues")
        axc.set_title(f"{cond} confusion", fontsize=10)
        axc.set_xticks(range(len(labels))); axc.set_yticks(range(len(labels)))
        axc.set_xticklabels(labels, fontsize=7); axc.set_yticklabels(labels, fontsize=7)
        vmax = cm.max() if cm.size else 1
        for i in range(len(labels)):
            for k in range(len(labels)):
                axc.text(k, i, int(cm[i, k]), ha="center", va="center", fontsize=7,
                         color="white" if cm[i, k] > vmax * 0.6 else "black")
    fig.suptitle("Severity head evaluation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return _save(fig, name)


# ---------------------------------------------------------------------------
# ROC (per-head micro / D one-vs-rest), conditions overlaid
# ---------------------------------------------------------------------------

def plot_roc(roc_data, name="eval_roc.png"):
    """
    roc_data: {condition: {head: (fpr, tpr, auc)}} for heads in B/C ('micro')
    and 'D' ('macro one-vs-rest'). Skips heads with no curve.
    """
    heads = ["B", "C", "D"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    flat = axes.flatten()
    for ax, h in zip(flat, heads):
        any_curve = False
        for i, (cond, hd) in enumerate(roc_data.items()):
            if h in hd and hd[h] is not None:
                fpr, tpr, auc = hd[h]
                ax.plot(fpr, tpr, color=_cond_color(i), lw=1.6,
                        label=f"{cond} (AUC={auc:.2f})")
                any_curve = True
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
        ax.set_title(f"{HEAD_LABEL[h]} "
                     f"({'one-vs-rest' if h == 'D' else 'micro-avg'})")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        if any_curve:
            ax.legend(fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    for extra in flat[len(heads):]:        # hide any unused panels
        extra.axis("off")
    fig.suptitle("ROC by causal step (conditions overlaid)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return _save(fig, name)


# ---------------------------------------------------------------------------
# Feature importance: sensitivity (perturbation) and SHAP
# ---------------------------------------------------------------------------

def plot_sensitivity(sens, feature_names, name="eval_sensitivity.png"):
    """sens: {condition: importance_vector (len=#features)}."""
    conds = list(sens.keys())
    n = len(feature_names)
    fig, ax = plt.subplots(figsize=(14, max(6, 0.3 * n)))
    y = np.arange(n); h = 0.8 / max(len(conds), 1)
    for i, c in enumerate(conds):
        ax.barh(y + (i - (len(conds) - 1) / 2) * h, sens[c], h,
                label=c, color=_cond_color(i))
    ax.set_yticks(y); ax.set_yticklabels(feature_names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Δ output when feature perturbed (importance)")
    ax.set_title("Input sensitivity (perturbation importance)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="x", alpha=0.3)
    return _save(fig, name)


def plot_shap(shap_by_cond, feature_names, name="eval_shap.png"):
    """shap_by_cond: {condition: mean_abs_shap (len=#features)}."""
    conds = list(shap_by_cond.keys())
    fig, axes = plt.subplots(1, max(len(conds), 1), figsize=(9 * max(len(conds), 1), 8),
                             squeeze=False)
    order = None
    for ax, c in zip(axes[0], conds):
        vals = np.asarray(shap_by_cond[c])
        if order is None:
            order = np.argsort(vals)[::-1][:min(20, len(vals))]
        ax.barh([feature_names[j] for j in order][::-1], vals[order][::-1], color="#5A189A")
        ax.set_title(f"{c}: mean |SHAP|"); ax.tick_params(axis="y", labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Feature importance (SHAP, top-20)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return _save(fig, name)


# ---------------------------------------------------------------------------
# Summary table + adjacency heatmap (causal discovery)
# ---------------------------------------------------------------------------

def plot_summary_table(summary, columns, name="eval_summary_table.png"):
    fig, ax = plt.subplots(figsize=(min(2 + 1.4 * len(columns), 24),
                                    1 + 0.5 * len(summary)))
    ax.axis("off")
    cell = [[f"{s.get(c, ''):.3f}" if isinstance(s.get(c), float) else str(s.get(c, ""))
             for c in columns] for s in summary]
    tbl = ax.table(cellText=cell, colLabels=columns,
                   rowLabels=[s["condition"] for s in summary],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.4)
    ax.set_title("Condition comparison summary", fontsize=13, fontweight="bold")
    return _save(fig, name)


def plot_adjacency_heatmap(adj, feature_names, reference=None,
                           name="causal_adjacency_heatmap.png"):
    """adj: square 0/1 (or weighted) matrix. reference: optional 0/1 expected-edge
    matrix; cells confirmed by reference are outlined."""
    n = len(feature_names)
    fig, ax = plt.subplots(figsize=(max(10, 0.7 * n), max(9, 0.7 * n)))
    im = ax.imshow(adj, cmap="Blues", vmin=0)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(feature_names, fontsize=8)
    ax.set_title("PC discovered adjacency (rows→cols)", fontsize=13, fontweight="bold")
    for i in range(n):
        for j in range(n):
            if adj[i, j]:
                ax.text(j, i, "1", ha="center", va="center", fontsize=7, color="white")
            if reference is not None and reference[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="#C44E52", lw=1.5))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, name)
