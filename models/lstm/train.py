"""
train.py  (Stage 4)
===================
Multi-label causal-chain LSTM over the NTSB corpus:

    context: step_ctx (economic)            -> root node (NOT predicted)
    Step B : step_b                         -> B: Preconditions     (multi-label, n_B)
    Step C : proj_c([soft_B | env | oper])  -> C: Unsafe Acts       (multi-label, n_C)
    Step D : proj_d([soft_C | soft_B | env])-> D: Severity          (single-class, n_D)

Organizational/Supervisory influences are NOT predicted: the upper HFACS tier is
structured economic context fed on a non-predicted root node (step_ctx) that
seeds Preconditions. B/C are multi-label (sigmoid + focal BCE); D (severity) is a
single-class ordinal head (focal CE). Hidden state flows ctx->B->C->D; soft
predictions hand off B->C and C->D. Per-head decision thresholds are tuned on the
validation split (stored in the checkpoint) to counter class imbalance.

The same class serves C1 (no RAG) and C4 (RAG priors appended; larger input
dims). Dimensions are inferred from the first batch — nothing is hardcoded.
ASIAS/ASRS never enter training.

CLI:
    python models/lstm/train.py                       # C1 baseline (no RAG)
    python models/lstm/train.py --rag-strategy hybrid # C4 (RAG priors via Stage 5)
"""

import argparse
import copy
import os
import sys

import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, balanced_accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from data.ntsbdataloader import (get_dataloaders, ENV_SLICE, OPER_SLICE,  # noqa: E402
                                 STEP_B_BASE, NTSB_CLEAN, FEWSHOT_DIM, FS_B, FS_C,
                                 FS_D, FS_HAS_BC, FS_HAS_D, FS_SIM)


# ---------------------------------------------------------------------------
# Losses & class weighting
# ---------------------------------------------------------------------------

def class_weights(labels_1d, n_classes, device, power=0.5, clip=3.0):
    """Single-class inverse-frequency weights for D — dampened, clipped, normalized.

    `power` softens the inverse-frequency (0.5 = sqrt); `clip` caps the per-class
    weight ratio so a rare class can't dominate the loss and push the model to
    over-predict it (the prior cause of severity's sub-chance accuracy)."""
    labels = np.asarray(labels_1d)
    counts = np.bincount(labels, minlength=n_classes).astype("float64")
    counts[counts == 0] = 1.0
    w = (counts.sum() / (n_classes * counts)) ** power
    w = w / w.mean()
    w = np.clip(w, 1.0 / clip, clip)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=device)


def pos_weights(multihot_2d, device):
    """Per-label BCE pos_weight = neg/pos, sqrt-dampened, clamped (for B/C)."""
    y = np.asarray(multihot_2d, dtype="float64")
    n = y.shape[0]
    pos = y.sum(axis=0)
    pos[pos == 0] = 1.0
    neg = n - pos
    pw = np.sqrt(np.clip(neg / pos, 1.0, None))
    return torch.tensor(pw, dtype=torch.float32, device=device)


class FocalLoss(nn.Module):
    """Single-class weighted focal cross-entropy (severity D and binary unsafe C).

    Honors ignore_index (default -100): masked targets (e.g. ASIAS rows for the
    NTSB-only D head) contribute nothing to the loss."""

    def __init__(self, weight=None, gamma=2.0, ignore_index=-100):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight,
                                         reduction="none", ignore_index=self.ignore_index)
        mask = targets != self.ignore_index
        if mask.sum() == 0:
            return logits.sum() * 0.0          # keep graph; no valid targets
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce)[mask].mean()


class MultiLabelFocalLoss(nn.Module):
    """Sigmoid focal BCE for multi-label heads (B/C)."""

    def __init__(self, pos_weight=None, gamma=2.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none")
        pt = torch.exp(-bce)
        return ((1 - pt) ** self.gamma * bce).mean()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

# step_b layout when RAG priors are present:
#   [ base(STEP_B_BASE) | precond(n_B) | unsafe(n_C) | severity(n_D) ]
# B reads base(+precond); C also reads the unsafe prior; D also the severity prior.
# Without RAG, step_b is just the base scalars.
def _prior_layout(n_B, n_C, n_D, step_b_dim):
    has = step_b_dim > STEP_B_BASE
    b_in = STEP_B_BASE + (n_B if has else 0)                       # base (+ precond) -> B
    uns = slice(STEP_B_BASE + n_B, STEP_B_BASE + n_B + n_C) if has else None
    sev = slice(STEP_B_BASE + n_B + n_C,
                STEP_B_BASE + n_B + n_C + n_D) if has else None
    return has, b_in, uns, sev, (n_C if has else 0), (n_D if has else 0)


class FewShotEncoder(nn.Module):
    """Encodes k retrieved labelled exemplars into one conditioning vector.

    Spec 2.3 "prompt augmentation: few-shot examples for LSTM sequence modeling".
    An LSTM has no prompt, so the analogue is to let the model READ a sequence of
    (features, labels) pairs before predicting — in-context learning by
    architecture rather than by prompt. Each exemplar is a retrieved train
    neighbour; the sequence is encoded with an LSTM (hence "sequence modeling")
    and masked-mean-pooled, so padding and empty retrievals contribute nothing.

    This differs from the RAG priors in what it preserves. A prior is the
    neighbours' label distribution with the features averaged away; an exemplar
    keeps feature and label bound together, which is what lets a shot teach a
    mapping rather than a base rate.
    """

    def __init__(self, fewshot_dim: int, out_dim: int = 32, dropout: float = 0.1,
                 tau: float = 0.1):
        super().__init__()
        if fewshot_dim != FEWSHOT_DIM:
            raise ValueError(f"exemplar width {fewshot_dim} != FEWSHOT_DIM {FEWSHOT_DIM}: "
                             f"this checkpoint predates the current exemplar layout "
                             f"and must be retrained")
        self.lstm = nn.LSTM(fewshot_dim, out_dim, batch_first=True)
        self.drop = nn.Dropout(dropout)
        # Vote sharpness. Cosines between narratives sit in a narrow band (0.70-0.85),
        # so weights proportional to similarity are nearly uniform; a softmax with a
        # small temperature lets closer neighbours count for more. Learned, because
        # the right sharpness differs between retrieval strategies.
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau)))
        # The encoder emits the learned summary AND the explicit similarity-weighted
        # vote, so `out_dim` seen by the rest of the model is the sum of the two.
        self.out_dim = out_dim + fewshot_dim

    def forward(self, fewshot, fewshot_mask):
        """-> [batch, out_dim + fewshot_dim]: learned summary || weighted vote.

        The learned half is an LSTM over the exemplar sequence, masked-mean-pooled.
        The second half is the similarity-weighted mean of the raw exemplar rows —
        the k-NN vote, computed rather than learned. Handing the model that
        aggregate is an inductive bias, not extra information: every value is a
        deterministic function of the exemplar block it already sees.

        Two details of the vote matter:

        * Weights are softmax(similarity / tau) over the real neighbours. They were
          similarity / sum(similarity) over a min-max-normalised score, which gave
          the weakest neighbour weight exactly 0 and ranked the rest by position.
        * Each LABEL block is averaged only over neighbours whose labels are known
          (has_factor_labels / has_severity). A neighbour with no label used to
          count as a confident negative. The vote's has_* entries end up holding
          the share of weight that was labelled, so the model can tell a vote of
          0.2 backed by every neighbour from one backed by a single neighbour.
        """
        if fewshot is None or fewshot.numel() == 0 or fewshot.size(1) == 0:
            n = fewshot.size(0) if fewshot is not None else 0
            return torch.zeros(n, self.out_dim, device=fewshot.device)
        m = fewshot_mask.unsqueeze(-1)                    # [batch, k, 1]
        any_real = (m.sum(1) > 0).float()                 # [batch, 1]

        out, _ = self.lstm(fewshot)                       # [batch, k, lstm_out]
        pooled = (out * m).sum(1) / m.sum(1).clamp(min=1.0)

        tau = self.log_tau.exp().clamp(0.02, 1.0)
        logit = (fewshot[..., FS_SIM:FS_SIM + 1] / tau).masked_fill(m == 0, -1e9)
        w = torch.softmax(logit, dim=1) * m               # padding -> exactly 0
        vote = (fewshot * w).sum(1)                       # [batch, fewshot_dim]

        def known(block, flag):                           # vote over labelled rows only
            wf = w * fewshot[..., flag:flag + 1]
            return (fewshot[..., block] * wf).sum(1) / wf.sum(1).clamp(min=1e-6)

        vote = torch.cat([
            vote[:, :FS_B.start],
            known(FS_B, FS_HAS_BC), known(FS_C, FS_HAS_BC), known(FS_D, FS_HAS_D),
            vote[:, FS_D.stop:]], dim=1)
        vote = vote * any_real                            # no exemplars -> zeros

        return self.drop(torch.cat([pooled, vote], dim=1))


class HFACSCausalLSTM(nn.Module):
    """Causal LSTM over the DAG: economic-context root (step_ctx) seeds
    B(Preconditions) -> C(Unsafe Acts) -> D(Severity). RAG priors when present
    (precond->B, unsafe->C, severity->D) enter via step_b. Few-shot exemplars,
    when enabled, are encoded and concatenated onto the CONTEXT ROOT so they
    condition the whole chain without disturbing the per-head prior structure.
    See module docstring."""

    def __init__(self, hidden_size, n_B, n_C, n_D,
                 step_ctx_dim, step_b_dim, dropout=0.2,
                 fewshot_dim=0, fewshot_out=32, c_multilabel=False):
        super().__init__()
        self.hidden_size = hidden_size
        # True: C is the four-tier multi-label head (sigmoid hand-off to D).
        # False: a checkpoint from the binary violation-vs-error head (softmax).
        self.c_multilabel = c_multilabel
        self.env_slice = ENV_SLICE                          # visual, light, tod (3)
        self.oper_slice = OPER_SLICE                        # person, pilot_hours (2)
        (self.has_priors, self._b_in, self.unsafe_slice, self.sev_slice,
         c_extra, d_extra) = _prior_layout(n_B, n_C, n_D, step_b_dim)

        # fewshot_dim=0 -> no encoder, and step_ctx_dim is unchanged, so existing
        # C1..C8 checkpoints load and behave identically.
        self.fewshot_enc = FewShotEncoder(fewshot_dim, fewshot_out, dropout) \
            if fewshot_dim > 0 else None
        # The encoder emits learned summary || weighted vote, so its width is
        # fewshot_out + fewshot_dim. Read it off the module rather than assuming.
        fs_width = self.fewshot_enc.out_dim if self.fewshot_enc is not None else 0
        ctx_in = step_ctx_dim + fs_width

        self.cell_ctx = nn.LSTMCell(ctx_in, hidden_size)   # context root (no head)
        # B takes the exemplars as DIRECT input too. Routing them only through
        # hCtx left B the one head without first-hand access to retrieval,
        # while C and D received fs_emb concatenated into their projections.
        self.cell_b = nn.LSTMCell(self._b_in + fs_width, hidden_size)
        self.cell_c = nn.LSTMCell(hidden_size, hidden_size)
        self.cell_d = nn.LSTMCell(hidden_size, hidden_size)
        self.drop = nn.Dropout(dropout)

        # Exemplars feed EVERY node, not just the root. Reaching only the root made
        # them useless to C and D: the chain detaches its soft predictions, so by the
        # time information travelled ctx -> B -> C -> D there was no path left for
        # retrieval to inform severity. The prior mechanism fed each head directly
        # (precond->B, unsafe->C, severity->D); exemplars now do the same.
        fs = fs_width
        self._fs = fs

        # Skip-edges (causal_discovery validates these); prior slots add when present.
        self.proj_c = nn.Linear(n_B + 3 + 2 + c_extra + fs, hidden_size)   # [soft_B|env|oper|prior|fewshot]
        self.proj_d = nn.Linear(n_C + n_B + 3 + d_extra + fs, hidden_size) # [soft_C|soft_B|env|prior|fewshot]

        self.head_b = nn.Linear(hidden_size, n_B)
        self.head_c = nn.Linear(hidden_size, n_C)
        self.head_d = nn.Linear(hidden_size, n_D)

    def forward(self, step_ctx, step_b, fewshot=None, fewshot_mask=None):
        """context -> B -> C -> D, with hidden state carrying GRADIENT.

        The hidden-state handoffs used to be detached (`hCtx.detach()` etc.). That
        was a bug, not a design choice: `cell_ctx` has no head of its own, so the
        handoff was its only gradient path. Detaching it left the context root AND
        the FewShotEncoder feeding it randomly initialised and never updated —
        retrieval could not influence any prediction, and `step_ctx` (the economic
        context) was inert.

        What legitimately stays detached is the SOFT PREDICTIONS (`soft_B`,
        `soft_C`). Those carry a head's *output* forward as evidence, and detaching
        them keeps each head trained by its own loss rather than by the losses of
        heads downstream. The causal chain is unchanged; it can now learn.
        """
        batch = step_b.size(0)
        zeros = lambda: torch.zeros(batch, self.hidden_size, device=step_b.device)

        fs_emb = None
        if self.fewshot_enc is not None:
            fs_emb = self.fewshot_enc(fewshot, fewshot_mask)
            step_ctx = torch.cat([step_ctx, fs_emb], 1)
        hCtx, cCtx = self.cell_ctx(step_ctx, (zeros(), zeros()))

        # B <- base step_b (+ precond prior) + exemplars, seeded by the context root
        b_in = step_b[:, :self._b_in]
        if fs_emb is not None:
            b_in = torch.cat([b_in, fs_emb], 1)          # retrieval -> B, directly
        hB, cB = self.cell_b(b_in, (hCtx, cCtx))
        logits_B = self.head_b(self.drop(hB))
        soft_B = torch.sigmoid(logits_B).detach()

        env = step_b[:, self.env_slice]
        oper = step_b[:, self.oper_slice]

        c_parts = [soft_B, env, oper]
        if self.has_priors:
            c_parts.append(step_b[:, self.unsafe_slice])
        if fs_emb is not None:
            c_parts.append(fs_emb)                     # retrieval -> C, directly
        hC, cC = self.cell_c(self.proj_c(torch.cat(c_parts, 1)), (hB, cB))
        logits_C = self.head_c(self.drop(hC))
        soft_C = (torch.sigmoid(logits_C) if self.c_multilabel
                  else torch.softmax(logits_C, 1)).detach()

        d_parts = [soft_C, soft_B, env]
        if self.has_priors:
            d_parts.append(step_b[:, self.sev_slice])
        if fs_emb is not None:
            d_parts.append(fs_emb)                     # retrieval -> D, directly
        hD, _ = self.cell_d(self.proj_d(torch.cat(d_parts, 1)), (hC, cC))
        logits_D = self.head_d(self.drop(hD))
        return logits_B, logits_C, logits_D


class HFACSCausalSCM(nn.Module):
    """Neural Structural Causal Model: one MLP per node over the SAME DAG and
    skip-edges as the LSTM (drop-in; identical forward signature). Each node is a
    structural equation f(parents), making do()/counterfactual semantics explicit
    — override a node's output and propagate."""

    def __init__(self, hidden_size, n_B, n_C, n_D,
                 step_ctx_dim, step_b_dim, dropout=0.2,
                 fewshot_dim=0, fewshot_out=32, c_multilabel=False):
        super().__init__()
        self.c_multilabel = c_multilabel
        self.env_slice = ENV_SLICE
        self.oper_slice = OPER_SLICE
        (self.has_priors, self._b_in, self.unsafe_slice, self.sev_slice,
         c_extra, d_extra) = _prior_layout(n_B, n_C, n_D, step_b_dim)

        def mlp(d_in, d_out):
            return nn.Sequential(nn.Linear(d_in, hidden_size), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden_size, d_out))

        # Exemplars enter on the context (root) equation, matching the LSTM variant.
        self.fewshot_enc = FewShotEncoder(fewshot_dim, fewshot_out, dropout) \
            if fewshot_dim > 0 else None
        ctx_in = step_ctx_dim + (self.fewshot_enc.out_dim
                                 if self.fewshot_enc is not None else 0)

        self.f_B = mlp(ctx_in + self._b_in, n_B)              # B <- context + base(+precond)
        self.f_C = mlp(n_B + 3 + 2 + c_extra, n_C)            # C <- soft_B|env|oper|unsafe_prior
        self.f_D = mlp(n_C + n_B + 3 + d_extra, n_D)          # D <- soft_C|soft_B|env|sev_prior

    def forward(self, step_ctx, step_b, fewshot=None, fewshot_mask=None):
        if self.fewshot_enc is not None:
            step_ctx = torch.cat([step_ctx,
                                  self.fewshot_enc(fewshot, fewshot_mask)], 1)
        logits_B = self.f_B(torch.cat([step_ctx, step_b[:, :self._b_in]], 1))
        soft_B = torch.sigmoid(logits_B).detach()
        env = step_b[:, self.env_slice]
        oper = step_b[:, self.oper_slice]
        c_parts = [soft_B, env, oper]
        if self.has_priors:
            c_parts.append(step_b[:, self.unsafe_slice])
        logits_C = self.f_C(torch.cat(c_parts, 1))
        soft_C = (torch.sigmoid(logits_C) if self.c_multilabel
                  else torch.softmax(logits_C, 1)).detach()
        d_parts = [soft_C, soft_B, env]
        if self.has_priors:
            d_parts.append(step_b[:, self.sev_slice])
        logits_D = self.f_D(torch.cat(d_parts, 1))
        return logits_B, logits_C, logits_D


_ARCHS = {"lstm": HFACSCausalLSTM, "scm": HFACSCausalSCM}


def make_model(config: dict):
    """Build a model from a checkpoint/config dict. config['arch'] selects 'lstm'
    (default) or 'scm'. Backward-compatible: missing 'arch' -> lstm."""
    cfg = dict(config)
    arch = cfg.pop("arch", "lstm")
    # Bookkeeping recorded in the checkpoint so eval can rebuild the condition
    # (exemplar count, retrieval strategy, raw mode) — none are model arguments.
    for k in ("fewshot_k", "condition", "strategy", "raw_mode", "seed",
              "kg_factor_labels", "gate"):
        cfg.pop(k, None)
    return _ARCHS[arch](**cfg)


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------

LOSS_WEIGHTS = (1.0, 1.5, 1.0)  # B, C, D


def _joint_loss(logits, targets, crits):
    return sum(w * c(l, t) for w, c, l, t in zip(LOSS_WEIGHTS, crits, logits, targets))


def train_epoch(model, loader, optimizer, crits, device):
    model.train()
    total = 0.0
    for step_ctx, step_b, yB, yC, yD, fs, fsm in loader:
        step_ctx, step_b = step_ctx.to(device), step_b.to(device)
        fs, fsm = fs.to(device), fsm.to(device)
        ys = [t.to(device) for t in (yB, yC, yD)]
        optimizer.zero_grad()
        loss = _joint_loss(model(step_ctx, step_b, fs, fsm), ys, crits)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


# Per-head multi-label decision thresholds. B and C are both multi-label; D is
# argmax-decoded.
_ML_KEYS = ("B", "C")

# What each head's thresholds are tuned to maximise on validation. B keeps F1 so
# its numbers stay comparable with earlier runs. C uses balanced accuracy: its
# tiers run from 9% to 71% positive, and F1 on a 71% tier is maximised by calling
# everything positive, which is the degenerate solution the README warns about.
THRESHOLD_OBJECTIVE = {"B": "f1", "C": "balanced_acc"}


def _apply_thr(logits, thr):
    """logits -> binary multi-hot via sigmoid, per-column threshold (default 0.5)."""
    prob = torch.sigmoid(logits)
    if thr is None:
        return (prob > 0.5).int().cpu().numpy()
    t = torch.as_tensor(thr, dtype=prob.dtype, device=prob.device)
    return (prob >= t).int().cpu().numpy()


def evaluate(model, loader, device, thresholds=None):
    """
    Returns 6 values:
        all_B, all_C, all_D, pred_B, pred_C, pred_D
    B and C are multi-hot numpy arrays (preds use the tuned thresholds, or
    sigmoid>0.5 if none). D is a 1-D array of class indices (argmax). A model from
    the binary-C era (c_multilabel False) still gets argmax class indices for C.
    `thresholds` is an optional {'B': vec, 'C': vec}.
    """
    thr = thresholds or {}
    multi_c = bool(getattr(model, "c_multilabel", False))
    model.eval()
    aB, aC, aD = [], [], []
    pB, pC, pD = [], [], []
    with torch.no_grad():
        for step_ctx, step_b, yB, yC, yD, fs, fsm in loader:
            step_ctx, step_b = step_ctx.to(device), step_b.to(device)
            fs, fsm = fs.to(device), fsm.to(device)
            lB, lC, lD = model(step_ctx, step_b, fs, fsm)
            pB.append(_apply_thr(lB, thr.get("B")))
            pC.append(_apply_thr(lC, thr.get("C")) if multi_c
                      else lC.argmax(1).cpu().numpy())
            pD.extend(lD.argmax(1).cpu().tolist())
            aB.append(yB.int().cpu().numpy())
            aC.append(yC.int().cpu().numpy() if multi_c else yC.cpu().numpy())
            aD.extend(yD.cpu().tolist())
    cat = lambda xs: np.concatenate(xs, axis=0) if xs else np.empty((0,))
    return (cat(aB), cat(aC), np.asarray(aD),
            cat(pB), cat(pC), np.asarray(pD))


def _threshold_score(y, pred, objective):
    if objective == "balanced_acc":
        return balanced_accuracy_score(y, pred) if 0 < y.sum() < len(y) else 0.0
    return f1_score(y, pred, zero_division=0)


def tune_thresholds(model, loader, device, grid=None):
    """Per-label thresholds for B and C on a (validation) loader, each head tuned
    to its own THRESHOLD_OBJECTIVE.

    Counters the focal/imbalance collapse where sigmoid stays < 0.5 for rare-but-
    present classes. Classes with no positives in the split keep the 0.5 default.
    """
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    model.eval()
    probs = {k: [] for k in _ML_KEYS}
    truth = {k: [] for k in _ML_KEYS}
    with torch.no_grad():
        for step_ctx, step_b, yB, yC, yD, fs, fsm in loader:
            step_ctx, step_b = step_ctx.to(device), step_b.to(device)
            fs, fsm = fs.to(device), fsm.to(device)
            lB, lC, lD = model(step_ctx, step_b, fs, fsm)
            for key, logits, y in (("B", lB, yB), ("C", lC, yC)):
                probs[key].append(torch.sigmoid(logits).cpu().numpy())
                truth[key].append(y.cpu().numpy())
    out = {}
    for key in _ML_KEYS:
        P = np.concatenate(probs[key], 0) if probs[key] else np.empty((0, 0))
        Y = np.concatenate(truth[key], 0) if truth[key] else np.empty((0, 0))
        thr = np.full(P.shape[1], 0.5, dtype="float32")
        for j in range(P.shape[1]):
            if Y[:, j].sum() == 0:
                continue
            best, best_t = -1.0, 0.5
            for t in grid:
                sc = _threshold_score(Y[:, j], (P[:, j] >= t).astype(int),
                                      THRESHOLD_OBJECTIVE.get(key, "f1"))
                if sc > best:
                    best, best_t = sc, t
            thr[j] = best_t
        out[key] = thr
    return out


def _build_criteria(train_set, n_C, n_D, device, baseline=False):
    """Losses for the three heads.

    `baseline=True` is the stripped-back configuration: plain BCE / cross-entropy
    with no focal term and no class weighting. Use it to see what the causal LSTM
    does on its own, before the imbalance machinery is layered on. It matters more
    than it used to: B's groups are now ~72% positive, so `pos_weights` clips to
    1.0 and the focal path is quietly doing nothing for them anyway.
    """
    if baseline:
        return (nn.BCEWithLogitsLoss(),
                nn.BCEWithLogitsLoss(),
                nn.CrossEntropyLoss(ignore_index=-100))
    crit_B = MultiLabelFocalLoss(pos_weight=pos_weights(train_set.y_B.numpy(), device))
    # C is multi-label over the four unsafe-act tiers, same loss family as B; its
    # per-tier pos_weight matters more here (violation ~9%, perception ~14%).
    # D is severity (NTSB-only — drop the -100 masked ASIAS rows before weighting).
    crit_C = MultiLabelFocalLoss(pos_weight=pos_weights(train_set.y_C.numpy(), device))
    yD = train_set.y_D.numpy()
    crit_D = FocalLoss(weight=class_weights(yD[yD != -100], n_D, device))
    return crit_B, crit_C, crit_D


def train_model(train_loader, val_loader, encoders, hidden_size=128, lr=1e-4,
                dropout=0.1, epochs=500, patience=200, device=None, verbose=True,
                arch="lstm", baseline=False, seed=None):
    """Train one condition. `seed` fixes weight init AND batch order.

    Without it, two runs of the SAME condition differed by ~0.04 kappa on head D —
    larger than any gap between the retrieval conditions, which made the comparison
    meaningless. Seeding is what lets repeated runs be averaged and compared.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if seed is not None:
        torch.manual_seed(seed)                 # weight init + DataLoader shuffling
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # Infer dims from the first batch — never hardcoded. fewshot is [batch, k, dim];
    # k == 0 means exemplars are disabled, so fewshot_dim stays 0 and no encoder is
    # built (keeping C1..C8 bit-identical).
    s_ctx, s_b, _yb, _yc, _yd, s_fs, _fsm = next(iter(train_loader))
    fewshot_dim = int(s_fs.shape[2]) if s_fs.dim() == 3 and s_fs.shape[1] > 0 else 0
    config = dict(arch=arch, hidden_size=hidden_size,
                  n_B=encoders.n_B, n_C=encoders.n_C, n_D=encoders.n_severity,
                  step_ctx_dim=s_ctx.shape[1], step_b_dim=s_b.shape[1], dropout=dropout,
                  fewshot_dim=fewshot_dim, c_multilabel=True,
                  # k is not a model hyperparameter (the encoder handles any
                  # sequence length) but eval must rebuild the SAME exemplar
                  # count, so it is recorded here.
                  fewshot_k=int(s_fs.shape[1]) if fewshot_dim else 0)
    model = make_model(config).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-5)
    crits = _build_criteria(train_loader.dataset, config["n_C"], config["n_D"], device,
                            baseline=baseline)

    best_val, best_state, no_improve = float("inf"), None, 0
    history = {"loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, crits, device)
        history["loss"].append(loss)

        model.eval()
        vtotal = 0.0
        with torch.no_grad():
            for step_ctx, step_b, yB, yC, yD, fs, fsm in val_loader:
                step_ctx, step_b = step_ctx.to(device), step_b.to(device)
                fs, fsm = fs.to(device), fsm.to(device)
                ys = [t.to(device) for t in (yB, yC, yD)]
                vtotal += _joint_loss(model(step_ctx, step_b, fs, fsm), ys, crits).item()
        val_loss = vtotal / max(len(val_loader), 1)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val, best_state, no_improve = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1

        if verbose:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:03d}/{epochs} | loss {loss:.4f} | "
                  f"val {val_loss:.4f} (best {best_val:.4f}) | lr {lr_now:.2e}")
        if no_improve >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Per-head decision thresholds tuned on the validation split. In baseline mode
    # they stay fixed at 0.5 — tuned thresholds are part of the machinery being
    # stripped, and with B now ~72% positive they would drift toward all-positive.
    if baseline:
        thresholds = {"B": np.full(config["n_B"], 0.5, dtype="float32"),
                      "C": np.full(config["n_C"], 0.5, dtype="float32")}
        if verbose:
            print("Baseline mode: thresholds fixed at 0.50 (not tuned).")
    else:
        thresholds = tune_thresholds(model, val_loader, device)
        if verbose:
            print("Tuned thresholds (mean per head): "
                  + ", ".join(f"{k}={float(np.mean(v)):.2f}" for k, v in thresholds.items()))
    return model, history, config, thresholds


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(model, encoders, path, config, thresholds=None):
    """Save weights + the constructor config + tuned per-head thresholds."""
    payload = {"state_dict": model.state_dict(), "config": config}
    if thresholds is not None:
        payload["thresholds"] = {k: np.asarray(v) for k, v in thresholds.items()}
    torch.save(payload, path)
    print(f"Saved checkpoint to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_retriever(strategy, model=None, asias_weight=0.34, asrs_weight=0.33,
                     ntsb_weight=0.33, top_k=5, factor_priors=True):
    """Construct a Stage-5 RAG retriever if available; None otherwise (C1)."""
    if not strategy:
        return None
    try:
        from data.rag_retriever import build_retriever  # Stage 5
        kw = {"model": model} if model else {}
        return build_retriever(strategy=strategy, asias_weight=asias_weight,
                               asrs_weight=asrs_weight, ntsb_weight=ntsb_weight,
                               k=top_k, factor_priors=factor_priors, **kw)
    except Exception as e:
        print(f"WARNING: RAG retriever unavailable ({e}); falling back to no-RAG.")
        return None


def main():
    parser = argparse.ArgumentParser(description="Train the HFACS causal LSTM")
    parser.add_argument("--rag-strategy", default=None,
                        help="RAG strategy for C2-C4 (e.g. 'hybrid'); omit for C1.")
    parser.add_argument("--arch", choices=["lstm", "scm"], default="lstm",
                        help="Node model: 'lstm' (default) or 'scm' (neural SCM).")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None,
                        help="Use only the first N records (smoke-test subset).")
    parser.add_argument("--model", default=None,
                        help="Ollama model for the RAG retriever's Cypher (C2-C4).")
    parser.add_argument("--input", default=NTSB_CLEAN,
                        help="NTSB CSV to train on (e.g. data/ntsb_subset.csv).")
    parser.add_argument("--save-path", default=None,
                        help="Where to save the checkpoint (e.g. results/c4.pt). "
                             "Default: models/lstm/hfacs_lstm.pt.")
    parser.add_argument("--asias-weight", type=float, default=0.34,
                        help="FAISS ASIAS weight (set 0 for single-source ablations).")
    parser.add_argument("--asrs-weight", type=float, default=0.33,
                        help="FAISS ASRS weight (set 0 for single-source ablations).")
    parser.add_argument("--ntsb-weight", type=float, default=0.33,
                        help="FAISS in-distribution NTSB-KG weight (set 0 to exclude).")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Retrieval depth k.")
    parser.add_argument("--no-factor-priors", dest="factor_priors", action="store_false",
                        help="C8: disable LLM-mined HFACS factor priors (held uniform); "
                             "keep retrieval + structured severity-outcome prior.")
    parser.add_argument("--baseline", action="store_true",
                        help="Stripped-back baseline: plain BCE/CE (no focal term, "
                             "no class weighting) and fixed 0.5 thresholds. Use to "
                             "see the causal LSTM's own performance before the "
                             "imbalance machinery is layered on.")
    parser.add_argument("--fewshot-k", type=int, default=0,
                        help="C9: prompt augmentation. Retrieve k labelled train "
                             "exemplars per record and encode them onto the context "
                             "root (see FewShotEncoder). 0 disables. Independent of "
                             "--rag-strategy: exemplars and priors can be used "
                             "together or separately.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    retriever = _build_retriever(
        args.rag_strategy, args.model, asias_weight=args.asias_weight,
        asrs_weight=args.asrs_weight, ntsb_weight=args.ntsb_weight,
        top_k=args.top_k, factor_priors=args.factor_priors)

    train_loader, val_loader, test_loader, encoders = get_dataloaders(
        filepath=args.input, batch_size=args.batch_size, retriever=retriever,
        limit=args.limit, fewshot_k=args.fewshot_k)

    cond = "RAG " + args.rag_strategy if retriever else "C1 (no RAG)"
    if args.fewshot_k:
        cond += f" + few-shot k={args.fewshot_k}"
    print("=" * 60)
    print(f"Arch: {args.arch} | Condition: {cond}")
    print(f"Heads — B:{encoders.n_B} C:{encoders.n_C} D:{encoders.n_severity}  "
          f"(org/sup context is a structured input, not predicted)")
    print("=" * 60)

    model, history, config, thresholds = train_model(
        train_loader, val_loader, encoders, hidden_size=args.hidden_size,
        lr=args.lr, dropout=args.dropout, epochs=args.epochs, device=device,
        baseline=args.baseline,
        arch=args.arch)

    fig_dir = os.path.join(os.path.dirname(__file__), "..", "..", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(history["loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("epoch"); plt.ylabel("joint focal loss"); plt.legend(); plt.grid(True)
    plt.title("HFACS Causal LSTM training")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "lstm_training_curves.png"))

    save_path = args.save_path or os.path.join(os.path.dirname(__file__), "hfacs_lstm.pt")
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    save_checkpoint(model, encoders, save_path, config, thresholds)


if __name__ == "__main__":
    main()
