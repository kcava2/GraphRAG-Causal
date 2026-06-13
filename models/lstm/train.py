"""
train.py  (Stage 4)
===================
Multi-label causal-chain LSTM over the NTSB corpus:

    Step O : step_o                         -> O: Org. Influences   (multi-label, n_O)
    Step A : step_a (org influences)        -> A: Supervisory       (multi-label, n_A)
    Step B : step_b                         -> B: Preconditions     (multi-label, n_B)
    Step C : proj_c([soft_B | supervisory]) -> C: Unsafe Acts       (multi-label, n_C)
    Step D : proj_d([soft_C | env])         -> D: Severity          (single-class, n_D)

O/A/B/C are multi-label (sigmoid + focal BCE) so co-occurring HFACS factors —
within a step and across the chain — can be predicted. D (severity) is a
single-class ordinal head (focal CE). Hidden state flows O->A->B->C->D; lower
levels are teacher-forced (true org influences into A, true supervisory into B);
soft predictions hand off B->C and C->D.

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

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from data.ntsbdataloader import get_dataloaders, ENV_SLICE, SUP_START, NTSB_CLEAN  # noqa: E402


# ---------------------------------------------------------------------------
# Losses & class weighting
# ---------------------------------------------------------------------------

def class_weights(labels_1d, n_classes, device):
    """Single-class inverse-frequency weights, sqrt-dampened, mean-normalized (for D)."""
    labels = np.asarray(labels_1d)
    counts = np.bincount(labels, minlength=n_classes).astype("float64")
    counts[counts == 0] = 1.0
    w = counts.sum() / (n_classes * counts)
    w = np.sqrt(w)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=device)


def pos_weights(multihot_2d, device):
    """Per-label BCE pos_weight = neg/pos, sqrt-dampened, clamped (for O/A/B/C)."""
    y = np.asarray(multihot_2d, dtype="float64")
    n = y.shape[0]
    pos = y.sum(axis=0)
    pos[pos == 0] = 1.0
    neg = n - pos
    pw = np.sqrt(np.clip(neg / pos, 1.0, None))
    return torch.tensor(pw, dtype=torch.float32, device=device)


class FocalLoss(nn.Module):
    """Single-class weighted focal cross-entropy (for severity D)."""

    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight,
                                         reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class MultiLabelFocalLoss(nn.Module):
    """Sigmoid focal BCE for multi-label heads (O/A/B/C)."""

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

class HFACSCausalLSTM(nn.Module):
    """Five-step causal LSTM; O/A/B/C multi-label, D single-class. See module docstring."""

    def __init__(self, hidden_size, n_O, n_A, n_B, n_C, n_D,
                 step_o_dim, step_a_dim, step_b_dim, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        # Slices into step_b (base layout; priors are appended after, so valid in C4).
        self.env_slice = ENV_SLICE                          # visual, light, sky
        self.sup_slice = slice(SUP_START, SUP_START + n_A)  # supervisory multi-hot

        self.cell_o = nn.LSTMCell(step_o_dim, hidden_size)
        self.cell_a = nn.LSTMCell(step_a_dim, hidden_size)
        self.cell_b = nn.LSTMCell(step_b_dim, hidden_size)
        self.cell_c = nn.LSTMCell(step_b_dim, hidden_size)
        self.cell_d = nn.LSTMCell(step_b_dim, hidden_size)
        self.drop = nn.Dropout(dropout)

        self.proj_c = nn.Linear(n_B + n_A, step_b_dim)   # [soft_B | supervisory]
        self.proj_d = nn.Linear(n_C + 3, step_b_dim)     # [soft_C | env(3)]

        self.head_o = nn.Linear(hidden_size, n_O)
        self.head_a = nn.Linear(hidden_size, n_A)
        self.head_b = nn.Linear(hidden_size, n_B)
        self.head_c = nn.Linear(hidden_size, n_C)
        self.head_d = nn.Linear(hidden_size, n_D)

    def forward(self, step_o, step_a, step_b):
        batch = step_b.size(0)
        zeros = lambda: torch.zeros(batch, self.hidden_size, device=step_b.device)

        # Step O -> Organizational Influences
        hO, cO = self.cell_o(step_o, (zeros(), zeros()))
        logits_O = self.head_o(self.drop(hO))

        # Step A -> Supervisory (step_a carries teacher-forced org influences)
        hA, cA = self.cell_a(step_a, (hO.detach(), cO.detach()))
        logits_A = self.head_a(self.drop(hA))

        # Step B -> Preconditions (step_b carries teacher-forced supervisory block)
        hB, cB = self.cell_b(step_b, (hA.detach(), cA.detach()))
        logits_B = self.head_b(self.drop(hB))
        soft_B = torch.sigmoid(logits_B).detach()

        # Step C -> Unsafe Acts
        sup = step_b[:, self.sup_slice]
        c_in = self.proj_c(torch.cat([soft_B, sup], dim=1))
        hC, cC = self.cell_c(c_in, (hB.detach(), cB.detach()))
        logits_C = self.head_c(self.drop(hC))
        soft_C = torch.sigmoid(logits_C).detach()

        # Step D -> Severity (single-class)
        env = step_b[:, self.env_slice]
        d_in = self.proj_d(torch.cat([soft_C, env], dim=1))
        hD, _ = self.cell_d(d_in, (hC.detach(), cC.detach()))
        logits_D = self.head_d(self.drop(hD))

        return logits_O, logits_A, logits_B, logits_C, logits_D


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------

LOSS_WEIGHTS = (1.0, 1.5, 1.0, 1.5, 1.0)  # O, A, B, C, D


def _joint_loss(logits, targets, crits):
    return sum(w * c(l, t) for w, c, l, t in zip(LOSS_WEIGHTS, crits, logits, targets))


def train_epoch(model, loader, optimizer, crits, device):
    model.train()
    total = 0.0
    for step_o, step_a, step_b, yO, yA, yB, yC, yD in loader:
        step_o, step_a, step_b = step_o.to(device), step_a.to(device), step_b.to(device)
        ys = [t.to(device) for t in (yO, yA, yB, yC, yD)]
        optimizer.zero_grad()
        loss = _joint_loss(model(step_o, step_a, step_b), ys, crits)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


def evaluate(model, loader, device):
    """
    Returns 10 values:
        all_O, all_A, all_B, all_C, all_D, pred_O, pred_A, pred_B, pred_C, pred_D
    O/A/B/C are multi-hot numpy arrays (preds use sigmoid>0.5); D is a 1-D array
    of class indices (argmax). All callers must unpack exactly 10 values.
    """
    model.eval()
    aO, aA, aB, aC, aD = [], [], [], [], []
    pO, pA, pB, pC, pD = [], [], [], [], []
    with torch.no_grad():
        for step_o, step_a, step_b, yO, yA, yB, yC, yD in loader:
            step_o, step_a, step_b = step_o.to(device), step_a.to(device), step_b.to(device)
            lO, lA, lB, lC, lD = model(step_o, step_a, step_b)
            for logits, store in ((lO, pO), (lA, pA), (lB, pB), (lC, pC)):
                store.append((torch.sigmoid(logits) > 0.5).int().cpu().numpy())
            pD.extend(lD.argmax(1).cpu().tolist())
            for y, store in ((yO, aO), (yA, aA), (yB, aB), (yC, aC)):
                store.append(y.int().cpu().numpy())
            aD.extend(yD.cpu().tolist())
    cat = lambda xs: np.concatenate(xs, axis=0) if xs else np.empty((0,))
    return (cat(aO), cat(aA), cat(aB), cat(aC), np.asarray(aD),
            cat(pO), cat(pA), cat(pB), cat(pC), np.asarray(pD))


def _build_criteria(train_set, n_D, device):
    crit_O = MultiLabelFocalLoss(pos_weight=pos_weights(train_set.y_O.numpy(), device))
    crit_A = MultiLabelFocalLoss(pos_weight=pos_weights(train_set.y_A.numpy(), device))
    crit_B = MultiLabelFocalLoss(pos_weight=pos_weights(train_set.y_B.numpy(), device))
    crit_C = MultiLabelFocalLoss(pos_weight=pos_weights(train_set.y_C.numpy(), device))
    crit_D = FocalLoss(weight=class_weights(train_set.y_D.numpy(), n_D, device))
    return crit_O, crit_A, crit_B, crit_C, crit_D


def train_model(train_loader, val_loader, encoders, hidden_size=128, lr=1e-4,
                dropout=0.1, epochs=500, patience=200, device=None, verbose=True):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Infer dims from the first batch — never hardcoded.
    s_o, s_a, s_b, *_ = next(iter(train_loader))
    config = dict(hidden_size=hidden_size,
                  n_O=encoders.n_O, n_A=encoders.n_A, n_B=encoders.n_B,
                  n_C=encoders.n_C, n_D=encoders.n_severity,
                  step_o_dim=s_o.shape[1], step_a_dim=s_a.shape[1],
                  step_b_dim=s_b.shape[1], dropout=dropout)
    model = HFACSCausalLSTM(**config).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-5)
    crits = _build_criteria(train_loader.dataset, config["n_D"], device)

    best_val, best_state, no_improve = float("inf"), None, 0
    history = {"loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, crits, device)
        history["loss"].append(loss)

        model.eval()
        vtotal = 0.0
        with torch.no_grad():
            for step_o, step_a, step_b, yO, yA, yB, yC, yD in val_loader:
                step_o, step_a, step_b = step_o.to(device), step_a.to(device), step_b.to(device)
                ys = [t.to(device) for t in (yO, yA, yB, yC, yD)]
                vtotal += _joint_loss(model(step_o, step_a, step_b), ys, crits).item()
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
    return model, history, config


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(model, encoders, path, config):
    """Save weights + the full constructor config needed to reconstruct the model."""
    torch.save({"state_dict": model.state_dict(), "config": config}, path)
    print(f"Saved checkpoint to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_retriever(strategy, model=None, asias_weight=0.6, asrs_weight=0.4, top_k=5):
    """Construct a Stage-5 RAG retriever if available; None otherwise (C1)."""
    if not strategy:
        return None
    try:
        from data.rag_retriever import build_retriever  # Stage 5
        kw = {"model": model} if model else {}
        return build_retriever(strategy=strategy, asias_weight=asias_weight,
                               asrs_weight=asrs_weight, k=top_k, **kw)
    except Exception as e:
        print(f"WARNING: RAG retriever unavailable ({e}); falling back to no-RAG.")
        return None


def main():
    parser = argparse.ArgumentParser(description="Train the HFACS causal LSTM")
    parser.add_argument("--rag-strategy", default=None,
                        help="RAG strategy for C2-C4 (e.g. 'hybrid'); omit for C1.")
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
    parser.add_argument("--asias-weight", type=float, default=0.6,
                        help="FAISS ASIAS weight (C5/C6 ablations).")
    parser.add_argument("--asrs-weight", type=float, default=0.4,
                        help="FAISS ASRS weight (C5/C6 ablations).")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Retrieval depth k (C7/C8 ablations).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    retriever = _build_retriever(args.rag_strategy, args.model,
                                 asias_weight=args.asias_weight,
                                 asrs_weight=args.asrs_weight, top_k=args.top_k)

    train_loader, val_loader, test_loader, encoders = get_dataloaders(
        filepath=args.input, batch_size=args.batch_size, retriever=retriever,
        limit=args.limit)

    print("=" * 60)
    print(f"Condition: {'C4 (RAG ' + args.rag_strategy + ')' if retriever else 'C1 (no RAG)'}")
    print(f"Classes — O:{encoders.n_O} A:{encoders.n_A} B:{encoders.n_B} "
          f"C:{encoders.n_C} D:{encoders.n_severity}")
    print("=" * 60)

    model, history, config = train_model(
        train_loader, val_loader, encoders, hidden_size=args.hidden_size,
        lr=args.lr, dropout=args.dropout, epochs=args.epochs, device=device)

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
    save_checkpoint(model, encoders, save_path, config)


if __name__ == "__main__":
    main()
