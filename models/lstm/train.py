"""
train.py
========
Four-step causal LSTM with an optional ordinal-severity head.

Steps:
    A : Org. Climate + Employment        →  Supervisory Conditions  (logits_A)
    0 : Weather/Time/Sky/Personnel/Sup.  →  Operator Conditions     (logits_B)
    1 : soft_B | Supervisory             →  Unsafe Acts             (logits_C)
    2 : soft_C | Supervisory             →  Severity (4-class)      (logits_D)

Requirements: neo4j, ollama, pandas, torch, scikit-learn, anthropic (optional)
"""

import copy
import os
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Allow importing from data/ and config/
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from data.real_dataloader import get_dataloaders  # noqa: E402


def class_weights(labels_tensor, n_classes, device):
    """Square-root-dampened class weights (softer than full inverse-frequency)."""
    labels = labels_tensor.numpy() if hasattr(labels_tensor, "numpy") else np.asarray(labels_tensor)
    weights = compute_class_weight("balanced", classes=np.arange(n_classes), y=labels)
    weights = np.sqrt(weights)           # dampen extremes
    weights = weights / weights.mean()   # re-normalize for stable scale
    return torch.tensor(weights, dtype=torch.float32, device=device)


class FocalLoss(nn.Module):
    """Weighted focal loss with (1-pt)^gamma modulator."""
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma  = gamma

    def forward(self, logits, targets):
        ce  = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt  = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ── Model ────────────────────────────────────────────────────────────────────

class HFACSCausalLSTM(nn.Module):
    """
    Four-step causal LSTM following the consolidated HFACS DAG.

    Step A: [Organizational Climate | Employment]  dim=2
              → predict A: Supervisory Conditions
    Step 0: [Weather | TimeOfDay | SkyCondNonceil | Personnel | Supervisory]
              → predict B: Operator Conditions
    Step 1: embed_proj([soft_B | Supervisory])
              → predict C: Unsafe Acts
    Step 2: embed_proj_D([soft_C | Supervisory])
              → predict D: Severity (4-class categorical, 0..3)
    """

    STEP_A_SIZE = 2  # Organizational Climate (encoded), Employment (numerical)
    STEP0_SIZE  = 5  # 5 label-encoded categorical inputs

    def __init__(self, hidden_size, n_A, n_B, n_C, n_D, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size

        self.cell_a     = nn.LSTMCell(self.STEP_A_SIZE, hidden_size)
        self.cell_0     = nn.LSTMCell(self.STEP0_SIZE,  hidden_size)
        self.cell_1     = nn.LSTMCell(self.STEP0_SIZE,  hidden_size)
        self.cell_2     = nn.LSTMCell(self.STEP0_SIZE,  hidden_size)
        self.drop       = nn.Dropout(dropout)

        # Project [soft_B (n_B) | Supervisory (1)] → STEP0_SIZE for cell_1
        self.embed_proj = nn.Linear(n_B + 1, self.STEP0_SIZE)
        # Project [soft_C (n_C) | Supervisory (1)] → STEP0_SIZE for cell_2
        self.embed_proj_D = nn.Linear(n_C + 1, self.STEP0_SIZE)

        self.head_A = nn.Linear(hidden_size, n_A)
        self.head_B = nn.Linear(hidden_size, n_B)
        self.head_C = nn.Linear(hidden_size, n_C)
        self.head_D = nn.Linear(hidden_size, n_D)

    def forward(self, step_a, step0):
        batch = step0.size(0)
        zeros = lambda: torch.zeros(batch, self.hidden_size, device=step0.device)

        # ── Step A → predict A (Supervisory Conditions) ───────────────────────
        hA, _        = self.cell_a(step_a, (zeros(), zeros()))
        logits_A     = self.head_A(self.drop(hA))

        # ── Step 0 → predict B (Operator Conditions) ─────────────────────────
        h0, c0       = self.cell_0(step0, (zeros(), zeros()))
        logits_B     = self.head_B(self.drop(h0))
        soft_B       = torch.softmax(logits_B, dim=1)

        # ── Step 1 → predict C (Unsafe Acts) ─────────────────────────────────
        supervisory  = step0[:, 4:5]
        step1_in     = self.embed_proj(torch.cat([soft_B.detach(), supervisory], dim=1))
        h1, c1       = self.cell_1(step1_in, (h0.detach(), c0.detach()))
        logits_C     = self.head_C(self.drop(h1))

        # ── Step 2 → predict D (Severity, 4-class categorical) ──────────────
        soft_C       = torch.softmax(logits_C, dim=1)
        step2_in     = self.embed_proj_D(torch.cat([soft_C.detach(), supervisory], dim=1))
        h2, _        = self.cell_2(step2_in, (h1.detach(), c1.detach()))
        logits_D     = self.head_D(self.drop(h2))

        return logits_A, logits_B, logits_C, logits_D


def train_epoch(model, loader, optimizer, crit_A, crit_B, crit_C, crit_D, device):
    """One epoch of training; returns (loss, bA, bB, bC, bD, bal_avg)."""
    model.train()
    total_loss = 0
    all_A, all_B, all_C, all_D = [], [], [], []
    pred_A, pred_B, pred_C, pred_D = [], [], [], []

    for batch in loader:
        s_a, s0, y_A, y_B, y_C, y_D = batch
        s_a, s0 = s_a.to(device), s0.to(device)
        y_A = y_A.to(device); y_B = y_B.to(device)
        y_C = y_C.to(device); y_D = y_D.to(device)

        optimizer.zero_grad()
        lA, lB, lC, lD = model(s_a, s0)

        loss = (
            1.5 * crit_A(lA, y_A)
            + 1.0 * crit_B(lB, y_B)
            + 1.5 * crit_C(lC, y_C)
            + 1.0 * crit_D(lD, y_D)
        )

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        all_A.extend(y_A.cpu().tolist());  pred_A.extend(lA.argmax(1).cpu().tolist())
        all_B.extend(y_B.cpu().tolist());  pred_B.extend(lB.argmax(1).cpu().tolist())
        all_C.extend(y_C.cpu().tolist());  pred_C.extend(lC.argmax(1).cpu().tolist())
        all_D.extend(y_D.cpu().tolist());  pred_D.extend(lD.argmax(1).cpu().tolist())

    bal_A   = balanced_accuracy_score(all_A, pred_A)
    bal_B   = balanced_accuracy_score(all_B, pred_B)
    bal_C   = balanced_accuracy_score(all_C, pred_C)
    bal_D   = balanced_accuracy_score(all_D, pred_D)
    bal_avg = (bal_A + bal_B + bal_C + bal_D) / 4
    return total_loss / len(loader), bal_A, bal_B, bal_C, bal_D, bal_avg


def evaluate(model, loader, device):
    """Inference helper; returns (all_A, all_B, all_C, all_D, pred_A, ...)."""
    model.eval()
    all_A,  all_B,  all_C,  all_D  = [], [], [], []
    pred_A, pred_B, pred_C, pred_D = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            s_a, s0, y_A, y_B, y_C, y_D = batch
            s_a = s_a.to(device); s0 = s0.to(device)
            lA, lB, lC, lD = model(s_a, s0)
            pred_A.extend(lA.argmax(1).cpu().tolist())
            pred_B.extend(lB.argmax(1).cpu().tolist())
            pred_C.extend(lC.argmax(1).cpu().tolist())
            pred_D.extend(lD.argmax(1).cpu().tolist())
            all_A.extend(y_A.tolist())
            all_B.extend(y_B.tolist())
            all_C.extend(y_C.tolist())
            all_D.extend(y_D.tolist())

    return all_A, all_B, all_C, all_D, pred_A, pred_B, pred_C, pred_D


def evaluate_with_confidence(model, loader, device):
    """
    Collect softmax probabilities for all four output heads.

    Returns a dict keyed by head ('A', 'B', 'C', 'D') with sub-keys
    'true' (int list), 'pred' (int list) and 'probs' (numpy array shape
    (N, n_classes)).
    """
    model.eval()
    keys = ("A", "B", "C", "D")
    bucket = {k: {"true": [], "pred": [], "probs": []} for k in keys}

    with torch.no_grad():
        for batch in loader:
            s_a, s0, y_A, y_B, y_C, y_D = batch
            s_a = s_a.to(device); s0 = s0.to(device)
            lA, lB, lC, lD = model(s_a, s0)
            for k, logits, y in zip(
                keys, (lA, lB, lC, lD), (y_A, y_B, y_C, y_D),
            ):
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                bucket[k]["probs"].append(probs)
                bucket[k]["pred"].extend(logits.argmax(1).cpu().tolist())
                bucket[k]["true"].extend(y.tolist())
    for k in keys:
        bucket[k]["probs"] = np.concatenate(bucket[k]["probs"], axis=0)
    return bucket


# ── Reusable training function ────────────────────────────────────────────────

def train_model(
    train_loader,
    encoders,
    val_loader=None,
    hidden_size=64,
    lr=3e-4,
    dropout=0.2,
    epochs=500,
    patience=200,
    n_D=None,
    device=None,
    verbose=True,
):
    """Train HFACSCausalLSTM and return (model, history)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_A = len(encoders.enc_supervisory.classes_)
    n_B = len(encoders.enc_operator.classes_)
    n_C = len(encoders.enc_unsafe.classes_)
    if n_D is None:
        n_D = len(encoders.enc_severity.classes_)

    model = HFACSCausalLSTM(
        hidden_size=hidden_size,
        n_A=n_A, n_B=n_B, n_C=n_C, n_D=n_D,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-5
    )

    train_set = train_loader.dataset
    crit_A = FocalLoss(weight=class_weights(train_set.y_A, n_A, device), gamma=1.0)
    crit_B = FocalLoss(weight=class_weights(train_set.y_B, n_B, device), gamma=2.0)
    crit_C = FocalLoss(weight=class_weights(train_set.y_C, n_C, device), gamma=1.0)
    crit_D = FocalLoss(weight=class_weights(train_set.y_D, n_D, device), gamma=1.0)

    history = {"loss": [], "acc_A": [], "acc_B": [], "acc_C": [], "acc_D": [], "acc_avg": []}

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0

    for epoch in range(1, epochs + 1):
        loss, acc_A, acc_B, acc_C, acc_D, acc_avg = train_epoch(
            model, train_loader, optimizer, crit_A, crit_B, crit_C, crit_D, device
        )
        history["loss"].append(loss)
        history["acc_A"].append(acc_A)
        history["acc_B"].append(acc_B)
        history["acc_C"].append(acc_C)
        history["acc_D"].append(acc_D)
        history["acc_avg"].append(acc_avg)

        val_loss = None
        if val_loader is not None:
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    s_a, s0, y_A, y_B, y_C, y_D = batch
                    s_a = s_a.to(device); s0 = s0.to(device)
                    y_A = y_A.to(device); y_B = y_B.to(device)
                    y_C = y_C.to(device); y_D = y_D.to(device)
                    lA, lB, lC, lD = model(s_a, s0)
                    batch_loss = (
                        1.5 * crit_A(lA, y_A)
                        + 1.0 * crit_B(lB, y_B)
                        + 1.5 * crit_C(lC, y_C)
                        + 1.0 * crit_D(lD, y_D)
                    )
                    total_val_loss += batch_loss.item()
            model.train()
            val_loss = total_val_loss / len(val_loader)
            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = copy.deepcopy(model.state_dict())
                no_improve    = 0
            else:
                no_improve += 1
        else:
            scheduler.step(loss)

        if verbose:
            current_lr = optimizer.param_groups[0]["lr"]
            val_str = (f"  ValLoss: {val_loss:.4f} (best {best_val_loss:.4f})"
                       if val_loader is not None else "")
            print(
                f"Epoch {epoch:03d}/{epochs} | Loss: {loss:.4f} | "
                f"BalAcc A: {acc_A:.2%}  B: {acc_B:.2%}  C: {acc_C:.2%}  "
                f"D: {acc_D:.2%}  Avg: {acc_avg:.2%} | "
                f"LR: {current_lr:.2e}{val_str}"
            )

        if val_loader is not None and no_improve >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch} (no val loss improvement for {patience} epochs).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    FILEPATH    = os.path.join(os.path.dirname(__file__), "..", "..", "data", "dataset.csv")
    HIDDEN_SIZE = 128
    BATCH_SIZE  = 32
    EPOCHS      = 500
    LR          = 1e-4
    DROPOUT     = 0.1
    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _, encoders = get_dataloaders(FILEPATH, batch_size=BATCH_SIZE)

    n_A = len(encoders.enc_supervisory.classes_)
    n_B = len(encoders.enc_operator.classes_)
    n_C = len(encoders.enc_unsafe.classes_)
    n_D = len(encoders.enc_severity.classes_)

    print("=" * 60)
    print("LSTM Causal Architecture — Input/Output per Step")
    print("=" * 60)
    print("Step A → predict: Supervisory Conditions")
    print(f"  Classes: {list(encoders.enc_supervisory.classes_)}")
    print("Step 0 → predict: Operator Conditions")
    print(f"  Classes: {list(encoders.enc_operator.classes_)}")
    print("Step 1 → predict: Unsafe Acts")
    print(f"  Classes: {list(encoders.enc_unsafe.classes_)}")
    print("Step 2 → predict: Severity (4-class)")
    print(f"  Classes: {list(encoders.enc_severity.classes_)}")
    print("=" * 60)
    print(f"Classes — A: {n_A}  B: {n_B}  C: {n_C}  D: {n_D}")
    print()

    model, history = train_model(
        train_loader, encoders,
        val_loader=val_loader,
        hidden_size=HIDDEN_SIZE, lr=LR, dropout=DROPOUT, epochs=EPOCHS,
        n_D=n_D,
        device=DEVICE, verbose=True,
    )

    # ── Final training summary ────────────────────────────────────────────────
    all_A, all_B, all_C, all_D, pred_A, pred_B, pred_C, pred_D = evaluate(
        model, train_loader, DEVICE
    )
    bal_A  = balanced_accuracy_score(all_A, pred_A)
    bal_B  = balanced_accuracy_score(all_B, pred_B)
    bal_C  = balanced_accuracy_score(all_C, pred_C)
    bal_D  = balanced_accuracy_score(all_D, pred_D)
    f1_A   = f1_score(all_A, pred_A, average="macro", zero_division=0)
    f1_B   = f1_score(all_B, pred_B, average="macro", zero_division=0)
    f1_C   = f1_score(all_C, pred_C, average="macro", zero_division=0)
    f1_D   = f1_score(all_D, pred_D, average="macro", zero_division=0)
    kap_A  = cohen_kappa_score(all_A, pred_A)
    kap_B  = cohen_kappa_score(all_B, pred_B)
    kap_C  = cohen_kappa_score(all_C, pred_C)
    kap_D  = cohen_kappa_score(all_D, pred_D)

    print(f"\n{'─' * 90}")
    print("Final Training Metrics")
    print(f"{'─' * 90}")
    print(f"{'Metric':<22} {'A (Sup.)':>14} {'B (Op.)':>14} {'C (Unsafe)':>14} {'D (Severity)':>18}")
    print(f"{'─' * 90}")
    print(f"{'Balanced Accuracy':<22} {bal_A:>14.2%} {bal_B:>14.2%} {bal_C:>14.2%} {bal_D:>18.2%}")
    print(f"{'Macro F1':<22} {f1_A:>14.4f} {f1_B:>14.4f} {f1_C:>14.4f} {f1_D:>18.4f}")
    print(f"{'Cohen Kappa':<22} {kap_A:>14.4f} {kap_B:>14.4f} {kap_C:>14.4f} {kap_D:>18.4f}")
    print(f"{'─' * 90}\n")

    # ── Plots ─────────────────────────────────────────────────────────────────
    epoch_range = range(1, len(history["loss"]) + 1)
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epoch_range, history["loss"], color="steelblue")
    ax1.set_title("Training Loss per Epoch")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.grid(True)

    ax2.plot(epoch_range, history["acc_A"], label="Supervisory (A)")
    ax2.plot(epoch_range, history["acc_B"], label="Operator (B)")
    ax2.plot(epoch_range, history["acc_C"], label="Unsafe Acts (C)")
    ax2.plot(epoch_range, history["acc_D"], label="Severity (D)")
    ax2.plot(epoch_range, history["acc_avg"], linestyle="--", color="black", label="Average")
    ax2.set_title("Training Balanced Accuracy per Epoch")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Balanced Accuracy")
    ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "..", "..", "figures", "lstm_training_curves.png")
    plt.savefig(plot_path)
    print(f"\nPlots saved to {plot_path}")
    plt.show()

    model_path = os.path.join(os.path.dirname(__file__), "hfacs_lstm.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()
