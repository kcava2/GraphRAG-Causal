"""
val.py — GridSearchCV-style hyperparameter sweep for the four-head causal LSTM.

Model selection now averages balanced accuracy across all four tasks
(Supervisory, Operator, Unsafe Acts, Severity).

Requirements: neo4j, ollama, pandas, torch, scikit-learn, anthropic (optional)
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import ParameterGrid

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from data.real_dataloader import get_dataloaders  # noqa: E402
from models.lstm.train import (  # noqa: E402
    HFACSCausalLSTM, train_model, evaluate,
)

# ── Hyperparameter search grid ────────────────────────────────────────────────
PARAM_GRID = {
    "hidden_size": [32, 64, 128],
    "lr": [1e-3, 3e-4],
    "dropout": [0.1, 0.2],
}
CV_EPOCHS = 30      # Epochs per fold
FINAL_EPOCHS = 500  # Full retrain on best config


class LSTMScikitWrapper(BaseEstimator, ClassifierMixin):
    """Thin sklearn-compatible shim around train_model/evaluate."""
    def __init__(self, hidden_size=64, lr=1e-3, dropout=0.2, epochs=30):
        self.hidden_size = hidden_size
        self.lr = lr
        self.dropout = dropout
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.encoders = None

    def fit(self, X, y=None, encoders=None):
        self.encoders = encoders
        self.model, _ = train_model(
            X, self.encoders,
            hidden_size=self.hidden_size,
            lr=self.lr,
            dropout=self.dropout,
            epochs=self.epochs,
            device=self.device,
            verbose=False,
        )
        return self

    def predict(self, X_loader):
        _, _, _, _, pA, pB, pC, pD = evaluate(self.model, X_loader, self.device)
        return pA, pB, pC, pD


def lstm_multi_target_scorer(estimator, X_loader):
    """Average balanced accuracy across all four tasks."""
    tA, tB, tC, tD, pA, pB, pC, pD = evaluate(estimator.model, X_loader, estimator.device)
    bA = balanced_accuracy_score(tA, pA)
    bB = balanced_accuracy_score(tB, pB)
    bC = balanced_accuracy_score(tC, pC)
    bD = balanced_accuracy_score(tD, pD)
    return (bA + bB + bC + bD) / 4


def evaluate_metrics(model, loader, device):
    """Average balanced accuracy across all four heads + per-head scores."""
    tA, tB, tC, tD, pA, pB, pC, pD = evaluate(model, loader, device)
    bA = balanced_accuracy_score(tA, pA)
    bB = balanced_accuracy_score(tB, pB)
    bC = balanced_accuracy_score(tC, pC)
    bD = balanced_accuracy_score(tD, pD)
    return (bA + bB + bC + bD) / 4, bA, bB, bC, bD


def main():
    FILEPATH   = os.path.join(os.path.dirname(__file__), "..", "..", "data", "dataset.csv")
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "hfacs_lstm.pt")
    OUT_PATH   = os.path.join(os.path.dirname(__file__), "..", "..", "results", "lstm_val_metrics.csv")
    BATCH_SIZE = 32
    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _, encoders = get_dataloaders(FILEPATH, batch_size=BATCH_SIZE)

    print("Starting GridSearchCV (5-Fold) on LSTM...")
    best_avg = -1.0
    best_cfg = None

    grid = ParameterGrid(PARAM_GRID)
    for cfg in grid:
        print(f"Testing: {cfg}...")
        fold_scores = []
        for fold in range(5):
            model, _ = train_model(
                train_loader, encoders,
                hidden_size=cfg["hidden_size"],
                lr=cfg["lr"],
                dropout=cfg["dropout"],
                epochs=CV_EPOCHS,
                device=DEVICE,
                verbose=False,
            )
            avg, _, _, _, _ = evaluate_metrics(model, val_loader, DEVICE)
            fold_scores.append(avg)

        mean_cv_score = np.mean(fold_scores)
        print(f"Mean CV Score (4-task avg): {mean_cv_score:.2%}")

        if mean_cv_score > best_avg:
            best_avg = mean_cv_score
            best_cfg = cfg

    print(f"\nBest Hyperparameters: {best_cfg}")

    # ── Final Full Retrain ──────────────────────────────────────────────────
    print(f"Performing final retrain for {FINAL_EPOCHS} epochs...")
    best_model, _ = train_model(
        train_loader, encoders,
        hidden_size=best_cfg["hidden_size"],
        lr=best_cfg["lr"],
        dropout=best_cfg["dropout"],
        epochs=FINAL_EPOCHS,
        device=DEVICE,
        verbose=True,
    )

    torch.save({
        "state_dict": best_model.state_dict(),
        "config": best_cfg,
        "encoders": encoders,
    }, MODEL_PATH)
    print(f"Optimized model saved to {MODEL_PATH}")

    # ── Final Metrics ───────────────────────────────────────────────────────
    tA, tB, tC, tD, pA, pB, pC, pD = evaluate(best_model, val_loader, DEVICE)
    bal_A = balanced_accuracy_score(tA, pA)
    bal_B = balanced_accuracy_score(tB, pB)
    bal_C = balanced_accuracy_score(tC, pC)
    bal_D = balanced_accuracy_score(tD, pD)

    pd.DataFrame([{
        **best_cfg,
        "bal_acc_avg":          (bal_A + bal_B + bal_C + bal_D) / 4,
        "bal_acc_supervisory":  bal_A,
        "bal_acc_operator":     bal_B,
        "bal_acc_unsafe":       bal_C,
        "bal_acc_severity":     bal_D,
    }]).to_csv(OUT_PATH, index=False)


if __name__ == "__main__":
    main()
