"""
test.py — write LSTM predictions for the held-out test split, including the
Severity (D) head alongside the three HFACS tier tasks.

Requirements: neo4j, ollama, pandas, torch, scikit-learn, anthropic (optional)
"""

import os
import sys
import torch
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from data.real_dataloader import get_dataloaders          # noqa: E402
from models.lstm.train import HFACSCausalLSTM, evaluate   # noqa: E402


def main():
    FILEPATH   = os.path.join(os.path.dirname(__file__), "..", "..", "data", "dataset.csv")
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "hfacs_lstm.pt")
    OUT_PATH   = os.path.join(os.path.dirname(__file__), "..", "..", "results", "lstm_test_predictions.csv")
    BATCH_SIZE = 32
    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, encoders = get_dataloaders(FILEPATH, batch_size=BATCH_SIZE)

    n_A = len(encoders.enc_supervisory.classes_)
    n_B = len(encoders.enc_operator.classes_)
    n_C = len(encoders.enc_unsafe.classes_)
    n_D = len(encoders.enc_severity.classes_)

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    sd = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    hidden_size = sd["cell_a.weight_hh"].shape[1]

    model = HFACSCausalLSTM(
        hidden_size=hidden_size, n_A=n_A, n_B=n_B, n_C=n_C, n_D=n_D,
    ).to(DEVICE)
    model.load_state_dict(sd)

    all_A, all_B, all_C, all_D, pred_A, pred_B, pred_C, pred_D = evaluate(
        model, test_loader, DEVICE,
    )

    results = pd.DataFrame({
        "true_supervisory": encoders.enc_supervisory.inverse_transform(all_A),
        "pred_supervisory": encoders.enc_supervisory.inverse_transform(pred_A),
        "true_operator":    encoders.enc_operator.inverse_transform(all_B),
        "pred_operator":    encoders.enc_operator.inverse_transform(pred_B),
        "true_unsafe_acts": encoders.enc_unsafe.inverse_transform(all_C),
        "pred_unsafe_acts": encoders.enc_unsafe.inverse_transform(pred_C),
        "true_severity":    encoders.enc_severity.inverse_transform(all_D),
        "pred_severity":    encoders.enc_severity.inverse_transform(pred_D),
    })

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    results.to_csv(OUT_PATH, index=False)
    print(f"Test samples: {len(results)}")
    print(f"Predictions saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
