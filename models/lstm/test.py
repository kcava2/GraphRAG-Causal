"""
test.py  —  single-checkpoint TEST-split metrics (3-head model)
=================================================================================
Loads one checkpoint, builds the test loader (with the matching retriever if the
checkpoint is a RAG condition), calls train.evaluate() and unpacks ALL 6 values
(3 heads: B/C multi-label + D severity; org/sup is a structured context input),
applies the checkpoint's tuned thresholds, and prints per-step metrics.

Usage:
  python models/lstm/test.py --input data/ntsb_subset.csv --checkpoint results/c1.pt
  python models/lstm/test.py --input data/ntsb_subset.csv --checkpoint results/c4.pt --model qwen2.5:3b
"""

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..", "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "data"))
import ntsbdataloader as N                              # noqa: E402
from ntsbdataloader import NTSBEncoders, load_and_join, _split  # noqa: E402
from models.lstm.train import make_model, evaluate  # noqa: E402
from models.lstm import eval as E                       # noqa: E402  (reuse helpers)


def run(split_name, df_split, df_train, args):
    ck = torch.load(args.checkpoint, weights_only=False)
    cfg = ck["config"]
    if "step_ctx_dim" not in cfg:
        raise SystemExit("Pre-redesign checkpoint — retrain with the current train.py.")
    n_D = cfg["n_D"]
    thr = ck.get("thresholds")
    is_rag = cfg["step_b_dim"] > N.STEP_B_BASE     # precond prior appended -> larger step_b
    retr = E._retriever(args.strategy, args.model, {}) if is_rag else None
    encoders = NTSBEncoders(df_train)
    loader = E._loader(df_split, encoders, retr, args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(cfg).to(device)
    model.load_state_dict(ck["state_dict"]); model.eval()

    # exactly 6 values (B/C multi-label + D severity)
    aB, aC, aD, pB, pC, pD = evaluate(model, loader, device, thr)
    print(f"\n{split_name} metrics  ({len(aD)} records, "
          f"{'C4-style RAG' if is_rag else 'C1 no-RAG'})")
    for pre, y, p in (("B Preconditions", aB, pB), ("C Unsafe Acts", aC, pC)):
        k = pre.split()[0]
        m = E.ml_metrics(k, y, p)
        print(f"  {pre:18} F1={m[k+'_F1']:.3f} acc={m[k+'_accuracy']:.3f} "
              f"(pos={m[k+'_support']})")
    sev, _ = E.severity_metrics(aD, pD, n_D)
    print(f"  D Severity         F1={sev['D_F1']:.3f} acc={sev['D_accuracy']:.3f} "
          f"bal_acc={sev['D_balanced_acc']:.3f} kappa={sev['D_kappa']:.3f}")
    print(f"  chain_completion_rate="
          f"{E.chain_completion_rate(aB,aC,aD,pB,pC,pD):.3f}")
    if retr is not None:
        retr.close()


def main():
    ap = argparse.ArgumentParser(description="Single-checkpoint TEST metrics")
    ap.add_argument("--input", default=N.NTSB_CLEAN)
    ap.add_argument("--checkpoint", default=os.path.join(_HERE, "hfacs_lstm.pt"))
    ap.add_argument("--model", default=None, help="Ollama model for RAG Cypher.")
    ap.add_argument("--strategy", default="hybrid",
                    help="Retriever strategy if the checkpoint is a RAG model.")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    E.EU._utf8()

    df = load_and_join(args.input)
    df_train, _, df_test = _split(df)
    run("TEST", df_test, df_train, args)


if __name__ == "__main__":
    main()
