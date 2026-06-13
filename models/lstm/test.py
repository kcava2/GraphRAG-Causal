"""
test.py  —  single-checkpoint TEST-split metrics (reconciled to the 5-head model)
=================================================================================
Loads one checkpoint, builds the test loader (with the matching retriever if the
checkpoint is a RAG condition), calls train.evaluate() and unpacks ALL 10 values
(5 heads: O/A/B/C multi-label + D severity), and prints per-head metrics.

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
from models.lstm.train import HFACSCausalLSTM, evaluate  # noqa: E402
from models.lstm import eval as E                       # noqa: E402  (reuse helpers)


def run(split_name, df_split, df_train, args):
    ck = torch.load(args.checkpoint, weights_only=False)
    cfg = ck["config"]; n_D = cfg["n_D"]
    is_rag = cfg["step_a_dim"] > N.N_O
    retr = E._retriever(args.strategy, args.model, {}) if is_rag else None
    encoders = NTSBEncoders(df_train)
    loader = E._loader(df_split, encoders, retr, args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HFACSCausalLSTM(**cfg).to(device)
    model.load_state_dict(ck["state_dict"]); model.eval()

    # exactly 10 values
    aO, aA, aB, aC, aD, pO, pA, pB, pC, pD = evaluate(model, loader, device)
    print(f"\n{split_name} metrics  ({len(aD)} records, "
          f"{'C4-style RAG' if is_rag else 'C1 no-RAG'})")
    for pre, y, p in (("O Organizational", aO, pO), ("A Supervisory", aA, pA),
                      ("B Preconditions", aB, pB), ("C Unsafe Acts", aC, pC)):
        k = pre.split()[0]
        m = E.head_ml_metrics(k, y, p)
        print(f"  {pre:18} microF1={m[k+'_microF1']:.3f} macroF1={m[k+'_macroF1']:.3f} "
              f"P={m[k+'_P']:.3f} R={m[k+'_R']:.3f} exact={m[k+'_exact']:.3f} "
              f"(pos={m[k+'_support']})")
    sev, _ = E.severity_metrics(aD, pD, n_D)
    print(f"  D Severity         acc={sev['D_accuracy']:.3f} bal_acc={sev['D_balanced_acc']:.3f} "
          f"macroF1={sev['D_macroF1']:.3f} kappa={sev['D_kappa']:.3f}")
    print(f"  chain_completion_rate="
          f"{E.chain_completion_rate(aO,aA,aB,aC,aD,pO,pA,pB,pC,pD):.3f}")
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
