"""
val.py  —  single-checkpoint VALIDATION-split metrics (reconciled to the 5-head model)
======================================================================================
Same as test.py but reports on the VALIDATION split (useful for model selection /
sanity-checking before the held-out test report). Reuses the eval helpers; calls
train.evaluate() and unpacks all 10 values.

Usage:
  python models/lstm/val.py --input data/ntsb_subset.csv --checkpoint results/c1.pt
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
from ntsbdataloader import load_and_join, _split        # noqa: E402
from models.lstm import test as T                       # noqa: E402  (reuse run())


def main():
    ap = argparse.ArgumentParser(description="Single-checkpoint VALIDATION metrics")
    ap.add_argument("--input", default=N.NTSB_CLEAN)
    ap.add_argument("--checkpoint", default=os.path.join(_HERE, "hfacs_lstm.pt"))
    ap.add_argument("--model", default=None, help="Ollama model for RAG Cypher.")
    ap.add_argument("--strategy", default="hybrid",
                    help="Retriever strategy if the checkpoint is a RAG model.")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    T.E.EU._utf8()

    df = load_and_join(args.input)
    df_train, df_val, _ = _split(df)
    T.run("VAL", df_val, df_train, args)


if __name__ == "__main__":
    main()
