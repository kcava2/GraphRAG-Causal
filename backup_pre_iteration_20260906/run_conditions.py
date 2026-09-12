#!/usr/bin/env python
"""
run_conditions.py — train the five deep-learning conditions
===========================================================
Retrieval-STRATEGY ablation over the sequential causal LSTM (spec 2.2/2.3):

    C1  Original LSTM, no RAG                 baseline
    C2  LSTM + Semantic RAG    (faiss)        Strategy A — narrative similarity
    C3  LSTM + Structural RAG  (cypher)       Strategy B — shared context nodes
    C4  LSTM + Hybrid RAG      (faiss+cypher) Strategy C — 50/50 combination
    C5  LSTM + Raw RAG, no text mining        hybrid, LLM-mined labels removed

Retrieval reaches the model as **few-shot exemplars**, not priors: each retrieved
neighbour enters as an intact (features, labels) row that the FewShotEncoder reads
(see GraphFewShotSource). C5 is the exemplar equivalent of `--no-factor-priors` —
the neighbours' y_B/y_C columns are zeroed and only their structured severity
survives, isolating what retrieval contributes without any LLM-mined content.

The model is the BASELINE configuration: the ctx -> B -> C -> D causal chain is
kept, but the accumulated machinery (focal loss, class weighting, tuned decision
thresholds) is stripped, so the conditions are compared on the architecture rather
than on tuning.

    python run_conditions.py                       # all five
    python run_conditions.py --only C1 C2          # a subset
    python run_conditions.py --epochs 200          # shorter run

Needs Neo4j for C3/C4/C5 (structural retrieval) and data/*.faiss for C2/C4.
Writes results/c1.pt .. results/c5.pt, then run models/lstm/eval_conditions.py.
"""

import argparse
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "data"))

from data.ntsbdataloader import get_dataloaders, GraphFewShotSource, NTSB_CLEAN  # noqa: E402
from models.lstm.train import train_model  # noqa: E402

RESULTS = os.path.join(_HERE, "results")

# name -> (checkpoint, retrieval strategy or None, raw_mode, description)
CONDITIONS = {
    "C1": ("c1.pt", None,     False, "Original LSTM, no RAG"),
    "C2": ("c2.pt", "faiss",  False, "Semantic RAG (Strategy A)"),
    "C3": ("c3.pt", "cypher", False, "Structural RAG (Strategy B)"),
    "C4": ("c4.pt", "hybrid", False, "Hybrid RAG (Strategy C)"),
    "C5": ("c5.pt", "hybrid", True,  "Raw RAG, no text mining"),
}


def build_source(strategy, raw_mode):
    """GraphFewShotSource for a retrieval condition; None for C1."""
    if strategy is None:
        return None, None
    from data.rag_retriever import build_retriever
    retr = build_retriever(strategy=strategy)
    return GraphFewShotSource(retr, raw_mode=raw_mode), retr


def run_one(name, args, device):
    ckpt, strategy, raw_mode, desc = CONDITIONS[name]
    print("\n" + "=" * 68)
    print(f"{name}: {desc}"
          + (f"   [strategy={strategy}{', raw' if raw_mode else ''}]" if strategy else ""))
    print("=" * 68)

    source, retr = build_source(strategy, raw_mode)
    k = 0 if strategy is None else args.fewshot_k
    t0 = time.time()
    try:
        train_loader, val_loader, _test, encoders = get_dataloaders(
            filepath=args.input, batch_size=args.batch_size,
            retriever=None,                  # priors OFF — exemplars are the mechanism
            build_faiss=False,               # never rebuild data/ntsb.faiss: it is the
                                             # Stage-2 few-shot index and rebuilding it
                                             # silently changes extraction prompts
            fewshot_k=k, fewshot_source=source)

        model, history, config, thresholds = train_model(
            train_loader, val_loader, encoders,
            hidden_size=args.hidden_size, lr=args.lr, dropout=args.dropout,
            epochs=args.epochs, device=device, arch="lstm",
            baseline=not args.no_baseline)

        config["condition"] = name
        config["strategy"] = strategy
        config["raw_mode"] = raw_mode
        os.makedirs(RESULTS, exist_ok=True)
        path = os.path.join(RESULTS, ckpt)
        torch.save({"state_dict": model.state_dict(), "config": config,
                    "thresholds": thresholds, "history": history}, path)
        print(f"  saved {path}   ({time.time() - t0:.0f}s)")
        return True
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        return False
    finally:
        if retr is not None:
            try:
                retr.close()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser(description="Train the five LSTM conditions.")
    ap.add_argument("--input", default=NTSB_CLEAN)
    ap.add_argument("--only", nargs="+", choices=list(CONDITIONS), default=None,
                    help="Subset of conditions (default: all five).")
    ap.add_argument("--fewshot-k", type=int, default=5,
                    help="Exemplars retrieved per record for C2-C5.")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--hidden-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--no-baseline", action="store_true",
                    help="Keep focal loss, class weighting and tuned thresholds "
                         "instead of the stripped-back baseline configuration.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = args.only or list(CONDITIONS)
    print(f"Device: {device} | conditions: {', '.join(names)} | "
          f"fewshot_k={args.fewshot_k} | "
          f"{'baseline' if not args.no_baseline else 'full'} configuration")

    ok = {n: run_one(n, args, device) for n in names}

    print("\n" + "=" * 68)
    for n, good in ok.items():
        print(f"  {n:<4}{'trained' if good else 'FAILED'}   {CONDITIONS[n][3]}")
    if all(ok.values()):
        print("\nNext:  python models/lstm/eval_conditions.py")
    else:
        print("\nSome conditions failed — see the messages above before evaluating.")
    return 0 if all(ok.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
