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

The causal chain context -> B -> C -> D is preserved throughout. Losses are
class-balanced by default (focal + pos_weight, thresholds tuned on validation);
`--baseline` strips that, which on this data produces degenerate heads and is a
diagnostic rather than a reporting configuration.

Each condition is trained once per seed and evaluated as mean +/- sd. Unseeded
runs of the SAME condition differed by ~0.04 kappa on head D — larger than any
gap between conditions — so single runs cannot distinguish them.

    python run_conditions.py                          # all five, 5 seeds
    python run_conditions.py --only C1 C2 --seeds 0   # quick subset
    python run_conditions.py --baseline               # degenerate diagnostic

Needs Neo4j for C3/C4/C5 (structural retrieval) and data/*.faiss for C2/C4.
Writes results/seeds/c{n}_s{seed}.pt, then run models/lstm/eval_conditions.py.
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
SEED_DIR = os.path.join(RESULTS, "seeds")

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


def run_condition(name, args, device):
    """Train one condition once per seed.

    Retrieval does not depend on the seed, so the exemplar source and the loaders
    are built ONCE and reused across seeds. Rebuilding per seed would re-run
    structural Cypher over every record five times for no benefit. Only weight
    init and batch order vary, which is what `seed` controls.
    """
    ckpt, strategy, raw_mode, desc = CONDITIONS[name]
    header = f"{name}: {desc}"
    if strategy:
        header += f"   [strategy={strategy}{', raw' if raw_mode else ''}]"
    print(os.linesep + "=" * 68)
    print(header)
    print("=" * 68)

    source, retr = build_source(strategy, raw_mode)
    k = 0 if strategy is None else args.fewshot_k
    try:
        train_loader, val_loader, _test, encoders = get_dataloaders(
            filepath=args.input, batch_size=args.batch_size,
            retriever=None,                  # priors OFF — exemplars are the mechanism
            build_faiss=False,               # never rebuild data/ntsb.faiss: it is the
                                             # Stage-2 few-shot index, and rebuilding
                                             # it silently changes extraction prompts
            fewshot_k=k, fewshot_source=source)
    except Exception as e:
        print(f"  FAILED building data: {type(e).__name__}: {e}")
        return False
    finally:
        if retr is not None:
            try:
                retr.close()
            except Exception:
                pass

    os.makedirs(SEED_DIR, exist_ok=True)
    all_ok = True
    for seed in args.seeds:
        t0 = time.time()
        try:
            model, history, config, thresholds = train_model(
                train_loader, val_loader, encoders,
                hidden_size=args.hidden_size, lr=args.lr, dropout=args.dropout,
                epochs=args.epochs, device=device, arch="lstm",
                baseline=args.baseline, seed=seed, verbose=False)
            config.update(condition=name, strategy=strategy,
                          raw_mode=raw_mode, seed=seed)
            payload = {"state_dict": model.state_dict(), "config": config,
                       "thresholds": thresholds, "history": history}
            torch.save(payload, os.path.join(SEED_DIR, f"{name.lower()}_s{seed}.pt"))
            if seed == args.seeds[0]:        # first seed also lands at the plain name
                torch.save(payload, os.path.join(RESULTS, ckpt))
            print(f"  seed {seed}: trained ({time.time() - t0:.0f}s)")
        except Exception as e:
            print(f"  seed {seed}: FAILED {type(e).__name__}: {e}")
            all_ok = False
    return all_ok


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
    ap.add_argument("--baseline", action="store_true",
                    help="Strip class balancing: plain BCE/CE, fixed 0.5 thresholds. "
                         "DEGENERATE on this data (B all-ones at 72%% positive, C "
                         "all-zeros at 9%%) - a diagnostic, not for reporting.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                    help="Train each condition once per seed; eval reports mean +/- sd.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = args.only or list(CONDITIONS)
    print(f"Device: {device} | conditions: {', '.join(names)} | "
          f"fewshot_k={args.fewshot_k} | seeds={args.seeds} | "
          f"{'baseline (degenerate)' if args.baseline else 'class-balanced'} losses")

    ok = {n: run_condition(n, args, device) for n in names}

    print(os.linesep + "=" * 68)
    for n, good in ok.items():
        print(f"  {n:<4}{'trained' if good else 'FAILED'}   {CONDITIONS[n][3]}")
    if all(ok.values()):
        print(os.linesep + "Next:  python models/lstm/eval_conditions.py")
    else:
        print(os.linesep + "Some conditions failed — see the messages above.")
    return 0 if all(ok.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
