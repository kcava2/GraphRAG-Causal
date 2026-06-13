"""
system_eval.py  —  end-to-end system inspection report (NOT Stage-6 evaluation)
===============================================================================
For a few specific NTSB records, shows the whole pipeline working together so you
can eyeball performance:

  1. The record (severity, structured features, narrative)
  2. HFACS extraction labels (the multi-label "truth" from hfacs_results.csv)
  3. Few-shot block the extractor would inject for this narrative
  4. RAG retrieval — most-similar ASIAS/ASRS accidents (FAISS) + their KG factors
  5. RAG priors — soft O/A/B/C distributions appended to the LSTM
  6. LSTM output — predicted probabilities for the full causal chain O->A->B->C->D
  7. Predicted-vs-true agreement (per-step Jaccard) + severity hit

Requires the artifacts from Stages 1-5 (ntsb_clean.csv, hfacs_results.csv,
ntsb/asias/asrs .faiss, the Neo4j KG, and models/lstm/hfacs_lstm.pt) plus
NEO4J_PASSWORD in the environment.

Usage:
  python system_eval.py --records 3
  python system_eval.py --ev-ids 20080107X00026,20080109X00036
  python system_eval.py --records 3 --model qwen2.5:7b
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ntsbdataloader as N          # noqa: E402
import hfacs_extractor as H         # noqa: E402
from rag_retriever import build_retriever            # noqa: E402
from models.lstm.train import HFACSCausalLSTM        # noqa: E402

_SEP = "=" * 78


def _split(df, seed=42, test=0.2, val=0.1):
    n = len(df)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    n_test, n_val = int(n * test), int(n * val)
    n_train = n - n_test - n_val
    return (df.iloc[perm[:n_train]].reset_index(drop=True),
            df.iloc[perm[n_train + n_val:]].reset_index(drop=True))


def _topk(probs, vocab, k=4):
    idx = np.argsort(probs)[::-1][:k]
    return [(vocab[i], float(probs[i])) for i in idx]


def _true(rowset, vocab):
    return [v for v in vocab if v in rowset] or ["(none)"]


def _jaccard(pred_set, true_set):
    if not pred_set and not true_set:
        return 1.0
    u = pred_set | true_set
    return len(pred_set & true_set) / len(u) if u else 0.0


def _src_narratives():
    """event_id -> (source, narrative snippet) for retrieved ASIAS/ASRS events."""
    import pandas as pd
    out = {}
    a = pd.read_csv(os.path.join("data", "asias_clean.csv"), dtype=str)
    for _, r in a.iterrows():
        out[str(r["accident_id"])] = ("ASIAS", str(r.get("combined_narrative", ""))[:140])
    s = pd.read_csv(os.path.join("data", "asrs_clean.csv"), dtype=str)
    for _, r in s.iterrows():
        out[str(r["acn"])] = ("ASRS", (str(r.get("narrative", "")) + " " +
                                       str(r.get("synopsis", "")))[:140])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=int, default=3)
    ap.add_argument("--ev-ids", default=None, help="comma-separated ev_ids to inspect")
    ap.add_argument("--strategy", default="hybrid")
    ap.add_argument("--model", default=None, help="Ollama model for retriever Cypher")
    ap.add_argument("--no-rag", action="store_true", help="skip retriever (C1-only)")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # ---- data, split, encoders (mirror training) ----
    df = N.load_and_join()
    df_train, df_test = _split(df)
    enc = N.NTSBEncoders(df_train)
    narr_map = _src_narratives()

    # ---- few-shot cache (read-only) ----
    H._seed_caches_from_existing(H.RESULTS_CSV)
    for _, r in df_train.iterrows():
        H._SNIPPET_CACHE[str(r["ev_id"])] = H._clean(r.get("combined_text"))

    # ---- model ----
    ckpt = torch.load(os.path.join("models", "lstm", "hfacs_lstm.pt"), weights_only=False)
    model = HFACSCausalLSTM(**ckpt["config"]); model.load_state_dict(ckpt["state_dict"]); model.eval()
    is_c4 = ckpt["config"]["step_a_dim"] > N.N_O          # priors appended?

    # ---- retriever ----
    retriever = None
    if not args.no_rag:
        kw = {"model": args.model} if args.model else {}
        retriever = build_retriever(strategy=args.strategy, **kw)

    # ---- pick records ----
    if args.ev_ids:
        want = set(args.ev_ids.split(","))
        sample = df[df["ev_id"].astype(str).isin(want)].reset_index(drop=True)
    else:
        sample = df_test.head(args.records).reset_index(drop=True)

    ds = N.NTSBSequenceDataset(sample, enc, retriever=retriever if is_c4 else None)

    print(f"\n{_SEP}\nSYSTEM EVALUATION — {len(sample)} records | model={ckpt['config']} "
          f"| RAG={'on' if (retriever and is_c4) else 'off'}\n{_SEP}")

    agg = {"O": [], "A": [], "B": [], "C": [], "sev": []}
    for i in range(len(sample)):
        row = sample.iloc[i]
        ev = str(row["ev_id"])
        narrative = H._clean(row.get("combined_text"))
        print(f"\n{_SEP}\nRECORD {i+1}: ev_id={ev}  severity_class(true)={row['severity_class']}")
        print(f"  features: visual={row['visual_condition']} light={row['light_conditions']} "
              f"person={row['person_involved']} hours={row['pilot_hours_bracket']} "
              f"emp={row['employment_bracket']} fuel={row['fuel_bracket']}")
        print(f"  narrative: {narrative[:200].replace(chr(10),' ')}...")

        # 1) HFACS extraction truth
        print("\n  [1] HFACS extraction (multi-label truth from hfacs_results.csv):")
        for label, col, vocab in (("O org", "_org", N.ORG_SUBS), ("A sup", "_sup", N.SUP_SUBS),
                                   ("B pre", "_pre", N.PRECOND_SUBS), ("C unsafe", "_uns", N.UNSAFE_SUBS)):
            print(f"      {label:10}: {_true(row[col], vocab)}")

        # 2) Few-shot block
        fb = H.get_ntsb_fewshot_examples(narrative, n=2)
        print(f"\n  [2] Few-shot block: {len(fb)} chars" + (" (empty — no labeled neighbors)" if not fb else ""))
        if fb:
            print("      " + fb[:240].replace(chr(10), " ") + " ...")

        # 3) RAG retrieval — similar accidents
        if retriever is not None:
            fs = retriever._faiss_scores(narrative)
            print("\n  [3] RAG retrieval — most similar ASIAS/ASRS accidents (FAISS):")
            for (eid, src), sc in list(fs.items())[:3]:
                facs = retriever._fetch_factors(eid, src)
                snip = narr_map.get(eid, (src, ""))[1]
                print(f"      {src} {eid} (sim={sc:.2f}) factors={facs[:4]}")
                print(f"          {snip}...")

        # 4) RAG priors (slice the appended portion of the step tensors)
        s_o, s_a, s_b, yO, yA, yB, yC, yD = ds[i]
        if is_c4:
            org_p = s_o[7:].numpy(); sup_p = s_a[N.N_O:].numpy(); pre_p = s_b[N.STEP_B_BASE:].numpy()
            print("\n  [4] RAG priors (appended to LSTM input):")
            print("      org top:", [(v, round(p, 3)) for v, p in _topk(org_p, N.ORG_SUBS, 3)])
            print("      sup top:", [(v, round(p, 3)) for v, p in _topk(sup_p, N.SUP_SUBS, 3)])
            print("      pre top:", [(v, round(p, 3)) for v, p in _topk(pre_p, N.PRECOND_SUBS, 3)])
        else:
            print("\n  [4] RAG priors: model is C1 (no priors).")

        # 5) LSTM output — predicted causal chain
        with torch.no_grad():
            lO, lA, lB, lC, lD = model(s_o.unsqueeze(0), s_a.unsqueeze(0), s_b.unsqueeze(0))
        pO, pA, pB, pC = (torch.sigmoid(x)[0].numpy() for x in (lO, lA, lB, lC))
        pD = torch.softmax(lD[0], 0).numpy()
        print("\n  [5] LSTM predicted causal chain (probabilities):")
        print("      O Organizational:", [(v, round(p, 2)) for v, p in _topk(pO, N.ORG_SUBS, 3)])
        print("      A Supervisory   :", [(v, round(p, 2)) for v, p in _topk(pA, N.SUP_SUBS, 3)])
        print("      B Preconditions :", [(v, round(p, 2)) for v, p in _topk(pB, N.PRECOND_SUBS, 3)])
        print("      C Unsafe Acts   :", [(v, round(p, 2)) for v, p in _topk(pC, N.UNSAFE_SUBS, 3)])
        print(f"      D Severity      : class {int(pD.argmax())} (probs {pD.round(2).tolist()})")

        # 6) agreement (predicted>0.5 vs true)
        def pset(probs, vocab):
            return {vocab[j] for j in range(len(vocab)) if probs[j] > 0.5}
        jO = _jaccard(pset(pO, N.ORG_SUBS), set(row["_org"]))
        jA = _jaccard(pset(pA, N.SUP_SUBS), set(row["_sup"]))
        jB = _jaccard(pset(pB, N.PRECOND_SUBS), set(row["_pre"]))
        jC = _jaccard(pset(pC, N.UNSAFE_SUBS), set(row["_uns"]))
        sev_true = int(float(row["severity_class"]))
        sev_hit = int(int(pD.argmax()) == enc.enc_severity.transform([sev_true])[0]
                      if sev_true in set(enc.enc_severity.classes_) else 0)
        print(f"\n  [6] agreement (Jaccard pred>0.5 vs true): "
              f"O={jO:.2f} A={jA:.2f} B={jB:.2f} C={jC:.2f} | severity_hit={sev_hit}")
        for k, v in zip(("O", "A", "B", "C", "sev"), (jO, jA, jB, jC, sev_hit)):
            agg[k].append(v)

    print(f"\n{_SEP}\nAGGREGATE over {len(sample)} records (mean):")
    print("  Jaccard  O={:.2f} A={:.2f} B={:.2f} C={:.2f} | severity_acc={:.2f}".format(
        *[np.mean(agg[k]) if agg[k] else 0.0 for k in ("O", "A", "B", "C", "sev")]))
    print("  NOTE: values reflect the smoke-trained model; this tool is for inspection, "
          "not Stage-6 metrics.\n" + _SEP)

    if retriever is not None:
        retriever.close()


if __name__ == "__main__":
    main()
