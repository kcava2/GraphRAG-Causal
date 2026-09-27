#!/usr/bin/env python
"""
leakage_audit.py — does a reduced query view still give the answer away?
=======================================================================
`build_query_views.py` is told not to mention outcomes or causes. This script
measures whether it obeyed, because a prompt instruction is a request, not a
guarantee. For every view of the TEST text it reports three things:

  transfer probe   A TF-IDF logistic model is trained on the FULL training
                   narratives to predict each target, then applied to the test text
                   under the view. This is the leak that matters for retrieval:
                   wording in the query that lines up with the wording of past
                   reports. On full test text severity scores ~0.97.
  within probe     5-fold cross-validated AUC using the test briefs alone. Catches
                   outcome wording the rewrite may have introduced in its own
                   vocabulary, which the transfer probe would not recognise.
  blocklist        Count of briefs still containing a forbidden term.

Two gates, one per view that is reported:

  L2 (the standard view)  must contain no outcome wording and no causal
      attribution: zero blocklist hits. Its severity AUC is reported but NOT gated,
      because L2 names the kind of event on purpose, and in this corpus the kind of
      event largely determines severity. That is why head D is not scored under L2.
  L1 (the lower bound)    must not predict severity: both probes at or below 0.65.

Reading it: the SEVERITY row is the leakage meter. Structured fields alone predict
severity at ~0.57 on this split; a clean L1 view should sit near that. The gate is
0.65, which with 202 records is inside the noise of 0.57 (95% half-width ~0.08).
L1b (circumstances INCLUDING the phase of flight at the event) is audited but not
gated: it carries no outcome wording and still predicts severity strongly, because
in this corpus the phase of flight all but determines the kind of event.
The B and C rows are not leakage — they are what an analyst could legitimately
read off the text at that stage, and they say how much signal each view keeps.

    python data/leakage_audit.py

Writes results/leakage_audit.csv and prints PASS/FAIL per view.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from ntsbdataloader import load_and_join, _split, PRECOND_SUBS, UNSAFE_SUBS  # noqa: E402
from standardize import strip_outcome                                         # noqa: E402
from build_query_views import blocklist_hits, OUT as VIEWS                    # noqa: E402

RESULTS = os.path.join(_HERE, "..", "results")
GATE = 0.65


def _targets(d):
    t = {"D severity": pd.to_numeric(d.severity_class).astype(int).to_numpy()}
    for g in PRECOND_SUBS:
        t["B " + g.replace("precond_", "")] = np.array([int(g in s) for s in d._pre])
    for u in UNSAFE_SUBS:
        t["C " + u.replace("unsafe_", "")] = np.array([int(u in s) for s in d._uns])
    return t


def main():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline

    df = load_and_join()
    tr, va, te = _split(df)
    trv = pd.concat([tr, va]).reset_index(drop=True)
    if not os.path.exists(VIEWS):
        raise SystemExit(f"{VIEWS} not found. Run: python data/build_query_views.py")
    v = pd.read_csv(VIEWS, dtype=str).fillna("").set_index("ev_id")
    ids = te["ev_id"].astype(str)
    have = ids.isin(v.index)
    if not have.all():
        print(f"WARNING: {int((~have).sum())} test records have no brief yet; "
              f"auditing the {int(have.sum())} that do.")
    te = te[have.to_numpy()].reset_index(drop=True)
    ids = te["ev_id"].astype(str)

    full = te.combined_text.fillna("").astype(str)
    col = lambda c: pd.Series([v.loc[e, c] if c in v.columns else "" for e in ids])
    views = {"full narrative": full,
             "outcome-stripped (old C2 query)": full.map(strip_outcome),
             "L2 preliminary": col("l2_text"),
             "L1b circumstances+phase": col("l1b_text"),
             "L1 pre-departure": col("l1_text")}
    tier_of = {"L2 preliminary": "L2", "L1b circumstances+phase": "L1b",
               "L1 pre-departure": "L1"}
    # A tier that is not fully written yet is left out rather than half-audited.
    views = {k: t for k, t in views.items()
             if k not in tier_of or t.str.strip().str.len().gt(0).all()}

    Ytr, Yte = _targets(trv), _targets(te)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=60000, sublinear_tf=True)
    Xtr = vec.fit_transform(trv.combined_text.fillna("").astype(str))
    models = {k: LogisticRegression(max_iter=3000, class_weight="balanced", C=3.0).fit(Xtr, y)
              for k, y in Ytr.items() if y.sum() >= 5}

    rows = []
    for name, text in views.items():
        X = vec.transform(text)
        hits = (sum(1 for t in text if blocklist_hits(t, tier_of[name]))
                if name in tier_of else np.nan)
        for k, m in models.items():
            y = Yte[k]
            if not 0 < y.sum() < len(y):
                continue
            transfer = roc_auc_score(y, m.predict_proba(X)[:, 1])
            within = np.nan
            if y.sum() >= 10:
                pipe = make_pipeline(
                    TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                    LogisticRegression(max_iter=3000, class_weight="balanced", C=3.0))
                cv = StratifiedKFold(5, shuffle=True, random_state=0)
                p = cross_val_predict(pipe, text.to_numpy(), y, cv=cv,
                                      method="predict_proba")[:, 1]
                within = roc_auc_score(y, p)
            rows.append({"view": name, "target": k, "transfer_auc": round(transfer, 3),
                         "within_auc": round(within, 3) if within == within else np.nan,
                         "mean_chars": int(text.str.len().mean()),
                         "briefs_with_blocklist_hit": hits})
    R = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    R.to_csv(os.path.join(RESULTS, "leakage_audit.csv"), index=False)

    pd.set_option("display.width", 200)
    for col, title in (("transfer_auc", "TRANSFER probe (trained on full train narratives)"),
                       ("within_auc", "WITHIN probe (5-fold CV on the test text itself)")):
        print(f"\n{title}")
        print(R.pivot(index="target", columns="view", values=col)[list(views)].to_string())
    print("\nBlocklist hits:",
          R.groupby("view")["briefs_with_blocklist_hit"].first().dropna().astype(int).to_dict())

    sev = R[R.target == "D severity"].set_index("view")
    ok = True
    print()
    hits = R.groupby("view")["briefs_with_blocklist_hit"].first()
    if "L2 preliminary" in sev.index:
        l2 = sev.loc["L2 preliminary"]
        l2_ok = int(hits.get("L2 preliminary", 0) or 0) == 0
        ok = ok and l2_ok
        print(f"L2  (standard view) wording gate: {int(hits.get('L2 preliminary', 0) or 0)} "
              f"briefs with outcome or cause wording -> {'PASS' if l2_ok else 'FAIL'}")
        print(f"L2  severity from the text alone: transfer {l2.transfer_auc:.3f}, within "
              f"{l2.within_auc:.3f} (reported, not gated: L2 names the kind of event, so "
              f"head D is not scored under it)")
    else:
        ok = False
        print("L2  (standard view): briefs missing -> FAIL. Run data/build_query_views.py")
    if "L1 pre-departure" in sev.index:
        l1 = sev.loc["L1 pre-departure"]
        worst = np.nanmax([l1.transfer_auc, l1.within_auc])
        l1_ok = worst <= GATE and int(hits.get("L1 pre-departure", 0) or 0) == 0
        ok = ok and l1_ok
        print(f"L1  (lower bound) severity gate: transfer {l1.transfer_auc:.3f}, within "
              f"{l1.within_auc:.3f} (gate {GATE}; structured-only is ~0.57) -> "
              f"{'PASS' if l1_ok else 'FAIL'}")
    if "L1b circumstances+phase" in sev.index:
        r = sev.loc["L1b circumstances+phase"]
        print(f"L1b (optional diagnostic): transfer {r.transfer_auc:.3f}, within "
              f"{r.within_auc:.3f} (not gated, not part of the standard run)")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
