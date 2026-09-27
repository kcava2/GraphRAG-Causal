# GraphRAG-Causal

**Predicting the HFACS causal chain of commercial aviation accidents, with LLM text-mining, a Neo4j knowledge graph, and retrieval-augmented few-shot exemplars.**

This document is written for someone joining the project cold. It covers what the
system is trying to do, how each piece works, what the data is, where the numbers
currently stand, and — most importantly — an honest diagnosis of *why the numbers
are bad* and what to try first.

---

## Intent

Accident investigation reports are narrative documents. Safety analysts read them and
mentally map them onto **HFACS** (the Human Factors Analysis and Classification System):
organizational pressure enables unsafe supervision, which enables latent *preconditions*
in the crew, which produce *unsafe acts*, which produce an *outcome*.

The research question here is:

> Given only what is knowable *before* the narrative is written — the operating
> environment, the crew profile, the airline's economic pressure, and evidence from
> *similar past events* — can a model reconstruct that causal chain and predict how
> severe the outcome will be?

Concretely the system:

1. **Text-mines** HFACS factors out of accident narratives with a local LLM, turning
   free text into structured multi-label targets.
2. **Builds a knowledge graph** of events → HFACS factors → context, from a
   *different* corpus than the one it trains on.
3. **Trains a causal-chain model** whose structure mirrors the HFACS DAG:
   `context → Preconditions → Unsafe Acts → Severity`.
4. **Augments it with retrieval** — for each new event, pull the most similar past
   events and feed them to the model as **few-shot exemplars**: each retrieved
   neighbour enters as an intact (features, labels) pair, not as an averaged summary.
5. **Ablates** the retrieval sources and augmentation strategies against a no-RAG
   baseline to measure whether the graph/retrieval actually contributes anything.

---

## TL;DR of current state (iteration 2026-09-18, L2 standard since 2026-09-27)

Held-out test split, `n = 202`. Balanced accuracy, mean over **5 seeds**, k = 15
exemplars, four-tier head C, repaired violation labels. Full detail and the list of
what changed: [CHANGELOG_2026-09-18.md](CHANGELOG_2026-09-18.md).

**The standard view is L2.** The test record is looked up by a *preliminary brief*: the
circumstances of the flight, the kind of event and its sequence, with no injury or
damage wording and no causal attribution. That is what an analyst holds a few days to
two weeks after an occurrence. Only the 202 test records are rewritten; training text,
the exemplar pool, the knowledge graph, the indexes and the labels are untouched.

| head, L2 query | C1 no RAG | **C2 semantic** | C3 structural | C4 hybrid | C5 raw RAG |
|---|---|---|---|---|---|
| **B** preconditions | 0.514 | **0.596** | 0.526 | 0.546 | 0.535 |
| **C** unsafe acts, 3 error tiers | 0.525 | **0.671** | 0.536 | 0.663 | 0.642 |
| **C** incl. violation (2 test positives) | 0.515 | **0.623** | 0.524 | 0.616 | 0.602 |
| **D** severity | not scored | not scored | not scored | not scored | not scored |

Head D is not scored under L2 because injuries and damage are already known when a
preliminary report exists, and the L2 text alone predicts severity at AUC 0.95. Scored
on request (`--include-severity`) C2 reaches 0.883, which measures recognising the kind
of event, not predicting the outcome.

**Bounds, C2 semantic (same checkpoints, only the test records' retrieval text changes):**

| query view | B | C, 3 error tiers | D |
|---|---|---|---|
| full narrative incl. probable cause (upper bound, retrospective) | 0.597 | 0.700 | 0.899 |
| **L2 preliminary brief (standard)** | **0.596** | **0.671** | not scored |
| L1 pre-departure brief (lower bound, passes the severity leakage gate) | 0.507 | 0.513 | 0.559 |
| C1, no retrieval | 0.514 | 0.525 | 0.569 |

Read it as three findings. **Head C works now**: 0.53 (kappa 0.02) on the old binary
target, 0.67 (kappa 0.27) on the three error tiers at L2. **L2 costs little**: moving
from the full report to the preliminary brief takes head C from 0.70 to 0.67 and leaves
head B unchanged, so most of what retrieval contributes does not depend on reading the
outcome or the probable cause. **Nothing is predictable before departure**: with a query
that reads as a flight that has not happened yet, every head falls to the no-retrieval
baseline, severity included. Caveats that matter: Neo4j was not running for this run, so
C3-C5 used the in-distribution structural match only and the KG exemplar path is
untested; the strict violation tier has 21 positives in the corpus and 2 in the test
split, so it cannot be scored; severity and the head C tiers are correlated (decision
errors: 85% of high-severity events, 59% of low), so part of head C's skill is knowing
the kind of event, which L2 states outright.

---

## TL;DR before the 2026-09-18 iteration (binary head C, k = 5, kept for reference)

Held-out test split, `n = 202` NTSB Part-121 events. Mean +/- sd over **5 seeds**.
`*` = paired t-test vs C1 across seeds, p < 0.05. Conditions are a retrieval-STRATEGY
ablation; retrieval reaches the model as **few-shot exemplars**, not priors.

| head | metric | MAJORITY | C1 no RAG | **C2 semantic** | C3 structural | C4 hybrid | C5 raw RAG |
|---|---|---|---|---|---|---|---|
| **B** preconditions | F1 | 0.790 | 0.707 | **0.790\*** | 0.722 | 0.721 | 0.716 |
| | bal-acc | 0.500 | 0.506 | **0.596\*** | 0.514 | 0.517 | 0.513 |
| | kappa | 0.000 | 0.007 | **0.183\*** | 0.023 | 0.025 | 0.020 |
| **C** unsafe acts | F1 | 0.481 | 0.308 | **0.423\*** | 0.231 | 0.289 | 0.290 |
| | bal-acc | 0.500 | 0.475 | **0.529\*** | 0.465 | 0.438 | 0.450 |
| | kappa | 0.000 | −0.011 | **0.018\*** | −0.014 | −0.026 | −0.021 |
| **D** severity | F1 | 0.355 | 0.550 | **0.901\*** | 0.568 | 0.708\* | 0.708\* |
| | bal-acc | 0.500 | 0.555 | **0.897\*** | 0.571 | 0.708\* | 0.709\* |
| | kappa | 0.000 | 0.108 | **0.802\*** | 0.140 | 0.416\* | 0.418\* |

**Read balanced accuracy and kappa, not F1.** B's groups are ~72% positive, so a
constant all-ones predictor scores 0.790 micro-F1 while carrying no information — which
is exactly the MAJORITY row. C2 reaching the same F1 with bal-acc 0.596 and kappa 0.183
is the meaningful result.

Read the table as four findings:

- **Semantic retrieval (C2) improves all three heads significantly.** B and C are above
  their chance floors for the first time in this project.
- **C5 is the text-mining ablation.** Strip the LLM-mined HFACS labels from the
  exemplars and B falls 0.596 -> 0.513 and C 0.529 -> 0.450 (neither significant), while
  D holds at 0.709. **The mined content carries B and C; structured severity carries D.**
  This reverses the earlier C8 conclusion that mined content contributed nothing — that
  was measured on a pipeline where retrieval could not reach the model at all (see
  [ITERATION_2026-09-06.md](ITERATION_2026-09-06.md)).
- **Structural retrieval (C3) helps almost nothing**, and is significant on no metric
  except marginally on D (p = 0.055). It matches on pre-narrative context only.
- **D's 0.897 is NOT a defensible prediction result — see the warning below.**

### The severity caveat you must not skip

Retrieval finds neighbours by narrative similarity, and NTSB narratives *describe the
outcome*. Measured directly:

| what predicts severity | bal-acc | kappa |
|---|---|---|
| raw narrative (TF-IDF, no retrieval) | 0.923 | 0.858 |
| narrative with outcome sentences stripped | 0.914 | 0.839 |
| **retrieval on pre-narrative context only (C3)** | **0.571** | **0.140** |
| retrieval on narrative (C2) | 0.897 | 0.802 |

C2's severity result is substantially **outcome-text matching**, not prediction from
pre-narrative knowables. `strip_outcome()` was added to address this and **does not
work**: it removes only ~0.01 of the signal, because narratives reveal outcomes in far
more ways than damage and injury sentences. **C3's 0.571 is the number defensible
against the research question as written.** Fixing this properly needs an LLM pass that
rewrites narratives as circumstances-only; a regex cannot get there.

Full numbers: `results/conditions_{metrics,summary,stats,mcnemar}.csv`.
Figures: `figures/cond_{metrics_table,stats_table,performance,kappa}.png`.

---

## Repo map

```
data/
  build_carol_ntsb.py    Stage 1a  NTSB CAROL + Access DBs -> ntsb_clean.csv (Part-121)
  data_assembler.py      Stage 1   all raw sources -> {ntsb,asias,asrs}_clean.csv
  standardize.py         Stage 1   pure functions: the shared feature vocabulary
  sdr_defect_rate.py     Stage 1   FAA SDRs -> maintenance-reliability brackets
  build_lstm_corpus.py   Stage 1   NTSB + disjoint ASIAS slice -> lstm_corpus.csv
  hfacs_extractor.py     Stage 2   LLM text-mining -> hfacs_results.csv  (HFACS_SCHEMA lives here)
  kg_builder.py          Stage 3   Neo4j KG + asias/asrs/ntsb_kg FAISS indexes
  ntsbdataloader.py      Stage 4   corpus -> tensors; label spaces; train/val/test split
                                   (FewShotSource: retrieved exemplars live here)
  rag_retriever.py       Stage 5   hybrid FAISS + Cypher retrieval -> soft priors
                                   (legacy input-augmentation path)
  compare_extractions.py           per-tier prevalence of two extraction runs
  adjudicate_violation.py Stage 2b strict, consensus re-labelling of unsafe_violation
                                   -> violation_adjudication.csv (a label OVERRIDE)
  build_query_views.py   Stage 6b  L2 (standard) and L1 (lower bound) briefs for the
                                   TEST split only -> test_query_views.csv
  leakage_audit.py       Stage 6b  the gate: L2 free of outcome/cause wording, L1 must
                                   not predict severity
  ollama_json.py                   schema-constrained Ollama helper for the two above
  standardize.py         Stage 1   shared vocabulary + strip_outcome() for retrieval text
  hfacs_analysis.py      figures: extraction distributions / co-occurrence / coverage
  hfacs_tier_counts.py   figure:  event counts per HFACS tier
models/
  lstm/train.py          Stage 4   the causal model (LSTM and SCM variants) + training loop
  lstm/eval.py           Stage 6   cross-condition evaluation -> results/*.csv + figures
  lstm/test.py           single-checkpoint test-split metrics
  lstm/val.py            single-checkpoint validation-split metrics
  lstm/ensemble.py       Stage 6   RAG-as-a-model, blended at alpha tuned on val
  lstm/eval_conditions.py Stage 6  C1..C5 metrics per head AND per label, bootstrap
                                   intervals, paired t-tests, --query-view L2|L1|full
                                   (L2 is the default)
  causal_discovery.py    PC algorithm vs the theoretical HFACS DAG
  eval_utils.py          shared plotting for Stage 6
run_conditions.py        Stage 4   trains C1..C5 across seeds -> results/seeds/
select_subset.py         curates the per-source subsets (and the disjoint KG slice)
eval_lstm.py             single-checkpoint eval + figure (see the caveat in §11)
system_eval.py           end-to-end inspection of a few records — read this first
visualize.py             DAG schema, data quality, KG figures
results/                 c1.pt..c5.pt, seeds/, conditions_*.csv
figures/                 all generated PNGs
```

**If you are new: run `python system_eval.py --records 3` first.** It walks three real
records through every stage — record → extracted HFACS labels → extraction few-shot
block → retrieved neighbours → model output → agreement — and is the fastest way to
build a mental model of the pipeline. (It needs a checkpoint; see §11.)

> **Two different things are called "few-shot" in this project.** Stage 2 few-shot is
> a real LLM prompt block (retrieved narratives + their HFACS labels pasted into the
> extraction prompt, answering RQ1). Stage 4 few-shot is the neural analogue — an LSTM
> has no prompt, so retrieved (features, labels) pairs are encoded and fed to the model
> (answering RQ2). They are separate mechanisms at separate stages; neither implements
> the other.

---

## HFACS — the framework and how it is mapped here

HFACS organizes accident causation into four tiers, each *enabling* the next:

```
Organizational Influences -> Unsafe Supervision -> Preconditions -> Unsafe Acts -> Outcome
```

The canonical 15-tier taxonomy is defined once, in `HFACS_SCHEMA` at
[data/hfacs_extractor.py](data/hfacs_extractor.py#L78). It is the single source of
truth — the extractor validates against it, the KG stores its tiers, the dataloader
derives its label spaces from it. Every other module imports it. **Do not fork it.**

### How the four tiers are handled

| HFACS tier | Treatment here | Why |
|---|---|---|
| **Organizational / Supervisory** | *Not* text-mined, *not* predicted. Represented as **structured economic context** (`step_ctx`): airline employment, fuel cost, operating revenue and load factor, quarter-over-quarter. | Narratives almost never state organizational causes, so the mined labels were near-empty and the heads sat at chance. The economic proxy preserves the HFACS edge (organizational pressure → preconditions) without a data-starved head. |
| **Preconditions — physical environment** (`situational_phys`: Weather/Lighting/Terrain) | *Not* mined. Supplied as structured inputs (`visual_condition`, `light_conditions`). | These are recorded fields; mining them from text would be strictly worse. |
| **Preconditions — the rest** (operator mental/physical/limits, personnel CRM/readiness, situational tech) | **Head B.** Mined, then collapsed from 6 tiers to **3 multi-label groups**. | The raw tiers were unlearnably rare (`operator_limits` 4%, `personnel_readiness` 1% = 13 records). Groups: `precond_operator`, `precond_personnel`, `precond_situational`. |
| **Unsafe Acts** (skill / decision / perception / violation) | **Head C**, **four tiers, multi-label** (one sigmoid per tier, thresholds tuned on validation). | It was a binary violation-vs-error target while `unsafe_perception` sat at 6%. On the current extraction the tiers are 71 / 31 / 14 / 9%, and the binary target was the worst of them: 15 test positives and a label two extraction runs agree on at kappa 0.10. Multi-label because 23% of records carry two or more tiers. The violation tier is re-labelled under the strict HFACS definition by `data/adjudicate_violation.py`. |
| **Outcome** | **Head D**, binary severity (high / low). | See §5. |

Those collapses are defined at
[data/ntsbdataloader.py](data/ntsbdataloader.py#L60-L105). They were the right call for
learnability, but they are also why the results are hard to read: you are no longer
predicting HFACS, you are predicting a 3-bit summary of it.

### Extraction (Stage 2)

Two deterministic (`temperature=0`) local-Ollama calls per record, in two passes:

- **Pass 1 — Unsafe Acts.** Evidence-gated: the model must ground each label in the
  text; "at least one" is required, but blanket-all-four is rejected.
- **Pass 2 — Preconditions.** Latent operator/personnel/tech states *inferred from the
  unsafe acts found in pass 1*, since preconditions are almost never stated outright.

Every response is JSON-parsed best-effort and validated against `HFACS_SCHEMA`;
anything off-schema is silently dropped. Few-shot examples come from `ntsb.faiss`,
which is **built from the training split only**, so labelling a val/test record never
puts a val/test example in its prompt.

Output is one row per record in `data/hfacs_results.csv`:
`ev_id, entities_json, hfacs_json, relationships_json, extraction_status`.

**Extraction coverage on the 1013 NTSB records** (998 `success`, 15 `parse_error`):

| tier | events | rate |
|---|---|---|
| unsafe_skill | 608 | 61% |
| unsafe_decision | 361 | 36% |
| operator_mental | 202 | 20% |
| personnel_crm | 157 | 16% |
| operator_physical | 133 | 13% |
| unsafe_violation | 131 | 13% |
| situational_tech | 83 | 8% |
| unsafe_perception | 58 | 6% |
| operator_limits | 40 | 4% |
| personnel_readiness | 13 | 1% |

After grouping, head B's three targets have base rates of **20% / 15% / 8%**, and
**809 of 1013 records (80%) carry no precondition label at all**. Head C's positive
rate is **13%**. Hold that thought for §10.

---

## Data

Raw inputs live in `data/rawdata/` (gitignored — they are large and partly licensed).

| Source | Role | Cleaned file | Rows |
|---|---|---|---|
| **NTSB CAROL + `avall.mdb` / `pre2008.mdb`** | **The only training corpus.** Part-121 (commercial) events, 1999–2026. Narratives, findings, crew, flight time, weather/light, injuries, damage. | `data/ntsb_clean.csv` | **1013** |
| **FAA ASIAS** | KG / retrieval only. Never trains the model. | `data/asias_clean.csv` | 4819 |
| **NASA ASRS** | KG / retrieval only. Voluntary incident reports; no injury data. | `data/asrs_clean.csv` | ~97 MB |
| **BTS employment / fuel / operating revenue / load factor** | Macro-economic context, joined by year+month, expressed as QoQ % change + bracket. | merged into the above | — |
| **FAA Service Difficulty Reports (SDR-2000...2026)** | Maintenance reliability. Aggregated to `defects_per_tail(make, year)` and tertile-bracketed. KG context node only — never an LSTM feature. | `data/sdr_defect_brackets.csv` | 631 cells |

### The shared feature vocabulary

Every source is funnelled through the pure functions in
[data/standardize.py](data/standardize.py) so the KG and the training corpus speak the
same language:

- `visual_condition` in {VMC, IMC, Unknown}
- `light_conditions` in {Daylight, Night, Dusk, Dawn, Unknown}
- `person_involved` in {PIC, CoPilot, Maintenance, ATC, Other, Unknown}
- `pilot_hours_bracket` in {<500, 500-2000, 2000-5000, 5000+, Unknown}
- QoQ brackets for employment / fuel / revenue / load factor

Nothing outside `standardize.py` performs these transforms.

### The severity target

Only **21 of 1013 (2%)** commercial events involve a fatality — far too rare to learn.
So severity is *gravity-based, not injury-count-based*
([standardize.py](data/standardize.py#L302)), which also makes it size-invariant across
a regional jet and a widebody:

```
FATL or aircraft DEST  -> 4
SERS (serious injury)  -> 3     HIGH  = ordinal >= 3
SUBS (substantial dmg) -> 2
MINR                   -> 1     LOW
NONE                   -> 0
```

Binarized at `SEVERITY_HIGH_THRESHOLD = 3` → **441 high / 572 low (43.5% high)**.
Balanced enough to be learnable; note it is a *constructed proxy*, not "fatal accident".

### Leakage discipline

This is designed in, and worth understanding before you change anything:

- **NTSB never enters the knowledge graph as a retrievable neighbour of itself.**
  `select_subset.py` writes `ntsb_kg_subset.csv` as the top-scored NTSB records
  *excluded* from the LSTM subset.
- **`ntsb.faiss` (few-shot + LOFO) is built from the training split only**
  ([ntsbdataloader.py](data/ntsbdataloader.py)), so val/test narratives are never
  retrievable examples.
- **The in-distribution NTSB retrieval source is leave-one-out.** `LOFORetriever`
  ([rag_retriever.py](data/rag_retriever.py#L429)) pools the neighbours of a query from
  the *train* split while excluding the query's own `ev_id`. A training record never
  sees its own label; a test record is not in the source at all.
- **ASIAS rows in `lstm_corpus.csv` carry `y_D = -100` (ignore_index)** so they train B
  and C off their narratives but contribute nothing to severity — ASIAS severity is
  gravity-coded and nearly all low, and is trivially separable from NTSB by
  side-channels (`sky='UNK'`, empty `crew_age`), which would let the model predict
  *source* instead of severity.
- **`invest_type` is excluded from the features** — it encodes accident-vs-incident,
  which directly leaks the target.
- No SMOTE, no synthetic data anywhere.

### Subsets

`select_subset.py` scores every record by `0.5*completeness + 0.4*narrative richness +
0.1*context-field bonus` and takes the top N per source. It is a deliberately curated,
non-random subset for a methods run — say so in any write-up.

⚠️ **`data/ntsb_subset.csv` (1191 rows) is stale.** It predates the CAROL commercial
rebuild and is a *superset* of the current 1013-row `ntsb_clean.csv`. Likewise
`ntsb_kg_subset.csv` (100 rows) comes from the old general-aviation population and
shares **zero** `ev_id`s with the current corpus. **The current results were produced
on `data/ntsb_clean.csv`** (1013 rows → 710 train / 101 val / 202 test), which is why
`n_test = 202`. Re-run `select_subset.py` before trusting anything that reads
`ntsb_subset.csv` — which is what most of the module docstrings still suggest.

---

## The knowledge graph (Stage 3)

Built into Neo4j from ASIAS + ASRS + the disjoint NTSB-KG slice. Same Stage-2 LLM
passes, run inline. MERGE/upsert only; read-only afterwards.

```
(EventNode {event_id, source})
  -[:HAS_FACTOR]->             (HFACSFactorNode {tier, value})
  -[:HAS_ENV_CONTEXT]->        (EnvironmentalContextNode {feature, value})
  -[:HAS_PERSONNEL_CONTEXT]->  (PersonnelContextNode {feature, value})
  -[:HAS_ORG_CONTEXT]->        (OrganizationalContextNode {feature, value_bracket})
  -[:HAS_TECH_CONTEXT]->       (TechnologicalContextNode {maintenance_defect_rate})

(HFACSFactorNode)-[:LEADS_TO {weight, evidence}]->(HFACSFactorNode)
(HFACSFactorNode)-[:CO_OCCURS_WITH {weight, evidence}]-(HFACSFactorNode)
```

Last build tally (`data/kg_build.log`): 903 EventNodes, 2094 HFACSFactorNodes, 1806 of
each context type, 1377 `LEADS_TO`, 14663 `CO_OCCURS_WITH`.

Also written: read-only FAISS indexes `asias.faiss`, `asrs.faiss`, `ntsb_kg.faiss`
(SBERT `all-MiniLM-L6-v2`, normalized inner product).

Note the ratio: 1377 directed causal edges against 14663 co-occurrence edges. The graph
is mostly an association network, and **nothing downstream currently reads `LEADS_TO`
weights at all** — see §10.

---

## The model (Stage 4)

[models/lstm/train.py](models/lstm/train.py). Two interchangeable architectures over the
*same* DAG (`--arch lstm` default, `--arch scm`):

```
step_ctx (economic)  ->  [ctx cell]                 root, not predicted
                              |
step_b (env/person)  ->  [B cell]  -> head_B   Preconditions (3, multi-label, sigmoid)
                              |
   [soft_B | env | oper]  ->  [C cell]  -> head_C   Unsafe Acts (binary, softmax)
                              |
   [soft_C | soft_B | env] -> [D cell]  -> head_D   Severity    (binary, softmax)
```

`HFACSCausalSCM` is the same graph expressed as one MLP per node — a neural structural
causal model, so `do()` / counterfactual semantics are explicit (override a node's
output and propagate). Drop-in identical forward signature.

Hidden state flows `ctx → B → C → D`; soft predictions hand off `B → C` and `C → D`
(detached, so each head trains on its own loss). Skip-edges (`env → C`, `env → D`,
`oper → C`) are the ones the PC-algorithm check in
[models/causal_discovery.py](models/causal_discovery.py) is meant to validate.

### What the model actually sees

This matters more than anything else in this README:

```
step_ctx (8) : employment_qoq, fuel_qoq, revenue_qoq, loadfactor_qoq,
               + the 4 corresponding brackets                    <- macro-economic
step_b   (5) : visual_condition, light_conditions, time_of_day,
               person_involved, pilot_hours_bracket              <- environment / crew

few-shot     : k exemplars x 37 (k = 15), retrieved from the TRAIN split  <- current design
               [ the 5 features above | y_B(3) | y_C(4) | y_D(2) | causal roles(20)
                 | has_factor_labels | has_severity | similarity ]
               encoded by FewShotEncoder -> 32 dims + the 37-dim weighted vote

[legacy]     : precond_prior(3) | unsafe_prior(4) | severity_prior(2) appended to
               step_b — the input-augmentation path used for C4..C8 (see below)
```

**Thirteen structured features. No narrative text reaches the model.** The narrative is
used to *create* the B/C labels and to *find* neighbours — never as an input.

The exemplar block is where retrieval now enters. Note what it does and does not add:
its five feature columns are the *same five* the model already sees for the query, so
the new information is the neighbours' labels **plus the pairing** — which features went
with which label. That pairing is the entire difference from a prior, and the reason the
two are different conditions rather than the same one.

### Training details

- Losses: sigmoid focal BCE for B (with sqrt-dampened per-label `pos_weight`), focal CE
  for C and D (with clipped inverse-frequency class weights). Head weights
  `(B, C, D) = (1.0, 1.5, 1.0)`.
- Adam, `lr 1e-4`, `ReduceLROnPlateau`, grad clip 1.0, dropout 0.1, hidden 128, up to
  500 epochs with early stopping.
- **Per-class decision thresholds for B are tuned on the validation split** and stored
  in the checkpoint (`tune_thresholds`), which counters the focal-loss collapse where
  sigmoid never crosses 0.5 for a rare-but-present class.
- Split: 70/10/10 via a seeded `torch.randperm(seed=42)`, defined once in
  `ntsbdataloader._split` and mirrored bit-for-bit by `hfacs_extractor.ntsb_train_ids`.
  **If you change the split, change it in both places.**

---

## Retrieval (Stage 5)

[data/rag_retriever.py](data/rag_retriever.py). For each record, two score sets:

1. **FAISS semantic search** over the narrative (`combined_text`), across up to three
   sources with configurable weights: `asias.faiss`, `asrs.faiss`, and the
   in-distribution **NTSB LOFO** source built from the train split.
2. **Deterministic structural Cypher** — scores each `EventNode` by how many
   `{feature, value}` context nodes it shares with the record (environment, personnel,
   organizational bracket, SDR maintenance bracket). This *replaced* LLM text-to-Cypher,
   which hallucinated node labels on small models.

Both are min-max normalized to [0,1] and combined 50/50.

### Retrieval text and causal chains

**Retrieval text is outcome-stripped.** `standardize.strip_outcome()` removes sentences
stating damage or injury, and is applied at every point where text is embedded for
RETRIEVAL — never in Stage-2 extraction, so `hfacs_results.csv` stays comparable. It
**does not achieve its goal**: stripped text still predicts severity at balanced
accuracy 0.914 versus 0.923 raw. See the severity caveat in the TL;DR.

**Exemplars carry the extracted causal chain.** Each exemplar is 37 dims (the layout
is defined once, as the `FS_*` slices in `data/ntsbdataloader.py`):

```
[ 5 base features | y_B(3) | y_C(4) | y_D(2) | causal roles(20)
  | has_factor_labels(1) | has_severity(1) | similarity(1) ]
```

The causal block is a role vector over the 10 mined tiers — for each tier, did it act as
a *cause* and did it act as an *effect*. It comes from the **LLM-extracted** directed
links, not the graph's deterministic ones:

- Graph `LEADS_TO` written by `kg_builder.classify_edge` is derived from `DAG_EDGES`
  given which factors co-occur. It is a pure function of the tier set the exemplar
  already encodes in `y_B`/`y_C`, so feeding it back adds **exactly zero information** —
  which is why nothing having read those edges has cost nothing.
- The LLM's own links are independent evidence and frequently contradict the DAG. The
  three most common are `unsafe_decision -> unsafe_skill`, `unsafe_perception ->
  unsafe_decision`, `unsafe_decision -> unsafe_violation`, none of which is a DAG edge.

KG exemplars take them from edges filtered on `l.evidence IS NOT NULL` (976/2042 events);
train exemplars from `relationships_json` (199/710 records).

### Augmentation strategies — how retrieval reaches the model

Three ways, per the project spec. They are independent and can be combined.

| | mechanism | code | status |
|---|---|---|---|
| **Prompt** *(current design)* | k retrieved neighbours enter as intact **(features, labels) exemplars**, encoded by `FewShotEncoder` onto the context root | `FewShotSource`, `--fewshot-k` | **primary** |
| **Input** *(legacy)* | top-k neighbours collapsed into 7 soft priors appended to `step_b` | `_retrieve_priors` | retained for C4–C8 reproduction |
| **Ensemble** | retrieval as a standalone predictor, blended with the model at weight α tuned on validation | `models/lstm/ensemble.py` | available |

**The project now uses few-shot exemplars, not priors.** A prior is the neighbours'
label distribution with the features averaged away; an exemplar keeps feature and label
bound together, so the model can learn a locally-weighted mapping instead of a global
base rate. The prior path is kept in the code because C1–C8 are a published ablation
and must stay reproducible — it is not the design going forward.

Exemplars are drawn from the **NTSB train split only**, self-excluded, the same
discipline as `LOFORetriever`. ASIAS/ASRS are deliberately excluded as exemplar sources:
their label space differs (ASIAS severity is `ignore_index`), so exemplars from them
would teach the model from labels it is never scored on.

Any retrieval failure (Neo4j down, FAISS missing, bad Cypher) degrades silently —
uniform priors on the legacy path, zero-filled and masked-out exemplars on the few-shot
path. Training never breaks, which means **a silent failure looks like a working run**.
The dataloader prints per-record coverage for both (`RAG priors non-uniform: …` and
`Few-shot exemplars: n/N records got >=1`). Watch those lines; they are the honest
measure of whether retrieval is carrying anything.

---

## Evaluation (Stage 6) and current results

[models/lstm/eval.py](models/lstm/eval.py) evaluates every checkpoint in `results/` on
the same held-out test split and writes `results/eval_summary.csv`,
`results/mcnemar_by_head.csv` and the `figures/eval_*.png` set.

### Conditions

A retrieval-STRATEGY ablation (spec 2.2/2.3). Retrieval reaches the model as few-shot
exemplars in every case; only the strategy changes.

| | strategy | what it tests |
|---|---|---|
| C1 | none | structured baseline |
| C2 | `faiss` — narrative similarity | Strategy A, semantic |
| C3 | `cypher` — shared structured context | Strategy B, structural. **The only pre-narrative-clean condition.** |
| C4 | `hybrid` — 50/50 of both | Strategy C |
| C5 | `hybrid`, LLM-mined labels zeroed | the text-mining ablation: what survives without mined content |

Run them with `python run_conditions.py`, evaluate with
`python models/lstm/eval_conditions.py`. Each condition trains once per seed
(default 5) and the headline test is a **paired t-test with seeds as replicates** —
unseeded repeats of C1 alone varied by ~0.04 kappa on D, larger than the gaps between
conditions, so a single run cannot separate them. McNemar on seed 0 is kept as a
per-item view.

The legacy prior-based conditions (`--rag-strategy`, C4..C8 of the old source ablation)
still work and produced the historical results, but are not the design going forward.

Metrics per head: F1 (micro for the multi-label B, macro for C/D), accuracy, balanced
accuracy, Cohen's kappa, support, and a **generalization error** = train-minus-test on
the same metric over a capped train subsample. Plus `chain_completion_rate` (exact
`B and C and D` match) and per-head **McNemar** tests against C1.

### What the numbers say

**Generalization error is tiny across the board** (B 0.00–0.07, C 0.03–0.09, D
0.05–0.07). The model is *not* overfitting. It is underfitting because the inputs do
not contain the answer. More capacity, more regularization, more epochs will not help.

**McNemar vs C1** (`results/mcnemar_by_head.csv`), net = helps − hurts:

- **B:** every RAG condition is a *significant net loss* — C4 −95, C5 −133, C6 −159,
  C7 −49, C8 −78, all p < 1e-4. Retrieval actively damages the precondition head.
- **C:** nothing helps. C6 (ASRS only) hurts badly (−72, p ≈ 1e-11).
- **D:** C4 +48, C7 +51, C8 +50, all p < 1e-6. C5 and C6 are not significant.

**Chain completion is a trap.** C1's 0.203 is the *highest*, but that is mostly the
degenerate case: B predicts all-zeros, and 80% of records have all-zero B targets, so
the "chain" completes by predicting nothing. Do not headline this number.

**Causal discovery** ([results/causal_metrics.csv](results/causal_metrics.csv)): the PC
algorithm on the NTSB training data recovers **1 of 27** hypothesized DAG edges
(`operator → severity_class`), precision 1.0, recall 0.037. Read that as: at this sample
size and with these coarsened variables, the data does not support the theoretical HFACS
edge set. It does not by itself falsify HFACS — 1013 records with 3-bit variables is
thin for constraint-based discovery — but it is consistent with everything else here.

---

## Setup

```bash
pip install pandas numpy torch scikit-learn matplotlib networkx tqdm \
            faiss-cpu sentence-transformers ollama neo4j statsmodels \
            beautifulsoup4 access_parser shap causal-learn
```

`shap` and `causal-learn` are optional (SHAP falls back to permutation importance;
`causal_discovery.py` needs `causal-learn`). Local LLM via
[Ollama](https://ollama.com) — default model `qwen2.5:7b`, fallback `llama3.1:8b`
(`gemma3`/`gemma4` OOM on 16 GB). Neo4j at `bolt://localhost:7687`; set `NEO4J_URI`,
`NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`.

### End-to-end run

```bash
# Stage 1 - data
python data/build_carol_ntsb.py                    # -> ntsb_clean.csv (Part-121)
python data/data_assembler.py                      # -> asias_clean.csv, asrs_clean.csv
python data/sdr_defect_rate.py                     # -> sdr_defect_brackets.csv
python select_subset.py --ntsb 1000 --asias 1000 --asrs 1000 --ntsb-kg 500
python data/build_lstm_corpus.py                   # optional: NTSB + disjoint ASIAS

# Stage 2 - text mining  (build the train-only FAISS first so few-shot can fire)
python data/ntsbdataloader.py --build-faiss-only --input data/ntsb_clean.csv
python data/hfacs_extractor.py --force-binary --model qwen2.5:7b

# Stage 3 - knowledge graph  (slow; chunk with --limit)
python data/kg_builder.py --source both
python data/kg_builder.py --faiss-only

# Stage 2b - repair the violation tier (strict HFACS definition, majority of 3 runs).
#   Writes data/violation_adjudication.csv, which the dataloader applies as an
#   override on top of hfacs_results.csv. HFACS_VIOLATION_OVERRIDE=0 ignores it.
python data/adjudicate_violation.py

# Stage 4/6 - train and evaluate the five conditions (5 seeds each, ~45 min on CPU)
python run_conditions.py --epochs 500 --seeds 0 1 2 3 4      # k = 15 exemplars

# Stage 6b - what the TEST record is allowed to know. Rewrites the retrieval text of
#   the 202 test records only; nothing is retrained and no index is rebuilt.
python data/build_query_views.py          # L2 preliminary (standard) + L1 pre-departure
python data/leakage_audit.py              # gate, exit 2 = FAIL
python models/lstm/eval_conditions.py                       # L2, the standard view
python models/lstm/eval_conditions.py --query-view L1       # lower bound
python models/lstm/eval_conditions.py --query-view full     # upper bound (retrospective)
#   optional: --include-severity scores head D under L2 (not a prediction, see TL;DR)
#   optional diagnostic: build_query_views.py --tiers L1b ; eval --query-view L1b

#   subset / quick check
python run_conditions.py --only C1 C2 --seeds 0 --epochs 120

#   NOTE: changing RETRIEVAL_MODEL or strip_outcome() means the FAISS indexes must be
#   rebuilt, or 768-dim queries hit a stale index and retrieval silently degrades:
python data/kg_builder.py --faiss-only --asias-csv data/asias_subset.csv        --asrs-csv data/asrs_subset.csv --ntsb-csv data/ntsb_kg_subset.csv

python models/causal_discovery.py --input data/ntsb_clean.csv

# Figures + inspection
python visualize.py
python data/hfacs_analysis.py
python system_eval.py --records 3
```
