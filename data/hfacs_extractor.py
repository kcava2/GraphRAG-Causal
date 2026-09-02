"""
HFACS LLM Text-Mining Extractor (Stage 2)
=========================================
Reads ``data/ntsb_clean.csv`` (Stage 1 output) and, for every record, makes two
deterministic local-Ollama (Gemma) calls. By default ``--split all`` processes
**every** NTSB record so Stage 4 has HFACS targets (y_A/y_B/y_C) for the train,
val, and test splits; ``--split train`` restricts processing to the training
split. Either way the few-shot index/retrieval (``ntsb.faiss``) is **train-only**,
so labeling val/test records never leaks val/test examples into a prompt.

    Task 1  entity + HFACS classification  -> {entities, hfacs_classifications}
    Task 2  causal relationship extraction -> {relationships}

Both calls use **Ollama structured outputs**: a JSON Schema derived from
``HFACS_SCHEMA`` is passed as ``format=``, so decoding is grammar-constrained and
the model cannot emit an unknown tier, a bad relation, or unparseable JSON. The
schema constrains the *vocabulary* only — every tier is optional, so nothing is
forced into a label. ``_extract_json`` and the ``_validate_*`` helpers remain as a
safety net for ``--no-structured`` runs and models without grammar support, where
off-schema values are still silently dropped. Output is one row per record in
``data/hfacs_results.csv``:

    ev_id, entities_json, hfacs_json, relationships_json, extraction_status

This stage is **read-only** w.r.t. all indexes and the KG: no Neo4j writes, no
FAISS writes, and no edits to any Stage-1 CSV. Few-shot examples are retrieved
from a read-only ``data/ntsb.faiss`` index built later in Stage 4; if that index
(or its id-map sidecar) is absent the few-shot block is simply empty.

Run order: Stage 1 must complete first. For real few-shot, build the NTSB
training FAISS index in Stage 4 before running ``--force-binary`` on the full
corpus.

CLI:
    python data/hfacs_extractor.py --force-binary --model gemma3

Requirements: ollama (+ a pulled gemma3 model), pandas, torch, faiss-cpu,
sentence-transformers (the last two only needed when few-shot is active).
"""

import argparse
import json
import logging
import os
import time

import pandas as pd
import torch
from tqdm import tqdm
import ollama


# ---------------------------------------------------------------------------
# Config / paths
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
NTSB_CLEAN = os.path.join(_HERE, "ntsb_clean.csv")
RESULTS_CSV = os.path.join(_HERE, "hfacs_results.csv")
FAISS_INDEX = os.path.join(_HERE, "ntsb.faiss")          # built in Stage 4
FAISS_IDMAP = os.path.join(_HERE, "ntsb_faiss_ids.json")  # sidecar (Stage 4)

DEFAULT_MODEL = "qwen2.5:7b"    # gemma4 (9.6 GB) OOMs on 16 GB; this is the best
FALLBACK_MODEL = "llama3.1:8b"  # installed model that fits. Override with --model.
SBERT_MODEL = "all-MiniLM-L6-v2"
CHECKPOINT_EVERY = 10           # default; override with --checkpoint-every

# Generation options sent to every Ollama call. num_ctx is set from --num-ctx in
# main(); a smaller context window lowers KV-cache memory (helps on 16 GB RAM)
# but can truncate very long narratives.
_GEN_OPTIONS = {"temperature": 0.0}


# ---------------------------------------------------------------------------
# HFACS schema — single source of truth
# ---------------------------------------------------------------------------

HFACS_SCHEMA = {
    "org_climate":         ["Culture", "Policy", "Structure"],
    "resource_mgmt":       ["Human Resource Mgmt", "Budget Constraints",
                            "Equipment/Facility Resources"],
    "org_process":         ["Operations", "Procedures", "Oversight",
                            "Safety Programs", "Risk Management",
                            "Schedules"],
    "supervisory":         ["Inadequate Supervision",
                            "Planned Inappropriate Ops",
                            "Failed to Correct Known Problem",
                            "Supervisory Violations"],
    "situational_phys":    ["Weather", "Lighting", "Terrain"],
    "situational_tech":    ["Equipment Design", "Automation", "Interface"],
    "operator_mental":     ["Channelized Attention", "Complacency",
                            "Distraction", "Mental Fatigue",
                            "Get-home-itis", "Haste", "Task Saturation",
                            "Loss of Situational Awareness",
                            "Misplaced Motivation"],
    "operator_physical":   ["Impaired Physiological State",
                            "Medical Illness",
                            "Physiological Incapacitation",
                            "Physical Fatigue"],
    "operator_limits":     ["Insufficient Reaction Time",
                            "Visual Limitation",
                            "Incompatible Intelligence/Aptitude"],
    "personnel_crm":       ["Failed to Communicate",
                            "Failed to Coordinate",
                            "Failed to Back-up",
                            "Failure of Leadership",
                            "Misinterpretation of Traffic Calls"],
    "personnel_readiness": ["Crew Rest Violation", "Self-medicating",
                            "Excessive Physical Training"],
    "unsafe_skill":        ["Breakdown in Visual Scan",
                            "Omitted Checklist Step",
                            "Failed to Prioritize Attention",
                            "Poor Technique",
                            "Inadvertent Flight Control Use",
                            "Over-controlled Aircraft"],
    "unsafe_decision":     ["Improper Procedure",
                            "Misdiagnosed Emergency",
                            "Exceeded Ability",
                            "Inappropriate Maneuver",
                            "Poor Decision"],
    "unsafe_perception":   ["Misjudged Distance/Altitude/Airspeed",
                            "Spatial Disorientation",
                            "Visual Illusion"],
    "unsafe_violation":    ["Unauthorized Approach",
                            "Violated Training Rules",
                            "Failed to Prepare for Flight",
                            "Not Current/Qualified",
                            "Intentionally Exceeded Aircraft Limits"],
}

# Flattened lookups for O(1) validation.
VALID_SUBS = {sub: tier for tier, subs in HFACS_SCHEMA.items() for sub in subs}
VALID_RELATIONS = {"LEADS_TO", "CO_OCCURS_WITH"}

# Tiers TEXT-MINED, split into two passes (see extract_row):
#   PASS 1 — UNSAFE ACTS: evidence-gated, >=1 required, NOT blanket-all-four.
#   PASS 2 — PRECONDITIONS: latent operator/personnel/tech states INFERRED from the
#            unsafe acts (they are usually implied, not stated).
# Excluded entirely: Organizational + Supervisory (-> economic context input) and
# situational_phys (Weather/Lighting/Terrain -> structured visual/light input).
# HFACS_SCHEMA stays complete as the reference vocabulary.
PRECOND_EXTRACT_TIERS = ["operator_mental", "operator_physical", "operator_limits",
                         "situational_tech", "personnel_crm", "personnel_readiness"]
UNSAFE_EXTRACT_TIERS = ["unsafe_skill", "unsafe_decision", "unsafe_perception",
                        "unsafe_violation"]
EXTRACT_TIERS = PRECOND_EXTRACT_TIERS + UNSAFE_EXTRACT_TIERS
PRECOND_SCHEMA = {t: HFACS_SCHEMA[t] for t in PRECOND_EXTRACT_TIERS}
UNSAFE_SCHEMA = {t: HFACS_SCHEMA[t] for t in UNSAFE_EXTRACT_TIERS}
EXTRACT_SCHEMA = {t: HFACS_SCHEMA[t] for t in EXTRACT_TIERS}
EXTRACT_VALID_SUBS = {sub: t for t in EXTRACT_TIERS for sub in HFACS_SCHEMA[t]}


# ---------------------------------------------------------------------------
# JSON Schemas for Ollama structured outputs
# ---------------------------------------------------------------------------
# Passed as ``format=`` on every chat call. Ollama constrains decoding to the
# schema, so the model *cannot* emit an unknown tier, a malformed relation, or
# unparseable JSON — the failure modes `_extract_json` and the `_validate_*`
# helpers used to absorb silently. Those two layers are kept as a safety net
# (the schema is not enforced when --no-structured is passed, and a model
# without grammar support falls back to free-form).
#
# Every property is OPTIONAL: the schema constrains the *vocabulary*, never
# which tiers must appear. Requiring tiers would manufacture labels, which is
# exactly the bias this stage must not introduce.


def classification_schema(tiers: list[str]) -> dict:
    """`{"hfacs_classifications": {<tier>: ["<evidence phrase>", ...]}}`."""
    return {
        "type": "object",
        "properties": {
            tier: {"type": "array", "items": {"type": "string"}}
            for tier in tiers
        },
        "additionalProperties": False,
    }


def task1_schema(tiers: list[str], with_entities: bool = True) -> dict:
    """Pass-1 shape: entities + tier-keyed classifications."""
    props = {"hfacs_classifications": classification_schema(tiers)}
    if with_entities:
        props["entities"] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "role": {"type": "string"},
                    "tier": {"type": "string", "enum": tiers},
                },
                "required": ["text", "role", "tier"],
                "additionalProperties": False,
            },
        }
    return {
        "type": "object",
        "properties": props,
        "required": ["hfacs_classifications"],
        "additionalProperties": False,
    }


def task2_schema(vocab: list[str]) -> dict:
    """Task-2 shape: relationships over `vocab` (TIER names — see note below).

    `_validate_relationships` accepts subject/object only when they are TIERS,
    so the grammar must emit tiers too; a subcategory-level vocabulary here
    would be discarded downstream.
    """
    return {
        "type": "object",
        "properties": {
            "relationships": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string", "enum": vocab},
                        "relation": {"type": "string",
                                     "enum": sorted(VALID_RELATIONS)},
                        "object": {"type": "string", "enum": vocab},
                        "evidence": {"type": "string"},
                    },
                    "required": ["subject", "relation", "object", "evidence"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["relationships"],
        "additionalProperties": False,
    }


UNSAFE_FORMAT = task1_schema(UNSAFE_EXTRACT_TIERS)
PRECOND_FORMAT = task1_schema(PRECOND_EXTRACT_TIERS, with_entities=False)
RELATION_FORMAT = task2_schema(EXTRACT_TIERS)

# Toggled off by --no-structured (kept global so kg_builder inherits the choice).
_STRUCTURED = True

# Reasoning models (gemma4, qwen3, ...) advertise a `thinking` capability and
# may enable it by default. Thinking tokens are pure cost here: the task is
# bounded schema-constrained extraction, and the reasoning text is returned
# separately from `message.content` so it never reaches the parser anyway.
# Off by default; --think re-enables it.
_THINK = False


# ---------------------------------------------------------------------------
# Train split — deterministic, mirrors data/dataloader.py exactly
# ---------------------------------------------------------------------------

def ntsb_train_ids(df: pd.DataFrame, seed: int = 42,
                   test_split: float = 0.2, val_split: float = 0.1) -> set:
    """
    Return the set of training-split ``ev_id`` values.

    Bit-identical to data/dataloader.py's split: a single ``torch.randperm``
    over the dataframe's row order with the same seed and ratios, taking the
    first ``n - n_test - n_val`` indices as train. Stage 4's real_dataloader.py
    MUST reuse this logic (same seed, ratios, and ntsb_clean.csv row order) so
    that ntsb.faiss and this stage agree on train membership.
    """
    n = len(df)
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=rng).tolist()
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_test - n_val
    train_pos = perm[:n_train]
    return set(df.iloc[train_pos]["ev_id"].astype(str))


# ---------------------------------------------------------------------------
# Text / JSON helpers
# ---------------------------------------------------------------------------

def _clean(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    s = str(value).strip()
    return "" if s.lower() in ("nan", "none") else s


def _extract_json(raw: str):
    """
    Best-effort JSON parse of an LLM response: strip ``` fences, slice from the
    first ``{`` to the last ``}``. Returns the parsed object or ``None``.
    """
    if not raw:
        return None
    clean = raw.strip()
    if "```" in clean:
        for part in clean.split("```"):
            part = part.strip()
            if part.lower().startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                clean = part
                break
    start = clean.find("{")
    end = clean.rfind("}") + 1
    if start != -1 and end > start:
        clean = clean[start:end]
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return None


_SCHEMA_UNSUPPORTED = False   # set once if the server rejects `format=`
_THINK_UNSUPPORTED = False    # set once if the model rejects `think=`


def _call_ollama(model_name: str, system: str, user: str,
                 retries: int = 3, schema: dict | None = None) -> str | None:
    """Single deterministic Ollama chat with retries. Raw text or None.

    When `schema` is given and structured output is enabled, it is passed as
    Ollama's ``format=`` so decoding is grammar-constrained to that shape.
    Falls back to free-form generation (once, permanently) if the server or
    model does not support constrained decoding.
    """
    global _SCHEMA_UNSUPPORTED, _THINK_UNSUPPORTED
    use_schema = schema if (schema and _STRUCTURED and not _SCHEMA_UNSUPPORTED) else None
    send_think = not _THINK_UNSUPPORTED
    for attempt in range(1, retries + 1):
        try:
            kwargs = {"format": use_schema} if use_schema else {}
            if send_think:
                kwargs["think"] = _THINK
            response = ollama.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                options=_GEN_OPTIONS,
                **kwargs,
            )
            return response["message"]["content"]
        except Exception as e:
            err = str(e).lower()
            if "connection" in err or "refused" in err:
                raise SystemExit(
                    "\nERROR: Cannot connect to Ollama.\n"
                    "Start it with 'ollama serve' or open the Ollama app."
                )
            if send_think and "think" in err:
                logging.warning(
                    "Model does not accept think= (%s) — omitting it from here on.", e)
                _THINK_UNSUPPORTED = True
                send_think = False
                continue
            if use_schema and ("format" in err or "schema" in err or "grammar" in err):
                logging.warning(
                    "Model/server rejected structured output (%s) — falling back "
                    "to free-form JSON for the rest of the run.", e)
                _SCHEMA_UNSUPPORTED = True
                use_schema = None
                continue
            logging.error(f"LLM error attempt {attempt}: {e}")
            if attempt < retries:
                time.sleep(2)
    return None


# ---------------------------------------------------------------------------
# Few-shot retrieval (read-only; degrades to "" when unavailable)
# ---------------------------------------------------------------------------

_SBERT = None
_FAISS = None          # (index, [ev_id, ...]) tuple once loaded
_FEWSHOT_DISABLED = False
_RESULTS_CACHE: dict[str, str] = {}   # ev_id -> hfacs_json (successful rows)
_SNIPPET_CACHE: dict[str, str] = {}   # ev_id -> narrative snippet


def _load_fewshot_backends():
    """Lazy-load SBERT + FAISS index + id-map. Returns False if unavailable."""
    global _SBERT, _FAISS, _FEWSHOT_DISABLED
    if _FEWSHOT_DISABLED:
        return False
    if _SBERT is not None and _FAISS is not None:
        return True
    if not (os.path.exists(FAISS_INDEX) and os.path.exists(FAISS_IDMAP)):
        logging.warning(
            "Few-shot disabled: %s or %s not found (built in Stage 4). "
            "Continuing with empty few-shot blocks.",
            os.path.basename(FAISS_INDEX), os.path.basename(FAISS_IDMAP),
        )
        _FEWSHOT_DISABLED = True
        return False
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
        _SBERT = SentenceTransformer(SBERT_MODEL)
        index = faiss.read_index(FAISS_INDEX)
        with open(FAISS_IDMAP, "r", encoding="utf-8") as f:
            ids = [str(x) for x in json.load(f)]
        _FAISS = (index, ids)
        return True
    except Exception as e:  # missing deps, model download failure, corrupt index
        logging.warning("Few-shot disabled (backend load failed: %s).", e)
        _FEWSHOT_DISABLED = True
        return False


def get_ntsb_fewshot_examples(narrative_text: str, n: int = 5,
                              exclude_ev_id: str | None = None) -> str:
    """
    Encode ``narrative_text`` with Sentence-BERT, search the read-only
    ``ntsb.faiss`` index (training split only), and format the HFACS
    classifications of the ``n`` nearest already-extracted records as a few-shot
    block for the Task-1 prompt. Returns "" when few-shot is unavailable or no
    successful neighbours exist yet. Never writes to any index or CSV.
    """
    if not _clean(narrative_text):
        return ""
    if not _load_fewshot_backends():
        return ""

    index, ids = _FAISS
    emb = _SBERT.encode([narrative_text], normalize_embeddings=True)
    import numpy as np
    emb = np.asarray(emb, dtype="float32")
    k = min(n + 1, len(ids))
    _, idx = index.search(emb, k)

    blocks = []
    for pos in idx[0]:
        if pos < 0 or pos >= len(ids):
            continue
        ev = ids[pos]
        if exclude_ev_id is not None and ev == exclude_ev_id:
            continue
        hfacs = _RESULTS_CACHE.get(ev)
        if not hfacs:
            continue
        snippet = _SNIPPET_CACHE.get(ev, "")[:400]
        blocks.append(f"NARRATIVE: {snippet}\nCLASSIFICATION: {hfacs}")
        if len(blocks) >= n:
            break
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_TASK1 = (
    "You are an HFACS (Human Factors Analysis and Classification System) expert "
    "analyst for aviation accident investigation. You classify the human factors "
    "in an accident at the HFACS TIER level. Human error applies to ANY person in "
    "the system — pilot, maintenance crew, ATC, dispatch, ground crew — not just "
    "the pilot. Factors are frequently IMPLIED by the narrative rather than stated "
    "outright; infer them. Respond ONLY with valid JSON — no preamble, no markdown."
)

SYSTEM_TASK2 = (
    "You are an HFACS expert. From the narrative and the listed HFACS entities, "
    "extract directed causal relationships between HFACS subcategories. Respond "
    "ONLY with valid JSON — no preamble, no markdown, no code fences."
)


def _structured_context(row: pd.Series) -> str:
    """Build the STRUCTURED CONTEXT block from non-null NTSB fields only."""
    lines = []
    inv = _clean(row.get("invest_type_binary"))
    if inv != "":
        ev_type = {"0": "ACC", "1": "INC"}.get(inv.split(".")[0], inv)
        lines.append(f"- Event type: {ev_type}")
    for label, col in (
        ("Visual condition", "visual_condition"),
        ("Light condition", "light_conditions"),
        ("NTSB finding path", "finding_description_agg"),
    ):
        val = _clean(row.get(col))
        if val:
            lines.append(f"- {label}: {val}")
    return "\n".join(lines)


def _build_unsafe_prompt(row: pd.Series, fewshot: str) -> str:
    """PASS 1 — unsafe acts, evidence-gated (cite a phrase per tier; >=1; not all)."""
    narrative = _clean(row.get("combined_text"))
    context = _structured_context(row)
    schema_json = json.dumps(UNSAFE_SCHEMA, indent=2)
    parts = [f"NARRATIVE:\n{narrative}"]
    if context:
        parts.append(f"STRUCTURED CONTEXT:\n{context}")
    if fewshot:
        parts.append(f"FEW-SHOT EXAMPLES:\n{fewshot}")
    parts.append("UNSAFE-ACT TIERS (example factors per tier — recognize the tier, "
                 f"not limited to these words):\n{schema_json}")
    parts.append(
        'Identify the UNSAFE ACTS in this accident, committed by ANY human (pilot, '
        'maintenance, ATC, dispatch, ...). Rules:\n'
        '- Assign an unsafe-act TIER only when the narrative SPECIFICALLY supports '
        'that type of act; quote the supporting phrase as the factor and name the '
        'human. Do NOT assign tiers by default — most accidents involve one or two, '
        'rarely all four.\n'
        '- At least ONE unsafe-act tier is required (every accident has one).\n'
        'Respond with valid JSON only, in the shape:\n'
        '{"entities": [{"text": "...", "role": "...", "tier": "..."}], '
        '"hfacs_classifications": {"<unsafe_tier>": ["<evidence phrase + human role>"]}}'
    )
    return "\n\n".join(parts)


def _build_precond_prompt(row: pd.Series, unsafe: dict) -> str:
    """PASS 2 — infer the latent preconditions that SET UP the unsafe acts.

    Conditioned on the unsafe acts from pass 1; NO few-shot (the corpus is
    precondition-sparse, so few-shot reinforces under-extraction — we rely on
    explicit inference instructions + worked implied->precondition examples)."""
    narrative = _clean(row.get("combined_text"))
    context = _structured_context(row)
    findings = _clean(row.get("finding_description_agg"))
    schema_json = json.dumps(PRECOND_SCHEMA, indent=2)
    parts = [f"NARRATIVE:\n{narrative}"]
    if context:
        parts.append(f"STRUCTURED CONTEXT:\n{context}")
    parts.append(f"UNSAFE ACTS already identified: {sorted(unsafe.keys())}")
    if findings:
        parts.append(
            "NTSB FINDINGS (official coded cause factors — these frequently NAME the "
            "human factor directly; MAP each 'Personnel issues' finding to its tier "
            "FIRST, then infer more from the narrative):\n" + findings + "\n\n"
            "NTSB finding -> precondition-tier guide:\n"
            "  Personnel issues - Psychological - Attention/monitoring/perception "
            "-> operator_mental\n"
            "  Personnel issues - ... - info processing / decision / judgment "
            "-> operator_mental\n"
            "  Personnel issues - ... - crew resource mgmt / coordination / "
            "communication -> personnel_crm\n"
            "  Personnel issues - Physical - fatigue / impairment / medical "
            "-> operator_physical\n"
            "  Personnel issues - ... - experience / knowledge / qualification "
            "-> operator_limits\n"
            "  Aircraft - systems / automation / display / interface "
            "-> situational_tech")
    parts.append("PRECONDITION TIERS (example factors per tier — not limited to "
                 f"these words):\n{schema_json}")
    parts.append(
        'Now infer the PRECONDITIONS that SET UP those unsafe acts. Use the NTSB '
        'FINDINGS above as direct evidence (a "Personnel issues" finding almost '
        'always implies a precondition tier). Beyond the findings, they are usually '
        'IMPLIED in the narrative. Consider EACH tier SEPARATELY — do NOT default '
        'everything to operator_mental; several tiers may apply:\n'
        '  operator_mental    - attention / awareness / motivation state '
        '(distraction, loss of situational awareness, get-home-itis, complacency)\n'
        '  operator_physical  - bodily state (fatigue, illness, impairment, '
        'incapacitation)\n'
        '  operator_limits    - capability / experience limits (low total time, '
        'inexperience, insufficient reaction time, visual limitation)\n'
        '  situational_tech   - equipment / automation / interface factors\n'
        '  personnel_crm      - COMMUNICATION & COORDINATION failures with the crew, '
        'ATC, OR others; applies even to a single pilot (failed to communicate, no '
        'read-back, misread a traffic/clearance call, no approach briefing, poor '
        'crew coordination, instructor failed to intervene)\n'
        '  personnel_readiness- readiness failures (crew rest violation, '
        'self-medicating, inadequate preparation)\n'
        'Worked IMPLIED examples (note the variety of tiers):\n'
        '  "did not read back / misread the ATC clearance"  -> personnel_crm\n'
        '  "the instructor did not take control in time"     -> personnel_crm\n'
        '  "low-time pilot in a high-performance airplane"   -> operator_limits\n'
        '  "on the fourth leg of a long duty day"            -> operator_physical\n'
        '  "confusing autopilot mode annunciations"          -> situational_tech\n'
        '  "continued a night approach with no visual cues"  -> operator_mental\n'
        'Include a tier when the narrative reasonably supports it (inferred is fine; '
        'invented is not). Name the responsible human. Respond with valid JSON only:\n'
        '{"hfacs_classifications": {"<precond_tier>": ["<inferred factor + human role>"]}}'
    )
    return "\n\n".join(parts)


def _build_task2_prompt(row: pd.Series, entities: list) -> str:
    narrative = _clean(row.get("combined_text"))
    return (
        f"NARRATIVE:\n{narrative}\n\n"
        f"HFACS ENTITIES:\n{json.dumps(entities)}\n\n"
        f"VALID HFACS TIERS (subject/object must be one of these tier names):\n"
        f"{json.dumps(EXTRACT_TIERS)}\n\n"
        'Extract directed causal relationships between TIERS. relation must be '
        '"LEADS_TO" or "CO_OCCURS_WITH". Respond with valid JSON only, in the shape:\n'
        '{"relationships": [{"subject": "<tier>", "relation": "LEADS_TO", '
        '"object": "<tier>", "evidence": "<narrative phrase>"}]}'
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_classifications(obj, allowed_tiers=None) -> dict:
    """Keep only valid HFACS TIERS; values are free-text factor descriptions.

    The mined label is the TIER (subcategories were only prompt examples). Factors
    are free text for the KG / interpretability; the LSTM uses tier PRESENCE.
    `allowed_tiers` restricts to one pass's tiers (unsafe vs precondition).
    """
    allowed = set(allowed_tiers) if allowed_tiers is not None else set(EXTRACT_SCHEMA)
    out: dict[str, list] = {}
    if not isinstance(obj, dict):
        return out
    for tier, facts in obj.items():
        if tier not in allowed:
            continue
        if isinstance(facts, str):
            facts = [facts]
        if not isinstance(facts, list):
            continue
        kept = [str(s).strip() for s in facts if str(s).strip()]
        if kept:
            out[tier] = kept
    return out


def _validate_relationships(obj) -> list:
    """Keep relationships whose subject/object are valid TIERS and relation valid."""
    rels = []
    if isinstance(obj, dict):
        obj = obj.get("relationships", [])
    if not isinstance(obj, list):
        return rels
    for r in obj:
        if not isinstance(r, dict):
            continue
        subj, rel, objc = r.get("subject"), r.get("relation"), r.get("object")
        if subj in EXTRACT_SCHEMA and objc in EXTRACT_SCHEMA and rel in VALID_RELATIONS:
            rels.append({
                "subject": subj,
                "relation": rel,
                "object": objc,
                "evidence": _clean(r.get("evidence")),
            })
    return rels


# ---------------------------------------------------------------------------
# Per-record extraction
# ---------------------------------------------------------------------------

def extract_row(model_name: str, row: pd.Series, n_fewshot: int = 5) -> dict:
    """Run both LLM tasks for one record; return the five output fields."""
    ev_id = str(row["ev_id"])
    narrative = _clean(row.get("combined_text"))

    fewshot = get_ntsb_fewshot_examples(narrative, n=n_fewshot, exclude_ev_id=ev_id)

    # Pass 1 — UNSAFE ACTS (evidence-gated, >=1 required)
    raw1 = _call_ollama(model_name, SYSTEM_TASK1, _build_unsafe_prompt(row, fewshot),
                        schema=UNSAFE_FORMAT)
    parsed1 = _extract_json(raw1)
    if parsed1 is None:
        return {
            "ev_id": ev_id, "entities_json": "[]", "hfacs_json": "{}",
            "relationships_json": "[]", "extraction_status": "parse_error",
        }
    entities = parsed1.get("entities", [])
    if not isinstance(entities, list):
        entities = []
    unsafe = _validate_classifications(parsed1.get("hfacs_classifications"),
                                       UNSAFE_EXTRACT_TIERS)

    # Pass 2 — PRECONDITIONS (infer the latent states that set up the unsafe acts)
    raw1b = _call_ollama(model_name, SYSTEM_TASK1, _build_precond_prompt(row, unsafe),
                         schema=PRECOND_FORMAT)
    parsed1b = _extract_json(raw1b)
    precond = _validate_classifications(
        parsed1b.get("hfacs_classifications") if isinstance(parsed1b, dict) else None,
        PRECOND_EXTRACT_TIERS)

    classifications = {**precond, **unsafe}
    status = "success" if (entities or classifications) else "empty"

    relationships = []
    if status == "success":
        raw2 = _call_ollama(model_name, SYSTEM_TASK2,
                            _build_task2_prompt(row, entities),
                            schema=RELATION_FORMAT)
        relationships = _validate_relationships(_extract_json(raw2))

    return {
        "ev_id": ev_id,
        "entities_json": json.dumps(entities),
        "hfacs_json": json.dumps(classifications),
        "relationships_json": json.dumps(relationships),
        "extraction_status": status,
    }


# ---------------------------------------------------------------------------
# Save / resume
# ---------------------------------------------------------------------------

OUTPUT_COLUMNS = ["ev_id", "entities_json", "hfacs_json",
                  "relationships_json", "extraction_status"]


def _save(result_rows: list[dict], output_path: str) -> None:
    """Append new rows, dedup on ev_id keeping the latest extraction."""
    new_df = pd.DataFrame(result_rows, columns=OUTPUT_COLUMNS)
    if os.path.exists(output_path):
        existing = pd.read_csv(output_path, dtype=str)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.drop_duplicates(subset=["ev_id"], keep="last", inplace=True)
        combined.to_csv(output_path, index=False)
    else:
        new_df.to_csv(output_path, index=False)


def _seed_caches_from_existing(output_path: str) -> set:
    """Load prior results into the few-shot cache; return processed ev_id set."""
    done = set()
    if not os.path.exists(output_path):
        return done
    df = pd.read_csv(output_path, dtype=str)
    for _, r in df.iterrows():
        ev = str(r["ev_id"])
        done.add(ev)
        if str(r.get("extraction_status")) == "success":
            _RESULTS_CACHE[ev] = str(r.get("hfacs_json", "{}"))
    return done


def _load_results_cache(path: str) -> int:
    """
    Populate the few-shot RESULTS cache from a *reference* results file (e.g. a
    zero-shot pass-1 output) WITHOUT marking those rows as already processed.
    Used by --fewshot-from so a RAG pass-2 can re-extract every record while
    retrieving pass-1 labels of its (train-split) neighbours as examples.
    """
    if not os.path.exists(path):
        logging.warning("--fewshot-from: %s not found — few-shot corpus empty.", path)
        return 0
    df = pd.read_csv(path, dtype=str)
    n = 0
    for _, r in df.iterrows():
        if str(r.get("extraction_status")) == "success":
            _RESULTS_CACHE[str(r["ev_id"])] = str(r.get("hfacs_json", "{}"))
            n += 1
    return n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _resolve_model(requested: str) -> str:
    """Verify the model is available in Ollama; fall back where sensible."""
    try:
        available = [m.model for m in ollama.list().models]
    except Exception:
        raise SystemExit(
            "\nERROR: Cannot connect to Ollama. Run 'ollama serve' first."
        )
    logging.info("Ollama models available: %s", available)
    if any(requested in m for m in available):
        return requested
    if requested == DEFAULT_MODEL and any(FALLBACK_MODEL in m for m in available):
        logging.warning("%s not found — falling back to %s.", requested, FALLBACK_MODEL)
        return FALLBACK_MODEL
    raise SystemExit(
        f"\nModel '{requested}' not found in Ollama.\n"
        f"Run: ollama pull {requested}\nAvailable: {available}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="HFACS LLM extractor (Gemma via Ollama, NTSB train split)"
    )
    parser.add_argument("--input", default=NTSB_CLEAN)
    parser.add_argument("--output", default=RESULTS_CSV)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n-fewshot", type=int, default=5)
    parser.add_argument("--force-binary", action="store_true",
                        help="Re-extract all records even if output exists "
                             "(default: resume, skipping processed ev_ids).")
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="Seconds to pause between records — throttles "
                             "sustained CPU/thermal load on low-RAM machines.")
    parser.add_argument("--num-ctx", type=int, default=8192,
                        help="Ollama context window. Lower (e.g. 4096) saves "
                             "KV-cache memory on 16 GB RAM but may truncate "
                             "very long narratives.")
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY,
                        help="Save to disk every N records (smaller = less "
                             "re-work if interrupted).")
    parser.add_argument("--split", choices=["all", "train"], default="all",
                        help="Which NTSB records to PROCESS. 'all' (default) "
                             "labels every record so Stage 4 has y_A/y_B/y_C for "
                             "all splits; 'train' restricts to the training "
                             "split. Few-shot retrieval (ntsb.faiss) is ALWAYS "
                             "train-only either way, so there is no leakage.")
    parser.add_argument("--num-predict", type=int, default=None,
                        help="Cap generation length (Ollama num_predict). Bounds "
                             "the occasional multi-minute outlier; Task-1/2 JSON "
                             "is short so ~512 is safe.")
    parser.add_argument("--think", action="store_true",
                        help="Enable the model's reasoning mode (gemma4, qwen3, "
                             "...). Off by default: it multiplies runtime and the "
                             "reasoning text is returned outside message.content, "
                             "so it never reaches the JSON parser.")
    parser.add_argument("--no-structured", action="store_true",
                        help="Disable Ollama structured outputs (format=JSON "
                             "schema) and fall back to free-form JSON + "
                             "best-effort parsing. Use to A/B whether "
                             "constrained decoding changes what a model "
                             "extracts, or for a model without grammar support.")
    parser.add_argument("--fewshot-from", default=None,
                        help="Reference results CSV (e.g. a zero-shot pass-1) to "
                             "seed the few-shot retrieval corpus for a RAG pass-2. "
                             "Combine with --force-binary to re-extract every "
                             "record with retrieval active.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    _GEN_OPTIONS["num_ctx"] = args.num_ctx
    if args.num_predict is not None:
        _GEN_OPTIONS["num_predict"] = args.num_predict
    global _STRUCTURED, _THINK
    _STRUCTURED = not args.no_structured
    _THINK = args.think
    model_name = _resolve_model(args.model)
    logging.info("Using model: %s (num_ctx=%d, sleep=%.1fs, structured=%s, "
                 "think=%s)", model_name, args.num_ctx, args.sleep,
                 _STRUCTURED, _THINK)

    df = pd.read_csv(args.input, dtype=str)
    if args.split == "train":
        train_ids = ntsb_train_ids(df)
        df = df[df["ev_id"].astype(str).isin(train_ids)].reset_index(drop=True)
        logging.info("NTSB training-split records: %d", len(df))
    else:
        df = df.reset_index(drop=True)
        logging.info("NTSB records (all splits): %d "
                     "(few-shot index stays train-only — no leakage)", len(df))

    # Seed few-shot snippet cache from the narratives we may process.
    for _, r in df.iterrows():
        _SNIPPET_CACHE[str(r["ev_id"])] = _clean(r.get("combined_text"))

    # RAG pass-2: seed the few-shot retrieval corpus from a reference (pass-1)
    # results file — independent of the output, so --force-binary re-extracts
    # every record while retrieving pass-1 labels of its neighbours.
    if args.fewshot_from:
        n_ref = _load_results_cache(args.fewshot_from)
        logging.info("Few-shot corpus from %s: %d labeled records.",
                     args.fewshot_from, n_ref)

    already_done = set()
    if not args.force_binary:
        already_done = _seed_caches_from_existing(args.output)
        logging.info("Resume: %d records already processed.", len(already_done))
    elif os.path.exists(args.output):
        os.remove(args.output)
        logging.info("--force-binary: cleared existing %s", args.output)

    todo = df[~df["ev_id"].astype(str).isin(already_done)]
    if args.limit:
        todo = todo.head(args.limit)
    logging.info("Records to process: %d", len(todo))
    if todo.empty:
        logging.info("Nothing to do.")
        return

    buffer: list[dict] = []
    for _, row in tqdm(todo.iterrows(), total=len(todo), desc="Extracting"):
        result = extract_row(model_name, row, n_fewshot=args.n_fewshot)
        buffer.append(result)
        # Make this record available to later few-shot lookups in-run.
        if result["extraction_status"] == "success":
            _RESULTS_CACHE[result["ev_id"]] = result["hfacs_json"]
        if len(buffer) % args.checkpoint_every == 0:
            _save(buffer, args.output)
            logging.info("Checkpoint: %d rows saved.", len(buffer))
        if args.sleep:
            time.sleep(args.sleep)

    _save(buffer, args.output)
    logging.info("Done. Results saved to %s", args.output)

    # Summary.
    final = pd.read_csv(args.output, dtype=str)
    counts = final["extraction_status"].value_counts().to_dict()
    print(f"\n--- HFACS Extraction Summary ({len(final)} rows) ---")
    for status, cnt in counts.items():
        print(f"  {status:<12} {cnt}")


if __name__ == "__main__":
    main()
