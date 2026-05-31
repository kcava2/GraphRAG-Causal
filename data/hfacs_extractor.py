"""
HFACS LLM Text-Mining Extractor (Stage 2)
=========================================
Reads the NTSB **training split** of ``data/ntsb_clean.csv`` (Stage 1 output)
and, for every record, makes two deterministic local-Ollama (Gemma) calls:

    Task 1  entity + HFACS classification  -> {entities, hfacs_classifications}
    Task 2  causal relationship extraction -> {relationships}

Both responses are validated against the 15-tier ``HFACS_SCHEMA`` defined here
(the single source of truth — there is no ``config/`` package). Any tier,
subcategory, or relation value not in the schema is silently dropped. Output is
one row per record in ``data/hfacs_results.csv``:

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

DEFAULT_MODEL = "gemma4"        # gemma4:latest (8B, Q4_K_M) — pulled locally
FALLBACK_MODEL = "gemma2"
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


def _call_ollama(model_name: str, system: str, user: str,
                 retries: int = 3) -> str | None:
    """Single deterministic Ollama chat with retries. Raw text or None."""
    for attempt in range(1, retries + 1):
        try:
            response = ollama.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                options=_GEN_OPTIONS,
            )
            return response["message"]["content"]
        except Exception as e:
            err = str(e).lower()
            if "connection" in err or "refused" in err:
                raise SystemExit(
                    "\nERROR: Cannot connect to Ollama.\n"
                    "Start it with 'ollama serve' or open the Ollama app."
                )
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
    "analyst for aviation accident investigation. You respond ONLY with valid "
    "JSON — no preamble, no markdown, no code fences. Use only the tier names "
    "and subcategory values provided in the schema."
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


def _build_task1_prompt(row: pd.Series, fewshot: str) -> str:
    narrative = _clean(row.get("combined_text"))
    context = _structured_context(row)
    schema_json = json.dumps(HFACS_SCHEMA, indent=2)
    parts = [f"NARRATIVE:\n{narrative}"]
    if context:
        parts.append(f"STRUCTURED CONTEXT:\n{context}")
    if fewshot:
        parts.append(f"FEW-SHOT EXAMPLES:\n{fewshot}")
    parts.append(f"HFACS SCHEMA:\n{schema_json}")
    parts.append(
        'Classify only tiers and subcategories present in the schema above. '
        'Respond with valid JSON only, in the shape:\n'
        '{"entities": [{"text": "...", "role": "...", "tier": "..."}], '
        '"hfacs_classifications": {"<tier_name>": ["<sub_category>"]}}'
    )
    return "\n\n".join(parts)


def _build_task2_prompt(row: pd.Series, entities: list) -> str:
    narrative = _clean(row.get("combined_text"))
    valid_subs = sorted(VALID_SUBS.keys())
    return (
        f"NARRATIVE:\n{narrative}\n\n"
        f"HFACS ENTITIES:\n{json.dumps(entities)}\n\n"
        f"VALID SUBCATEGORIES (subject/object must be one of these):\n"
        f"{json.dumps(valid_subs)}\n\n"
        'Extract directed causal relationships. relation must be "LEADS_TO" or '
        '"CO_OCCURS_WITH". Respond with valid JSON only, in the shape:\n'
        '{"relationships": [{"subject": "<sub>", "relation": "LEADS_TO", '
        '"object": "<sub>", "evidence": "<narrative phrase>"}]}'
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_classifications(obj) -> dict:
    """Keep only schema tiers and, within each, only valid subcategory values."""
    out: dict[str, list] = {}
    if not isinstance(obj, dict):
        return out
    for tier, subs in obj.items():
        if tier not in HFACS_SCHEMA:
            continue
        if not isinstance(subs, list):
            continue
        allowed = set(HFACS_SCHEMA[tier])
        kept = [s for s in subs if isinstance(s, str) and s in allowed]
        if kept:
            out[tier] = kept
    return out


def _validate_relationships(obj) -> list:
    """Keep relationships whose subject/object are valid subs and relation valid."""
    rels = []
    if isinstance(obj, dict):
        obj = obj.get("relationships", [])
    if not isinstance(obj, list):
        return rels
    for r in obj:
        if not isinstance(r, dict):
            continue
        subj, rel, objc = r.get("subject"), r.get("relation"), r.get("object")
        if subj in VALID_SUBS and objc in VALID_SUBS and rel in VALID_RELATIONS:
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
    raw1 = _call_ollama(model_name, SYSTEM_TASK1, _build_task1_prompt(row, fewshot))
    parsed1 = _extract_json(raw1)

    if parsed1 is None:
        return {
            "ev_id": ev_id, "entities_json": "[]", "hfacs_json": "{}",
            "relationships_json": "[]", "extraction_status": "parse_error",
        }

    entities = parsed1.get("entities", [])
    if not isinstance(entities, list):
        entities = []
    classifications = _validate_classifications(parsed1.get("hfacs_classifications"))

    status = "success" if (entities or classifications) else "empty"

    relationships = []
    if status == "success":
        raw2 = _call_ollama(model_name, SYSTEM_TASK2,
                            _build_task2_prompt(row, entities))
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    _GEN_OPTIONS["num_ctx"] = args.num_ctx
    model_name = _resolve_model(args.model)
    logging.info("Using model: %s (num_ctx=%d, sleep=%.1fs)",
                 model_name, args.num_ctx, args.sleep)

    df = pd.read_csv(args.input, dtype=str)
    train_ids = ntsb_train_ids(df)
    df = df[df["ev_id"].astype(str).isin(train_ids)].reset_index(drop=True)
    logging.info("NTSB training-split records: %d", len(df))

    # Seed few-shot snippet cache from the narratives we may process.
    for _, r in df.iterrows():
        _SNIPPET_CACHE[str(r["ev_id"])] = _clean(r.get("combined_text"))

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
