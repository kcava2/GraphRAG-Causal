"""
HFACS Hybrid Extractor for NTSB Data
=====================================
Tier-by-tier HFACS classification driven by the consolidated HFACS_SCHEMA
defined in config/dag_config.py. Each accident is classified independently
for every tier (Organizational Climate, Supervisory Conditions, Personnel
Conditions, Operator Conditions, Unsafe Acts) by an Ollama LLM, with a
validate-and-retry pass for missing fields and a rule-based keyword fallback
for any subcategory the LLM never returns.

Pipeline per tier:
    1. build_tier_prompt() asks the LLM for a flat YES/NO JSON for the
       subcategories of that tier only.
    2. validate_and_retry() re-prompts once for any field returned blank,
       None or UNKNOWN.
    3. rule_based_fallback() fills any subcategory still missing by keyword
       matching against HFACS_KEYWORDS.

Output: one row per accident with one boolean column per subcategory in the
consolidated schema. Optionally writes the same structure to a Neo4j
property graph when --neo4j is passed.

Requirements: neo4j, ollama, pandas, torch, scikit-learn, anthropic (optional)
"""

import json
import math
import os
import re
import sys
import time
import argparse
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import ollama

# Ensure the repo root is importable when this file is run as a script
# (e.g. `python data/hfacs_extractor.py`). config/ lives next to data/.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Single source of truth — HFACS tier structure, keyword lists, Neo4j config
from config.dag_config import (
    HFACS_SCHEMA,
    HFACS_KEYWORDS,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASS,
    DAG_EDGES,
)

try:
    from neo4j import GraphDatabase
except ImportError:  # neo4j is optional at runtime
    GraphDatabase = None

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "qwen2.5:14b"

# Flat list of every subcategory across all tiers — column order is stable.
ALL_SUBS: list[str] = [
    sub for tier in HFACS_SCHEMA.values() for sub in tier["subs"]
]

YES_NO_TOKENS = {"yes", "no"}


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def _clean(text) -> str:
    if text is None:
        return ""
    if isinstance(text, float) and math.isnan(text):
        return ""
    s = str(text).strip()
    return "" if s.lower() in ("nan", "none", "") else s


def _truncate(text: str, max_chars: int) -> str:
    return text[:max_chars] + "…" if len(text) > max_chars else text


def _combine_fields(probable_cause: str, findings: str, narrative: str,
                    extra: str = "") -> str:
    """Pack the four NTSB text fields into a single LLM prompt body."""
    parts = []
    if probable_cause:
        parts.append(f"PROBABLE CAUSE:\n{probable_cause}")
    if findings:
        parts.append(f"FINDINGS:\n{findings}")
    if narrative:
        parts.append(f"NARRATIVE:\n{narrative}")
    if extra:
        parts.append(f"OTHER:\n{extra}")
    return _truncate("\n\n".join(parts), 3000)


# ---------------------------------------------------------------------------
# Tier prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an HFACS (Human Factors Analysis and Classification System) "
    "expert analyst specializing in aviation accident investigation. You "
    "respond ONLY with valid JSON. Never include markdown, code fences, or "
    "any explanation outside the JSON.\n\n"
    "LEAN TOWARD YES. Aviation-accident narratives often only IMPLICITLY "
    "reference organizational, supervisory, physiological and mental-state "
    "factors. If there is any plausible evidence — explicit or implicit — "
    "that a factor contributed, answer YES. Examples of implicit cues that "
    "still warrant YES:\n"
    "  * Inadequate training, deferred maintenance, scheduling problems, or "
    "regulatory non-compliance → supervisory / organizational factors.\n"
    "  * Long duty days, early-morning departures, consecutive flights, "
    "repeated errors, or rushed actions → adverse mental state.\n"
    "  * Any medication, illness, fatigue, or post-accident medical finding "
    "→ adverse physiological state.\n"
    "  * Company-wide procedures, operations manual gaps, or production/cost "
    "pressure → organizational climate.\n\n"
    "Answer NO only when the text gives clear evidence the factor was NOT "
    "present, or when no part of the narrative could reasonably be read as "
    "touching on it. Never leave a field blank, and never use UNKNOWN — if "
    "uncertain, lean YES."
)


def build_tier_prompt(
    tier_id: str,
    probable_cause: str,
    findings: str,
    narrative: str,
    extra: str = "",
    extra_instruction: str | None = None,
) -> str:
    """
    Build a tier-scoped HFACS classification prompt.

    The prompt asks the model to return a flat JSON object with one key per
    subcategory of the specified tier (taken from HFACS_SCHEMA) and the
    string value YES or NO. The `extra_instruction` argument is prepended to
    the user message and is used by validate_and_retry() to demand corrected
    values for specific missing fields.
    """
    tier = HFACS_SCHEMA[tier_id]
    label = tier["label"]
    subs = tier["subs"]

    combined = _combine_fields(probable_cause, findings, narrative, extra)
    schema_json = json.dumps({s: "YES_or_NO" for s in subs}, indent=2)

    header = (
        f"Classify the following NTSB accident text for the HFACS tier "
        f"'{label}'. Consider ONLY these subcategories:\n"
        + "\n".join(f"  - {s}" for s in subs)
    )

    instructions = (
        "Return ONLY a flat JSON object with these subcategory names as keys "
        "and the string value YES or NO. If a factor is genuinely absent, "
        "mark it NO. Never leave a field blank and never use UNKNOWN."
    )

    extra_block = f"{extra_instruction}\n\n" if extra_instruction else ""

    return (
        f"{extra_block}{header}\n\n"
        f"ACCIDENT TEXT:\n{combined}\n\n"
        f"{instructions}\n\n"
        f"Respond with ONLY this JSON shape (replace YES_or_NO with YES or NO):\n"
        f"{schema_json}"
    )


def parse_tier_response(raw: str, tier_id: str) -> dict[str, str]:
    """Parse the LLM's JSON and coerce each value to 'YES' or 'NO' or ''."""
    subs = HFACS_SCHEMA[tier_id]["subs"]
    clean = raw.strip()
    if "```" in clean:
        for part in clean.split("```"):
            part = part.strip().lstrip("json").strip()
            if part.startswith("{"):
                clean = part
                break
    start = clean.find("{")
    end = clean.rfind("}") + 1
    if start != -1 and end > start:
        clean = clean[start:end]
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        return {s: "" for s in subs}

    result: dict[str, str] = {}
    for s in subs:
        v = parsed.get(s, "")
        if v is None:
            v = ""
        v_str = str(v).strip().lower()
        if v_str in {"yes", "y", "true", "1"}:
            result[s] = "YES"
        elif v_str in {"no", "n", "false", "0"}:
            result[s] = "NO"
        else:
            result[s] = ""  # missing — caller will retry / fallback
    return result


def _call_ollama(model_name: str, prompt: str, retries: int = 3) -> str | None:
    """Single Ollama call with retries. Returns raw text or None on failure."""
    for attempt in range(1, retries + 1):
        try:
            response = ollama.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.0},
            )
            return response["message"]["content"]
        except Exception as e:
            err = str(e)
            if "connection" in err.lower() or "refused" in err.lower():
                raise SystemExit(
                    "\nERROR: Cannot connect to Ollama.\n"
                    "Make sure Ollama is running: open the Ollama app or run "
                    "'ollama serve'."
                )
            logging.error(f"LLM error attempt {attempt}: {e}")
            if attempt < retries:
                time.sleep(2)
    return None


def validate_and_retry(
    result: dict[str, str],
    tier_id: str,
    probable_cause: str,
    findings: str,
    narrative: str,
    extra: str,
    model_name: str,
) -> dict[str, str]:
    """
    Look for keys with missing/empty values and re-prompt once for them.

    Identifies any field that came back None/empty/UNKNOWN. If at least one
    such field exists, builds a new tier prompt with an extra_instruction
    naming those fields explicitly and calls Ollama one more time. Returns
    the corrected dict — any field that is still missing after the retry is
    left for rule_based_fallback() to handle.
    """
    missing = [
        k for k, v in result.items()
        if v is None or str(v).strip() == "" or str(v).strip().upper() == "UNKNOWN"
    ]
    if not missing:
        return result

    extra_instruction = (
        "RETRY: in your previous response the following subcategories were "
        "missing, blank, or UNKNOWN: "
        + ", ".join(missing)
        + ". You MUST return YES or NO for every one of them in this attempt."
    )
    prompt = build_tier_prompt(
        tier_id, probable_cause, findings, narrative, extra,
        extra_instruction=extra_instruction,
    )
    raw = _call_ollama(model_name, prompt, retries=2)
    if raw is None:
        return result

    retried = parse_tier_response(raw, tier_id)
    for k in missing:
        v = retried.get(k, "")
        if v in ("YES", "NO"):
            result[k] = v
    return result


def rule_based_fallback(tier_id: str, combined_text: str) -> dict[str, str]:
    """
    Pure-keyword fallback for any subcategory the LLM never returned.

    Looks up the subcategories of the tier from HFACS_SCHEMA and the keyword
    lists from HFACS_KEYWORDS. Returns YES if any keyword for the subcategory
    occurs in the lowercased combined text, otherwise NO.
    """
    subs = HFACS_SCHEMA[tier_id]["subs"]
    lower = combined_text.lower()
    out: dict[str, str] = {}
    for sub in subs:
        keywords = HFACS_KEYWORDS.get(sub, [])
        hit = any(kw.lower() in lower for kw in keywords)
        out[sub] = "YES" if hit else "NO"
    return out


def _keyword_score(text_lower: str, sub: str) -> int:
    """Count keyword hits for one subcategory in lowercased text."""
    return sum(1 for kw in HFACS_KEYWORDS.get(sub, []) if kw.lower() in text_lower)


# ---------------------------------------------------------------------------
# Structured NTSB Findings tag map — high-precision signal
# These map the NTSB's curated Findings tags (substrings) onto the consolidated
# subcategory names. A hit here is treated as ground truth (cannot be flipped
# to NO downstream) because Findings tags are human-coded, not free text.
# ---------------------------------------------------------------------------

FINDINGS_TAG_MAP: dict[str, list[str]] = {
    # Organizational Climate
    "Safety Culture": [
        "organizational issues - safety",
        "safety management",
        "safety program",
        "production pressure",
        "organizational issues - culture",
    ],
    "Structure": [
        "organizational issues - staffing",
        "organizational issues - structure",
        "staffing",
    ],
    # Supervisory Conditions
    "Inadequate Supervision": [
        "supervision - inadequate", "oversight - inadequate",
        "supervision - lack of", "organizational issues - training",
        "training - inadequate", "training - lack of",
        "training - not provided", "qualification - not met",
        "currency - not met", "proficiency - not checked",
        "management - inadequate", "oversight - lack of",
        # Supervisory Violations tags absorbed here per the consolidation
        "supervisory violation", "management - violation",
        "regulatory - violation", "far - violation",
        "organizational issues - regulatory",
        "organizational issues - violation",
    ],
    "Planned Inappropriate Operations": [
        "planned inappropriate operation", "unauthorized operation",
        "exceeded limitations", "scheduling - inadequate",
        "duty time - exceeded", "rest - inadequate",
        "fatigue risk management", "dispatch - inappropriate",
        "organizational issues - scheduling",
    ],
    "Failed to Correct Known Problem": [
        "failure to correct", "known problem",
        "previously identified", "recurring",
        "maintenance - deferred", "airworthiness - not addressed",
        "corrective action - not taken",
        "organizational issues - corrective action",
    ],
    # Personnel Conditions
    "Crew Resource Management": [
        "crew resource management", "crm",
        "crew - coordination", "crew - communication",
        "communication - inadequate", "callout - missed",
    ],
    "Personal Readiness": [
        "personnel issues - physical - alertness",
        "alertness/fatigue", "fatigue",
        "rest - inadequate", "duty time",
        "fitness for duty",
    ],
    # Operator Conditions
    # (Adverse Mental State absorbs the former Mental Limitations tags)
    "Adverse Mental State": [
        "alertness/fatigue", "fatigue", "mental state",
        "attention - divided", "complacency", "distraction",
        "task saturation", "psychological", "mental/emotional",
        "emotional state", "anxiety", "stress", "fixation",
        "channelized attention", "workload", "inattention",
        "overconfidence", "personnel issues - psychological",
        "personnel issues - physical - alertness",
    ],
    # (Adverse Physiological State absorbs the former Physical Limitations tags)
    "Adverse Physiological State": [
        "medical/health - history", "medical/health",
        "physiological", "incapacitation",
        "substance", "alcohol", "hypoxia",
        "physical - health", "medical - condition",
        "drug", "medication", "illness",
        "carbon monoxide",
        "personnel issues - physical - health",
        "predisposing condition",
        "physical - limitation", "visual limitation",
        "hearing - limited", "color vision",
        "physical capability", "not rated",
    ],
    # Unsafe Acts
    "Decision Errors": [
        "decision making/judgment", "poor decision",
        "inadequate decision", "continued vfr into imc",
        "risk assessment", "planning/preparation",
        "preflight planning", "fuel planning",
        "weight and balance", "improper planning",
        "inadequate preflight", "judgment - poor",
        "crew decision",
    ],
    "Skill-based Errors": [
        "aircraft control", "procedure - improper",
        "loss of control", "technique - improper",
        "skill/knowledge", "aircraft handling",
        "directional control", "airspeed control",
        "altitude control", "stall/spin",
        "landing - hard", "gear - retracted",
        "checklist - not followed", "checklist - improper",
        "fuel mismanagement", "automation", "configuration",
        "inadvertent",
    ],
    "Perceptual Errors": [
        "spatial disorientation", "visual illusion",
        "terrain - awareness", "see and avoid",
        "situational awareness", "obstacle - awareness",
        "altitude awareness", "wire strike",
        "midair collision", "traffic - not seen",
        "terrain - not seen", "obstacle - not seen",
        "ground proximity",
    ],
    "Routine Violations": [
        "procedures - not followed",
        "checklist - not followed",
        "sop - not followed",
        "routine violation",
    ],
}


def score_findings_tags(findings_text: str, tier_id: str) -> dict[str, str]:
    """
    Look for high-precision NTSB Findings tag substrings.

    Returns YES for any subcategory whose tag list has a hit in the lowercased
    findings text, NO otherwise. Findings-tag YES is treated as ground truth
    in extract_row (it cannot be flipped to NO downstream).
    """
    subs = HFACS_SCHEMA[tier_id]["subs"]
    if not findings_text:
        return {s: "NO" for s in subs}
    lower = findings_text.lower()
    out: dict[str, str] = {}
    for sub in subs:
        tags = FINDINGS_TAG_MAP.get(sub, [])
        out[sub] = "YES" if any(t in lower for t in tags) else "NO"
    return out


# ---------------------------------------------------------------------------
# Tier-level force-pick — re-prompt when the LLM returned all NOs for a tier
# the dataset almost always has something for. Mirrors the OLD extractor's
# FORCE_PICK_CATEGORIES policy, restricted to supervisory & operator.
# ---------------------------------------------------------------------------

FORCE_PICK_TIERS: set[str] = {"supervisory", "operator"}

FORCE_PICK_GUIDANCE: dict[str, str] = {
    "supervisory": (
        "You previously returned NO for every subcategory of Supervisory "
        "Conditions. Reconsider — supervisory failures are often IMPLICIT in "
        "NTSB text. Inadequate training, poor scheduling, deferred "
        "maintenance, known problems ignored, or regulatory non-compliance "
        "by the organization all qualify. Did the organization do everything "
        "it should have to prevent this accident? Pick at least the single "
        "most plausible YES if any indirect evidence exists. Only confirm "
        "all-NO if the text makes absolutely clear the organization was "
        "fully compliant and proactive."
    ),
    "operator": (
        "You previously returned NO for every subcategory of Operator "
        "Conditions. Reconsider — aviation accidents almost always involve "
        "some degraded operator state. Was the pilot fatigued, stressed, "
        "distracted, or rushed? Was there any medical condition, medication, "
        "or physical limitation? Infer from context: a pilot who made "
        "repeated errors likely had an adverse mental state. Pick at least "
        "the single most plausible YES if any indirect evidence exists."
    ),
}


def _force_pick_retry(
    tier_id: str,
    probable_cause: str,
    findings: str,
    narrative: str,
    extra: str,
    model_name: str,
) -> dict[str, str]:
    """One extra LLM pass that demands at least one YES for the given tier."""
    extra_instruction = FORCE_PICK_GUIDANCE[tier_id]
    prompt = build_tier_prompt(
        tier_id, probable_cause, findings, narrative, extra,
        extra_instruction=extra_instruction,
    )
    raw = _call_ollama(model_name, prompt, retries=2)
    if raw is None:
        return {s: "NO" for s in HFACS_SCHEMA[tier_id]["subs"]}
    return parse_tier_response(raw, tier_id)


# ---------------------------------------------------------------------------
# Combined extraction for one row
# ---------------------------------------------------------------------------

def _merge_yes(*results: dict[str, str]) -> dict[str, str]:
    """OR-merge several {sub: YES|NO} dicts — YES if any source said YES."""
    out: dict[str, str] = {}
    keys: list[str] = []
    for r in results:
        for k in r:
            if k not in out:
                keys.append(k)
                out[k] = "NO"
    for r in results:
        for k, v in r.items():
            if v == "YES":
                out[k] = "YES"
    return out


def _force_unsafe_pick(combined_text: str) -> dict[str, str]:
    """
    Guarantee at least one Unsafe Acts YES per row.

    Every record in this dataset is an accident, so HFACS doctrine says an
    unsafe act is always present. Strategy:
      1. Pick the subcategory with the highest keyword score.
      2. If no keyword fires, fall back to 'Skill-based Errors' — the most
         generic execution-failure bucket and the highest base rate in our
         own extraction history.
    """
    subs = HFACS_SCHEMA["unsafe"]["subs"]
    lower = combined_text.lower()
    scores = {s: _keyword_score(lower, s) for s in subs}
    best_sub = max(scores, key=scores.get)
    if scores[best_sub] == 0:
        best_sub = "Skill-based Errors"
    return {s: ("YES" if s == best_sub else "NO") for s in subs}


def extract_row(
    model_name: str,
    row: pd.Series,
    rules_only: bool = False,
    llm_only: bool = False,
    aggressive: bool = True,
) -> dict[str, str]:
    """
    Run the full tier-by-tier extraction for one accident row.

    Aggressive mode (default) layers five signals per tier:
        1. Structured NTSB Findings tag matches (highest precision — sticky YES)
        2. LLM tier prompt (YES/NO per subcategory)
        3. validate_and_retry() — re-prompt once for blank/UNKNOWN cells
        4. rule_based_fallback() — keyword OR over the LLM output
        5. Tier-level force-pick retry for supervisory & operator when both
           previous passes returned all-NO

    Plus a hard guarantee: the Unsafe Acts tier always has at least one YES
    (these are aviation accidents — an unsafe act is always present).

    rules_only / llm_only flags preserve legacy behavior for ablations.
    """
    probable_cause = _clean(row.get("ProbableCause"))
    findings = _clean(row.get("Findings"))
    narrative = _clean(row.get("narratives.narr_cause"))
    extra = _clean(row.get("narratives.narr_accp"))

    combined_text = _combine_fields(probable_cause, findings, narrative, extra)

    flat: dict[str, str] = {}
    for tier_id in HFACS_SCHEMA:
        subs = HFACS_SCHEMA[tier_id]["subs"]

        # ── Rules-only ablation path ─────────────────────────────────────
        if rules_only:
            tier_result = _merge_yes(
                score_findings_tags(findings, tier_id),
                rule_based_fallback(tier_id, combined_text),
            )
            flat.update(tier_result)
            continue

        # ── LLM pass ─────────────────────────────────────────────────────
        prompt = build_tier_prompt(
            tier_id, probable_cause, findings, narrative, extra,
        )
        raw = _call_ollama(model_name, prompt)
        llm_result = parse_tier_response(raw, tier_id) if raw is not None else {s: "" for s in subs}

        if not llm_only:
            llm_result = validate_and_retry(
                llm_result, tier_id, probable_cause, findings, narrative,
                extra, model_name,
            )

        # Coerce any remaining blanks to NO so downstream OR is well-defined
        llm_clean = {s: (llm_result.get(s) if llm_result.get(s) in ("YES", "NO") else "NO") for s in subs}

        if aggressive and not llm_only:
            # ── OR keyword + findings-tag hits on top of the LLM verdict ─
            tier_result = _merge_yes(
                llm_clean,
                rule_based_fallback(tier_id, combined_text),
                score_findings_tags(findings, tier_id),
            )

            # ── Force-pick retry for tiers that almost always have YES ───
            if (
                tier_id in FORCE_PICK_TIERS
                and all(v == "NO" for v in tier_result.values())
            ):
                retry = _force_pick_retry(
                    tier_id, probable_cause, findings, narrative, extra,
                    model_name,
                )
                tier_result = _merge_yes(tier_result, retry)
        else:
            # Conservative mode: rule-based only fills gaps, no OR fusion
            tier_result = llm_clean
            still_missing = [k for k, v in tier_result.items() if v not in ("YES", "NO")]
            if still_missing:
                fb = rule_based_fallback(tier_id, combined_text)
                for k in still_missing:
                    tier_result[k] = fb[k]

        # ── Hard guarantee: Unsafe Acts must have at least one YES ───────
        if tier_id == "unsafe" and all(v == "NO" for v in tier_result.values()):
            tier_result = _force_unsafe_pick(combined_text)

        flat.update(tier_result)

    return flat


# ---------------------------------------------------------------------------
# Neo4j writer
# ---------------------------------------------------------------------------

# Tier-to-tier causal edges projected from DAG_EDGES for the Neo4j writer.
# These are the canonical HFACS chain links used between tier nodes.
_TIER_NEO4J_EDGES = [
    ("org_climate", "supervisory"),
    ("supervisory", "operator"),
    ("operator", "unsafe"),
]


def write_to_neo4j(driver, ntsb_no: str, result: dict[str, str]) -> None:
    """
    Persist one accident's HFACS classification to Neo4j as a small subgraph.

    Creates an AccidentEvent node keyed on NtsbNo, plus one node per HFACS
    tier whose label comes from HFACS_SCHEMA. Subcategory YES/NO values are
    stored as properties on the tier nodes. Edges follow the causal ordering
    in DAG_EDGES (org_climate → supervisory → operator → unsafe). All writes
    use MERGE so reruns are idempotent. Failures are swallowed so extraction
    is never blocked by Neo4j issues.
    """
    if driver is None:
        return
    try:
        with driver.session() as session:
            session.run(
                "MERGE (a:AccidentEvent {ntsb_no: $ntsb_no})",
                ntsb_no=ntsb_no,
            )
            for tier_id, tier in HFACS_SCHEMA.items():
                label = tier["label"]
                props = {s: result.get(s, "") for s in tier["subs"]}
                props["source"] = "llm"
                # MERGE on (label, ntsb_no) so each tier node is per-accident.
                session.run(
                    f"""
                    MERGE (n:`{label}` {{ntsb_no: $ntsb_no}})
                    SET n += $props
                    WITH n
                    MATCH (a:AccidentEvent {{ntsb_no: $ntsb_no}})
                    MERGE (a)-[:HAS_TIER {{tier: $tier_id}}]->(n)
                    """,
                    ntsb_no=ntsb_no, props=props, tier_id=tier_id,
                )
            for parent_id, child_id in _TIER_NEO4J_EDGES:
                p_label = HFACS_SCHEMA[parent_id]["label"]
                c_label = HFACS_SCHEMA[child_id]["label"]
                session.run(
                    f"""
                    MATCH (p:`{p_label}` {{ntsb_no: $ntsb_no}})
                    MATCH (c:`{c_label}` {{ntsb_no: $ntsb_no}})
                    MERGE (p)-[:CAUSES]->(c)
                    """,
                    ntsb_no=ntsb_no,
                )
    except Exception as e:  # pragma: no cover — Neo4j optional at runtime
        logging.warning(f"Neo4j write failed for {ntsb_no}: {e}")


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save(result_rows: list[dict], output_path: str, out_path: Path) -> None:
    new_df = pd.DataFrame(result_rows)
    if out_path.exists():
        existing = pd.read_csv(output_path, encoding="latin1")
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.drop_duplicates(subset=["NtsbNo"], keep="last", inplace=True)
        combined.to_csv(output_path, index=False)
    else:
        new_df.to_csv(output_path, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="HFACS hybrid extractor (tier-based LLM + rule fallback)"
    )
    _here = Path(__file__).parent
    parser.add_argument("--input",   default=str(_here / "ntsb text fields.csv"))
    parser.add_argument("--output",  default=str(_here / "hfacs_results.csv"))
    parser.add_argument("--model",   default=DEFAULT_MODEL)
    parser.add_argument("--limit",   type=int, default=None)
    parser.add_argument("--resume",  action="store_true")
    parser.add_argument("--rerun-errors", action="store_true")
    parser.add_argument("--rules-only", action="store_true",
                        help="Use rule-based fallback only, no Ollama")
    parser.add_argument("--llm-only",   action="store_true",
                        help="Use Ollama only, skip validate-and-retry")
    parser.add_argument("--neo4j", action="store_true",
                        help="Also write each row's classification to Neo4j")
    # Aggressive mode is ON by default — pass --no-aggressive to disable.
    parser.add_argument(
        "--aggressive", dest="aggressive", action="store_true", default=True,
        help="OR keyword + findings-tag hits over LLM verdicts, force-pick "
             "supervisory/operator all-NO tiers, and guarantee at least one "
             "Unsafe Acts YES per row (default: on).",
    )
    parser.add_argument(
        "--no-aggressive", dest="aggressive", action="store_false",
        help="Disable aggressive extraction (conservative: LLM + rule "
             "gap-fill only).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if not args.rules_only:
        try:
            models = [m.model for m in ollama.list().models]
            logging.info(f"Ollama running. Available models: {models}")
            if not any(args.model in m for m in models):
                raise SystemExit(
                    f"\nModel '{args.model}' not found.\nRun: ollama pull "
                    f"{args.model}\nAvailable: {models}"
                )
        except SystemExit:
            raise
        except Exception:
            raise SystemExit(
                "\nERROR: Cannot connect to Ollama.\n"
                "Run 'ollama serve' or open the Ollama app first."
            )
        logging.info(f"Model: {args.model}")

    # Optional Neo4j driver — print warning and continue on failure
    driver = None
    if args.neo4j:
        if GraphDatabase is None:
            print("WARNING: neo4j package not installed — continuing without graph writes.")
        else:
            try:
                driver = GraphDatabase.driver(
                    NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS)
                )
                # Test connection eagerly so the warning fires once at startup
                driver.verify_connectivity()
                logging.info(f"Connected to Neo4j at {NEO4J_URI}")
            except Exception as e:
                print(f"WARNING: cannot reach Neo4j ({e}) — continuing without graph writes.")
                driver = None

    df = pd.read_csv(args.input, encoding="latin1")
    if args.limit:
        df = df.head(args.limit)
    logging.info(f"Loaded {len(df)} rows from {args.input}")

    already_done: set[str] = set()
    out_path = Path(args.output)
    if out_path.exists():
        done_df = pd.read_csv(args.output, encoding="latin1")
        if args.rerun_errors:
            # Treat any row with a blank cell across the subcategory columns as bad
            sub_cols = [c for c in ALL_SUBS if c in done_df.columns]
            if sub_cols:
                clean_mask = done_df[sub_cols].applymap(
                    lambda v: str(v).strip().upper() in {"YES", "NO"}
                ).all(axis=1)
                already_done = set(done_df.loc[clean_mask, "NtsbNo"].astype(str))
                logging.info(f"Rerun-errors: skipping {len(already_done)} clean rows")
        elif args.resume:
            already_done = set(done_df["NtsbNo"].astype(str))
            logging.info(f"Resume: {len(already_done)} rows already done")

    rows_to_process = df[~df["NtsbNo"].astype(str).isin(already_done)]
    logging.info(f"Rows to process: {len(rows_to_process)}")

    if rows_to_process.empty:
        logging.info("Nothing to do.")
        return

    result_rows: list[dict] = []
    for _, row in tqdm(rows_to_process.iterrows(),
                       total=len(rows_to_process), desc="Extracting"):
        ntsb_no = str(row["NtsbNo"])
        features = extract_row(
            args.model, row,
            rules_only=args.rules_only,
            llm_only=args.llm_only,
            aggressive=args.aggressive,
        )
        base = {"NtsbNo": ntsb_no, "Event.Id": row.get("Event.Id", "")}
        base.update(features)
        result_rows.append(base)

        if driver is not None:
            write_to_neo4j(driver, ntsb_no, features)

        if len(result_rows) % 25 == 0:
            _save(result_rows, args.output, out_path)
            logging.info(f"Checkpoint: {len(result_rows)} rows saved")

    _save(result_rows, args.output, out_path)
    logging.info(f"Done. Results saved to {args.output}")

    if driver is not None:
        driver.close()

    # ── Summary ──────────────────────────────────────────────────────────
    result_df = pd.read_csv(args.output, encoding="latin1")
    total = len(result_df)
    print(f"\n--- HFACS Extraction Summary ({total} total rows) ---")
    for tier_id, tier in HFACS_SCHEMA.items():
        print(f"\n  {tier['label']}:")
        for sub in tier["subs"]:
            if sub not in result_df.columns:
                continue
            n_yes = (result_df[sub].astype(str).str.upper() == "YES").sum()
            bar = "█" * int(30 * n_yes / total) if total else ""
            print(f"    {sub:<38} {n_yes:4d} ({100*n_yes/total:5.1f}%)  {bar}")


if __name__ == "__main__":
    main()
