#!/usr/bin/env python
"""
run_all.py — unattended driver for the HFACS extraction + knowledge-graph pipeline.

Automates the sequence in HANDOFF.md end to end, with one change: every Ollama call
uses the model ``qwen3.8:27b`` instead of ``gemma4:12b-it-qat``. The model is assumed
to be installed already — this script never pulls it, never checks the registry for a
newer digest, and never installs anything.

Stages (in order):

    1. pilot extraction          data/hfacs_extractor.py --limit 150      ~15 min
    2. comparison vs current     data/compare_extractions.py             seconds
    3. full extraction           data/hfacs_extractor.py --split all      ~1.5 h
    4. violation adjudication    data/adjudicate_violation.py             ~40 min
    5. clear the Neo4j graph     MATCH (n) DETACH DELETE n               seconds
    6. preflight KG build        data/kg_builder.py --limit 5 --dry-run   ~1 min
    7. KG build, graph only      data/kg_builder.py --source all          ~1.5 h
                                 --skip-faiss
    8. FAISS indexes, then the   data/kg_builder.py --faiss-only          minutes
       NTSB training events      data/kg_builder.py --ingest-ntsb-train
       into the graph
    9. Neo4j dump                neo4j-admin database dump                minutes

Total: roughly 4 hours on the RTX 5090 with qwen3.8:27b (timings observed on
2026-09-27), almost all of it in stages 3, 4 and 7.

Stage 8 also writes the NTSB TRAINING events into the graph (no model calls: they
are written from the labels stages 3 and 4 produced). Retrieval in the experiment
reads the graph and nothing else, and those events are where almost all of the
graph's outcome information is. Validation and test events are never written. This
has to follow stage 5, which empties the graph, and stages 3-4, which set the labels;
after a labels-only run (--stop-after 4) the closing message says to run it by hand.

Load on the machine. Stages 1-7 are GPU work: the model runs in Ollama and the CPU
is mostly idle. The one CPU-heavy step is embedding the KG narratives for the FAISS
indexes, and three things keep it as light as possible:

  * It happens ONCE. Stage 7 used to build the three indexes at the end of the graph
    build and stage 8 then rebuilt the same three, doubling the heaviest stretch of
    the run. Stage 7 now passes --skip-faiss; stage 8 builds them, with the
    alignment check that proves they match the graph.
  * The model is unloaded as soon as the last LLM stage in the run finishes
    (`ollama stop`). Left alone, Ollama keeps it for five minutes and then unloads
    and re-probes the GPU, which landed in the middle of the embedding.
  * Stage 8 runs with a capped CPU thread count (EMBED_CPU_THREADS, default 6;
    override with RUN_ALL_EMBED_THREADS). With a CUDA build of PyTorch the
    embedding itself runs on the GPU and the cap only bounds the tokenizer.

Stage numbering changed on 2026-09-20: the violation adjudication was inserted as
stage 4, so the old stages 4-8 are now 5-9. ``--start-at 8`` used to mean the dump;
it now means the FAISS build, and the dump is ``--start-at 9``.

Backups. Before stage 1 (and before any stage 3 that starts over) the script copies
the CURRENT ``data/hfacs_results.csv`` to a run-stamped file,
``data/hfacs_results.pre-run.<timestamp>.bak.csv``. That backup is not optional
bookkeeping: stage 3 deletes ``hfacs_results.csv`` and rebuilds it, every trained
checkpoint and every published number rests on it, and stage 2 compares the pilot
against it. The previous scheme wrote one fixed backup name and never overwrote it,
so from the second run onward nothing was backed up at all and stage 2 compared
against a stale qwen2.5 extraction. Identical content is not backed up twice.

Stage 4 exists because ``hfacs_results.csv`` is not the last word on one label. The
``unsafe_violation`` tier is re-decided under the strict HFACS definition by
``adjudicate_violation.py`` and applied by the dataloader as an override
(``data/violation_adjudication.csv``). After a re-extraction that file is out of
date: its consensus candidates were chosen from the old labels, and the script
would treat every record as already done. Stage 4 moves the old file aside and
adjudicates afresh. It is skipped when the adjudication is already newer than the
extraction (``--force-adjudication`` overrides that).

Everything is checked before any long job starts: required files, the Neo4j
environment, Ollama reachability, the presence of ``qwen3.8:27b`` in ``ollama list``,
and — statically, by reading the source — that extraction still runs at
``temperature=0`` with structured outputs on and thinking off. Stage 4 is the one
deliberate exception to temperature 0: two of its three votes are sampled at 0.7,
because a majority vote over identical deterministic runs would be a vote of one.

Usage
-----
    python run_all.py                     # all nine stages
    python run_all.py --preflight-only    # run the checks only, then exit
    python run_all.py --plan              # print the exact commands, run nothing
    python run_all.py --start-at 6        # resume from stage 6 onward
    python run_all.py --resume-extraction # stage 3 without --force-binary (resumes)
    python run_all.py --stop-after 4      # labels only: no Neo4j needed at all
    python run_all.py --skip-dump         # stages 1-8 only

Environment
-----------
    NEO4J_URI       default bolt://localhost:7687
    NEO4J_USER      default neo4j
    NEO4J_PASSWORD  required, no default
    NEO4J_DATABASE  default neo4j

    NEO4J_CONTAINER default neo4j-graphrag   (stage 9, Docker backend)
    NEO4J_VOLUME    default neo4j-data       (stage 9, Docker backend)
    NEO4J_IMAGE     default neo4j:5          (stage 9, Docker backend)
    NEO4J_HOME      autodetected             (stage 9, Neo4j Desktop backend)

Stage 9 picks its backend at run time. If a container named NEO4J_CONTAINER
exists it uses HANDOFF.md's Docker recipe; otherwise it falls back to a Neo4j
Desktop DBMS (autodetected under ~/.Neo4jDesktop2/Data/dbmss/, or pinned with
NEO4J_HOME). The Desktop path needs the instance stopped first, and says so
rather than killing the server itself.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants — absolute paths anchored at the repo root
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
DATA = REPO_ROOT / "data"
PY = sys.executable

MODEL = "qwen3.8:27b"          # the only model this pipeline may use
NUM_CTX = 32768
CHECKPOINT_EVERY = 25
PILOT_LIMIT = 150

N_STAGES = 9
DUMP_STAGE = 9
OLLAMA_STAGES = (1, 3, 4, 6, 7)   # stages that call the model
NEO4J_STAGES = (5, 6, 7, 8)       # clear, KG preflight, KG build, FAISS (stamps the graph)

# CPU threads allowed for the stage-8 embedding. All cores at full load for minutes,
# then straight back to idle, is the harshest thing this pipeline does to the CPU.
# A value already set in the shell (OMP_NUM_THREADS etc.) is respected.
EMBED_CPU_THREADS = int(os.environ.get("RUN_ALL_EMBED_THREADS", "6"))
_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS")

# Scripts
HFACS_EXTRACTOR = DATA / "hfacs_extractor.py"
COMPARE_EXTRACTIONS = DATA / "compare_extractions.py"
KG_BUILDER = DATA / "kg_builder.py"
ADJUDICATE = DATA / "adjudicate_violation.py"
OLLAMA_JSON = DATA / "ollama_json.py"      # the adjudicator's Ollama helper

# Inputs — the three --csv flags below are mandatory for the KG build. Without
# them --source all defaults to asrs_clean.csv (44,448 records) and an 8-hour
# stage becomes a multi-week one.
NTSB_CLEAN = DATA / "ntsb_clean.csv"
ASIAS_SUBSET = DATA / "asias_subset.csv"
ASRS_SUBSET = DATA / "asrs_subset.csv"
NTSB_KG_SUBSET = DATA / "ntsb_kg_subset.csv"
NTSB_FAISS = DATA / "ntsb.faiss"
NTSB_FAISS_IDS = DATA / "ntsb_faiss_ids.json"

# Outputs — names kept exactly as HANDOFF.md's "what to send back" list has them.
# ("gemma4" in three of them is historical: every stage here runs qwen3.8:27b. The
# names are kept so existing files and the handoff list still line up.)
RESULTS_CSV = DATA / "hfacs_results.csv"
LEGACY_BASELINE_CSV = DATA / "hfacs_results.qwen25-7b.bak.csv"   # read-only history
PRE_RUN_GLOB = "hfacs_results.pre-run.*.bak.csv"
ADJUDICATION_CSV = DATA / "violation_adjudication.csv"
ADJUDICATION_PARTIAL = DATA / "violation_adjudication.partial.csv"
PILOT_CSV = DATA / "pilot_gemma4.csv"
EXTRACT_LOG = DATA / "extract_gemma4.log"
KG_BUILD_LOG = DATA / "kg_build_gemma4.log"

# Logs for the stages HANDOFF.md does not name explicitly.
PILOT_LOG = DATA / "pilot_extract.log"
COMPARE_LOG = DATA / "compare_extractions.log"
ADJUDICATE_LOG = DATA / "adjudicate_violation.log"
CLEAR_GRAPH_LOG = DATA / "clear_graph.log"
KG_PREFLIGHT_LOG = DATA / "kg_preflight.log"
KG_FAISS_LOG = DATA / "kg_faiss_only.log"
KG_INGEST_LOG = DATA / "kg_ingest_ntsb_train.log"
NTSB_TRAIN_FAISS = DATA / "ntsb_train_kg.faiss"
NTSB_TRAIN_IDMAP = DATA / "ntsb_train_kg_id_map.csv"
DUMP_LOG = DATA / "neo4j_dump.log"

NEO4J_DEFAULTS = {
    "NEO4J_URI": "bolt://localhost:7687",
    "NEO4J_USER": "neo4j",
    "NEO4J_DATABASE": "neo4j",
}

# Stage 9 container settings (Docker path from HANDOFF.md §6).
CONTAINER = os.environ.get("NEO4J_CONTAINER", "neo4j-graphrag")
VOLUME = os.environ.get("NEO4J_VOLUME", "neo4j-data")
IMAGE = os.environ.get("NEO4J_IMAGE", "neo4j:5")

# Stage 9, native path. HANDOFF.md assumes Neo4j runs in Docker; a Neo4j Desktop
# install has no container and no named volume, so the Docker recipe fails on its
# first command. NEO4J_HOME pins the DBMS directory (the one holding bin/, conf/
# and data/databases/); left unset, _desktop_home() looks for exactly one under
# ~/.Neo4jDesktop2/Data/dbmss/.
NEO4J_HOME = os.environ.get("NEO4J_HOME", "")
DESKTOP_DBMS_ROOT = Path.home() / ".Neo4jDesktop2" / "Data" / "dbmss"

TAIL_LINES = 50   # how much log to echo when a stage fails


# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------

def _stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def say(msg: str = "") -> None:
    print(f"[{_stamp()}] {msg}" if msg else "", flush=True)


def banner(msg: str) -> None:
    print("", flush=True)
    print("=" * 78, flush=True)
    print(f"  {msg}", flush=True)
    print("=" * 78, flush=True)


def human(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def quote(argv: list[str]) -> str:
    """Render a command the way a human would retype it."""
    out = []
    for a in argv:
        out.append(f'"{a}"' if " " in a else a)
    return " ".join(out)


def tail(path: Path, n: int = TAIL_LINES) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"(could not read {path}: {exc})"
    return "\n".join(lines[-n:])


class StageFailure(RuntimeError):
    """A pipeline stage exited non-zero, or a precondition was not met."""


# ---------------------------------------------------------------------------
# Command execution — every external command goes through here
# ---------------------------------------------------------------------------

def run_logged(argv: list[str], log_path: Path, *, label: str,
               cwd: Path = REPO_ROOT, check: bool = True,
               extra_env: dict | None = None) -> int:
    """Run `argv`, streaming combined stdout/stderr into `log_path`.

    Returns the exit code. Raises StageFailure on a non-zero exit when
    `check` is set. The child inherits this process's environment, so the
    NEO4J_* variables normalised in check_neo4j_env() reach it. `extra_env`
    adds variables for this one child only (the stage-8 thread cap).
    """
    say(f"{label}: {quote(argv)}")
    say(f"{label}: logging to {log_path}")
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
        say(f"{label}: with " + ", ".join(f"{k}={v}" for k, v in extra_env.items()))
    started = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as fh:
        fh.write(f"# {label}\n# {_stamp()}\n# {quote(argv)}\n")
        if extra_env:
            fh.write("# env: " + " ".join(f"{k}={v}" for k, v in extra_env.items()) + "\n")
        fh.write("\n")
        fh.flush()
        proc = subprocess.run(
            argv,
            cwd=str(cwd),
            stdout=fh,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    elapsed = time.time() - started
    if proc.returncode != 0:
        say(f"{label}: FAILED with exit code {proc.returncode} after {human(elapsed)}")
        print("", flush=True)
        print(f"--- last {TAIL_LINES} lines of {log_path.name} ---", flush=True)
        print(tail(log_path), flush=True)
        print("--- end of log tail ---", flush=True)
        if check:
            raise StageFailure(f"{label} exited {proc.returncode}")
    else:
        say(f"{label}: done in {human(elapsed)}")
    return proc.returncode


def run_capture(argv: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run a short command and capture its output (used by the checks only)."""
    return subprocess.run(
        argv,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


# ---------------------------------------------------------------------------
# Preflight checks — all of these run before any long job starts
# ---------------------------------------------------------------------------

def check_files() -> None:
    """Every script and input the run depends on must already exist."""
    required = [
        HFACS_EXTRACTOR, COMPARE_EXTRACTIONS, KG_BUILDER, ADJUDICATE, OLLAMA_JSON,
        NTSB_CLEAN, ASIAS_SUBSET, ASRS_SUBSET, NTSB_KG_SUBSET,
        NTSB_FAISS, NTSB_FAISS_IDS,
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise StageFailure(
            "Missing required files:\n  " + "\n  ".join(str(p) for p in missing)
        )
    say(f"Files: all {len(required)} required inputs present.")
    # ntsb.faiss and ntsb_faiss_ids.json feed few-shot examples into the prompts.
    # Rebuilding them silently changes what the model extracts, so this script
    # only ever reads them.
    say("Files: ntsb.faiss / ntsb_faiss_ids.json will be read, never rebuilt.")


def check_neo4j_env() -> None:
    """Fill in the documented defaults, require a password, normalise os.environ."""
    for key, default in NEO4J_DEFAULTS.items():
        if not os.environ.get(key):
            os.environ[key] = default
            say(f"Neo4j env: {key} unset — using default {default!r}")
    if not os.environ.get("NEO4J_PASSWORD"):
        raise StageFailure(
            "NEO4J_PASSWORD is not set in this terminal.\n"
            "  PowerShell : $env:NEO4J_PASSWORD=\"graphrag123\"\n"
            "  Git Bash   : export NEO4J_PASSWORD=graphrag123\n"
            "Without it the KG stage fails, or worse, silently writes nothing."
        )
    say(f"Neo4j env: URI={os.environ['NEO4J_URI']} "
        f"USER={os.environ['NEO4J_USER']} DATABASE={os.environ['NEO4J_DATABASE']} "
        "PASSWORD=(set)")


NEO4J_PING = """
import os, sys
from neo4j import GraphDatabase
drv = GraphDatabase.driver(
    os.environ["NEO4J_URI"],
    auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
)
drv.verify_connectivity()
with drv.session(database=os.environ.get("NEO4J_DATABASE", "neo4j")) as s:
    n = s.run("MATCH (n) RETURN count(n) AS n").single()["n"]
print("Neo4j OK - %d nodes currently in the graph" % n)
drv.close()
"""


def check_neo4j_connection() -> None:
    """Fail now rather than eight hours into the run."""
    res = run_capture([PY, "-c", NEO4J_PING])
    if res.returncode != 0:
        raise StageFailure(
            "Cannot connect to Neo4j.\n"
            "Check the container is up (docker ps) and that the NEO4J_* variables\n"
            "are set in *this* terminal.\n\n" + res.stdout.strip()
        )
    say(f"Neo4j: {res.stdout.strip()}")


def check_ollama_model() -> None:
    """Confirm qwen3.8:27b is present. Never pull, never check for updates."""
    if shutil.which("ollama") is None:
        raise StageFailure(
            "The 'ollama' executable is not on PATH. Install Ollama and make sure "
            "the server is running (the desktop app starts it, or 'ollama serve')."
        )
    res = run_capture(["ollama", "list"])
    if res.returncode != 0:
        raise StageFailure(
            "Cannot connect to Ollama ('ollama list' failed). Start the server "
            "with 'ollama serve' in its own terminal.\n\n" + res.stdout.strip()
        )
    tags = []
    for line in res.stdout.splitlines()[1:]:
        parts = line.split()
        if parts:
            tags.append(parts[0])
    if MODEL not in tags:
        raise StageFailure(
            f"Model '{MODEL}' not found in Ollama.\n"
            f"Installed: {tags or '(none)'}\n"
            "This script does not install models — it is assumed to be present. "
            "The extractor does a substring match on the tag, so a near-miss can "
            "resolve to something that does not exist."
        )
    say(f"Ollama: '{MODEL}' present. No pull attempted, no update check performed.")


def check_generation_settings() -> None:
    """Static verification that decoding is deterministic and unchanged.

    Reads the sources rather than importing them — no model is loaded and
    nothing is generated. CLAUDE.md: temperature=0 everywhere, structured
    outputs on by default, thinking off by default.
    """
    src = HFACS_EXTRACTOR.read_text(encoding="utf-8", errors="replace")
    if '_GEN_OPTIONS = {"temperature": 0.0}' not in src:
        raise StageFailure(
            f"{HFACS_EXTRACTOR} no longer sets _GEN_OPTIONS to temperature 0.0. "
            "Extraction must be deterministic; fix that before running."
        )
    if "_THINK = False" not in src:
        raise StageFailure(
            f"{HFACS_EXTRACTOR}: _THINK is no longer False by default. Thinking is "
            "pure runtime cost on a bounded extraction task and never reaches the "
            "parser."
        )
    say("Generation: temperature=0.0, structured outputs on, thinking off "
        "(verified statically; --think and --no-structured are never passed).")
    say("Note: kg_builder imports _call_ollama from hfacs_extractor, so it "
        "inherits both toggles.")

    # The strict HFACS violation rule lives in the pass-1 prompt. Without it the
    # extractor goes back to labelling almost any non-compliance (passengers, an
    # operator skipping a bulletin) as a violation. Stage 4 would still repair the
    # NTSB labels, so this is a warning rather than a stop.
    if "unsafe_violation is ONLY for a KNOWING deviation" not in src:
        say("WARNING: the strict unsafe_violation rule is missing from the pass-1 "
            "prompt in hfacs_extractor.py. Extraction will over-assign violations; "
            "stage 4 repairs the NTSB labels but the pilot comparison will not "
            "show the expected drop.")
    else:
        say("Generation: pass-1 prompt carries the strict HFACS violation rule.")

    # Stage 4 goes through data/ollama_json.py, not _call_ollama. Thinking must be
    # off there too, and its default model should be the pipeline's model (run_all
    # passes --model explicitly either way).
    helper = OLLAMA_JSON.read_text(encoding="utf-8", errors="replace")
    if '"think": False' not in helper:
        raise StageFailure(
            f"{OLLAMA_JSON} no longer sends think=False. The adjudication would pay "
            "for reasoning tokens on ~1,400 calls.")
    if f'DEFAULT_MODEL = "{MODEL}"' not in helper:
        say(f"WARNING: {OLLAMA_JSON.name} DEFAULT_MODEL is not {MODEL!r}. Stage 4 "
            f"passes --model {MODEL} explicitly, so this run is unaffected, but a "
            "manual run of the adjudicator would use a different model.")
    say("Generation: stage 4 samples two of its three votes at temperature 0.7 by "
        "design (a majority over identical deterministic runs is a vote of one). "
        "Every other stage is temperature 0.")


def _same_bytes(a: Path, b: Path) -> bool:
    if a.stat().st_size != b.stat().st_size:
        return False
    import hashlib
    digest = lambda p: hashlib.sha1(p.read_bytes()).hexdigest()
    return digest(a) == digest(b)


def latest_pre_run_backup() -> Path | None:
    """Newest run-stamped backup of hfacs_results.csv, if any."""
    found = sorted(DATA.glob(PRE_RUN_GLOB), key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


def comparison_baseline() -> Path:
    """What stage 2 compares the pilot against: the extraction this run replaces.

    Falls back to the live results file (stage 3 has not overwritten it yet when
    stage 2 runs), then to the historical qwen2.5 baseline.
    """
    for cand in (latest_pre_run_backup(), RESULTS_CSV, LEGACY_BASELINE_CSV):
        if cand is not None and cand.exists():
            return cand
    raise StageFailure(
        f"No extraction to compare against: neither a pre-run backup, "
        f"{RESULTS_CSV.name} nor {LEGACY_BASELINE_CSV.name} exists.")


def backup_current_results() -> None:
    """Copy the CURRENT hfacs_results.csv to a run-stamped file before it can be lost.

    Stage 3 (without --resume-extraction) deletes hfacs_results.csv and rebuilds
    it. Every checkpoint, results table and the violation adjudication rest on that
    file. The old scheme wrote one fixed backup name and never overwrote it, which
    protected the first run only.
    """
    if not RESULTS_CSV.exists():
        say(f"Backup: {RESULTS_CSV.name} does not exist yet — nothing to back up.")
        return
    newest = latest_pre_run_backup()
    if newest is not None and _same_bytes(newest, RESULTS_CSV):
        say(f"Backup: {newest.name} already holds the current results — not duplicated.")
        return
    dest = DATA / f"hfacs_results.pre-run.{datetime.now():%Y%m%d-%H%M%S}.bak.csv"
    shutil.copy2(RESULTS_CSV, dest)
    say(f"Backup: copied {RESULTS_CSV.name} -> {dest.name}")


def check_dump_backend() -> None:
    """Work out now whether stage 9 has anything to dump from.

    The old check only asked whether `docker` was on PATH. On a machine running
    Neo4j Desktop it is, so the run sailed through preflight and the dump failed
    eight hours later on `docker stop`. Resolve the real backend instead.
    """
    backend, home = resolve_dump_backend()
    if backend == "docker":
        say(f"Dump backend: Docker container {CONTAINER!r} (volume {VOLUME!r}).")
    elif backend == "desktop":
        say(f"Dump backend: Neo4j Desktop at {home}")
        if _bolt_is_up():
            say("WARNING: that instance is running. neo4j-admin cannot dump a "
                "mounted database, so stage 9 will ask you to stop it in Desktop "
                f"and resume with --start-at {DUMP_STAGE}.")
    else:
        say(f"WARNING: no Docker container named {CONTAINER!r} and no Neo4j "
            "Desktop DBMS found — stage 9 (Neo4j dump) will be skipped. "
            "HANDOFF.md notes that part is recoverable.")


def preflight(args: argparse.Namespace) -> None:
    """Check only what the requested stages actually need.

    Three cases matter. A dump-only run is the awkward one: the Desktop dump
    requires the server *stopped*, so pinging bolt here would fail the run before
    it reached the stage. A labels-only run (--stop-after 4) never touches Neo4j,
    so demanding a password and a live server for it would be a false stop. And
    nothing from the FAISS build onward talks to Ollama.
    """
    banner("Preflight checks")
    check_files()
    first, last = args.start_at, args.stop_after
    in_range = lambda stages: any(first <= n <= last for n in stages)
    needs_ollama = in_range(OLLAMA_STAGES)         # extraction, adjudication, KG build
    needs_neo4j = in_range(NEO4J_STAGES)           # clear, KG build, FAISS stamps the graph
    runs_dump = last >= DUMP_STAGE and not args.skip_dump
    if needs_ollama:
        check_generation_settings()
        check_ollama_model()
    else:
        say("Preflight: no stage in range calls Ollama — skipping those checks.")
    if needs_neo4j:
        check_neo4j_env()
        check_neo4j_connection()
    elif first >= DUMP_STAGE:
        say("Preflight: dump only — skipping the Neo4j connection check "
            "(the dump needs the server stopped).")
    else:
        say(f"Preflight: stopping after stage {last} — no stage in range touches "
            "Neo4j, so its environment and connection are not checked.")
    if runs_dump:
        check_dump_backend()
    if in_range((8,)):
        check_embedding_device()
    say("Preflight complete.")


EMBED_DEVICE_PROBE = (
    "import torch; "
    "print('cuda:' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
)


def check_embedding_device() -> None:
    """Say where stage 8 will embed. Information only: both devices work, but on
    the CPU this is the heaviest sustained load of the whole run."""
    res = run_capture([PY, "-c", EMBED_DEVICE_PROBE])
    device = res.stdout.strip().splitlines()[-1] if res.returncode == 0 and res.stdout.strip() else "unknown"
    if device.startswith("cuda:"):
        say(f"Embedding: stage 8 will run on the GPU ({device[5:]}); CPU threads "
            f"capped at {EMBED_CPU_THREADS}.")
    else:
        say(f"Embedding: stage 8 will run on the CPU ({device}), capped at "
            f"{EMBED_CPU_THREADS} threads. A CUDA build of PyTorch would move it "
            "to the GPU.")


def embed_thread_env() -> dict:
    """Thread-cap variables for the embedding child. Anything the user already set
    in the shell wins."""
    return {v: str(EMBED_CPU_THREADS) for v in _THREAD_VARS if not os.environ.get(v)}


def unload_model(reason: str) -> None:
    """Ask Ollama to drop the model now instead of after its idle timer.

    Never fatal: a model that is already gone, or an Ollama that has exited, is
    exactly the state this is trying to reach.
    """
    if shutil.which("ollama") is None:
        return
    try:
        res = run_capture(["ollama", "stop", MODEL], timeout=60)
    except (subprocess.TimeoutExpired, OSError) as exc:
        say(f"Ollama: could not unload {MODEL} ({exc}); it will unload on its idle timer.")
        return
    if res.returncode == 0:
        say(f"Ollama: unloaded {MODEL} ({reason}). The GPU is free.")
    else:
        say(f"Ollama: 'ollama stop {MODEL}' returned {res.returncode} — the model was "
            "probably not loaded. Continuing.")


# ---------------------------------------------------------------------------
# Stage commands
# ---------------------------------------------------------------------------

def cmd_pilot() -> list[str]:
    return [
        PY, str(HFACS_EXTRACTOR),
        "--model", MODEL,
        "--limit", str(PILOT_LIMIT),
        "--force-binary",
        "--num-ctx", str(NUM_CTX),
        "--checkpoint-every", str(CHECKPOINT_EVERY),
        "--output", str(PILOT_CSV),
    ]


def cmd_compare() -> list[str]:
    # Baseline = the extraction this run is about to replace (its pre-run backup),
    # so the table shows what THIS prompt and model changed. It used to be the
    # qwen2.5 file from two extractions ago.
    try:
        baseline = comparison_baseline()
    except StageFailure:
        baseline = RESULTS_CSV                        # --plan on a bare checkout
    return [
        PY, str(COMPARE_EXTRACTIONS),
        "--baseline", str(baseline),
        "--candidate", str(PILOT_CSV),
    ]


def cmd_adjudicate() -> list[str]:
    # --model is passed explicitly so the stage cannot drift from the pipeline's
    # model if the helper's default is ever edited.
    return [PY, str(ADJUDICATE), "--model", MODEL]


def cmd_full_extraction(resume: bool) -> list[str]:
    argv = [PY, str(HFACS_EXTRACTOR), "--model", MODEL]
    if not resume:
        # --force-binary deletes the output and starts over; omitting it resumes
        # from the last checkpoint.
        argv.append("--force-binary")
    argv += [
        "--split", "all",
        "--num-ctx", str(NUM_CTX),
        "--checkpoint-every", str(CHECKPOINT_EVERY),
        "--output", str(RESULTS_CSV),
    ]
    return argv


CLEAR_GRAPH = """
import os
from neo4j import GraphDatabase
drv = GraphDatabase.driver(
    os.environ["NEO4J_URI"],
    auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
)
db = os.environ.get("NEO4J_DATABASE", "neo4j")
with drv.session(database=db) as s:
    before = s.run("MATCH (n) RETURN count(n) AS n").single()["n"]
    print("nodes before: %d" % before)
    s.run("MATCH (n) DETACH DELETE n")
    after = s.run("MATCH (n) RETURN count(n) AS n").single()["n"]
    print("nodes after : %d" % after)
    if after != 0:
        raise SystemExit("graph not empty after DETACH DELETE")
drv.close()
print("graph cleared")
"""


def cmd_clear_graph() -> list[str]:
    # HANDOFF.md clears the graph with `docker exec ... cypher-shell`. The same
    # MATCH (n) DETACH DELETE n is issued here through the driver instead, so it
    # works against Docker and Neo4j Desktop alike and uses the NEO4J_*
    # variables already validated above. On a fresh database this is a no-op --
    # run it anyway: kg_builder skips any event with e.processed = true, so
    # against a stale graph stage 7 finishes in minutes having changed nothing.
    return [PY, "-c", CLEAR_GRAPH]


def _kg_common() -> list[str]:
    return [
        "--source", "all",
        "--asias-csv", str(ASIAS_SUBSET),
        "--asrs-csv", str(ASRS_SUBSET),
        "--ntsb-csv", str(NTSB_KG_SUBSET),
        "--model", MODEL,
        "--num-ctx", str(NUM_CTX),
    ]


def cmd_kg_preflight() -> list[str]:
    argv = [PY, str(KG_BUILDER), "--source", "all", "--limit", "5", "--dry-run"]
    argv += [
        "--asias-csv", str(ASIAS_SUBSET),
        "--asrs-csv", str(ASRS_SUBSET),
        "--ntsb-csv", str(NTSB_KG_SUBSET),
        "--model", MODEL,
        "--num-ctx", str(NUM_CTX),
    ]
    return argv


def cmd_kg_build() -> list[str]:
    # --skip-faiss: stage 8 builds the indexes, once, and then verifies them
    # against the graph. Without the flag kg_builder also builds them here, so the
    # same three indexes were embedded twice in a row.
    return [PY, str(KG_BUILDER)] + _kg_common() + ["--skip-faiss"]


def cmd_faiss_only() -> list[str]:
    # The three --*-csv flags are as mandatory here as they are in stage 7.
    # --faiss-only leaves csv_for[src] = None, and kg_builder then falls back to
    # _DEFAULT_CSV -- asias_clean.csv (4,819 rows) and asrs_clean.csv (44,448).
    # That silently rebuilds the indexes over corpora the graph does not contain,
    # so retrieval returns neighbours with no EventNode, and set_embedding_index
    # stamps positions from the wrong ordering onto the events that do exist.
    return [PY, str(KG_BUILDER), "--faiss-only"] + [
        "--source", "all",
        "--asias-csv", str(ASIAS_SUBSET),
        "--asrs-csv", str(ASRS_SUBSET),
        "--ntsb-csv", str(NTSB_KG_SUBSET),
    ]


# ---------------------------------------------------------------------------
# Stage bodies
# ---------------------------------------------------------------------------

def stage_pilot() -> None:
    run_logged(cmd_pilot(), PILOT_LOG, label="stage 1 pilot")
    _report_pilot_health()


def _report_pilot_health() -> None:
    """Surface the two things HANDOFF 5b says to check, without stopping."""
    text = tail(PILOT_LOG, 400)
    if "structured=True" in text and "think=False" in text:
        say("Pilot: startup log confirms structured=True, think=False.")
    else:
        say("Pilot: WARNING — could not confirm 'structured=True' and "
            "'think=False' in the startup log. Inspect it before trusting the run.")
    for line in text.splitlines():
        if "parse_error" in line:
            say(f"Pilot: {line.strip()}")
            if line.split()[-1].strip() != "0":
                say("Pilot: WARNING — parse_error is not 0. Schema-constrained "
                    "decoding fell back for some records. Continuing, but report "
                    "this number with the results.")
            break
    else:
        say("Pilot: no parse_error line found in the summary — check the log.")
    say("Pilot: while the long stages run, check 'ollama ps'. Anything much "
        "below ~100% GPU means layers are on the CPU and the job will take "
        "multiples longer.")


def stage_compare() -> None:
    cmd = cmd_compare()
    say(f"stage 2: baseline is {Path(cmd[cmd.index('--baseline') + 1]).name}")
    run_logged(cmd, COMPARE_LOG, label="stage 2 comparison")
    banner("Prevalence comparison — current extraction vs pilot")
    print(COMPARE_LOG.read_text(encoding="utf-8", errors="replace"), flush=True)
    say("Read this table before trusting the full run. Same model, so most rows "
        "should barely move. The one deliberate change is the pass-1 prompt's strict "
        "HFACS violation rule: unsafe_violation should FALL sharply. If every row "
        "shifts by a similar amount, something other than that rule changed.")
    say(f"Reference, current {MODEL} extraction (1,013 NTSB records): "
        "unsafe_decision 70.7%, unsafe_skill 30.6%, unsafe_perception 13.8%, "
        "unsafe_violation 9.0%. The strict adjudication of those same records "
        "found 2.1% violations, which is roughly where the pilot should land.")
    say("The pilot covers 150 records and the baseline all of them, so expect a "
        "few points of sampling noise on every row.")
    say("HANDOFF.md treats this as a human go/no-go. This run is unattended, so "
        "the pipeline continues — send the table on regardless.")


def stage_full_extraction(resume: bool) -> None:
    say("stage 3: ~1.5 hours over 1,013 records, checkpointing every "
        f"{CHECKPOINT_EVERY}. Safe to leave unattended.")
    run_logged(cmd_full_extraction(resume), EXTRACT_LOG, label="stage 3 extraction")


def _adjudication_is_current() -> bool:
    """True when a COMPLETE adjudication is newer than the extraction it overrides."""
    return (ADJUDICATION_CSV.exists() and RESULTS_CSV.exists()
            and ADJUDICATION_CSV.stat().st_mtime >= RESULTS_CSV.stat().st_mtime)


def stage_adjudicate(force: bool) -> None:
    """Re-decide unsafe_violation under the strict HFACS definition.

    The dataloader applies violation_adjudication.csv as an override on top of
    hfacs_results.csv. After stage 3 rewrites the extraction that file is stale:
    its consensus candidates (the records that get three votes instead of one) were
    chosen from the OLD labels, and adjudicate_violation.py resumes from an existing
    output, so it would consider every record done and change nothing. The stale
    file is therefore moved aside first. A partial file newer than the extraction is
    an interrupted run of THIS stage and is left alone so the script resumes it.
    """
    if not RESULTS_CSV.exists():
        raise StageFailure(f"{RESULTS_CSV} does not exist — nothing to adjudicate.")
    if _adjudication_is_current() and not force:
        say(f"stage 4: {ADJUDICATION_CSV.name} is newer than {RESULTS_CSV.name} — "
            "already current, skipping. Use --force-adjudication to redo it.")
        return
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    results_mtime = RESULTS_CSV.stat().st_mtime
    if ADJUDICATION_PARTIAL.exists():
        if ADJUDICATION_PARTIAL.stat().st_mtime >= results_mtime and not force:
            say(f"stage 4: resuming from {ADJUDICATION_PARTIAL.name}.")
        else:
            dest = DATA / f"violation_adjudication.partial.pre-run.{stamp}.bak.csv"
            shutil.move(str(ADJUDICATION_PARTIAL), str(dest))
            say(f"stage 4: stale partial moved to {dest.name}")
    if ADJUDICATION_CSV.exists():
        dest = DATA / f"violation_adjudication.pre-run.{stamp}.bak.csv"
        shutil.move(str(ADJUDICATION_CSV), str(dest))
        say(f"stage 4: previous adjudication moved to {dest.name}. Until this stage "
            "finishes, the dataloader applies NO violation override.")
    say(f"stage 4: ~40 minutes. One deterministic pass over all 1,013 records, then two "
        f"sampled passes over the candidates plus a reliability sample, all on {MODEL}.")
    run_logged(cmd_adjudicate(), ADJUDICATE_LOG, label="stage 4 violation adjudication")
    banner("Violation adjudication — report")
    print(tail(ADJUDICATE_LOG, 14), flush=True)
    if not ADJUDICATION_CSV.exists():
        raise StageFailure(
            f"stage 4 exited cleanly but {ADJUDICATION_CSV.name} was not written. The "
            f"work so far is in {ADJUDICATION_PARTIAL.name}; re-run with --start-at 4.")
    say("Read the report: the strict rate should be a few percent, and the run-to-run "
        "kappa is the label's reliability ceiling — report both with the results.")


def stage_clear_graph() -> None:
    run_logged(cmd_clear_graph(), CLEAR_GRAPH_LOG, label="stage 5 clear graph")
    print(CLEAR_GRAPH_LOG.read_text(encoding="utf-8", errors="replace"), flush=True)


def stage_kg_preflight() -> None:
    run_logged(cmd_kg_preflight(), KG_PREFLIGHT_LOG, label="stage 6 KG preflight")
    print(tail(KG_PREFLIGHT_LOG, 40), flush=True)
    say("Preflight should show a node/edge tally with no traceback. It exercises "
        "a recently fixed code path — if it threw, the traceback is above.")


def stage_kg_build() -> None:
    say("stage 7: ~1.5 hours over ~2,100 records, graph only (the FAISS indexes are "
        "stage 8). The three --*-csv flags are mandatory; without them --source all "
        "would default to asrs_clean.csv (44,448 records) and run for weeks.")
    run_logged(cmd_kg_build(), KG_BUILD_LOG, label="stage 7 KG build")


def _csv_rows(path: Path) -> int:
    """Count data rows the way pandas would.

    CLAUDE.md: these narratives contain embedded newlines, so a line count is
    not a row count. csv.reader honours the quoting, which is all this needs.
    """
    # A single narrative field routinely exceeds csv's 128 KB default, which
    # raises _csv.Error rather than returning a wrong count. 2**31-1 is the
    # ceiling here: field_size_limit takes a C long, 32-bit on Windows.
    csv.field_size_limit(2 ** 31 - 1)
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        return max(sum(1 for _ in csv.reader(fh)) - 1, 0)


def cmd_ingest_ntsb_train() -> list[str]:
    return [PY, str(KG_BUILDER), "--ingest-ntsb-train"]


EVENT_PHASE = DATA / "event_phase.csv"
KG_PHASE_LOG = DATA / "kg_update_phase.log"


def stage_faiss_only() -> None:
    # The only CPU-heavy step of the run, so it gets a thread cap. With a CUDA
    # PyTorch the encoder is on the GPU and the cap just bounds the tokenizer.
    run_logged(cmd_faiss_only(), KG_FAISS_LOG, label="stage 8 FAISS indexes",
               extra_env=embed_thread_env())
    built = [DATA / n for n in (
        "asias.faiss", "asrs.faiss", "ntsb_kg.faiss",
        "asias_id_map.csv", "asrs_id_map.csv", "ntsb_kg_id_map.csv",
    )]
    for p in built:
        say(f"  {'OK  ' if p.exists() else 'MISSING'} {p.name}")
    _verify_faiss_alignment()

    # The NTSB training events go into the graph last: stage 5 emptied it, and this
    # needs the labels from stages 3-4. No model call is made. Without this step
    # the experiment's retrieval conditions refuse to run, by design.
    say("stage 8: writing the NTSB training events into the graph (no model calls; "
        "validation and test events are never written).")
    run_logged(cmd_ingest_ntsb_train(), KG_INGEST_LOG,
               label="stage 8 NTSB training events -> graph", extra_env=embed_thread_env())
    print(tail(KG_INGEST_LOG, 4), flush=True)
    for p in (NTSB_TRAIN_FAISS, NTSB_TRAIN_IDMAP):
        say(f"  {'OK  ' if p.exists() else 'MISSING'} {p.name}")
    if not NTSB_TRAIN_FAISS.exists():
        raise StageFailure(
            "Stage 8 did not produce the index for the NTSB training events. The "
            f"retrieval conditions cannot run without it. See {KG_INGEST_LOG.name}.")

    # Phase-of-flight nodes, the event-descriptive context structural retrieval
    # matches on. Stage 5 emptied the graph, so they are re-attached every run.
    # The phases themselves come from data/derive_phase.py (an LLM pass over the
    # narratives, run once; it depends on nothing this pipeline changes).
    if EVENT_PHASE.exists():
        run_logged([PY, str(KG_BUILDER), "--update-phase"], KG_PHASE_LOG,
                   label="stage 8 phase-of-flight nodes")
        print(tail(KG_PHASE_LOG, 2), flush=True)
    else:
        say("stage 8: WARNING - data/event_phase.csv is missing, so the graph has no "
            "phase-of-flight nodes and structural retrieval (C3, C4) falls back to "
            "weather, crew and calendar context. Run:  python data/derive_phase.py  "
            "then  python data/kg_builder.py --update-phase")


def _verify_faiss_alignment() -> None:
    """Each index must hold exactly the records stage 7 put in the graph.

    A wrong-corpus rebuild is otherwise silent -- every file is present and
    every log line says 'wrote', and the mismatch only surfaces much later as
    retrieval hits on events that have no node.
    """
    expected = {
        "asias": ASIAS_SUBSET,
        "asrs": ASRS_SUBSET,
        "ntsb_kg": NTSB_KG_SUBSET,
    }
    log = tail(KG_FAISS_LOG, 100000)
    ok = True
    for prefix, csv_path in expected.items():
        idmap = DATA / f"{prefix}_id_map.csv"
        if not idmap.exists():
            say(f"  FAISS check: {idmap.name} missing.")
            ok = False
            continue
        want = _csv_rows(csv_path)
        got = _csv_rows(idmap)
        match = re.search(rf"{re.escape(str(DATA / (prefix + '.faiss')))}"
                          r" \(ntotal=(\d+)", log)
        ntotal = int(match.group(1)) if match else None
        detail = f"id_map={got} ntotal={ntotal if ntotal is not None else '?'}"
        if got == want and (ntotal is None or ntotal == want):
            say(f"  FAISS check: {prefix} OK -- {detail}, {csv_path.name}={want}")
        else:
            ok = False
            say(f"  FAISS check: {prefix} MISMATCH -- {detail}, expected {want} "
                f"from {csv_path.name}. The index does not match the graph; "
                "re-run stage 8 with the --*-csv flags.")
    if not ok:
        raise StageFailure(
            "Stage 8 built indexes that do not match the KG subsets. Retrieval "
            "would return neighbours with no EventNode."
        )


def _desktop_admin(home: Path) -> Path | None:
    """The neo4j-admin entry point inside a DBMS home, if that home looks real."""
    for name in ("neo4j-admin.bat", "neo4j-admin"):
        cand = home / "bin" / name
        if cand.is_file():
            return cand
    return None


def _desktop_home() -> Path | None:
    """Locate a Neo4j Desktop DBMS directory, or None if that is ambiguous.

    NEO4J_HOME wins outright. Otherwise a single DBMS under
    ~/.Neo4jDesktop2/Data/dbmss/ is unambiguous and gets used; several means we
    cannot know which one holds the graph, so the caller is told to pin it.
    """
    if NEO4J_HOME:
        home = Path(NEO4J_HOME)
        if _desktop_admin(home) is None:
            say(f"stage 9: NEO4J_HOME={NEO4J_HOME} has no bin/neo4j-admin — ignoring it.")
        else:
            return home
    if not DESKTOP_DBMS_ROOT.is_dir():
        return None
    homes = sorted(p for p in DESKTOP_DBMS_ROOT.iterdir() if _desktop_admin(p))
    if len(homes) == 1:
        return homes[0]
    if len(homes) > 1:
        say(f"stage 9: {len(homes)} Neo4j Desktop instances under "
            f"{DESKTOP_DBMS_ROOT} — set NEO4J_HOME to the one holding the graph:")
        for h in homes:
            say(f"           {h}")
    return None


def _docker_container_exists() -> bool:
    """True only if a container actually named CONTAINER is present."""
    if shutil.which("docker") is None:
        return False
    res = run_capture(["docker", "ps", "-a", "--filter", f"name=^{CONTAINER}$",
                       "--format", "{{.Names}}"])
    return res.returncode == 0 and CONTAINER in res.stdout.split()


def _bolt_is_up() -> bool:
    """Is something listening on the bolt port from NEO4J_URI?

    `neo4j-admin database dump` refuses to touch a database mounted in a running
    server, so this is what decides whether stage 9 can proceed unattended.
    """
    uri = os.environ.get("NEO4J_URI", NEO4J_DEFAULTS["NEO4J_URI"])
    m = re.search(r"//(?:[^@/]*@)?([^:/]+)(?::(\d+))?", uri)
    host, port = (m.group(1), int(m.group(2) or 7687)) if m else ("localhost", 7687)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(2.0)
        return sock.connect_ex((host, port)) == 0


def resolve_dump_backend() -> tuple[str, Path | None]:
    """Decide how stage 9 should reach the store: 'docker', 'desktop' or 'none'.

    Docker wins when a container of the expected name really exists — that is
    HANDOFF.md's documented setup. A Neo4j Desktop install is the fallback,
    because it has neither a container nor a named volume for the Docker recipe
    to bind.
    """
    if _docker_container_exists():
        return "docker", None
    home = _desktop_home()
    if home is not None:
        return "desktop", home
    return "none", None


def _dump_docker() -> None:
    """HANDOFF §6: stop the container, dump from a throwaway one, restart."""
    stopped = False
    try:
        run_logged(["docker", "stop", CONTAINER], DUMP_LOG,
                   label="stage 9a stop container")
        stopped = True
        dump_cmd = [
            "docker", "run", "--rm",
            "-v", f"{VOLUME}:/data",
            "-v", f"{REPO_ROOT}:/backup",
            IMAGE,
            "neo4j-admin", "database", "dump", "neo4j", "--to-path=/backup",
        ]
        run_logged(dump_cmd, DATA / "neo4j_dump_run.log", label="stage 9b dump")
    finally:
        if stopped:
            run_logged(["docker", "start", CONTAINER], DATA / "neo4j_restart.log",
                       label="stage 9c restart container", check=False)


def _dump_desktop(home: Path) -> None:
    """Dump from a Neo4j Desktop DBMS, which must already be stopped.

    Desktop owns its server process, so this deliberately does not stop or start
    anything — killing the JVM out from under Desktop risks the store. The user
    stops the instance in the UI and resumes with --start-at 9.
    """
    if _bolt_is_up():
        raise StageFailure(
            "the Neo4j Desktop instance is still running, and neo4j-admin cannot "
            "dump a database mounted in a running server.\n"
            "  1. Stop the instance in Neo4j Desktop (or `neo4j-admin server stop`)\n"
            f"  2. python run_all.py --start-at {DUMP_STAGE}\n"
            "  3. Start it again in Desktop afterwards"
        )
    admin = _desktop_admin(home)
    if admin is None:                                  # pragma: no cover - guarded above
        raise StageFailure(f"no bin/neo4j-admin under {home}")
    argv = [str(admin), "database", "dump", "neo4j",
            f"--to-path={REPO_ROOT}", "--overwrite-destination"]
    if sys.platform == "win32":
        argv = ["cmd", "/c"] + argv                    # CreateProcess will not run .bat
    say(f"stage 9: dumping from Neo4j Desktop at {home}")
    run_logged(argv, DATA / "neo4j_dump_run.log", label="stage 9b dump (desktop)")


def cmd_dump() -> list[str]:
    """What stage 9 will actually run, for --plan. Backend decides the shape."""
    backend, home = resolve_dump_backend()
    if backend == "docker":
        return ["docker", "stop", CONTAINER, "&&", "docker", "run", "--rm",
                "-v", f"{VOLUME}:/data", "-v", f"{REPO_ROOT}:/backup", IMAGE,
                "neo4j-admin", "database", "dump", "neo4j", "--to-path=/backup",
                "&&", "docker", "start", CONTAINER]
    if backend == "desktop":
        return [str(_desktop_admin(home)), "database", "dump", "neo4j",
                f"--to-path={REPO_ROOT}", "--overwrite-destination"]
    return ["(no dump backend found — stage 9 will be skipped)"]


def stage_dump(strict: bool) -> None:
    """Produce neo4j.dump, from Docker or from a Neo4j Desktop install."""
    backend, home = resolve_dump_backend()
    if backend == "none":
        msg = (f"no Docker container named {CONTAINER!r} and no Neo4j Desktop "
               "DBMS found — cannot produce neo4j.dump. Send the CSV/FAISS files "
               "and flag it; HANDOFF.md calls this recoverable.")
        if strict:
            raise StageFailure(msg)
        say(f"stage 9: SKIPPED. {msg}")
        return

    try:
        if backend == "docker":
            _dump_docker()
        else:
            _dump_desktop(home)
    except StageFailure as exc:
        if strict:
            raise
        say(f"stage 9: dump failed ({exc}). Continuing — send the CSV/FAISS "
            "files anyway and flag it.")

    dump_file = REPO_ROOT / "neo4j.dump"
    say(f"stage 9: {'wrote' if dump_file.exists() else 'did NOT write'} "
        f"{dump_file}")
    if dump_file.exists() and backend == "desktop":
        # Desktop 2 ships Enterprise and defaults new databases to block format,
        # which Community cannot read at all. HANDOFF.md's neo4j:5 load will not
        # work on this dump; whoever receives it needs Enterprise of the same
        # major version.
        say("stage 9: NOTE — this dump came from Neo4j Desktop (Enterprise). If "
            "the store uses block format it will not load into Community or into "
            "neo4j:5; report the server version alongside the file.")


# ---------------------------------------------------------------------------
# Stage table
# ---------------------------------------------------------------------------

def build_stages(args: argparse.Namespace) -> list[dict]:
    return [
        {"n": 1, "name": "Pilot extraction (150 records)",
         "eta": "~15 min", "cmd": cmd_pilot(),
         "run": stage_pilot},
        {"n": 2, "name": "Comparison against the current extraction",
         "eta": "seconds", "cmd": cmd_compare(),
         "run": stage_compare},
        {"n": 3, "name": "Full extraction (1,013 records)",
         "eta": "~1.5 h", "cmd": cmd_full_extraction(args.resume_extraction),
         "run": lambda: stage_full_extraction(args.resume_extraction)},
        {"n": 4, "name": "Violation adjudication (strict HFACS, majority of 3)",
         "eta": "~40 min", "cmd": cmd_adjudicate(),
         "run": lambda: stage_adjudicate(args.force_adjudication)},
        {"n": 5, "name": "Clear the Neo4j graph",
         "eta": "seconds", "cmd": cmd_clear_graph(),
         "run": stage_clear_graph},
        {"n": 6, "name": "Preflight KG build (5 records, dry run)",
         "eta": "~1 min", "cmd": cmd_kg_preflight(),
         "run": stage_kg_preflight},
        {"n": 7, "name": "KG build, graph only (~2,100 records)",
         "eta": "~1.5 h", "cmd": cmd_kg_build(),
         "run": stage_kg_build},
        {"n": 8, "name": "FAISS indexes + NTSB training events into the graph",
         "eta": "minutes", "cmd": cmd_faiss_only(),
         "run": stage_faiss_only},
        {"n": 9, "name": "Neo4j dump",
         "eta": "minutes", "cmd": cmd_dump(),
         "run": lambda: stage_dump(args.strict_dump)},
    ]


def print_plan(stages: list[dict]) -> None:
    banner(f"Plan — model {MODEL}, nothing executed")
    for st in stages:
        print(f"\n[{st['n']}/{N_STAGES}] {st['name']}  ({st['eta']})", flush=True)
        print(f"      {quote(st['cmd'])}", flush=True)
    print("", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=f"Run the full HFACS + KG pipeline unattended using {MODEL}.",
    )
    p.add_argument("--plan", action="store_true",
                   help="Print the exact commands for all nine stages and exit "
                        "without running anything.")
    p.add_argument("--preflight-only", action="store_true",
                   help="Run the preflight checks and exit. No stage runs, no "
                        "model is loaded, nothing is generated or written.")
    p.add_argument("--start-at", type=int, default=1, choices=range(1, N_STAGES + 1),
                   metavar="N",
                   help=f"Resume from stage N (1-{N_STAGES}). Earlier stages are "
                        "skipped. NOTE: the adjudication became stage 4 on 2026-09-20, "
                        "so the old stages 4-8 are now 5-9.")
    p.add_argument("--stop-after", type=int, default=N_STAGES,
                   choices=range(1, N_STAGES + 1), metavar="N",
                   help=f"Stop after stage N (1-{N_STAGES}). --stop-after 4 is a "
                        "labels-only run and needs no Neo4j.")
    p.add_argument("--force-adjudication", action="store_true",
                   help="Redo stage 4 even when violation_adjudication.csv is already "
                        "newer than hfacs_results.csv.")
    p.add_argument("--resume-extraction", action="store_true",
                   help="Run stage 3 without --force-binary so it resumes from "
                        "the last checkpoint instead of starting over.")
    p.add_argument("--skip-dump", action="store_true",
                   help="Skip stage 9 (the Neo4j dump).")
    p.add_argument("--strict-dump", action="store_true",
                   help="Treat a failed stage 9 as a fatal error instead of a "
                        "warning.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.stop_after < args.start_at:
        print("--stop-after is before --start-at; nothing to do.", flush=True)
        return 2

    stages = build_stages(args)
    if args.plan:
        print_plan(stages)
        return 0

    banner(f"GraphRAG-Causal pipeline — model {MODEL}"
           + (" (preflight only)" if args.preflight_only else ""))
    say(f"Repo root      : {REPO_ROOT}")
    say(f"Interpreter    : {PY}")
    say(f"Model          : {MODEL} (assumed installed; never pulled)")
    say(f"Context window : {NUM_CTX}")
    if args.preflight_only:
        say("Expected total : seconds — checks only, no stage will run.")
    else:
        say("Expected total : roughly 4 hours, mostly unattended.")

    run_started = time.time()
    try:
        preflight(args)
        if args.preflight_only:
            banner("PREFLIGHT OK — nothing was run")
            say("Every precondition is satisfied. Start the real run with:")
            say(f"  {PY} run_all.py")
            return 0
        # Back up before anything that leads to stage 3 rewriting the labels. A
        # resumed extraction is the exception: hfacs_results.csv is then the
        # half-written NEW file, and the real pre-run backup already exists.
        rewrites_labels = args.start_at <= 3 <= args.stop_after and not args.resume_extraction
        if rewrites_labels or args.start_at <= 2:
            backup_current_results()
        # The stage table was built before that backup existed; stage 2's baseline
        # is resolved again when it runs, so refresh the table.
        stages = build_stages(args)

        for st in stages:
            if st["n"] < args.start_at or st["n"] > args.stop_after:
                say(f"[{st['n']}/{N_STAGES}] {st['name']} — skipped")
                continue
            if st["n"] == DUMP_STAGE and args.skip_dump:
                say(f"[{DUMP_STAGE}/{N_STAGES}] Neo4j dump — skipped (--skip-dump)")
                continue
            banner(f"[{st['n']}/{N_STAGES}] {st['name']}  ({st['eta']})")
            st["run"]()
            # Free the GPU the moment the last model stage of THIS run is done,
            # rather than leaving Ollama to unload mid-embedding five minutes later.
            later_llm = any(st["n"] < n <= args.stop_after for n in OLLAMA_STAGES)
            if st["n"] in OLLAMA_STAGES and not later_llm:
                unload_model(f"stage {st['n']} was the last model stage of this run")

    except StageFailure as exc:
        banner("PIPELINE FAILED")
        say(str(exc))
        say(f"Elapsed before failure: {human(time.time() - run_started)}")
        say("Nothing after this point ran. Fix the cause, then re-run with "
            "--start-at N to pick up from the failed stage. For an interrupted "
            "stage 3, add --resume-extraction so the extractor resumes instead "
            "of starting over.")
        return 1
    except KeyboardInterrupt:
        banner("INTERRUPTED")
        say("Stopped by the user. Re-run with --start-at N to continue; add "
            "--resume-extraction if stage 3 was in progress.")
        return 130

    banner("PIPELINE COMPLETE")
    say(f"Total elapsed: {human(time.time() - run_started)}")
    say("Send back:")
    for path in (RESULTS_CSV, ADJUDICATION_CSV, PILOT_CSV, EXTRACT_LOG, ADJUDICATE_LOG,
                 KG_BUILD_LOG, NTSB_TRAIN_FAISS, NTSB_TRAIN_IDMAP, DATA / "asias.faiss", DATA / "asrs.faiss", DATA / "ntsb_kg.faiss",
                 DATA / "asias_id_map.csv", DATA / "asrs_id_map.csv",
                 DATA / "ntsb_kg_id_map.csv", REPO_ROOT / "neo4j.dump"):
        say(f"  {'OK  ' if path.exists() else 'MISSING'} {path}")
    rev = run_capture(["git", "rev-parse", "HEAD"])
    if rev.returncode == 0:
        say(f"Also report: commit {rev.stdout.strip()}, model {MODEL}, and "
            "roughly how long each stage took (timings are in the logs above).")
    backup = latest_pre_run_backup()
    if backup is not None:
        say(f"Previous extraction kept as {backup.name}.")
    _say_downstream(args)
    return 0


def _say_downstream(args: argparse.Namespace) -> None:
    """What this run made stale, and the commands that bring it back in line."""
    labels_changed = args.start_at <= 4 and args.stop_after >= 3
    graph_changed = args.start_at <= 8 and args.stop_after >= 7
    if not (labels_changed or graph_changed):
        return
    banner("Downstream — now out of date")
    if labels_changed:
        say("The labels changed, so every trained checkpoint in results/seeds/ and "
            "every results/conditions_*.csv was produced from the OLD labels.")
    if graph_changed:
        say("The knowledge graph and its FAISS indexes changed, so the retrieval "
            "conditions C2-C5 retrieve different neighbours than before.")
    say("Still valid: data/test_query_views.csv (built from the narratives and the "
        "split, not from the labels) and data/.emb_cache/ (keyed by text).")
    if labels_changed and not (args.start_at <= 8 <= args.stop_after):
        say("The NTSB training events in the graph still carry the OLD labels. The "
            "retrieval conditions will refuse to run until they are rewritten:")
        say("  python data/kg_builder.py --ingest-ntsb-train")
    say("Bring it back in line with, Neo4j running (every retrieval condition "
        "reads the graph and stops without it):")
    for line in ("python run_conditions.py",
                 "python data/leakage_audit.py",
                 "python models/lstm/eval_conditions.py                    # L1b, the standard view",
                 "python models/lstm/eval_conditions.py --query-view L1    # lower bound",
                 "python models/lstm/eval_conditions.py --query-view L2    # comparison (B and C)",
                 "python models/lstm/eval_conditions.py --query-view full  # upper bound"):
        say(f"  {line}")


if __name__ == "__main__":
    sys.exit(main())
