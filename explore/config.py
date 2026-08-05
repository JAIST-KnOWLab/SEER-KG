"""Central configuration for the seer-kg pipeline.

All tunable paths and constants live here so runs are reproducible and easy to
audit. Modules import from this file rather than hard-coding values.
"""

import os


# --- Secrets / credentials (loaded from .env, never hard-coded) ---------------
def _load_env_file(path: str = ".env") -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ (without overriding
    values already present in the real environment). Dependency-free."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env_file()

# Hugging Face token — required only to DOWNLOAD the gated Llama weights. The HF
# libraries read HF_TOKEN from the environment automatically; exposed here too.
HF_TOKEN = os.environ.get("HF_TOKEN", "")
# Contact string embedded in the Wikidata User-Agent header (endpoint etiquette).
WIKIDATA_CONTACT = os.environ.get("WIKIDATA_CONTACT", "anonymous")


# --- Directories (relative to the project root; run scripts from there) -------
DATA_DIR = "data"          # inputs (e.g. counterfact.json)
RESULTS_DIR = "results"    # all outputs (checkpoints, finalized arrays, logs, chains)
CHAIN_DIR = os.path.join(RESULTS_DIR, "chain")

# Rich edit dataset (nested `requested_rewrite` schema with Wikidata QIDs +
# relation_id). This is the data source the probe reads.
INPUT_PATH = os.path.join(DATA_DIR, "counterfact.json")

# Optional split selector: probe only the case_ids present in this file. It may be a
# flat EasyEdit-style split (e.g. counterfact-train.json) that lacks QIDs — only its
# case_ids are used; the actual fields come from INPUT_PATH. Set to None for all cases.
SPLIT_PATH = os.path.join(DATA_DIR, "counterfact-train.json")
SKIPPED_LOG = os.path.join(RESULTS_DIR, "skipped.log")
FALLBACK_LOG = os.path.join(RESULTS_DIR, "fallback_resolved.log")
EXCLUDED_LOG = os.path.join(RESULTS_DIR, "excluded.log")

# --- Knowledge-effect probing -------------------------------------------------
# The BFS side-effect checker is run once per max_depth in DEPTHS, each producing
# its own resumable output file.
DEPTHS = [2, 3, 4, 5, 6]
# Dataset name for output filenames — derived from the split if one is set,
# else the input file (e.g. counterfact-train.json -> "counterfact-train").
DATASET_NAME = os.path.splitext(os.path.basename(SPLIT_PATH or INPUT_PATH))[0]
OUTPUT_STEM = DATASET_NAME + "_seer-kg_depth-{depth}"  # .jsonl checkpoint / .json array


def checkpoint_path(depth: int) -> str:
    return os.path.join(RESULTS_DIR, OUTPUT_STEM.format(depth=depth) + ".jsonl")


def array_path(depth: int) -> str:
    return os.path.join(RESULTS_DIR, OUTPUT_STEM.format(depth=depth) + ".json")


# --- Wikidata access ----------------------------------------------------------
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
USER_AGENT = f"seer-kg/1.0 ({WIKIDATA_CONTACT}; knowledge-effect probing)"

SEARCH_LIMIT = 7            # candidate QIDs per subject search
SLEEP_BETWEEN_CALLS = 1.0   # polite delay (seconds) between network calls
MAX_RETRIES = 4             # retries per failed request
BACKOFF_BASE = 2.0          # backoff grows as BACKOFF_BASE * 2**attempt (seconds)
SPARQL_TIMEOUT = 60         # per-query socket timeout (seconds); WDQS allows ~60s

# --- LLM chain / prompt generation --------------------------------------------
SEED = 42
MODEL_PATH = "./hugging_cache/llama-3.1-8b-instruct"
BATCH_SIZE = 32
MAX_NEW_TOKENS = 40
CHAIN_MAX_DEPTH = 5        # default max chain length when a record lacks a "depth" field
CHAIN_MAX_PER_CASE = 5000  # cap on chains emitted per case (guards combinatorial blowup); None disables
