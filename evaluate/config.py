"""Configuration for the evaluation stage.

Centralizes all paths and knobs so eval runs are reproducible. Chain-input paths
reuse the probe/chain naming from `explore.config`, so a change of dataset flows
through automatically.
"""

import os
import sys

from explore import config as kg

# --- Vendored easyeditor library ----------------------------------------------
# The knowledge-editing library is vendored under `evaluate/easyeditor/`. We add
# THIS directory (evaluate/) to sys.path so the package is importable as top-level
# `easyeditor` — which keeps its internal absolute imports and `import *` working.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

# hparams YAMLs are vendored under evaluate/hparams/ (resolve_hparams joins
# HPARAMS_ROOT + a relative "hparams/<ALG>/<model>.yaml" path). Override with
# SEER_KG_HPARAMS to point elsewhere. Model *weights* are still external — set each
# YAML's `model_name` to your local/HF model path.
HPARAMS_ROOT = os.environ.get("SEER_KG_HPARAMS", _EVAL_DIR)

# --- Inputs / outputs ---------------------------------------------------------
# Chain files produced by `python main.py chain`.
CHAIN_DIR = kg.CHAIN_DIR                       # results/chain
EVAL_RESULTS_DIR = os.path.join(kg.RESULTS_DIR, "eval")   # results/eval

def chain_input_path(depth: int) -> str:
    """results/chain/<dataset>_seer-kg_depth-{depth}_chains.json"""
    stem = kg.OUTPUT_STEM.format(depth=depth)
    return os.path.join(CHAIN_DIR, f"{stem}_chains.json")

# Training split used by in-context algorithms (IKE / RAG).
TRAIN_DATASET_PATH = kg.SPLIT_PATH or os.path.join(kg.DATA_DIR, "counterfact-train.json")

# --- Experiment grid ----------------------------------------------------------
TARGET_GPUS = [0]                # physical GPU ids to run workers on
DEPTHS = list(kg.DEPTHS)         # depths to evaluate (defaults to the probe depths)

# Editing algorithms and sequential-editing regimes. Selected at runtime via the
# evaluate.py CLI (--alg / --mode, with ALL to run every one).
SUPPORTED_ALGORITHMS = ["ROME", "MEND", "IKE", "RAG"]
MODES = ["one-time", "chain-stop", "chain-all"]
DEFAULT_MODEL = "llama-3.1-8b"   # hparams YAML stem; override with --model

# Optional MEND meta-training before editing.
RUN_MEND_TRAINING = False
MEND_TRAIN_FILE = "data/zsre/zsre_mend_train.json"
MEND_EVAL_FILE = "data/zsre/zsre_mend_eval.json"


def resolve_hparams(path: str) -> str:
    """Resolve an hparams path against HPARAMS_ROOT when it is relative."""
    return path if os.path.isabs(path) else os.path.join(HPARAMS_ROOT, path)


def hparams_path(alg: str, model: str) -> str:
    """hparams YAML for an (algorithm, model), e.g. ('MEND','llama-3.1-8b')."""
    return resolve_hparams(os.path.join("hparams", alg, f"{model}.yaml"))


def mend_training_hparams_path(model: str) -> str:
    """MEND meta-training hparams YAML for a model."""
    return resolve_hparams(os.path.join("hparams", "TRAINING", "MEND", f"{model}.yaml"))
