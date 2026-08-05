"""Model loading, per-hop verification, and per-chain editing regimes.

Wraps the external `easyeditor` library. Import of this module triggers loading
`easyeditor`, so `evaluate.config` (which puts easyeditor on sys.path) must be
imported first — it is, via the package `__init__`/entry point.
"""

import gc
import os

# Import config FIRST: it puts the vendored easyeditor on sys.path so the
# `easyeditor` imports below resolve to evaluate/easyeditor/.
from . import config  # noqa: F401  (import side effect: sys.path setup)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from easyeditor import (
    BaseEditor, CounterFactDataset, EditTrainer,
    IKEHyperParams, MENDHyperParams, MENDTrainingHparams,
    ROMEHyperParams, ZsreDataset, RAGHyperParams,
)
from easyeditor.models.ike.util import encode_ike_facts

HYPERPARAM_CLASS_MAP = {
    "IKE": IKEHyperParams,
    "ROME": ROMEHyperParams,
    "MEND": MENDHyperParams,
    "RAG": RAGHyperParams,
}

MODE_LOGIC_MAP = {
    "one-time": "one-time",
    "chain-stop": "chain-stop",
    "chain-all": "all",
}


def validate_algorithm(alg_name: str):
    if alg_name not in HYPERPARAM_CLASS_MAP:
        raise ValueError(f"Unsupported algorithm: {alg_name}")


def get_mode_logic(mode_key: str) -> str:
    return MODE_LOGIC_MAP[mode_key]


def cleanup_gpu(device=None):
    """Aggressive GPU memory cleanup between tasks / models."""
    gc.collect()
    if torch.cuda.is_available():
        if device:
            gpu_id = int(device.split(":")[-1]) if ":" in device else 0
            with torch.cuda.device(gpu_id):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
        else:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


# --- Algorithm preprocessing / optional training ------------------------------

def get_train_dataset_for_algorithm(alg_name: str):
    if alg_name in ["IKE", "RAG"]:
        return CounterFactDataset(config.TRAIN_DATASET_PATH)
    return None


def apply_algorithm_preprocessing(alg_name: str, hparams_path: str, device: str = None):
    if alg_name == "IKE":
        ensure_ike_embedding_cache(hparams_path, device)
    elif alg_name == "RAG":
        hparams = RAGHyperParams.from_hparams(hparams_path)
        print(f"[PREPROCESS] RAG HyperParams loaded: {hparams.model_name}")


def ensure_ike_embedding_cache(hparams_path: str, device: str = None):
    from sentence_transformers import SentenceTransformer

    hparams = IKEHyperParams.from_hparams(hparams_path)
    train_ds = CounterFactDataset(config.TRAIN_DATASET_PATH)
    safe_model_name = hparams.sentence_model_name.rsplit("/", 1)[-1]

    os.makedirs(f"{hparams.results_dir}/{hparams.alg_name}/embedding/", exist_ok=True)
    cache_file = (
        f"{hparams.results_dir}/{hparams.alg_name}/embedding/"
        f"{safe_model_name}_{type(train_ds).__name__}_{len(train_ds)}.pkl"
    )
    if os.path.exists(cache_file):
        print(f"[CACHE] IKE embedding cache found: {cache_file}")
        return

    print(f"[CACHE] IKE embedding cache missing. Building: {cache_file}")
    device = device or (f"cuda:{hparams.device}" if hasattr(hparams, "device") else "cuda:0")
    gpu_id = int(device.split(":")[-1]) if ":" in device else 0
    torch.cuda.set_device(gpu_id)

    sentence_model = SentenceTransformer(hparams.sentence_model_name).to(device)
    encode_ike_facts(sentence_model, train_ds, hparams)
    print("[CACHE] IKE embedding cache built.")


def apply_algorithm_optional_training(alg_name, run_mend_training, mend_training_hparams,
                                      mend_train_file, mend_eval_file):
    if not (alg_name == "MEND" and run_mend_training):
        return
    print("[MEND] Starting meta-training...")
    training_hparams = MENDTrainingHparams.from_hparams(mend_training_hparams)
    train_ds = ZsreDataset(mend_train_file, config=training_hparams)
    eval_ds = ZsreDataset(mend_eval_file, config=training_hparams)

    trainer = EditTrainer(config=training_hparams, train_set=train_ds, val_set=eval_ds)
    trainer.run()

    # Save only the MEND hypernetwork, not the full base model.
    mend_state_dict = {k: v for k, v in trainer.model.state_dict().items()
                       if "mend" in k or "edit_lrs" in k}
    obj = {
        "model": mend_state_dict,
        "opt": trainer.opt.state_dict() if getattr(trainer, "opt", None) is not None else None,
        "lr_opt": trainer.lr_opt.state_dict() if trainer.lr_opt is not None else None,
        "val_stats": {},
        "start_time": trainer.start_time,
        "elapsed_time": 0,
        "step": trainer.global_iter,
    }
    os.makedirs(os.path.dirname(trainer.save_path), exist_ok=True)
    torch.save(obj, trainer.save_path)
    print(f"[MEND] Checkpoint saved to: {trainer.save_path}")

    del trainer
    gc.collect()
    cleanup_gpu(getattr(training_hparams, "device", None))


# --- Model loading + editing --------------------------------------------------

def load_fresh_components(alg_name: str, hparams_path: str, internal_device: str, display_gpu_id: str):
    """Load the editor + model into VRAM (4-bit quantized for non-ROME algs)."""
    validate_algorithm(alg_name)
    hparams = HYPERPARAM_CLASS_MAP[alg_name].from_hparams(hparams_path)
    model_path = hparams.model_name
    local = os.path.isdir(model_path)
    gpu_id = int(internal_device.split(":")[-1]) if ":" in internal_device else 0

    if alg_name == "ROME":
        hparams.device = gpu_id
        print(f"[MODEL] Loading ROME editor model on GPU-{display_gpu_id}...")
        editor = BaseEditor.from_hparams(hparams)
        editor.tok.padding_side = "left"
        if editor.tok.pad_token is None:
            editor.tok.pad_token = editor.tok.eos_token
        return editor.tok, editor.model, editor

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"[MODEL] Loading 4-bit quantized model on GPU-{display_gpu_id}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        local_files_only=local,
        device_map={"": gpu_id},
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()

    hparams.model_name = (model, tokenizer)
    hparams.device = gpu_id
    editor = BaseEditor.from_hparams(hparams)

    hparams.model_name = model_path
    editor.model_name = model_path
    editor.hparams.model_name = model_path
    return tokenizer, model, editor


@torch.no_grad()
def verify_hop(model, tokenizer, hop, max_tokens=20):
    """Greedy-generate for a hop's prompt and check the expected new object appears."""
    prompt = hop["prompt"].replace("[X].", "").replace("[X]", "").strip()
    expected = [ans["str"] for ans in hop["new_obj"]]

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                         pad_token_id=tokenizer.eos_token_id)
    ans = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    return any(e.lower() in ans.lower() for e in expected), ans


def run_mode(tok, model, editor, chain_details, alg_name, train_ds=None, mode="all"):
    """Edit and verify along a chain under one regime; returns annotated hops."""
    import copy

    current_model = model
    recorded_details = copy.deepcopy(chain_details)
    stop_editing = False

    hops_to_edit_indices = (
        [i for i, h in enumerate(recorded_details) if h["hop_level"] == 0]
        if mode == "one-time" else list(range(len(recorded_details)))
    )

    for i, hop in enumerate(recorded_details):
        if not stop_editing and i in hops_to_edit_indices:
            edit_kwargs = {
                "prompts": [hop["prompt"]],
                "ground_truth": [hop["obj"]["str"]],
                "target_new": [hop["new_obj"][0]["str"]],
                "subject": [hop["subj"]["str"]],
                "sequential_edit": True,
                "keep_original_weight": True,
            }
            if train_ds:
                edit_kwargs["train_ds"] = train_ds

            _, current_model, _, _ = editor.edit(**edit_kwargs)
            if mode == "chain-stop":
                is_correct, _ = verify_hop(current_model, tok, hop)
                if not is_correct:
                    stop_editing = True

        is_correct, model_ans = verify_hop(current_model, tok, hop)
        hop.update({
            "predict": is_correct,
            "model_answer": model_ans,
            "was_edited": (i in hops_to_edit_indices and not stop_editing),
        })

    return recorded_details
