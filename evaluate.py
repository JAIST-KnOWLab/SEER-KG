#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Evaluate the seer-kg side-effect dataset via sequential knowledge editing.

Edits a base LLM hop-by-hop along each reasoning chain (using the vendored
`easyeditor` primitives) and records whether each hop is predicted correctly.
Consumes the chain files from the `chain` stage
(`results/chain/<dataset>_seer-kg_depth-{d}_chains.json`) and writes results under
`results/eval/<model>/<alg>/<mode>/`.

Run from the project root (after `python main.py chain`):

    python evaluate.py                                   # model llama-3.1-8b, all algs, all modes
    python evaluate.py --model llama-3.1-8b              # pick the LLM (hparams YAML stem)
    python evaluate.py --alg MEND                         # one algorithm (or ALL)
    python evaluate.py --mode chain-all                   # one regime (or ALL)
    python evaluate.py -m qwen2-7b -a ROME --mode chain-stop

Choices:
    --model   any hparams YAML stem present for the algorithm (e.g. llama-3.1-8b,
              llama-3.1-70b, qwen2-7b, mistral-7b). See evaluate/hparams/<ALG>/.
    --alg     ROME | MEND | IKE | RAG | ALL          (default ALL)
    --mode    one-time | chain-stop | chain-all | ALL (default ALL)

Depths and GPUs are configured in evaluate/config.py (DEPTHS, TARGET_GPUS).
"""

import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate import config

# Friendly aliases for the mode name.
_MODE_ALIASES = {"one-chain": "one-time", "onetime": "one-time"}


def _parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate seer-kg chains via sequential knowledge editing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-m", "--model", default=config.DEFAULT_MODEL,
                   help=f"LLM hparams YAML stem (default {config.DEFAULT_MODEL})")
    p.add_argument("-a", "--alg", default="ALL",
                   choices=config.SUPPORTED_ALGORITHMS + ["ALL"],
                   help="editing algorithm (default ALL)")
    p.add_argument("--mode", default="ALL",
                   choices=config.MODES + ["ALL"] + list(_MODE_ALIASES),
                   help="sequential-editing regime (default ALL)")
    return p.parse_args()


def main():
    args = _parse_args()
    model = args.model
    algorithms = config.SUPPORTED_ALGORITHMS if args.alg == "ALL" else [args.alg]
    if args.mode == "ALL":
        modes = list(config.MODES)
    else:
        modes = [_MODE_ALIASES.get(args.mode, args.mode)]

    print(f"\n{'=' * 70}\nEVALUATION PIPELINE INITIALIZING")
    print(f"Model: {model} | Algorithms: {algorithms} | Modes: {modes}")
    print(f"Depths: {config.DEPTHS} | Target GPUs: {config.TARGET_GPUS}\n{'=' * 70}\n")

    # Preflight (before heavy torch/easyeditor imports): chain inputs must exist.
    available = [d for d in config.DEPTHS if os.path.exists(config.chain_input_path(d))]
    if not available:
        print(f"[ERROR] No chain input files in {config.CHAIN_DIR} for depths {config.DEPTHS}.")
        print(f"        Expected e.g. {config.chain_input_path(config.DEPTHS[0])}")
        print("        Run the chain stage first:  python main.py chain")
        return
    missing = [d for d in config.DEPTHS if d not in available]
    if missing:
        print(f"[WARN] No chain input for depths {missing}; they will be skipped.")

    from evaluate.editing import apply_algorithm_preprocessing, apply_algorithm_optional_training
    from evaluate.manager import process_mode_with_checkpoint

    for alg_name in algorithms:
        hparams_path = config.hparams_path(alg_name, model)
        if not os.path.exists(hparams_path):
            print(f"[SKIP] {alg_name}: no hparams for model '{model}' at {hparams_path}")
            continue

        print(f"\n[INIT] Preprocessing for {alg_name} ({model})...")
        apply_algorithm_preprocessing(alg_name, hparams_path, device=f"cuda:{config.TARGET_GPUS[0]}")
        apply_algorithm_optional_training(
            alg_name,
            config.RUN_MEND_TRAINING,
            config.mend_training_hparams_path(model),
            config.MEND_TRAIN_FILE,
            config.MEND_EVAL_FILE,
        )

        for depth in available:
            for mode in modes:
                print(f"\n{'=' * 70}\nSTARTING: {model} | {alg_name} | Depth {depth} | {mode}\n{'=' * 70}")
                try:
                    process_mode_with_checkpoint(
                        depth=depth,
                        mode_key=mode,
                        alg_name=alg_name,
                        hparams_path=hparams_path,
                        target_gpus=config.TARGET_GPUS,
                        model=model,
                    )
                except Exception as exc:
                    print(f"Critical failure on {model}/{alg_name}/Depth {depth}/{mode}: {exc}")
                    traceback.print_exc()

    print("\n[DONE] All experiments completed.")


if __name__ == "__main__":
    main()
