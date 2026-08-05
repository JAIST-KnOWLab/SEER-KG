"""Evaluation of the seer-kg side-effect dataset via sequential knowledge editing.

Reads the chain files produced by the `chain` stage
(`results/chain/<dataset>_seer-kg_depth-{d}_chains.json`), applies an editing
algorithm (ROME / MEND / IKE / RAG via the external `easyeditor` library) hop-by-hop
along each reasoning chain, and records whether each hop is predicted correctly under
three editing regimes (one-time / chain-stop / chain-all).

Modules:
    config      -- paths, depths, modes, algorithm->hparams map, GPU list, easyeditor shim
    checkpoint  -- non-blocking background checkpoint saver
    editing     -- model loading, per-hop verify, per-chain edit/verify regimes
    worker      -- persistent per-GPU worker process (holds the model in VRAM)
    manager     -- queue manager with resumable checkpointing

The entry point is `evaluate.py` at the project root.
"""
