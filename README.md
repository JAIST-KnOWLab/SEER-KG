# SEER-KG

> **SEER-KG: Side Effect Exploration and Evaluation with Knowledge Graph-based Retrieval in Knowledge Editing for Large Language Models**

SEER-KG is a **knowledge graph-based retrieval framework** for exposing the hidden side effects of **Knowledge Editing (KE)** in Large Language Models (LLMs).

Rather than evaluating only whether an edited fact is successfully updated, SEER-KG traces how a localized edit propagates through **multi-hop Wikidata neighborhoods**. The framework constructs reasoning chains, retrieves semantically related facts, and evaluates how knowledge edits affect both nearby and distant knowledge.

SEER-KG measures editing performance using the three standard Knowledge Editing evaluation metrics:

- 🎯 **Generality** — Does the model correctly learn the edited knowledge?
- 🔗 **Portability** — Does the edited knowledge transfer to related facts?
- 🛡️ **Locality** — Does unrelated knowledge remain unchanged?

The framework supports three sequential editing settings:

- **One-Time Editing**
- **Early-Stop Editing**
- **Complete Editing**

SEER-KG is dataset-agnostic and currently supports popular knowledge editing methods including **ROME**, **MEND**, **IKE**, and **RAG**.

---

# Requirements

- Python **3.10**
- CUDA-enabled GPU (required for Stages 3–4)
- Hugging Face account (for downloading gated LLM checkpoints)
- Internet connection (Stage 1 queries Wikidata)

Install only the dependencies required for the stage you plan to run.

## Install dependencies

```bash
# Stage 1: Knowledge Graph Retrieval
pip install -r requirements.txt

# Stage 2: Reasoning Chain Generation
pip install -r requirements-llm.txt

# Stage 3: Knowledge Editing Evaluation
pip install -r requirements-eval.txt
```

## Environment Variables

Copy the template

```bash
cp .env.example .env
```

Then configure

```text
HF_TOKEN=<your_huggingface_token>
WIKIDATA_CONTACT=<your_email_or_contact>
```

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | Used to download gated Hugging Face models (only required once). |
| `WIKIDATA_CONTACT` | Contact information included in the Wikidata User-Agent. |

---

# Quick Start

Run the complete SEER-KG pipeline.

```bash
# Stage 1: Retrieve side effects from Wikidata
python main.py

# Stage 2: Finalize checkpoint files
python main.py finalize

# Stage 3: Generate reasoning chains
python main.py chain

# Stage 4: Evaluate Knowledge Editing
python evaluate.py
```

---

# Framework Overview

```text
Knowledge Editing Dataset
           │
           ▼
┌──────────────────────────┐
│ Stage 1:                 │
│ Knowledge Graph Retrieval│
└──────────────────────────┘
           │
           ▼
┌──────────────────────────┐
│ Stage 2:                 │
│ Reasoning Chain Builder  │
└──────────────────────────┘
           │
           ▼
┌──────────────────────────┐
│ Stage 3:                 │
│ Sequential Knowledge     │
│ Editing Evaluation       │
└──────────────────────────┘
           │
           ▼
┌──────────────────────────┐
│ Evaluation Stage:        │
│ • Generality             │
│ • Portability            │
| • Locality               |
└──────────────────────────┘
```

---

# Pipeline

The complete framework consists of four stages.

| Stage | Description |
|--------|-------------|
| **Stage 1 – Probe** | Retrieve multi-hop knowledge surrounding edited facts from Wikidata. |
| **Stage 2 – Finalize** | Merge resumable checkpoints into finalized datasets. |
| **Stage 3 – Chain Generation** | Convert retrieved facts into reasoning chains and masked prompts. |
| **Evaluation Stage** | Perform sequential knowledge editing and measure Generality, Portability, and Locality. |

---

# Project Structure

```text
seer-kg/
├── main.py
├── evaluate.py
├── data/
├── results/
├── explore/
│   ├── config.py
│   ├── probe.py
│   ├── side_effect_bfs.py
│   ├── wikidata.py
│   └── chain_converter.py
└── evaluate/
    ├── editing.py
    ├── manager.py
    ├── worker.py
    ├── checkpoint.py
    ├── easyeditor/
    └── hparams/
```

---

# Stage 1 — Probe

This stage explores the Wikidata knowledge graph around each editing case.

Features:

- Resolves entities to Wikidata IDs.
- Performs a single BFS traversal.
- Retrieves Generality, Portability, and Locality facts.
- Supports resumable checkpointing.
- Works with CounterFact-style datasets.

Example:

```bash
python main.py

python main.py 0 to 100

python main.py [2,3,4]

python main.py finalize
```

---

# Stage 2 — Chain Generation

Generate reasoning chains and masked cloze prompts.

```bash
python main.py chain
```

Outputs include:

- reasoning chains
- masked prompts
- ontology labels
- deduplicated paths

---

# Stage 3 — Knowledge Editing Evaluation

Evaluate side effects using sequential editing.

Supported methods

- ROME
- MEND
- IKE
- RAG

Supported settings

- One-Time
- Early-Stop
- Complete

Example

```bash
python evaluate.py

python evaluate.py --alg ROME

python evaluate.py --mode chain-stop

python evaluate.py --model llama-3.1-8b
```

---

# Outputs

All generated files are stored under `results/`.

| Output | Description |
|---------|-------------|
| `*_seer-kg_depth-*.jsonl` | Probe checkpoints |
| `*_seer-kg_depth-*.json` | Finalized datasets |
| `chain/*.json` | Reasoning chains |
| `eval/.../results_depth-*.json` | Sequential editing results |
| `skipped.log` | Failed entity resolutions |
| `fallback_resolved.log` | Entity resolution fallbacks |

---

# Reproducibility

SEER-KG is designed for reproducible large-scale experiments.

- Deterministic random seeds.
- Resumable checkpointing.
- Single-pass multi-depth retrieval.
- Dataset-agnostic pipeline.
- Centralized configuration.

---

# Citation

If you use SEER-KG in your research, please cite:

```bibtex
@inproceedings{seerkg,
  author    = {Patipon Wiangnak and Natthawut Kertkeidkachorn and Kiyoaki Shirai},
  title     = {SEER-KG: Side Effect Exploration and Evaluation with Knowledge Graph-based Retrieval in Knowledge Editing for LLMs},
  booktitle = {},
  year      = {},
  pages     = {},
  publisher = {}
}
```

---

# Acknowledgements

The evaluation pipeline builds upon **EasyEdit** by reusing its editing primitives while extending the framework to support sequential knowledge editing over reasoning chains. The EasyEdit source code is included under `evaluate/easyeditor/` with its original MIT license.

---

# License

This project is released under the **MIT License**.

Third-party code under `evaluate/easyeditor/` retains its original license.