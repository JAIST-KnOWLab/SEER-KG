# seer-kg

Knowledge-effect **probing** for model-editing datasets, backed by the Wikidata
knowledge graph.

Given an edit `(subject, predicate, object → new_object)`, seer-kg explores the
surrounding knowledge graph and partitions the reachable facts into three effect
sets — mirroring the standard knowledge-editing evaluation axes:

| Effect | Meaning | Graph scope |
|---|---|---|
| **generality** | the target knowledge (the edited fact and its immediate variants) | the edit triple |
| **portability** | neighborhood knowledge | boundary of `{subject, object, new_object}` (≈ depth 2) |
| **locality** | knowledge that should remain unaffected | farther-out (up to the run's `max_depth`) |

A second, optional stage turns those effects into natural-language **reasoning
chains** with masked cloze prompts using an instruction-tuned LLM.

The pipeline is dataset-agnostic: it works with any CounterFact-style edit file, not
only CounterFact itself.

## Pipeline at a glance

Four stages, run from the project root in order:

```bash
python main.py            # 1. probe    — search Wikidata side effects (generality/portability/locality)
python main.py finalize   # 2. finalize — compile per-depth checkpoints into JSON arrays
python main.py chain      # 3. chain    — build reasoning chains + masked cloze prompts (LLM)
python evaluate.py        # 4. evaluate — sequential knowledge editing (ROME/MEND/IKE/RAG)
```

Stages 1–2 are CPU + network only; stages 3–4 need a GPU. Each stage consumes the
previous stage's output (see [Outputs](#outputs-all-under-results)).

## Requirements

- **Python 3.10** (the vendored `easyeditor` targets 3.10).
- Per-stage dependencies, installed as needed:
  - probe: `pip install -r requirements.txt`
  - chain (LLM): `pip install -r requirements-llm.txt`
  - evaluate: `pip install -r requirements-eval.txt`
- A GPU for stages 3–4, and local model weights (see each stage below).

## Project layout

```
seer-kg/
├── main.py                     # probe/chain CLI entry point (loose positional args)
├── evaluate.py                 # evaluation entry point (sequential editing)
├── requirements.txt            # core deps (probe stage)
├── requirements-llm.txt        # deps for the LLM chain stage
├── requirements-eval.txt       # deps for the evaluation stage
├── .env.example                # credential template (copy to .env)
├── data/                       # input edit dataset(s)
├── results/                    # all outputs (git-ignored)
├── explore/                    # probe + chain package
│   ├── config.py               # all paths & tunables (single source of truth)
│   ├── wikidata.py             # subject→QID resolution, labels (cached, throttled)
│   ├── side_effect_bfs.py      # BFS side-effect checker over Wikidata
│   ├── probe.py                # probing driver (library API)
│   ├── chain_converter.py      # reasoning-chain + LLM cloze-prompt conversion
│   └── legacy/                 # earlier, unused pipeline (kept for provenance)
└── evaluate/                   # evaluation package
    ├── config.py               # eval paths, GPUs, algorithms→hparams, import setup
    ├── checkpoint.py           # async background checkpoint saver
    ├── editing.py              # model loading, per-hop verify, editing regimes
    ├── worker.py               # persistent per-GPU worker process
    ├── manager.py              # queue manager with resumable checkpointing
    ├── easyeditor/             # vendored EasyEdit library (importable as `easyeditor`)
    └── hparams/                # vendored EasyEdit hparams YAMLs (per algorithm/model)
```

## Setup

```bash
# core (probe stage)
pip install -r requirements.txt

# credentials
cp .env.example .env            # then fill in the values (see below)
```

### Configuration & credentials

All tunables live in [`explore/config.py`](explore/config.py). Secrets are **not**
in code — they are read from a git-ignored `.env` file:

| Key | Purpose |
|---|---|
| `HF_TOKEN` | Hugging Face token; needed only to *download* the gated Llama weights. |
| `WIKIDATA_CONTACT` | Contact string embedded in the Wikidata `User-Agent` (endpoint etiquette). |

## Stage 1 — Probe

The CLI takes loose positional arguments (run from the project root):

```bash
python main.py                  # full run: all cases, all depths
python main.py 0 to 10          # limit data: cases in index range [0, 10)
python main.py [2,3,4]          # limit depth: only these depths
python main.py [2,6]            # limit depth: depths 2 and 6 only
python main.py 0 to 10 [2,6]    # combine a data range and a depth set
python main.py finalize         # compile per-depth checkpoints into JSON arrays
python main.py finalize [2,6]   # ... for specific depths
```

A data range (`START to END`) and a depth set (`[...]`) may appear in any order;
depths must be a subset of `config.DEPTHS`.

- Reads the edit dataset at `config.INPUT_PATH` (default `data/counterfact.json`).
  Each case supplies `relation_id`, `target_true` (object) and `target_new`
  (new object) with QIDs; the **subject is a bare string** and is resolved to a QID
  via Wikidata `wbsearchentities` — keeping the first candidate that actually uses
  the relation, else falling back to the top hit (logged to
  `results/fallback_resolved.log`). Unresolvable cases go to `results/skipped.log`.
- **Single-pass over depths.** The BFS is run **once per case at `max(depths)`**;
  each requested depth is then derived by filtering triples to those discovered
  within that many hops (each triple carries its minimum discovery depth). This is
  ~5× cheaper than running the BFS separately per depth.
  - Note: a deeper run back-fills shallow depths — completing more chains surfaces
    additional genuinely-within-`k`-hop effects that a native depth-`k` run misses.
    So depth-`k` here is a **superset** of a standalone depth-`k` run (more complete;
    validated with `probe.validate()`).
- **Only complete cases are kept.** A case is written for a depth only if it has
  **all three** effect sets non-empty (generality *and* portability *and* locality);
  otherwise it is excluded and logged to `results/excluded.log` (so resume won't
  re-probe it).
- Writes one **resumable** checkpoint per depth,
  `results/<dataset>_seer-kg_depth-{d}.jsonl` (where `<dataset>` is the input
  filename stem, e.g. `counterfact`) — re-running skips `case_id`s already present
  in that depth's file (or in `excluded.log`), and only re-probes to the deepest
  *pending* depth. `finalize` sorts each into `results/<dataset>_seer-kg_depth-{d}.json`.

> ⚠️ A full run issues many live SPARQL/API calls; expect it to be slow and to hit
> rate limits. Queries retry with exponential backoff (honoring `Retry-After`) and
> checkpointing makes repeated resumes safe. Tune `SLEEP_BETWEEN_CALLS`,
> `MAX_RETRIES`, `BACKOFF_BASE`, `SPARQL_TIMEOUT`, and `DEPTHS` in `config.py`.

## Stage 2 — Chains + prompts (LLM)

```bash
pip install -r requirements-llm.txt
# download the gated weights once (needs HF_TOKEN):
#   huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
#     --local-dir ./hugging_cache/llama-3.1-8b-instruct
python main.py chain
```

Reads the finalized per-depth arrays from `results/`, walks each case's effects into
reasoning chains, renders each hop's `(subj, pred, obj)` triple into a masked cloze
prompt (object → `[X]`), tags cycle-closing chains with ontology types, and writes
`results/chain/*_chains.json`. The LLM is loaded lazily, so importing the package is
cheap and needs no GPU.

Ontology tags follow the closed cycle in a chain: a 2-hop loop `A→B→A` is `symmetric`
(same predicate) or `asymmetric` (different predicates), always also `inverse`; a 3+
hop loop `A→B→C→A` is `transitive`. Chain enumeration is deduplicated and capped per
case at `config.CHAIN_MAX_PER_CASE` to guard against combinatorial blowup on dense
effect graphs.

## Stage 3 — Evaluate (sequential knowledge editing)

Evaluates the side-effect dataset by editing a base model hop-by-hop along each
reasoning chain (via the **vendored `easyeditor`** library under
`evaluate/easyeditor/`) and checking whether each hop is predicted correctly under
three regimes — `one-time`, `chain-stop`, `chain-all`.

```bash
pip install -r requirements-eval.txt   # torch/transformers/bitsandbytes/... + EasyEdit's own deps

python evaluate.py                                  # model llama-3.1-8b, all algs, all modes
python evaluate.py --model llama-3.1-8b             # pick the LLM (hparams YAML stem)
python evaluate.py --alg MEND                        # one algorithm (ROME|MEND|IKE|RAG|ALL)
python evaluate.py --mode chain-all                  # one regime (one-time|chain-stop|chain-all|ALL)
python evaluate.py -m qwen2-7b -a ROME --mode chain-stop
```

- **CLI:** `--model` (any hparams stem under `evaluate/hparams/<ALG>/`, e.g.
  `llama-3.1-8b`, `llama-3.1-70b`, `qwen2-7b`, `mistral-7b`), `--alg`
  (`ROME|MEND|IKE|RAG|ALL`), `--mode` (`one-time|chain-stop|chain-all|ALL`). Depths
  and GPUs come from `evaluate/config.py` (`DEPTHS`, `TARGET_GPUS`).
- **Entry point:** `evaluate.py`; the logic lives in the `evaluate/` package
  (`config`, `checkpoint`, `editing`, `worker`, `manager`), with the editing library
  vendored at `evaluate/easyeditor/` (importable as top-level `easyeditor`).
- **Input:** the chain files `results/chain/<dataset>_seer-kg_depth-{d}_chains.json`.
- **Output:** `results/eval/<model>/<alg>/<mode>/results_depth-{d}.json` (with
  resumable `checkpoint_depth-{d}.json`, promoted to the final name on completion).
- **hparams** resolve under `HPARAMS_ROOT` (defaults to `evaluate/`, i.e. the vendored
  `evaluate/hparams/`; override with `SEER_KG_HPARAMS`).
- Runs one persistent worker per GPU (model held in VRAM, 4-bit quantized for
  non-ROME algorithms), with a background checkpoint saver.

> Both `easyeditor` and the `hparams/` YAMLs are vendored — no separate install or
> external checkout needed. You still need `requirements-eval.txt` deps, a GPU, and
> the **model weights** (not vendored): set each hparams YAML's `model_name` to your
> local or HF model path.
>
> The vendored `easyeditor`'s **multimodal** components (BLIP-2 / MiniGPT-4) and the
> **SERAC** algorithm are imported optionally (guarded `try/except`), so ROME / MEND /
> IKE / RAG on text LLMs do **not** require vision deps (`timm`, `iopath`, `opencv`,
> `torchvision`, …). If a multimodal component's deps are missing it warns and skips.

## Outputs (all under `results/`)

| File | Produced by | Contents |
|---|---|---|
| `<dataset>_seer-kg_depth-{d}.jsonl` | `python main.py` | resumable checkpoint — **one compact JSON record per line** (JSON Lines) |
| `<dataset>_seer-kg_depth-{d}.json` | `python main.py finalize` | pretty-printed, `case_id`-sorted JSON array |
| `skipped.log`, `fallback_resolved.log` | `python main.py` | resolution diagnostics |
| `chain/*_chains.json` | `python main.py chain` | reasoning chains + masked prompts + ontology tags |
| `eval/<model>/<alg>/<mode>/results_depth-{d}.json` | `python evaluate.py` | per-hop edit/verify results (with resumable checkpoint) |

The `.jsonl` checkpoint is intentionally one record per line (not indented) so writes
are append-only and O(1) — that's what makes runs resumable. Run `finalize` to get the
human-readable, indented `.json` array; **`finalize` then deletes the `.jsonl`
checkpoint** once the array is written (the `.json` is canonical, and the `chain`
stage reads only `.json`). Note: finalize a depth only after its probe run is
complete — deleting the checkpoint ends resumability for that depth.

**Storage lifecycle.** Each stage removes the previous stage's file as it consumes it:
`probe` → `.jsonl`; `finalize` deletes `.jsonl`, leaves `.json`; `chain` deletes the
`.json` and leaves only `results/chain/*_chains.json`. ⚠️ The final chain output keeps
`chains` / `onto_chains` but **not** the raw generality/portability/locality triples —
so once `chain` runs, the raw effect sets are gone. Keep a copy of the `.json` arrays
first if you need them.

## Reproducibility notes

- Deterministic seeds (`config.SEED`) for the LLM stage.
- The BFS engine in `side_effect_bfs.py` preserves its validated traversal logic;
  the refactor only added per-triple depth tracking so a single deep run can serve
  every depth.
- Single-pass depth derivation is a documented **superset** of native per-depth runs
  (see Stage 1); `probe.validate(case_index, shallow, deep)` reproduces the check.
- All paths/tunables are centralized in `config.py`; no secrets in code.

## Acknowledgments

The evaluation stage uses **[EasyEdit](https://github.com/zjunlp/EasyEdit)**
(MIT License, © 2022 Kevin Meng and contributors) **as a code base** — we do not use
the framework as-is or in whole. We take EasyEdit's editing primitives and modify /
extend the surrounding code to implement and test **sequential knowledge editing**
along reasoning chains (the `one-time` / `chain-stop` / `chain-all` regimes).

- The knowledge-editing library is vendored under
  [`evaluate/easyeditor/`](evaluate/easyeditor/) and its license is retained at
  [`evaluate/easyeditor/LICENSE`](evaluate/easyeditor/LICENSE). Only the parts needed
  for the enabled algorithms (ROME / MEND / IKE / RAG) are exercised.
- The driver (`evaluate.py` and the `evaluate/` modules) is our own, adapted from
  EasyEdit's chain-edit evaluation script and modified for sequential-editing
  evaluation over seer-kg chains.

If you use this evaluation in academic work, please also cite EasyEdit:

```bibtex
@misc{easyedit,
  title  = {EasyEdit: An Easy-to-use Knowledge Editing Framework for Large Language Models},
  author = {Wang, Peng and Zhang, Ningyu and others},
  year   = {2023},
  note   = {https://github.com/zjunlp/EasyEdit}
}
```

## License

Released under the [MIT License](LICENSE). Vendored third-party code under
`evaluate/easyeditor/` retains its own MIT license (see
[`evaluate/easyeditor/LICENSE`](evaluate/easyeditor/LICENSE)).
