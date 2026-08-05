"""Reasoning-chain construction and masked-prompt generation (LLM stage).

Reads the finalized per-depth effect arrays produced by the probe stage
(`results/<dataset>_seer-kg_depth-{d}.json`), walks each case's effects into
reasoning chains, and uses an instruction-tuned LLM to render every hop's
(subj, pred, obj) triple as a masked cloze prompt (object replaced with `[X]`).
Chains that close a cycle are additionally tagged with ontology types
(asymmetric / symmetric / transitive / inverse).

Outputs `<name>_chains.json` into `results/chain/`.

The heavy LLM is loaded lazily inside `run()`, so importing this module is cheap.
Tunables and paths live in `explore.config`.
"""

import copy
import json
import os
import random
import re
from collections import defaultdict
from typing import Dict, List, Tuple

from tqdm import tqdm

from . import config

# Only consume the finalized per-depth arrays (not checkpoints or logs).
# Derived from config so it always matches the probe stage's output naming.
INPUT_PREFIX = config.OUTPUT_STEM.split("{")[0]


# =============================================================================
# Utilities & ontology detection
# =============================================================================

def mask_object(prompt: str, obj_str: str) -> str:
    """Mask the exact object string with `[X]`.

    Prefers a match at the end of the fragment; otherwise falls back to a
    word-boundary match anywhere in the string.
    """
    if not prompt or not obj_str:
        return prompt

    exact_obj = re.escape(obj_str.strip())  # handles e.g. "Washington, D.C."

    end_pattern = re.compile(rf"{exact_obj}(?=[.,!?\s]*$)", re.IGNORECASE)
    if end_pattern.search(prompt):
        return end_pattern.sub("[X]", prompt).strip()

    fallback_pattern = re.compile(rf"\b{exact_obj}\b", re.IGNORECASE)
    return fallback_pattern.sub("[X]", prompt).strip()


def triple_signature(hop: Dict) -> Tuple:
    return (hop["subj"]["id"], hop["pred"]["id"], hop["obj"]["id"])


class OntologyDetector:
    """Tag a chain with the logical-property types of any cycle it closes."""

    @staticmethod
    def detect(chain: Dict) -> List[str]:
        hops = chain.get("detail", [])
        if len(hops) < 1:
            return []

        found_types = set()
        first_seen: Dict[str, int] = {}

        for i, hop in enumerate(hops):
            subj_id = hop["subj"]["id"]
            obj_id = hop["obj"]["id"]

            if subj_id not in first_seen:
                first_seen[subj_id] = i

            if obj_id in first_seen:
                start_idx = first_seen[obj_id]
                segment_len = i - start_idx + 1
                segment_preds = [hops[k]["pred"]["id"] for k in range(start_idx, i + 1)]
                unique_preds = set(segment_preds)

                # 2-hop cycle A -> B -> A
                if segment_len == 2:
                    if len(unique_preds) > 1:
                        found_types.add("asymmetric")   # e.g. parent / child
                    else:
                        found_types.add("symmetric")    # e.g. spouse / spouse
                    found_types.add("inverse")          # both are inverse directions

                # 3+ hop cycle A -> B -> C -> A: a relation composition closing a
                # loop -> transitive (whether or not the predicates all match).
                # (Previously a same-predicate loop was mislabeled "symmetric";
                # symmetry is strictly the 2-hop A -> B -> A property above.)
                if segment_len >= 3:
                    found_types.add("transitive")

        return sorted(found_types)


# =============================================================================
# Chain construction
# =============================================================================

class ChainBuilder:
    """Depth-first expansion of a case's effect graph into reasoning chains."""

    def build(self, entry: Dict, max_depth: int) -> List[Dict]:
        adjacency = defaultdict(list)
        for source in ("generality", "portability", "locality"):
            for edge in entry.get("effects", {}).get(source, []):
                edge["source"] = source
                adjacency[edge["subj"]["id"]].append(edge)

        roots = [edge for edges in adjacency.values() for edge in edges
                 if edge["source"] == "generality"]
        results: List[Dict] = []
        seen_chains = set()          # dedup identical chains (by ordered triple sigs)
        cap = config.CHAIN_MAX_PER_CASE

        def emit(new_path):
            signature = tuple(triple_signature(h) for h in new_path)
            if signature in seen_chains:
                return
            seen_chains.add(signature)
            results.append({"chain_id": len(results) + 1, "detail": new_path})

        def dfs(edge, path, visited_sigs, depth):
            if cap is not None and len(results) >= cap:
                return
            hop = {
                "hop_level": depth,
                "subj": edge["subj"], "pred": edge["pred"], "obj": edge["obj"],
                "new_obj": edge.get("new_obj", []), "prompt": None,
            }
            new_path = path + [hop]
            edge_sig = triple_signature(edge)

            # Neighbors not already in the current path (prevents cycles by signature).
            neighbors = [n for n in adjacency[edge["obj"]["id"]]
                         if triple_signature(n) not in visited_sigs]

            if depth >= max_depth or not neighbors:
                emit(new_path)
                return

            for neighbor in neighbors:
                if cap is not None and len(results) >= cap:
                    return
                dfs(neighbor, new_path, visited_sigs | {edge_sig}, depth + 1)

        for root in roots:
            if cap is not None and len(results) >= cap:
                break
            dfs(root, [], set(), 0)
        return results


# =============================================================================
# LLM prompt generation
# =============================================================================

class PromptManager:
    """Render triples into masked cloze prompts with an instruction-tuned LLM."""

    def __init__(self, generator, tokenizer):
        self.generator = generator
        self.tokenizer = tokenizer
        self.fact_cache: Dict[Tuple, str] = {}

    def bulk_generate(self, unique_triples: Dict[Tuple, Tuple[str, str, str]]) -> None:
        """Generate one sentence per triple using the exact subj/pred/obj strings."""
        pending = [sig for sig in unique_triples if sig not in self.fact_cache]
        if not pending:
            return

        prompts = []
        for sig in pending:
            subj_str, pred_str, obj_str = unique_triples[sig]
            chat = [
                {
                    "role": "system",
                    "content": (
                        "Task: Write a single sentence using the EXACT words provided. "
                        "Rules:\n"
                        "1. You must use the provided Subject, Predicate, and Object strings exactly.\n"
                        "2. No synonyms, no abbreviations, no shortened names.\n"
                        "3. The sentence must contain Subject, Predicate, and Object in that order.\n"
                        "Example: Subj: 'USA' | Pred: 'capital is' | Obj: 'Washington'\n"
                        "Output: The USA capital is Washington"
                    ),
                },
                {"role": "user", "content": f"Subj: '{subj_str}' | Pred: '{pred_str}' | Obj: '{obj_str}'"},
            ]
            prompts.append(self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True))

        outputs = self.generator(
            prompts, max_new_tokens=config.MAX_NEW_TOKENS,
            batch_size=config.BATCH_SIZE, return_full_text=False)

        for sig, output in zip(pending, outputs):
            text = output[0]["generated_text"].strip().split("\n")[0]
            text = re.sub(r"^(Output|Sentence|Response|Question):\s*", "", text, flags=re.IGNORECASE)
            self.fact_cache[sig] = text

    def apply_to_chains(self, chains: List[Dict]) -> None:
        """Fill each hop's `prompt` by masking the generated sentence, with a
        literal-template fallback when the LLM drifts off the exact object."""
        for chain in chains:
            for hop in chain["detail"]:
                if hop["prompt"] is not None:
                    continue
                raw_text = self.fact_cache.get(triple_signature(hop), "")
                masked = mask_object(raw_text, hop["obj"]["str"])
                if "[X]" not in masked:  # LLM used a synonym -> force a literal template
                    masked = f"{hop['subj']['str']} {hop['pred']['str']} [X]"
                hop["prompt"] = masked


# =============================================================================
# Orchestration
# =============================================================================

def _load_llm():
    """Load the instruction-tuned generator + tokenizer (heavy; call lazily)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map="auto", local_files_only=True)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    return generator, tokenizer


def run() -> None:
    """Build chains and masked prompts for every finalized per-depth array."""
    import torch  # local import so the module stays importable without torch installed

    random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    os.makedirs(config.CHAIN_DIR, exist_ok=True)
    generator, tokenizer = _load_llm()
    builder = ChainBuilder()
    prompter = PromptManager(generator, tokenizer)

    files = sorted(
        name for name in os.listdir(config.RESULTS_DIR)
        if name.startswith(INPUT_PREFIX) and name.endswith(".json")
    )
    all_file_data = []
    global_unique_triples: Dict[Tuple, Tuple[str, str, str]] = {}

    # Pass 1: build chains and collect the unique triples that need a prompt.
    for fname in tqdm(files, desc="Pass 1: build chains"):
        with open(os.path.join(config.RESULTS_DIR, fname), encoding="utf-8") as handle:
            data = json.load(handle)
        items = data if isinstance(data, list) else [data]

        file_entries = []
        for item in items:
            chains = builder.build(item, item.get("depth", config.CHAIN_MAX_DEPTH))
            for chain in chains:
                if chain["detail"]:
                    chain["detail"][0]["prompt"] = mask_object(
                        item.get("prompt", ""), chain["detail"][0]["obj"]["str"])
                for hop in chain["detail"]:
                    if hop["prompt"] is None:
                        global_unique_triples[triple_signature(hop)] = (
                            hop["subj"]["str"], hop["pred"]["str"], hop["obj"]["str"])
            file_entries.append({"item": item, "chains": chains})
        all_file_data.append((fname, file_entries))

    # Pass 2: one batched LLM inference over all unique triples.
    prompter.bulk_generate(global_unique_triples)

    # Pass 3: apply prompts, tag ontology, write per-file outputs.
    for fname, entries in tqdm(all_file_data, desc="Pass 3: save results"):
        final_output = []
        for entry in entries:
            item, chains = entry["item"], entry["chains"]
            prompter.apply_to_chains(chains)

            onto_chains = []
            for chain in chains:
                otypes = OntologyDetector.detect(chain)
                if otypes:
                    tagged = copy.deepcopy(chain)
                    tagged["onto_type"] = otypes
                    onto_chains.append(tagged)

            final_output.append({
                "case_id": item.get("case_id"),
                "subj": item.get("subj"),
                "pred": item.get("pred"),
                "obj": item.get("obj"),
                "new_obj": item.get("new_obj"),
                "effects": {
                    "chains": chains,
                    "onto_chains": onto_chains if onto_chains else None,
                },
            })

        out_name = f"{os.path.splitext(fname)[0]}_chains.json"
        with open(os.path.join(config.CHAIN_DIR, out_name), "w", encoding="utf-8") as handle:
            json.dump(final_output, handle, indent=2, ensure_ascii=False)

        # The chain output supersedes the finalized effect array; remove the source.
        os.remove(os.path.join(config.RESULTS_DIR, fname))
        print(f"[chain] wrote {out_name}; removed source {fname}")
