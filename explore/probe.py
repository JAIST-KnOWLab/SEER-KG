"""Knowledge-effect probing over an edit dataset.

For each case (CounterFact-style schema) we build the edit tuple
(subj, pred, obj, new_obj) -- resolving the bare subject string to a Wikidata QID --
then run the BFS side-effect checker at each `max_depth` in `config.DEPTHS`, producing
per-depth `generality` / `portability` / `locality` effect sets.

Each depth has its own resumable checkpoint (see `config.checkpoint_path`).
`finalize()` compiles a depth's checkpoint into a sorted JSON array.

This module exposes a library API (`run`, `finalize`); the CLI lives in `main.py`.
"""

import json
import os
from typing import Dict, List, Optional

from tqdm import tqdm

from . import config
from .side_effect_bfs import side_effects_checker_bfs
from .wikidata import get_property_label, resolve_subject


def _load_done_case_ids(path: str) -> set:
    """Case ids already present in a depth's checkpoint (for resume)."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["case_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _load_excluded_ids() -> Dict[int, set]:
    """Per-depth case_ids previously excluded (missing an effect set), for resume."""
    excluded: Dict[int, set] = {}
    if not os.path.exists(config.EXCLUDED_LOG):
        return excluded
    with open(config.EXCLUDED_LOG, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                excluded.setdefault(record["depth"], set()).add(record["case_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return excluded


def _append_log(path: str, payload: dict) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _log_skip(case_id, subject, relation_id, reason) -> None:
    _append_log(config.SKIPPED_LOG, {
        "case_id": case_id,
        "subject": subject,
        "relation_id": relation_id,
        "reason": reason,
    })


def resolve_case(case: dict) -> Optional[dict]:
    """Resolve the depth-independent fields of a case, or None to skip.

    Subject resolution and the predicate label are computed once here and reused
    across every depth.
    """
    case_id = case["case_id"]
    rewrite = case["requested_rewrite"]

    subject = rewrite["subject"]
    relation_id = rewrite["relation_id"]
    obj = {"str": rewrite["target_true"]["str"], "id": rewrite["target_true"]["id"]}
    new_obj = {"str": rewrite["target_new"]["str"], "id": rewrite["target_new"]["id"]}

    subject_id, method = resolve_subject(subject, relation_id)
    if subject_id is None:
        _log_skip(case_id, subject, relation_id, "search returned no candidates")
        return None
    if method == "fallback":
        _append_log(config.FALLBACK_LOG, {
            "case_id": case_id, "subject": subject,
            "relation_id": relation_id, "resolved_qid": subject_id,
        })
    subj = {"str": subject, "id": subject_id}

    predicate = {"str": get_property_label(relation_id), "id": relation_id}
    try:
        prompt = rewrite["prompt"].format(subject)
    except (KeyError, IndexError):
        prompt = rewrite["prompt"]

    return {
        "case_id": case_id,
        "pararel_idx": case.get("pararel_idx"),
        "subj": subj,
        "pred": predicate,
        "obj": obj,
        "new_obj": new_obj,
        "prompt": prompt,
    }


def _filter_to_depth(triples: List[dict], max_k: int) -> List[dict]:
    """Keep triples discovered within `max_k` hops, stripping the internal `depth`
    key so the emitted record format is unchanged."""
    out = []
    for triple in triples:
        if triple.get("depth", 0) <= max_k:
            out.append({key: value for key, value in triple.items() if key != "depth"})
    return out


def _assemble_record(base: dict, depth: int, generality, portability, locality) -> dict:
    return {
        "case_id": base["case_id"],
        "pararel_idx": base["pararel_idx"],
        "depth": depth,
        "subj": base["subj"],
        "pred": base["pred"],
        "obj": base["obj"],
        "new_obj": base["new_obj"],
        "prompt": base["prompt"],
        "effects": {
            "generality": generality,
            "portability": portability,
            "locality": locality,
        },
    }


def probe_case(base: dict, depths: List[int]) -> Dict[int, dict]:
    """Probe a resolved case at every requested depth with a SINGLE BFS run.

    The BFS is executed once at max(depths); each depth's effects are then derived
    by filtering triples to those discovered within that many hops (a deep run is a
    superset of every shallower one). Returns {depth: record}.
    """
    subj, pred, obj, new_obj = base["subj"], base["pred"], base["obj"], base["new_obj"]
    generality, portability, locality = side_effects_checker_bfs(
        {"str": subj["str"], "wikidata_id": subj["id"]},
        {"str": pred["str"], "wikidata_id": pred["id"]},
        {"str": obj["str"], "wikidata_id": obj["id"]},
        {"str": new_obj["str"], "wikidata_id": new_obj["id"]},
        max(depths),
    )
    records = {}
    for depth in depths:
        records[depth] = _assemble_record(
            base, depth,
            _filter_to_depth(generality, depth),
            _filter_to_depth(portability, depth),
            _filter_to_depth(locality, depth),
        )
    return records


def validate(case_index: int = 0, shallow: int = 2, deep: int = 3) -> bool:
    """Sanity-check the single-pass optimization on one case.

    Runs the engine natively at `shallow`, and at `deep` then filters to `shallow`,
    and compares the two per effect bucket by triple signature. Returns True if all
    three buckets match. Prints a per-bucket report.
    """
    def signature(triple):
        return (triple["subj"]["id"], triple["pred"]["id"], triple["obj"]["id"],
                frozenset(o["id"] for o in triple.get("new_obj", [])))

    with open(config.INPUT_PATH, "r", encoding="utf-8") as handle:
        dataset = json.load(handle)
    base = resolve_case(dataset[case_index])
    if base is None:
        print(f"case {case_index} unresolved; pick another index")
        return False

    subj, pred, obj, new_obj = base["subj"], base["pred"], base["obj"], base["new_obj"]

    def run_at(depth):
        return side_effects_checker_bfs(
            {"str": subj["str"], "wikidata_id": subj["id"]},
            {"str": pred["str"], "wikidata_id": pred["id"]},
            {"str": obj["str"], "wikidata_id": obj["id"]},
            {"str": new_obj["str"], "wikidata_id": new_obj["id"]},
            depth,
        )

    native = run_at(shallow)
    deep_sets = run_at(deep)
    derived = tuple(_filter_to_depth(s, shallow) for s in deep_sets)

    print(f"\n=== validate case {case_index} ('{subj['str']}'): native depth {shallow} "
          f"vs depth {deep} filtered to <= {shallow} ===")
    all_match = True
    for name, nat, der in zip(("generality", "portability", "locality"), native, derived):
        nat_sig = {signature(t) for t in nat}
        der_sig = {signature(t) for t in der}
        match = nat_sig == der_sig
        all_match &= match
        verdict = "MATCH" if match else (f"DIFF only_native={len(nat_sig - der_sig)} "
                                         f"only_derived={len(der_sig - nat_sig)}")
        print(f"  {name:12} native={len(nat_sig):4d} derived={len(der_sig):4d}  {verdict}")
    print(f"=> {'ALL MATCH — optimization is exact for this case' if all_match else 'DIVERGENCE (see above)'}")
    return all_match


def finalize(depths: List[int]) -> None:
    """Compile each depth's checkpoint into a pretty, case-id-sorted JSON array."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    for depth in depths:
        checkpoint = config.checkpoint_path(depth)
        if not os.path.exists(checkpoint):
            print(f"[depth {depth}] nothing to finalize: {checkpoint} not found.")
            continue
        records = []
        with open(checkpoint, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        records.sort(key=lambda record: record["case_id"])
        array = config.array_path(depth)
        with open(array, "w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False, indent=4)
        print(f"[depth {depth}] wrote {len(records)} records to {array}")
        # The .json array is now the canonical output; drop the .jsonl checkpoint.
        os.remove(checkpoint)
        print(f"[depth {depth}] removed checkpoint {checkpoint}")


def run(start: int = 0, end: Optional[int] = None,
        limit: Optional[int] = None, depths: Optional[List[int]] = None) -> None:
    """Probe dataset cases in [start, end) at each depth, resumably."""
    depths = depths or list(config.DEPTHS)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    with open(config.INPUT_PATH, "r", encoding="utf-8") as handle:
        dataset = json.load(handle)

    # Optional split: keep only case_ids listed in SPLIT_PATH (fields still come from
    # the rich INPUT_PATH). Lets us probe a train/val/edit split by id.
    if config.SPLIT_PATH:
        with open(config.SPLIT_PATH, "r", encoding="utf-8") as handle:
            split_ids = {case["case_id"] for case in json.load(handle)}
        before = len(dataset)
        dataset = [case for case in dataset if case["case_id"] in split_ids]
        print(f"=> split {os.path.basename(config.SPLIT_PATH)}: {len(dataset)}/{before} cases selected")

    end = end if end is not None else len(dataset)
    subset = dataset[start:end]
    if limit is not None:
        subset = subset[:limit]

    done = {depth: _load_done_case_ids(config.checkpoint_path(depth)) for depth in depths}
    excluded_ids = _load_excluded_ids()
    for depth in depths:  # excluded cases are "done" too — don't re-probe them on resume
        done[depth] |= excluded_ids.get(depth, set())
    handles = {depth: open(config.checkpoint_path(depth), "a", encoding="utf-8") for depth in depths}
    print(f"=> {len(subset)} cases in range [{start}:{end}]; depths={depths}")
    for depth in depths:
        print(f"   depth {depth}: {len(done[depth])} already done (will skip).")

    processed = {depth: 0 for depth in depths}
    excluded = {depth: 0 for depth in depths}
    skipped = 0
    try:
        for case in tqdm(subset, desc="=> Probing dataset"):
            case_id = case["case_id"]
            pending = [depth for depth in depths if case_id not in done[depth]]
            if not pending:
                continue

            try:
                base = resolve_case(case)
            except Exception as exc:
                rewrite = case.get("requested_rewrite", {})
                _log_skip(case_id, rewrite.get("subject"), rewrite.get("relation_id"), f"resolve error: {exc}")
                skipped += 1
                continue
            if base is None:
                skipped += 1
                continue

            # Single BFS at max(pending); derive each pending depth by filtering.
            try:
                records = probe_case(base, pending)
            except Exception as exc:
                _log_skip(case_id, base["subj"]["str"], base["pred"]["id"], f"probe error: {exc}")
                continue

            for depth in pending:
                record = records[depth]
                effects = record["effects"]
                # A case is only "considered" at a depth if it has ALL of
                # generality, portability, and locality; otherwise exclude it.
                if effects["generality"] and effects["portability"] and effects["locality"]:
                    handles[depth].write(json.dumps(record, ensure_ascii=False) + "\n")
                    handles[depth].flush()
                    processed[depth] += 1
                else:
                    _append_log(config.EXCLUDED_LOG, {
                        "case_id": case_id, "depth": depth,
                        "counts": {k: len(v) for k, v in effects.items()},
                    })
                    excluded[depth] += 1
                done[depth].add(case_id)
    finally:
        for handle in handles.values():
            handle.close()

    summary = ", ".join(f"d{depth}={processed[depth]}(excl {excluded[depth]})" for depth in depths)
    print(f"=> done. skipped(unresolved)={skipped}. kept(excluded) per depth: {summary}")
    print("=> run `main.py finalize` to produce the JSON arrays.")
