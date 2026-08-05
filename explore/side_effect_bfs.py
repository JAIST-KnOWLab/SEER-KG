"""BFS side-effect checker over the Wikidata knowledge graph.

Given an edit (subj, pred, obj -> new_obj), `side_effects_checker_bfs` explores the
graph around the subject, original object, and new object up to `max_depth` hops and
partitions the reachable triples into three effect sets:

    generality  -- the target knowledge (the edited triple and its immediate variants)
    portability -- neighborhood knowledge (boundary of the {subj, obj, new_obj} set)
    locality    -- farther-out knowledge that should remain unaffected

The engine is intentionally preserved as-is (a validated implementation); only the
module framing has been cleaned up. It depends on stdlib + SPARQLWrapper + tqdm.
"""

import copy
import re
import time
from collections import defaultdict, deque
from itertools import chain

from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm

from . import config

CACHE = {}


def _execute_sparql(sparql, description="query"):
    """Run a prepared SPARQLWrapper query with retry + exponential backoff.

    Returns the parsed JSON, or None if every attempt fails. Handles the common
    transient WDQS failures (read timeout, HTTP 502) and honors a server-provided
    Retry-After header when present.
    """
    for attempt in range(config.MAX_RETRIES):
        try:
            return sparql.query().convert()
        except Exception as exc:
            wait = config.BACKOFF_BASE * (2 ** attempt)
            headers = getattr(exc, "headers", None)
            if headers is not None:
                retry_after = headers.get("Retry-After")
                if retry_after:
                    try:
                        wait = max(wait, float(retry_after))
                    except (TypeError, ValueError):
                        pass
            print(f"Error during query execution ({description}, attempt "
                  f"{attempt + 1}/{config.MAX_RETRIES}): {exc} -> retry in {wait:.0f}s")
            time.sleep(wait)
    print(f"[bfs] giving up on {description} after {config.MAX_RETRIES} attempts")
    return None


def get_entity_instances_cached(s_id):
    if s_id in CACHE:
        return CACHE[s_id]  # Return cached result
    results = get_entity_instances(s_id)
    CACHE[s_id] = results  # Store in cache
    return results


def get_entity_instances(s_id):
    
    time.sleep(1)

    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(config.SPARQL_TIMEOUT)

    sparql.addCustomHttpHeader('User-Agent', config.USER_AGENT)

    query = f"""
    SELECT DISTINCT ?s ?sLabel ?p ?pLabel ?o ?oLabel WHERE {{
        BIND(wd:{s_id} AS ?s)
        ?s ?a ?o.


        FILTER(
            STRSTARTS(STR(?a), "http://www.wikidata.org/prop/direct/") &&
            STRSTARTS(STR(?o), "http://www.wikidata.org/entity/Q")
        )

        ?p wikibase:directClaim ?a.
        ?p rdfs:label ?pLabel.
        FILTER(LANG(?pLabel) = "en").

        # Exclude Gende, Instance Of, described by source, on focus list of Wikimedia project, subclass of
        FILTER(?a NOT IN (wdt:P21, wdt:P31, wdt:P1343, wdt:P5008, wdt:P279, wdt:P735, wdt:P1889, wdt:P1424, wdt:P1151, wdt:P910, wdt:P6104, wdt:P1963, wdt:P1855, wdt:P1659, wdt:P1629, wdt:P2302, wdt:P1687))

        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }} LIMIT 100
    """
    sparql.setQuery(query)

    results = _execute_sparql(sparql, f"entity {s_id}")
    if results is None:
        return []

    data = [
        {
            "subj": {"str": result["sLabel"]["value"], "id": result["s"]["value"].split('/')[-1]},
            "pred": {"str": result["pLabel"]["value"], "id": result["p"]["value"].split('/')[-1]},
            "obj": {"str": result["oLabel"]["value"], "id": result["o"]["value"].split('/')[-1]}
        }
        for result in results["results"]["bindings"]
    ]
    return data


def query_wikidata(subjects):
    """Fetch direct relationships for multiple subjects from Wikidata."""
    if not subjects:
        return {}

    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(config.SPARQL_TIMEOUT)

    sparql.addCustomHttpHeader('User-Agent', config.USER_AGENT)

    # Ensure only valid strings are included in the query
    subject_filter = " ".join([f"wd:{s}" for s in subjects if isinstance(s, str)])

    if not subject_filter:
        # print("No valid subjects were provided for the query.")
        return {}

    query = f"""
    SELECT DISTINCT ?s ?sLabel ?p ?pLabel ?o ?oLabel WHERE {{
      VALUES ?s {{ {subject_filter} }}  
      ?s ?a ?o.


      FILTER(
        STRSTARTS(STR(?a), "http://www.wikidata.org/prop/direct/") &&
        STRSTARTS(STR(?o), "http://www.wikidata.org/entity/Q")
      )
      
      ?p wikibase:directClaim ?a.
      ?p rdfs:label ?pLabel 
      FILTER(LANG(?pLabel) = "en").
    
      # Exclude Gende, Instance Of, described by source, on focus list of Wikimedia project, subclass of
      FILTER(?a NOT IN (wdt:P21, wdt:P31, wdt:P1343, wdt:P5008, wdt:P279, wdt:P735, wdt:P1889, wdt:P1424, wdt:P1151, wdt:P910, wdt:P6104, wdt:P1963, wdt:P1855, wdt:P1659, wdt:P1629, wdt:P2302, wdt:P1687))

      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }} LIMIT 100
    """

    sparql.setQuery(query)
    results = _execute_sparql(sparql, f"batch of {len(subjects)} subjects")
    if results is None:
        return {}

    triples = defaultdict(list)
    for res in results.get("results", {}).get("bindings", []):
        subj = res.get("s", {}).get("value", "").split("/")[-1]
        pred_url = res.get("p", {}).get("value", "")  # Full URL for the predicate
        pred = pred_url.split("/")[-1]  # Extract the predicate ID (e.g., 'P26')
        obj = res.get("o", {}).get("value", "").split("/")[-1]
        subj_label = res.get("sLabel", {}).get("value", "")
        obj_label = res.get("oLabel", {}).get("value", "")

        if subj and pred and obj:
            triples[subj].append({
                'subj': {'id': subj, 'str': subj_label},
                'pred': {
                    'id': pred,  # Use the actual predicate ID ('P26', 'P40', etc.)
                    'str': res.get("pLabel", {}).get("value", ""),  # Predicate label (e.g., 'spouse')
                },
                'obj': {'id': obj, 'str': obj_label}
            })
    
    return dict(triples)

def get_general_relationships(rel_id, node_subj, node_obj, triples, curr_depth):
    subj, obj = node_subj, node_obj

    # Direct relationships
    direct_triples = [
        {
            'subj': {'id': t['subj']['id'], 'str': t['subj']['str']},
            'pred': {'id': t['pred']['id'], 'str': t['pred']['str']},
            'obj': {'id': t['obj']['id'], 'str': t['obj']['str']},
            'depth': curr_depth
        }
        for t in triples
        if t['subj']['id'] != t['obj']['id'] and t['pred']['id'] == rel_id
    ]
    
    # Inverse relationships (swap subj and obj)
    inverse_triples = [
        {
            'subj': {'id': t['obj']['id'], 'str': t['obj']['str']},
            'pred': {'id': t['pred']['id'], 'str': t['pred']['str']},
            'obj': {'id': t['subj']['id'], 'str': t['subj']['str']},
            'depth': curr_depth
        }
        for t in triples
        if t['subj']['id'] != t['obj']['id'] 
        and ( ( t['subj']['id'] == subj or t['obj']['id'] == obj ) or ( t['subj']['id'] == obj or t['obj']['id'] == subj ) )
    ]
    
    results = direct_triples + inverse_triples
    return results


def find_inverse_relationships(ent_id, rel_id, tar_id, current_depth):

    # Regular expression pattern to match Wikidata IDs

    triples = []
    # Find triples that contain the rel_id (direct and inverse match)
    t1 = get_entity_instances_cached(ent_id)
    t2 = get_entity_instances_cached(tar_id)
    general_triples1 = get_general_relationships(rel_id, ent_id, tar_id, t1, current_depth)
    general_triples2 = get_general_relationships(rel_id, tar_id, ent_id, t2, current_depth)
    # Collect all relevant triples
    for triple in chain(general_triples1, general_triples2):
        if triple['subj']['id'] != triple['obj']['id'] and (triple['subj']['id'] != triple['subj']['str'] and triple['obj']['id'] != triple['obj']['str']):
            triples.append(triple) 
    return triples


def find_common_predicate_chains(ent_id, tar_id, rel_id, max_depth):
    """Find chains of triples that contain:
    - A common predicate between two entities
    - The specified `rel_id`
    - The inverse of `rel_id`
    """
    root1, root2 = ent_id, tar_id

    # BFS Queues for Both Trees (starting with the roots)
    queue1 = deque([(root1, [], 0)])  # (entity, path, depth)
    queue2 = deque([(root2, [], 0)])
    
    visited1 = {root1}
    visited2 = {root2}

    common_chains = []
    current_depth = 0
    
    # relevant_list = [find_inverse_relationships(root1, rel_id, root2, current_depth)]

    # print(f"Queue: {queue1} and {queue2}")
    
    for _ in tqdm(range(max_depth), desc=f"===> Exploring Chains between {root1} and {root2} at lv.{current_depth+1}"):
        if not queue1 and not queue2:
            break  # Stop if both queues are empty
        
        level_nodes1 = list(queue1)
        level_nodes2 = list(queue2)

        queue1.clear()
        queue2.clear()

        # Query both sets of nodes in level-by-level
        triples1 = query_wikidata([node[0] for node in level_nodes1])
        triples2 = query_wikidata([node[0] for node in level_nodes2])

        for (node1, path1, _), (node2, path2, _) in zip(level_nodes1, level_nodes2):
            new_path1, new_path2 = list(path1), list(path2)
            
            # Extract predicates
            preds1 = {p['pred']['id'] for p in triples1.get(node1, [])}
            preds2 = {p['pred']['id'] for p in triples2.get(node2, [])}
            
            # Find common predicates
            common_preds = preds1.intersection(preds2)
            
            if common_preds:
                # Get inverse relationships
                # print(common_preds)

                # inverse_preds = find_inverse_relationships(node1, common_preds, node2, current_depth)
                # all_preds = {p for p in common_preds}.union({p['pred']['id'] for p in inverse_preds})

                # print("common_preds:", common_preds)
                # print("inverse_preds:", inverse_preds)
                # print("Type of common_preds:", type(common_preds))
                # print("Type of inverse_preds:", type(inverse_preds))

                # Chain triples based on common and inverse predicates
                for triple1 in triples1.get(node1, []):
                    if triple1['pred']['id'] in common_preds and triple1['subj']['id'] != triple1['obj']['id'] and (triple1['subj']['id'] != triple1['subj']['str'] and triple1['obj']['id'] != triple1['obj']['str']):
                        
                        # Get inverse relationships
                        inverse_preds = find_inverse_relationships(triple1['subj']['id'], triple1['pred']['id'], triple1['obj']['id'], current_depth)
                        all_preds = {p for p in common_preds}.union({p['pred']['id'] for p in inverse_preds})  # Combine both predicates
                        
                        if triple1['pred']['id'] in all_preds:
                            # print(triple1['subj']['str'], triple1['pred']['str'], triple1['obj']['str'], current_depth)

                            new_triple = copy.deepcopy(triple1)
                            new_triple['depth'] = current_depth
                            new_path1.append(new_triple)

                            obj1 = triple1['obj']['id']
                            if obj1 not in visited1:
                                queue1.append((obj1, new_path1, current_depth + 1))
                                visited1.add(obj1)

                for triple2 in triples2.get(node2, []):
                    if triple2['pred']['id'] in common_preds and triple2['subj']['id'] != triple2['obj']['id'] and (triple2['subj']['id'] != triple2['subj']['str'] and triple2['obj']['id'] != triple2['obj']['str']):
                        
                        # Get inverse relationships
                        inverse_preds = find_inverse_relationships(triple2['subj']['id'], triple2['pred']['id'], triple2['obj']['id'], current_depth)
                        all_preds = {p for p in common_preds}.union({p['pred']['id'] for p in inverse_preds})  # Combine both predicates

                        # Check if the predicate is in the combined set
                        if triple2['pred']['id'] in all_preds:
                            new_triple = copy.deepcopy(triple2)
                            new_triple['depth'] = current_depth
                            new_path2.append(new_triple)

                            obj2 = triple2['obj']['id']
                            if obj2 not in visited2:
                                queue2.append((obj2, new_path2, current_depth + 1))
                                visited2.add(obj2)

                # Collect the chain paths at the current depth
                common_chains.append(new_path1)
                common_chains.append(new_path2)
        
        current_depth += 1
    
    return common_chains


def clean_triples(triples):
    
    seen = set()
    unique_triples = []

    for triple in triples:
        key = (triple['subj']['id'], triple['pred']['id'], triple['obj']['id'])
        if key not in seen:
            seen.add(key)
            unique_triples.append(triple)

    return unique_triples




def make_hashable(triple):
    """Convert nested dictionaries to a hashable format (tuple of tuples)."""
    return (
        ("subj", triple["subj"]["id"]),
        ("pred", triple["pred"]["id"]),
        ("obj", triple["obj"]["id"])
    )

def get_unique_triples(triples):
    # Deduplicate on (subj, pred, obj) while keeping the MINIMUM discovery depth
    # seen for each triple. Retaining depth lets a single deep run be filtered down
    # to any shallower max_depth (a depth-k result == depth-D result with depth<=k).
    best_depth = {}
    for triple in triples:
        key = (
            triple['subj']['id'], triple['subj'].get('str', ''),
            triple['pred']['id'], triple['pred'].get('str', ''),
            triple['obj']['id'], triple['obj'].get('str', ''),
        )
        depth = triple.get('depth')
        if key not in best_depth:
            best_depth[key] = depth
        elif depth is not None and (best_depth[key] is None or depth < best_depth[key]):
            best_depth[key] = depth

    unique_triples = []
    for t, depth in best_depth.items():
        triple = {
            "subj": {"id": t[0], "str": t[1]},
            "pred": {"id": t[2], "str": t[3]},
            "obj": {"id": t[4], "str": t[5]},
        }
        if depth is not None:
            triple["depth"] = depth
        unique_triples.append(triple)

    return unique_triples


def is_valid_triple(triple, id1, id2):
    return (triple['subj']['id'] != triple['obj']['id']) and (
        (triple['subj']['id'] == id1 and triple['obj']['id'] == id2) or 
        (triple['obj']['id'] == id1 and triple['subj']['id'] == id2)
    )

def deduplicate_triple_with_new_targets(triples):
    # Deduplicate on (subj, pred, obj, new_obj), keeping the first occurrence but
    # collapsing to the MINIMUM discovery depth across duplicates (so depth-based
    # filtering stays monotonic).
    seen = {}
    order = []

    for triple in triples:
        new_obj_tuple = tuple(tuple(obj.items()) for obj in triple['new_obj'])
        triple_key = (triple['subj']['id'], triple['pred']['id'], triple['obj']['id'], new_obj_tuple)

        if triple_key not in seen:
            seen[triple_key] = triple
            order.append(triple_key)
        else:
            existing = seen[triple_key]
            depth = triple.get('depth')
            if depth is not None and (existing.get('depth') is None or depth < existing['depth']):
                existing['depth'] = depth

    return [seen[key] for key in order]


def side_effects_checker_bfs(subj, pred, obj, new_obj, max_depth):
    
    CACHE.clear() 
    print(f"{subj['str']}({subj['wikidata_id']}), {pred['str']}({pred['wikidata_id']}), {obj['str']}({obj['wikidata_id']}), {new_obj['str']}({new_obj['wikidata_id']}), depth={max_depth}")

    entity_id, target_id, new_target_id = subj['wikidata_id'], obj['wikidata_id'], new_obj['wikidata_id']
    entity_str, target_str, new_target_str = subj['str'], obj['str'], new_obj['str']
    relation_id, relation_str = pred['wikidata_id'], pred['str']

    # Example Usage
    # entity_id = "Q11975"  # Example subject 1 (e.g., Britney Spears)
    # target_id = "Q96816262"  # Example subject 2 (e.g., James Parnell Spears)
    # new_target_id = "Q35723119"

    # Assuming 'find_common_predicate_chains' function is already defined and working
    b1 = find_common_predicate_chains(entity_id, target_id, relation_id, max_depth)
    b2 = find_common_predicate_chains(entity_id, new_target_id, relation_id, max_depth)
    b3 = find_common_predicate_chains(target_id, new_target_id, relation_id, max_depth)

    # Print the resulting chains in the desired format
    # print("\n===== Chains with Common Predicates =====")
    no = 0
    final_results = []
    eliminated_results = []
    for chain_triple in chain(b1, b2, b3):
        if entity_id in {t['subj']['id'] for t in chain_triple} or target_id in {t['subj']['id'] for t in chain_triple} or new_target_id in {t['subj']['id'] for t in chain_triple}:
            cleaned_result = clean_triples(chain_triple)
            if cleaned_result != []:
                no += 1
                max_depth_cnt = max(t['depth'] for t in cleaned_result)
                if not any(t['subj']['id'] in [new_target_id] or t['obj']['id'] in [new_target_id] for t in cleaned_result):
                    if max_depth_cnt > 0:
                        # print(f"\nChain {no}: {cleaned_result} ({len(cleaned_result)}) ({max_depth})")
                        final_results += cleaned_result
                    else:
                        # if any(
                        #     (t['subj']['id'] == entity_id and t['obj']['id'] == target_id) or 
                        #     (t['subj']['id'] == target_id and t['obj']['id'] == entity_id)
                        #     for t in cleaned_result
                        # ):
                        #     # print(f"\nChain {no}: {cleaned_result} ({len(cleaned_result)}) ({max_depth})")
                        #     final_results += cleaned_result
                        # else:
                        # print(f"\nEliminated Chain {no}: {cleaned_result} ({len(cleaned_result)}) ({max_depth_cnt})")
                        eliminated_results += cleaned_result # There are general fact (mostly are atomic data)
                else:
                    # print(f"\nChain {no}: {cleaned_result} ({len(cleaned_result)}) ({max_depth})")
                    final_results += cleaned_result

    # max_depth_found = max(triple['depth'] for t in final_results for triple in final_results) + 1
    # print(f"\n===== BFS of {max_depth} levels is complete (max depth found = {max_depth_found}) =====")
    # final_results = get_unique_triples([t for t in final_results if t not in eliminated_results])
    eliminated_results = get_unique_triples([t for t in eliminated_results if t['obj']['id'] not in {entity_id}])
    final_results = get_unique_triples([t for t in final_results if t not in eliminated_results])



    # Process triples and update with candidate swaps
    predsA = [t['pred']['id'] for t in final_results if t['subj']['id'] == entity_id]
    predsB = [t['pred']['id'] for t in final_results if t['subj']['id'] == target_id]
    predsC = [t['pred']['id'] for t in final_results if t['subj']['id'] == new_target_id]

    predsCommon = list(set(predsA) & set(predsB) & set(predsC))

    irrelevants = []
    for triple in final_results:
        sid, pid, oid = triple['subj']['id'], triple['pred']['id'], triple['obj']['id']
        triple['new_obj'] = []
        if sid == target_id:
            candidates = [
                {'id': t['obj']['id'], 'str': t['obj']['str']}
                for t in final_results if t['subj']['id'] == new_target_id and t['pred']['id'] == pid
            ]

            # print(f"\nCurrent: {triple}")
            # print(f"Candidates: {candidates}\n-----\n")
            # predsCommon = [_id for _id in predsA if _id in predsC]
            if pid in predsCommon:
                triple['new_obj'] = candidates # [] if oid != entity_id else [{'id': entity_id, 'str': entity_str}]
                # print(f"\n-----\nChilds: {triple['pred']['str']}({triple['pred']['id']}): {childs_of_new_targets}\n")
                # print(triple)
            else:
                # print("\n-----\nIrrelevant:")
                # print(triple)
                irrelevants.append(triple)

        elif sid == new_target_id:
            # E -> J (prop) : OK // (triple['obj']['id'] not in {entity_id, target_id, new_target_id} and triple['subj']['id'] in {entity_id, target_id, new_target_id})
            candidates = [
                {'id': t['obj']['id'], 'str': t['obj']['str']}
                for t in final_results if t['subj']['id'] == target_id and t['pred']['id'] == pid
            ]

            # predsCommon = [_id for _id in predsA if _id in predsB]
            if pid in predsCommon:
                triple['new_obj'] = candidates
            else:
                irrelevants.append(triple)

        elif sid == entity_id:
            childs_of_new_targets = [
                {'id': t['obj']['id'], 'str': t['obj']['str']}
                for t in final_results if t['subj']['id'] == new_target_id and t['pred']['id'] == pid
            ]
            child_ids = {t['id'] for t in childs_of_new_targets}

            candidates = [
                {'id': t['obj']['id'], 'str': t['obj']['str']}
                for t in final_results
                if t['subj']['id'] in child_ids and t['pred']['id'] == pid
            ]

            # predsCommon = [_id for _id in predsB if _id in predsC]
            if pid in predsCommon:
                # if pid == "P3373":
                #     print(triple)
                triple['new_obj'] = candidates
            else:
                irrelevants.append(triple)
        else:
            if triple['pred']['id'] in [t['pred']['id'] for t in irrelevants]:
                # print("in Irrelevant:", triple)
                continue
            else:
                # print("not in Irrelevant:", triple)
                triple['new_obj'] = [{'id': triple['obj']['id'], 'str': triple['obj']['str']}]
    # return 0
    

    # final_results = [
    #     triple for triple in final_results if 
    #     (make_hashable(triple) not in {make_hashable(t) for t in eliminated_results})]

    # print("---------- transitive ----------")
    # transitive_list = get_transitive_properties(final_results)
    # print(transitive_list)

    # print("---------- symmetric ----------")
    # symmetric_list = get_symmetric_properties(final_results)
    # print(symmetric_list)



    general = [{
        'subj': {'id': subj['wikidata_id'], 'str': subj['str']},
        'pred': {'id': pred['wikidata_id'], 'str': pred['str']},
        'obj': {'id': obj['wikidata_id'], 'str': obj['str']},
        'new_obj': [{'id': new_obj['wikidata_id'], 'str': new_obj['str']}],
        'depth': 0  # the edited fact itself: present at every max_depth
    }]
    set_a = [triple for triple in final_results if any(is_valid_triple(triple, entity_id, _id) for _id in [target_id, new_target_id])]
    set_b = [triple for triple in final_results if any(is_valid_triple(triple, target_id, _id) for _id in [new_target_id, entity_id])]
    set_c = [triple for triple in final_results if any(is_valid_triple(triple, new_target_id, _id) for _id in [entity_id, target_id])]
    generality_set = deduplicate_triple_with_new_targets(chain(general, set_a, set_b, set_c))
    
    # inverse_generality_set = copy.deepcopy(generality_set)
    # for triple in inverse_generality_set:
    #     if triple['subj']['id'] == entity_id and triple['obj']['id'] == target_id:
    #         triple['obj']['id'] = new_target_id
    #         triple['obj']['str'] =  new_target_str
    #     elif triple['subj']['id'] == target_id and triple['obj']['id'] == entity_id:
    #         triple['subj']['id'] = new_target_id
    #         triple['subj']['str'] =  new_target_str
    generality_in_depth_set = deduplicate_triple_with_new_targets([t for t in chain(generality_set, []) if t['new_obj'] != []])


    # common_predicates = {triple['pred']['id'] for triple in final_results}


    # irrelevant_portability = [
    #     triple for triple in final_results if 
    #     (triple['subj']['id'] != triple['obj']['id'])
    #     # and (triple['pred']['id'] in common_predicates)
    #     and ((triple['subj']['id'] == entity_id or triple['obj']['id'] == entity_id)
    #         or (triple['subj']['id'] == target_id or triple['obj']['id'] == target_id)
    #         # or (triple['subj']['id'] == new_target_id or triple['obj']['id'] == new_target_id)
    #         )
    #     and (make_hashable(triple) not in {make_hashable(t) for t in generality_in_depth_set})
    # ]
    # portability_in_depth_set = [triple for triple in eliminated_results if (make_hashable(triple) not in {make_hashable(t) for t in generality_in_depth_set})]

    portability_in_depth_set = [
        triple for triple in final_results if 
        (triple['subj']['id'] != triple['obj']['id'])
        and triple['new_obj'] != []
        and ((triple['subj']['id'] == entity_id or triple['obj']['id'] == entity_id)
            or (triple['subj']['id'] == target_id or triple['obj']['id'] == target_id)
            or (triple['subj']['id'] == new_target_id or triple['obj']['id'] == new_target_id))
        # left-to-right logic only
        and (
            (triple['subj']['id'] not in {entity_id, target_id, new_target_id} and triple['obj']['id'] in {entity_id, target_id, new_target_id})
            or (triple['subj']['id'] in {entity_id, target_id, new_target_id} and triple['obj']['id'] not in {entity_id, target_id, new_target_id})
        )
        and not (triple['subj']['id'] == entity_id and len(triple['new_obj'])>1)
        and (make_hashable(triple) not in {make_hashable(t) for t in generality_in_depth_set})]

    # for triple in portability_in_depth_set:
    #     if len(triple['new_obj']) > 1:
    #         common_objs_by_preds = [t['obj'] for t in portability_in_depth_set if t['pred']['id'] == triple['pred']['id']]
    #         # print("common:", common_objs_by_preds)
    #         temp = copy.deepcopy(triple['new_obj'])
    #         for common_obj in common_objs_by_preds:
    #             temp.append(common_obj)
    #         triple['new_obj'] = temp

    locality_in_depth__set = [
        triple for triple in final_results if 
        (triple['subj']['id'] != triple['obj']['id'])
        and triple['new_obj'] != []

        # left-to-right logic only
        and (triple['subj']['id'] not in {entity_id, target_id, new_target_id} and triple['obj']['id'] in [t['id'] for t in triple['new_obj']])

        and (make_hashable(triple) not in {make_hashable(t) for t in generality_in_depth_set}) 
        and (make_hashable(triple) not in {make_hashable(t) for t in portability_in_depth_set})] # [triple for triple in all_outscope_effects if triple['subj']['id'] != entity_id and triple['obj']['id'] != entity_id]


    print(f"\n----- generality in depth -----")
    print(generality_in_depth_set)
    print(len(generality_in_depth_set))

    print(f"\n----- portability in depth -----")
    print(portability_in_depth_set)
    print(len(portability_in_depth_set))

    print(f"\n----- locality in depth -----")
    print(locality_in_depth__set)
    print(len(locality_in_depth__set))


    # print(f"\n----- Eliminated results -----")
    # print(eliminated_results)
    # print(len(eliminated_results))
    

    print(f"""=====> Gen({len(generality_in_depth_set)}), Por({len(portability_in_depth_set)}), Loc({len(locality_in_depth__set)})""")
    return generality_in_depth_set, portability_in_depth_set, locality_in_depth__set


if __name__ == "__main__":

    subj = {'wikidata_id': 'Q11975', 'str': 'Britney Spears'}
    pred = {'wikidata_id': 'P22', 'str': 'father'}
    obj = {'wikidata_id': 'Q96816262', 'str': 'James Parnell Spears'}
    new_obj = {'wikidata_id': 'Q35723119', 'str': 'Errol Musk'}

    # subj = {'wikidata_id': 'Q473873', 'str': 'Vevo'}
    # pred = {'wikidata_id': 'P127', 'str': 'owned by'}
    # obj = {'wikidata_id': 'Q95', 'str': 'Google'}
    # new_obj = {'wikidata_id': 'Q2135', 'str': 'Winnipeg'}

    max_depth = 6

    # entity_id = "Q11975"  # Example subject 1 (e.g., Britney Spears)
    # target_id = "Q96816262"  # Example subject 2 (e.g., James Parnell Spears)
    # new_target_id = "Q35723119"
    side_effects_checker_bfs(subj, pred, obj, new_obj, max_depth)