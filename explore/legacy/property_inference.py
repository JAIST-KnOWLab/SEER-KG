from collections import defaultdict

def get_inverse_properties(triples):
    """
    Infer inverse relationships where:
    (A, r1, B) and (B, r2, A) and r1 ≠ r2
    """
    triple_set = set((t['subj']['id'], t['pred']['id'], t['obj']['id']) for t in triples)
    inferred = []
    seen = set()

    for t1 in triples:
        s1, p1, o1 = t1['subj']['id'], t1['pred']['id'], t1['obj']['id']
        for t2 in triples:
            s2, p2, o2 = t2['subj']['id'], t2['pred']['id'], t2['obj']['id']
            if s1 == o2 and o1 == s2 and p1 != p2:
                key = tuple(sorted([(s1, p1, o1), (s2, p2, o2)]))
                if key not in seen:
                    seen.add(key)
                    inferred.append({
                        'subj': t1['subj'],
                        'pred': t1['pred'],
                        'obj': t1['obj'],
                        'inferrences': [
                            { 'subj': t1['subj'], 'pred': t1['pred'], 'obj': t1['obj'] }, 
                            { 'subj': t2['subj'], 'pred': t2['pred'], 'obj': t2['obj'] }
                        ]
                    })
    return inferred


def get_symmetric_properties(triples):
    """
    Infer symmetric relationships where:
    (A, r, B) and (B, r, A)
    """
    triple_set = set((t['subj']['id'], t['pred']['id'], t['obj']['id']) for t in triples)
    inferred = []
    seen = set()

    for t in triples:
        s, p, o = t['subj']['id'], t['pred']['id'], t['obj']['id']
        if (o, p, s) in triple_set:
            key = tuple(sorted([(s, p, o), (o, p, s)]))
            if key not in seen:
                seen.add(key)
                inferred.append({
                    'subj': t['subj'],
                    'pred': t['pred'],
                    'obj': t['obj'],
                    'inferrences': [
                        { 'subj': t['subj'], 'pred': t['pred'], 'obj': t['obj'] }, 
                        { 'subj': t['obj'], 'pred': t['pred'], 'obj': t['subj'] }
                    ]
                })
    return inferred


def get_asymmetric_properties(triples):
    """
    Infer asymmetric relationships where:
    (A, r, B) holds but (B, r, A) does NOT ⇒ r behaves asymmetrically for that pair.

    Each result also carries a `violation` flag: True when the reverse triple
    (B, r, A) is unexpectedly present (an asymmetry violation), False otherwise.
    """
    triple_set = set((t['subj']['id'], t['pred']['id'], t['obj']['id']) for t in triples)
    inferred = []
    seen = set()

    for t in triples:
        s, p, o = t['subj']['id'], t['pred']['id'], t['obj']['id']
        if s == o:
            continue  # reflexive, not meaningful for asymmetry
        reverse_present = (o, p, s) in triple_set
        key = tuple(sorted([(s, p, o), (o, p, s)]))
        if key in seen:
            continue
        seen.add(key)
        inferred.append({
            'subj': t['subj'],
            'pred': t['pred'],
            'obj': t['obj'],
            'violation': reverse_present,
            'inferrences': [
                { 'subj': t['subj'], 'pred': t['pred'], 'obj': t['obj'] }
            ] + ([
                { 'subj': t['obj'], 'pred': t['pred'], 'obj': t['subj'] }
            ] if reverse_present else [])
        })
    return inferred


def get_transitive_properties(triples):
    """
    Infer transitive relationships where:
    (A, r, B) and (B, r, C) ⇒ (A, r, C)
    """
    inferred = []

    forward_map = defaultdict(set)
    str_map = {}

    for t in triples:
        s, p, o = t['subj']['id'], t['pred']['id'], t['obj']['id']
        forward_map[(s, p)].add(o)
        str_map[s] = t['subj']['str']
        str_map[p] = t['pred']['str']
        str_map[o] = t['obj']['str']

    seen = set()
    for (a, p), bs in forward_map.items():
        for b in bs:
            if (b, p) in forward_map:
                for c in forward_map[(b, p)]:
                    if c != a:
                        key = (a, p, c)
                        if key not in seen:
                            seen.add(key)
                            inferred.append({
                                'subj': {'id': a, 'str': str_map[a]},
                                'pred': {'id': p, 'str': str_map[p]},
                                'obj': {'id': c, 'str': str_map[c]},
                                'inferrences': [
                                    { 'subj': {'id': a, 'str': str_map[a]}, 'pred': {'id': p, 'str': str_map[p]}, 'obj': {'id': b, 'str': str_map[b]} }, 
                                    { 'subj': {'id': b, 'str': str_map[b]}, 'pred': {'id': p, 'str': str_map[p]}, 'obj': {'id': c, 'str': str_map[c]} }
                                ]
                            })

    return inferred


# Optional test usage
if __name__ == "__main__":
    triples = [
        # Inverse
        {'subj': {'id': 'Q1', 'str': 'Alice'}, 'pred': {'id': 'P1', 'str': 'employs'}, 'obj': {'id': 'Q2', 'str': 'Bob'}, 'new_obj': []},
        {'subj': {'id': 'Q2', 'str': 'Bob'}, 'pred': {'id': 'P2', 'str': 'employed_by'}, 'obj': {'id': 'Q1', 'str': 'Alice'}, 'new_obj': []},
        {'subj': {'id': 'Q3', 'str': 'Carol'}, 'pred': {'id': 'P1', 'str': 'employs'}, 'obj': {'id': 'Q4', 'str': 'Dan'}, 'new_obj': []},
        {'subj': {'id': 'Q4', 'str': 'Dan'}, 'pred': {'id': 'P2', 'str': 'employed_by'}, 'obj': {'id': 'Q3', 'str': 'Carol'}, 'new_obj': []},

        # Symmetric
        {'subj': {'id': 'Q5', 'str': 'Tom'}, 'pred': {'id': 'P3', 'str': 'married_to'}, 'obj': {'id': 'Q6', 'str': 'Jerry'}, 'new_obj': []},
        {'subj': {'id': 'Q6', 'str': 'Jerry'}, 'pred': {'id': 'P3', 'str': 'married_to'}, 'obj': {'id': 'Q5', 'str': 'Tom'}, 'new_obj': []},
        {'subj': {'id': 'Q7', 'str': 'Mickey'}, 'pred': {'id': 'P3', 'str': 'married_to'}, 'obj': {'id': 'Q8', 'str': 'Minnie'}, 'new_obj': []},
        {'subj': {'id': 'Q8', 'str': 'Minnie'}, 'pred': {'id': 'P3', 'str': 'married_to'}, 'obj': {'id': 'Q7', 'str': 'Mickey'}, 'new_obj': []},

        # Transitive
        {'subj': {'id': 'Q9', 'str': 'Paris'}, 'pred': {'id': 'P4', 'str': 'located_in'}, 'obj': {'id': 'Q10', 'str': 'France'}, 'new_obj': []},
        {'subj': {'id': 'Q10', 'str': 'France'}, 'pred': {'id': 'P4', 'str': 'located_in'}, 'obj': {'id': 'Q11', 'str': 'Europe'}, 'new_obj': []},
        {'subj': {'id': 'Q11', 'str': 'Europe'}, 'pred': {'id': 'P4', 'str': 'located_in'}, 'obj': {'id': 'Q12', 'str': 'Earth'}, 'new_obj': []},

        # Asymmetric (parent_of: reverse should never hold)
        {'subj': {'id': 'Q13', 'str': 'Homer'}, 'pred': {'id': 'P5', 'str': 'parent_of'}, 'obj': {'id': 'Q14', 'str': 'Bart'}, 'new_obj': []},
        # Asymmetry violation: both directions present for the same predicate
        {'subj': {'id': 'Q15', 'str': 'Loop_A'}, 'pred': {'id': 'P5', 'str': 'parent_of'}, 'obj': {'id': 'Q16', 'str': 'Loop_B'}, 'new_obj': []},
        {'subj': {'id': 'Q16', 'str': 'Loop_B'}, 'pred': {'id': 'P5', 'str': 'parent_of'}, 'obj': {'id': 'Q15', 'str': 'Loop_A'}, 'new_obj': []},
    ]


    print("=== Inverse Triples ===")
    for r in get_inverse_properties(triples):
        print(r)

    print("\n=== Symmetric Triples ===")
    for r in get_symmetric_properties(triples):
        print(r)

    print("\n=== Transitive Triples ===")
    for r in get_transitive_properties(triples):
        print(r)

    print("\n=== Asymmetric Triples ===")
    for r in get_asymmetric_properties(triples):
        print(r)
