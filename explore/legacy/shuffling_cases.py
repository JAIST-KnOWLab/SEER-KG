import random
from collections import defaultdict
import json

from SPARQLWrapper import SPARQLWrapper, JSON
from concurrent.futures import ThreadPoolExecutor

import time

import spacy
nlp = spacy.load("en_core_web_sm")

def check_class_instance(wikidata_id):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setReturnFormat(JSON)
    time.sleep(1) 

    data = []
    query = f"""
        SELECT ?class WHERE {{
            ?s wdt:{wikidata_id} ?o .
            ?o wdt:P31 ?class .
            FILTER(?class IN (wd:Q5)) # wd:Q43229, wd:Q2424752
        }} LIMIT 1
    """
    sparql.setQuery(query)
    
    try:
        results = sparql.query().convert()
    except Exception as e:
        print(f"Error during query execution: {e}")
        return []

    for result in results["results"]["bindings"]:
        classId = result["class"]["value"].split('/')[-1]
        data.append(classId)

    return data[0]


def get_shuffling_cases(dataset):
    shuffled_triples = []

    for data in dataset:
        pred_label, pred_id = data['pred']['str'], data['pred']['id']
        common_set = data['common']

        new_common = []
        visited_candidates = []

        print(pred_label)
        adp = False
        classId = check_class_instance(pred_id)
        question = "Who" if classId == "Q5" else "What" if classId == "Q2424752" else "Where" 
        tokens = nlp(pred_label.split(' ')[-1])
        for token in tokens:
            if token.pos_ == "ADP":
                adp = True

        for x in common_set:
            subj, obj = x['subj'], x['obj']

            candidates = [
                (minor['subj'], minor['obj']) for minor in common_set
                if len(minor['obj']) <= len(obj) and minor['subj']['id'] != subj['id']
            ] # Find matching elements based on length and ID
            if len(candidates) == 0:
                candidates = [(minor['subj'], minor['obj']) for minor in common_set if minor['subj']['id'] != subj['id']]
            
            sorted_candidates = sorted(candidates, key=lambda i: len(i[1]), reverse=True) # Sort by length similarity

            max_length = len(sorted_candidates[0][1])
            max_candidates = [candidate for candidate in sorted_candidates if len(candidate[1]) == max_length and candidate[0] not in visited_candidates]
            if len(max_candidates) == 0:
                max_candidates = [candidate for candidate in sorted_candidates if len(candidate[1]) == max_length]

            random_candidate = random.choice(max_candidates)
            
            visited_candidates.append(random_candidate[0])

            new_obj = {'subj': random_candidate[0], 'obj': random_candidate[1]}

            adpStr = " of " if adp == False else " "
            prompt = f"""{question} is the {pred_label}{adpStr}{subj['str']}?"""
            cleanedPrompt = prompt if len(pred_label.split(' ')) > 1 else prompt

            print(prompt)
            print(cleanedPrompt)
            
            new_common.append({
                'subj': subj,
                'obj': obj,
                'new_obj': new_obj,
                'general_prompt': cleanedPrompt
            })

        shuffled_triples.append({
            'pred': {'str': pred_label, 'id': pred_id},
            'common': new_common
        })

    return shuffled_triples


if __name__ == "__main__":

    # # Example list of triples
    # triples = [
    #     ('Britney_Spears', 'father', 'James_Parnell_Spears'),
    #     ('Elon_Musk', 'father', 'Errol_Musk'),
    #     ('Bill_Gates', 'father', 'William_Gates_Sr'),
    #     ('Brad_Pitt', 'father', 'John_Pitt'),
    #     ('Angelina_Jolie', 'father', 'Jon_Voight'),
    #     # Add more triples as needed
    # ]

    with open('model/characteristics/people5.json', 'r') as file:
        dataset = json.load(file)

    shuffled_triples = get_shuffling_cases(dataset)

    # Step 3: Output the generated cases (shuffled triples)
    for i in shuffled_triples:
        print(i['pred'])
        print(f"""----------""")
        for j in i['common']:
            print(f"""subj: {j['subj']}""")
            print(f"""obj: {j['obj']}""")
            print(f"""new_obj: {j['new_obj']}""")
            print()
            
    print(len(shuffled_triples))

    with open('model/characteristics/people_shuffled5.json', 'w', encoding='utf-8') as f:
        json.dump(shuffled_triples, f, ensure_ascii=False, indent=4)
