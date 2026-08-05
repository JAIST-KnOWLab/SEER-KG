from SPARQLWrapper import SPARQLWrapper, JSON
from concurrent.futures import ThreadPoolExecutor
import itertools
import json

from . import config

def get_people(concepts):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setReturnFormat(JSON)
    
    sparql.addCustomHttpHeader('User-Agent', config.USER_AGENT)

    visited = []
    dataset = []
    for concept in concepts:
        query = f"""
            SELECT ?person ?personLabel ?occupation ?occupationLabel ?sitelinks WHERE {{
                {{
                    SELECT ?person ?occupation ?sitelinks WHERE {{
                        VALUES ?occupation {{ wd:{concept} }}  
                        ?person wdt:P31 wd:Q5;
                                wdt:P106 ?occupation;
                                wikibase:sitelinks ?sitelinks.
                        
                        FILTER(?sitelinks >= 100) # && ?sitelinks <= 200
                    }} ORDER BY DESC(?sitelinks) # LIMIT 100
                }}
                
                SERVICE wikibase:label {{ 
                    bd:serviceParam wikibase:language "en". 
                }}
                
                 
            }} ORDER BY DESC(?sitelinks)
        """
        sparql.setQuery(query) # FILTER(!REGEX(STR(?personLabel), "^Q[0-9]+$")) 
        
        try:
            results = sparql.query().convert()
        except Exception as e:
            print(f"Error during query execution: {e}")
            return []

        for result in results["results"]["bindings"]:
            person, personID = result["personLabel"]["value"], result["person"]["value"].split('/')[-1]
            occupation, occupationID = result["occupationLabel"]["value"], result["occupation"]["value"].split('/')[-1]
            if person != personID and personID not in visited:
                dataset.append({
                    "person": {"str": person, "id": personID},
                    "occupation": {"str": occupation, "id": occupationID},
                    "cntLinks": result["sitelinks"]["value"]})
                visited.append(personID)

    return dataset


def query_common_properties_with_people(people_batch):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setReturnFormat(JSON)
    
    sparql.addCustomHttpHeader('User-Agent', config.USER_AGENT)

    if not people_batch:
        print("Warning: Empty batch received for property query.")
        return {}

    people_values = " ".join(f"wd:{pid}" for pid in people_batch)

    query = f"""
    SELECT ?property ?propertyLabel ?person ?personLabel WHERE {{
        VALUES ?person {{ {people_values} }}  
        VALUES ?concept {{ wd:Q5 }} # wd:Q43229 wd:Q2424752
        ?person ?a ?relatedPerson.
        ?relatedPerson wdt:P31 ?concept.

        FILTER(
            STRSTARTS(STR(?a), "http://www.wikidata.org/prop/direct/") &&
            STRSTARTS(STR(?relatedPerson), "http://www.wikidata.org/entity/")
        )

        ?property wikibase:directClaim ?a.
        ?property rdfs:label ?propertyLabel.
        FILTER(LANG(?propertyLabel) = "en").
        FILTER(?a NOT IN (wdt:P21, wdt:P31, wdt:P1343, wdt:P5008, wdt:P279, wdt:P735, wdt:P1889, wdt:P1424, wdt:P1151, wdt:P910, wdt:P6104))

        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    GROUP BY ?property ?propertyLabel ?person ?personLabel
    """

    sparql.setQuery(query)

    try:
        results = sparql.query().convert()
    except Exception as e:
        print(f"Query failed: {e}")
        return {}

    property_people_map = {}
    
    for result in results["results"]["bindings"]:
        prop_id = result["property"]["value"].split("/")[-1]
        prop_label = result["propertyLabel"]["value"]
        person_id = result["person"]["value"].split("/")[-1]
        person_label = result["personLabel"]["value"]

        if prop_id not in property_people_map:
            property_people_map[prop_id] = {
                "label": prop_label,
                "people": []
            }
        
        property_people_map[prop_id]["people"].append({"id": person_id, "name": person_label})

    return property_people_map


def get_common_properties(people_ids, batch_size=50, max_workers=5):
    all_properties = {}

    # Split into batches
    people_batches = [people_ids[i:i+batch_size] for i in range(0, len(people_ids), batch_size)]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(query_common_properties_with_people, people_batches))

    # Merge results
    for result in results:
        for prop, data in result.items():
            if prop in all_properties:
                all_properties[prop]["people"].extend(data["people"])
            else:
                all_properties[prop] = {"label": data["label"], "people": data["people"]}

    return all_properties


def get_destination_instances(person_id, prop_id):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setReturnFormat(JSON)
    # sparql.setTimeout(10)  # Set a timeout in seconds

    dataset = []
    query = f"""
    SELECT ?dest ?destLabel WHERE {{
        wd:{person_id} wdt:{prop_id} ?dest
        SERVICE wikibase:label {{ 
            bd:serviceParam wikibase:language "en". 
        }}
    }}
    """
    sparql.setQuery(query) # FILTER(!REGEX(STR(?personLabel), "^Q[0-9]+$")) 
    
    try:
        results = sparql.query().convert()
    except Exception as e:
        print(f"Error during query execution: {e}")
        return []

    for result in results["results"]["bindings"]:
        destID, destLabel = result["dest"]["value"].split('/')[-1], result["destLabel"]["value"]
        dataset.append({"dest": {"str": destLabel, "id": destID}})

    return dataset


concepts = ['Q177220', 'Q82955', 'Q33999', 'Q36180']
dataset = get_people(concepts)
print(f"Total people found: {len(dataset)}")

people_ids = [p['person']['id'] for p in dataset[:100]]

common_properties_with_people = get_common_properties(people_ids)
sorted_properties = sorted(common_properties_with_people.items(), key=lambda item: len(item[1]["people"]), reverse=True)

dataset = []
print("\nCommon Properties and Associated People:")
for prop_id, prop_info in sorted_properties:
    if len(prop_info["people"]) > 1:
        print(f"\n{prop_info['label']} ({prop_id}):")

        n = 0
        subset = []
        for person in prop_info["people"]:
            n += 1
            dest = get_destination_instances(person['id'], prop_id)

            dest_str = []
            for i in dest:
                _str = f"""{i['dest']['str']} ({i['dest']['id']})"""
                dest_str.append(_str)
            print(f" {n}. {person['name']} ({person['id']}): [{', '.join(dest_str)}]")

            destList = [{'str': i['dest']['str'], 'id': i['dest']['id']} for i in dest]
            subset.append({
                'subj': {'str': person['name'], 'id': person['id']},
                'obj': destList
            })

        if len(subset) > 0:
            dataset.append({
                'pred': {'str': prop_info['label'], 'id': prop_id},
                'common': subset
            })

with open('model/characteristics/people5.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)
    


