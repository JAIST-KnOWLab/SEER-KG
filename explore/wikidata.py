"""Wikidata access: resolve dataset subjects to QIDs and fetch labels.

Public API
----------
search_entity(label)              -> ranked candidate QIDs for a label.
entity_has_property(qid, pid)     -> whether the entity actually uses the predicate.
get_property_label(pid)           -> cached English label for a property.
resolve_subject(label, relation)  -> (qid, method) for the best subject match.

All network calls are memoized, throttled, and retried with exponential backoff so
a full-dataset run stays polite to the public Wikidata endpoints. Tunables live in
`explore.config`.
"""

import time
from typing import List, Optional, Tuple

import requests
from SPARQLWrapper import SPARQLWrapper, JSON

from . import config

# Memoization caches (keyed by label / (qid, pid) / pid).
_search_cache: dict = {}
_has_property_cache: dict = {}
_label_cache: dict = {}

_session = requests.Session()
_session.headers.update({"User-Agent": config.USER_AGENT})

_sparql = SPARQLWrapper(config.WIKIDATA_SPARQL, agent=config.USER_AGENT)
_sparql.setReturnFormat(JSON)


def _sleep() -> None:
    if config.SLEEP_BETWEEN_CALLS:
        time.sleep(config.SLEEP_BETWEEN_CALLS)


def _run_sparql(query: str) -> Optional[dict]:
    """Run a SPARQL query with retry/backoff. Returns parsed JSON, or None on failure."""
    _sparql.setQuery(query)
    for attempt in range(config.MAX_RETRIES):
        try:
            _sleep()
            return _sparql.query().convert()
        except Exception as exc:  # network / rate-limit / timeout
            wait = config.BACKOFF_BASE * (2 ** attempt)
            print(f"[sparql] error (attempt {attempt + 1}/{config.MAX_RETRIES}): {exc} -> retry in {wait:.0f}s")
            time.sleep(wait)
    print("[sparql] giving up on query")
    return None


def search_entity(label: str) -> List[str]:
    """Return rank-ordered candidate QIDs for `label` (best first), cached."""
    if label in _search_cache:
        return _search_cache[label]

    params = {
        "action": "wbsearchentities",
        "search": label,
        "language": "en",
        "uselang": "en",
        "type": "item",
        "format": "json",
        "limit": config.SEARCH_LIMIT,
    }
    qids: List[str] = []
    for attempt in range(config.MAX_RETRIES):
        try:
            _sleep()
            response = _session.get(config.WIKIDATA_API, params=params, timeout=30)
            response.raise_for_status()
            qids = [hit["id"] for hit in response.json().get("search", [])]
            break
        except Exception as exc:
            wait = config.BACKOFF_BASE * (2 ** attempt)
            print(f"[search] error for {label!r} (attempt {attempt + 1}/{config.MAX_RETRIES}): {exc} -> retry in {wait:.0f}s")
            time.sleep(wait)

    _search_cache[label] = qids
    return qids


def entity_has_property(qid: str, pid: str) -> bool:
    """True if `wd:qid wdt:pid ?o` has at least one binding, cached."""
    key = (qid, pid)
    if key in _has_property_cache:
        return _has_property_cache[key]

    result = _run_sparql(f"ASK {{ wd:{qid} wdt:{pid} ?o }}")
    present = bool(result.get("boolean", False)) if result else False
    _has_property_cache[key] = present
    return present


def get_property_label(pid: str) -> str:
    """English label for a property id (e.g. P103 -> 'native language'), cached."""
    if pid in _label_cache:
        return _label_cache[pid]

    query = f"""
        SELECT ?label WHERE {{
            wd:{pid} rdfs:label ?label .
            FILTER(LANG(?label) = "en")
        }} LIMIT 1
    """
    result = _run_sparql(query)
    label = pid
    if result:
        bindings = result["results"]["bindings"]
        if bindings:
            label = bindings[0]["label"]["value"]
    _label_cache[pid] = label
    return label


def resolve_subject(label: str, relation_id: str) -> Tuple[Optional[str], str]:
    """Resolve a subject string to a QID.

    Returns (qid, method):
      * ("Q...", "verified")  : first candidate that actually uses `relation_id`.
      * ("Q...", "fallback")  : no candidate has the predicate (common when the
                                statement drifted off Wikidata since the dataset was
                                built), so fall back to the top search hit.
      * (None,   "unresolved"): search returned no candidates at all.

    The dataset already supplies obj/new_obj QIDs, so a correct subject QID suffices
    for the BFS even if that exact predicate is no longer asserted; the predicate
    check is only a disambiguation aid.
    """
    candidates = search_entity(label)
    for qid in candidates:
        if entity_has_property(qid, relation_id):
            return qid, "verified"
    if candidates:
        return candidates[0], "fallback"
    return None, "unresolved"
