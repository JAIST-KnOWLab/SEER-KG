"""seer-kg: knowledge-effect probing over an edit dataset via Wikidata.

Pipeline stages:
    * probe        -- resolve each edit against Wikidata and run a BFS
                      side-effect checker to derive generality / portability / locality
                      effects, one output per max_depth (see `probe`).
    * chain        -- convert those effects into reasoning chains (chains + onto_chains)
                      with masked cloze prompts from an LLM (see `chain_converter`).

Configuration lives in `explore.config`.
"""

__version__ = "1.0.0"
