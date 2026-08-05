#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""seer-kg command-line entry point.

Knowledge-effect probing over an edit dataset, then LLM chain conversion.

Usage (run from the project root):

    python main.py                     # full run: all cases, all depths
    python main.py 0 to 10             # limit data: cases in index range [0, 10)
    python main.py [2,3,4]             # limit depth: only these depths
    python main.py [2,6]               # limit depth: only depths 2 and 6
    python main.py 0 to 10 [2,6]       # combine a data range and a depth set
    python main.py finalize            # compile per-depth checkpoints into JSON arrays
    python main.py finalize [2,6]      # ... for specific depths
    python main.py chain               # LLM chain conversion (chains + onto_chains)

Data range and depth set may appear in any order. Depths must be a subset of
config.DEPTHS. Outputs go to results/ (see README).
"""

import os
import re
import sys

# Make the explore package importable regardless of the invocation directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from explore import config


def _parse_depths(text: str):
    """Depths from a `[..]` token (e.g. '[2,6]'); default to config.DEPTHS."""
    match = re.search(r"\[([0-9,\s]+)\]", text)
    if not match:
        return list(config.DEPTHS)
    depths = []
    for part in match.group(1).split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value not in config.DEPTHS:
            raise SystemExit(f"unsupported depth {value}; choose from {config.DEPTHS}")
        depths.append(value)
    return depths or list(config.DEPTHS)


def _parse_range(text: str):
    """Data range from 'START to END' (or two bare ints); default (0, None)."""
    match = re.search(r"(\d+)\s+to\s+(\d+)", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    bare = re.findall(r"(?<!\[)(?<![\d,])\b\d+\b(?![\d,])", text)
    if len(bare) >= 2:
        return int(bare[0]), int(bare[1])
    return 0, None


def main():
    text = " ".join(sys.argv[1:]).strip()
    lowered = text.lower()

    if lowered in {"-h", "--help", "help"}:
        print(__doc__)
        return

    depths = _parse_depths(text)

    if "chain" in lowered:
        from explore import chain_converter
        chain_converter.run()
        return

    if "finalize" in lowered:
        from explore import probe
        probe.finalize(depths)
        return

    start, end = _parse_range(text)
    from explore import probe
    probe.run(start=start, end=end, depths=depths)


if __name__ == "__main__":
    main()
