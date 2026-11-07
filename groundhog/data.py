"""Reads the dataset: the 3,300 benchmark instances and the classifier training sets."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path


def _locate():
    """The dataset directory: inside the package once installed, data/ in a checkout."""
    here = Path(__file__).resolve().parent
    for candidate in (here / "dataset", here.parent / "data"):
        if (candidate / "groundhog.jsonl").exists():
            return candidate
    raise FileNotFoundError("the dataset is missing: reinstall groundhog-bench or check out data/")


DATA = _locate()
BENCHMARK = DATA / "groundhog.jsonl"


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def call(spec):
    return (spec["tool"], tuple(sorted(spec["args"].items())))


def instance(record):
    """A record in the form the harness and the scorer use."""
    return {
        "opening": record["request"],
        "effects": [call(c) for c in record["calls"]],
        "answer": record["answer"],
        "excludes": record["excludes"],
        "era": record["state"],
        "ordered": record["ordered"],
        "max_turns": record["max_turns"],
        "forbidden": [call(c) for c in record["forbidden"]],
        "domain": record["domain"],
    }


@lru_cache(maxsize=None)
def _benchmark():
    grouped = {}
    for record in read(BENCHMARK):
        grouped.setdefault((record["domain"], record["family"]), []).append(record)
    return grouped


def records(domain, family):
    return list(_benchmark().get((domain, family), []))


def instances(domain, family):
    return [instance(r) for r in records(domain, family)]


@lru_cache(maxsize=None)
def _training(name):
    return tuple(read(DATA / "train" / f"{name}.jsonl"))


def training(name):
    """A training set as (family, instance) pairs, in the order it was drawn."""
    return [(r["family"], instance(r)) for r in _training(name)]


def digest():
    return hashlib.sha256(BENCHMARK.read_bytes()).hexdigest()[:16]
