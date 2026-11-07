import collections
import hashlib
import json

import pytest

from groundhog import data
from groundhog.conditions import CONDITIONS, FAMILIES, condition_of
from groundhog.world import DOMAINS, WRITES

FIELDS = [
    "id",
    "domain",
    "family",
    "condition",
    "request",
    "state",
    "max_turns",
    "calls",
    "ordered",
    "forbidden",
    "answer",
    "excludes",
]


@pytest.fixture(scope="module")
def records():
    return data.read(data.BENCHMARK)


def test_the_files_match_their_published_checksums():
    sums = (data.DATA / "SHA256SUMS").read_text().split("\n")
    for line in filter(None, sums):
        digest, name = line.split()
        assert hashlib.sha256((data.DATA / name).read_bytes()).hexdigest() == digest, name


def test_size_and_balance(records):
    assert len(records) == 3300
    pairs = collections.Counter((r["domain"], r["family"]) for r in records)
    assert set(pairs) == {(d, f) for d in DOMAINS for f in FAMILIES}
    assert set(pairs.values()) == {100}
    assert len({r["id"] for r in records}) == len(records)


def test_every_record_has_the_documented_fields(records):
    for r in records:
        assert list(r) == FIELDS
        assert r["condition"] == condition_of(r["family"])
        assert all(set(c) == {"tool", "args"} for c in r["calls"] + r["forbidden"])


def test_eleven_families_in_five_conditions():
    assert len(FAMILIES) == 11
    assert list(CONDITIONS) == ["identity", "values", "meaning", "world", "depth"]


def test_only_refusals_require_no_call(records):
    empty = [r for r in records if not r["calls"]]
    assert empty and all(r["family"] == "reversal" for r in empty)
    assert all(r["forbidden"][0]["tool"] in WRITES for r in empty)


def test_world_families_change_halfway(records):
    for family, before, after in (("drift", 0, 1), ("quiet", 2, 3), ("standing", 4, 5)):
        for domain in DOMAINS:
            states = [
                r["state"] for r in records if r["family"] == family and r["domain"] == domain
            ]
            assert states == [before] * 50 + [after] * 50


def test_procedures_are_ordered(records):
    for r in records:
        assert r["ordered"] == (r["family"] in ("horizon", "quiet"))


def test_training_sets():
    for name in ("w5h2", "agentreuse", "agentreuse-check"):
        pairs = data.training(name)
        assert len(pairs) == 2400
        assert {family for family, _ in pairs} >= set(FAMILIES)


def test_instances_round_trip(records):
    inst = data.instance(records[0])
    assert inst["opening"] == records[0]["request"]
    assert inst["effects"] == [
        (c["tool"], tuple(sorted(c["args"].items()))) for c in records[0]["calls"]
    ]
    assert json.loads(json.dumps(records[0])) == records[0]
