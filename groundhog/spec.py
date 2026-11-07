"""The benchmark's identity: a version and a digest of everything that decides a number.

Two results are comparable only if they were produced from the same instances under the
same rules. The digest covers the dataset and the harness modules that run and score it,
and every result file carries it.
"""

import hashlib
from pathlib import Path

VERSION = "2.0"

HERE = Path(__file__).resolve().parent

DEFINES = ("data.py", "domains.py", "world.py", "model.py", "agent.py", "score.py", "run.py")

CHANGELOG = {
    "2.0": "The dataset is a file. The 3,300 instances are read from data/groundhog.jsonl; "
    "the file holds exactly the instances 1.33 used, and the agent, model, world and scorer "
    "behave as they did in 1.33.",
}


def digest():
    parts = hashlib.sha256()
    for name in DEFINES:
        parts.update(hashlib.sha256((HERE / name).read_bytes()).digest())
    parts.update(hashlib.sha256((HERE.parent / "data" / "groundhog.jsonl").read_bytes()).digest())
    return parts.hexdigest()[:16]


def stamp():
    return {"spec_version": VERSION, "spec_digest": digest()}
