"""The benchmark checking itself: the written policy, run without a cache, satisfies every
instance, and costs exactly what the published baseline says it costs."""

import json
from pathlib import Path

import pytest

from groundhog import agent, data
from groundhog.conditions import FAMILIES
from groundhog.model import Model
from groundhog.score import verify_control
from groundhog.world import DOMAINS

ROOT = Path(__file__).resolve().parent.parent


class InProcess:
    """The model without its HTTP loop, metered the same way."""

    def __init__(self, domain):
        self.model = Model(0)
        self.model.domain = domain
        self.spent = 0.0

    def post(self, url, body, mark=None):
        out, spent = self.model.respond(url[url.index("/v1") :], json.loads(json.dumps(body)))
        self.spent += spent
        return out


def baseline(domain, family, wire="anthropic"):
    model = InProcess(domain)
    runs = []
    original = agent.post
    agent.post = model.post
    try:
        for inst in data.instances(domain, family):
            model.model.era = inst["era"]
            before = model.spent
            outcome = agent.run(
                "http://model",
                inst["opening"],
                era=inst["era"],
                wire=wire,
                model="stub",
                domain=domain,
                max_turns=inst["max_turns"],
            )
            outcome["metered"] = model.spent - before
            runs.append(outcome)
    finally:
        agent.post = original
        model.model.server_close()
    return runs


PUBLISHED = {
    r["family"]: r["control_tokens"]
    for r in json.loads((ROOT / "results" / "control.json").read_text())["rows"]
}


@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize("family", FAMILIES)
def test_the_baseline_is_correct_and_costs_what_was_published(domain, family):
    runs = baseline(domain, family)
    assert verify_control(f"{domain}/{family}", data.instances(domain, family), runs) == []
    assert sum(r["metered"] for r in runs) == pytest.approx(
        PUBLISHED[f"{domain}/{family}"], abs=1e-6
    )


@pytest.mark.parametrize("domain", DOMAINS)
def test_the_openai_wire_is_correct_too(domain):
    for family in FAMILIES:
        runs = baseline(domain, family, wire="openai")
        assert verify_control(family, data.instances(domain, family), runs) == []
