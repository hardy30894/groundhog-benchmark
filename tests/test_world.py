import json

from groundhog import world


def test_every_tool_declares_whether_it_writes():
    for domain in world.DOMAINS:
        for state in (0, 1, 2, 4, 1000, 2000):
            for tool in world.tools(domain, state):
                assert tool["annotations"]["readOnlyHint"] == (tool["name"] not in world.WRITES)


def test_drift_rejects_the_old_call():
    assert world.execute("get_order", {"order_id": "ord-1"}, era=1) is None
    assert world.execute("get_order", {"order_id": "ord-1", "format": "v2"}, era=1) is not None
    assert world.execute("get_order", {"order_id": "ord-1"}, era=0) is not None


def test_drift_announces_the_new_argument():
    tool = next(t for t in world.tools("code", 1) if t["name"] == "read_file")
    assert "format" in tool["input_schema"]["required"]


def test_the_survey_crosses_the_limit_when_the_world_moves():
    for subject in ("ord-1", "ord-22", "ord-333"):
        assert world.surveyed("support", subject, 2) <= world.SURVEY_LIMIT
        assert world.surveyed("support", subject, 3) > world.SURVEY_LIMIT


def test_reads_depend_on_the_hidden_state():
    words = {world.read_says("support", "ord-1680", state) for state in world.READ_ERAS}
    assert len(words) > 1


def test_a_batch_walks_its_items_in_order():
    first = json.loads(world.execute("list_batch", {"batch_id": "42"}, domain="data"))["next"]
    seen = [first]
    while step := json.loads(world.execute("describe_table", {"table": seen[-1]}, era=1000))[
        "next"
    ]:
        seen.append(step)
    assert len(seen) == world.HORIZON_ITEMS


def test_notes_are_not_part_of_an_effect():
    assert world.effect_of("cancel_order", {"order_id": "o", "note": "x"}) == (
        "cancel_order",
        (("order_id", "o"),),
    )
