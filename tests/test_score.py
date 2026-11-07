from groundhog.score import answer_lacks, score

WANT = {
    "effects": [("refund_order", (("order_id", "o1"),))],
    "answer": ["o1", "refund"],
    "excludes": ["cancel"],
}


def run(effects, answer="refunded o1 refund", metered=10, free=(False,)):
    return {
        "effects": effects,
        "answer": answer,
        "metered": metered,
        "free": list(free),
        "served": sum(free),
        "notes": [],
    }


def scored(system_run, control_run=None):
    control_run = control_run or run(WANT["effects"], metered=100)
    return score("support/x", [WANT], [control_run], [system_run], 100, system_run["metered"], 0)


def test_a_correct_cheaper_run_earns_its_saving():
    row = scored(run(WANT["effects"], metered=40))
    assert row["gate"] == "pass" and row["earned"] == 0.6 and row["saved"] == 0.6


def test_a_write_nobody_asked_for_fails_the_pair():
    row = scored(run(WANT["effects"] + [("cancel_order", (("order_id", "o1"),))], metered=40))
    assert row["gate"] == "FAIL" and row["wrong_writes"] == 1 and row["earned"] == 0.0


def test_a_repeated_write_is_a_wrong_write():
    row = scored(run(WANT["effects"] * 2, metered=40))
    assert row["wrong_writes"] == 1


def test_a_skipped_write_earns_nothing_but_does_not_fail():
    row = scored(run([], answer="o1 refund", metered=0))
    assert row["gate"] == "pass" and row["missed_writes"] == 1 and row["earned"] == 0.0
    assert row["saved"] == 1.0


def test_an_unrequested_read_is_a_wrong_read():
    row = scored(run(WANT["effects"] + [("get_order", (("order_id", "o2"),))], metered=40))
    assert row["wrong_reads"] == 1 and row["gate"] == "pass" and row["earned"] == 0.0


def test_the_answer_must_name_everything_and_contradict_nothing():
    assert answer_lacks(WANT, {"answer": "o1 refund"}) == []
    assert answer_lacks(WANT, {"answer": "o1"}) == ["refund"]
    assert answer_lacks(WANT, {"answer": "o1 refund, or cancel"}) == ["also says cancel"]


def test_a_procedure_out_of_order_is_a_wrong_answer():
    want = dict(WANT, effects=[("a", ()), ("b", ())], ordered=True, answer=[""], excludes=[])
    row = score(
        "x/y",
        [want],
        [run(want["effects"], metered=100)],
        [run([("b", ()), ("a", ())], metered=10)],
        100,
        10,
        0,
    )
    assert row["wrong_answers"] == 1 and row["earned"] == 0.0
