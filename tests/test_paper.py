"""The result files reproduce the paper's numbers."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import tables  # noqa: E402
from paper import RUNS, TABLE1  # noqa: E402

# Table 1 of the paper: gross, earned, failed pairs, wrong writes, missed writes, wrong reads,
# wrong answers, served turns.
TABLE_1 = {
    "LexCache": (9.3, 9.9, 6, 924, 853, 3417, 597, 3100),
    "GPTCache": (8.7, 0.8, 12, 1555, 1481, 5381, 2561, 2847),
    "vCache": (0.1, 0.1, 1, 1, 1, 3, 3, 38),
    "W5H2": (11.9, 10.2, 3, 150, 150, 522, 593, 3214),
    "AgentReuse": (70.7, 11.7, 11, 945, 979, 522, 1230, 7443),
    "AgenticCache": (-14.9, 0.0, 12, 339, 0, 696, 386, 802),
    "GenCache": (93.8, 22.2, 5, 257, 859, 412, 1053, 7127),
}


def test_every_result_file_exists_and_covers_all_pairs():
    for run in RUNS:
        assert len(tables.rows(run.name)) == 33, run.name


def test_table_1():
    for system, name in TABLE1.items():
        s = tables.summary(tables.rows(name))
        got = (
            round(s["gross"], 1),
            round(s["earned"], 1),
            s["failed"],
            s["wrong_writes"],
            s["missed_writes"],
            s["wrong_reads"],
            s["wrong_answers"],
            s["served"],
        )
        assert got == TABLE_1[system], system


def test_table_11_counts_the_program_writer():
    assert tables.table11()[-1].endswith("| -8.2% | 63.4% |")


def test_table_15_horizon_column():
    horizon = {
        line.split("|")[1].strip(): line.split("|")[-2].strip() for line in tables.table15()[3:]
    }
    assert (
        horizon["W5H2"] == "1.2" and horizon["GenCache"] == "1.3" and horizon["AgentReuse"] == "1.6"
    )


def test_the_published_tables_are_current():
    rendered = "".join(
        f"### Table {n}: {title}\n\n" + "\n".join(make()) + "\n\n"
        for n, (title, make) in tables.TABLES.items()
    )
    assert (ROOT / "results" / "TABLES.md").read_text() == rendered
