"""Recomputes the paper's numeric tables from the result files in results/.

python scripts/tables.py            all tables, as Markdown
python scripts/tables.py 1 9        only Tables 1 and 9
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paper import BY_NAME, TABLE1  # noqa: E402

from groundhog.conditions import CONDITIONS, FAMILIES, condition_of  # noqa: E402
from groundhog.world import DOMAINS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def rows(name):
    """A result file's rows for the 33 benchmark family-domain pairs."""
    data = json.loads((ROOT / BY_NAME[name].path).read_text())
    return [r for r in data["rows"] if r["family"].split("/")[1] in FAMILIES]


def pooled(rs, field="earned"):
    """A saving pooled over pairs, weighted by each pair's cost without a cache, in percent."""
    control = sum(r["control_tokens"] for r in rs)
    return 100 * sum(r[field] * r["control_tokens"] for r in rs) / control if control else 0.0


def family(rs, name):
    return [r for r in rs if r["family"].split("/")[1] == name]


def domain(rs, name):
    return [r for r in rs if r["family"].split("/")[0] == name]


def summary(rs):
    return {
        "failed": sum(r["gate"] == "FAIL" for r in rs),
        "pairs": len(rs),
        "wrong_writes": sum(r["wrong_writes"] for r in rs),
        "missed_writes": sum(r["missed_writes"] for r in rs),
        "wrong_reads": sum(r["wrong_reads"] for r in rs),
        "wrong_answers": sum(r["wrong_answers"] for r in rs),
        "served": sum(r["served_turns"] for r in rs),
        "earned": pooled(rs),
        "gross": pooled(rs, "saved"),
    }


def program_writer_tokens(log):
    """The program writer's tokens, priced by the meter's rule: cached input at a tenth."""
    calls = [json.loads(line) for line in (ROOT / log).read_text().splitlines() if line.strip()]
    return sum(c["input_tokens"] + c["output_tokens"] + c["cache_read"] / 10 for c in calls)


def turns_per_run(control_rows):
    """A family's turns per run without a cache, averaged over its domains."""
    return sum(r["steps_per_run"] for r in control_rows) / len(control_rows)


def pct(x):
    return f"{x:.1f}%"


def table1():
    out = [
        "| system | gross | earned | failed pairs | wrong writes | missed writes | wrong reads "
        "| wrong answers | served |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for system, name in TABLE1.items():
        s = summary(rows(name))
        out.append(
            f"| {system} | {pct(s['gross'])} | {pct(s['earned'])} | {s['failed']}/{s['pairs']} | "
            f"{s['wrong_writes']:,} | {s['missed_writes']:,} | {s['wrong_reads']:,} | "
            f"{s['wrong_answers']:,} | {s['served']:,} |"
        )
    return out


def table3():
    control = rows("control")
    total = sum(r["control_tokens"] for r in control)
    out = ["| family | condition | turns per run | share of cost |", "|---|---|---:|---:|"]
    for name in FAMILIES:
        rs = family(control, name)
        share = 100 * sum(r["control_tokens"] for r in rs) / total
        out.append(f"| {name} | {condition_of(name)} | {turns_per_run(rs):.1f} | {pct(share)} |")
    return out


def sweep(title, names):
    out = [
        f"| {title} | failed pairs | wrong writes | missed writes | wrong reads | earned | gross |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, name in names:
        s = summary(rows(name))
        out.append(
            f"| {label} | {s['failed']}/{s['pairs']} | {s['wrong_writes']:,} | "
            f"{s['missed_writes']:,} | {s['wrong_reads']:,} | {pct(s['earned'])} | {pct(s['gross'])} |"
        )
    return out


def table11():
    name = "gencache/gpt-5-mini"
    rs = rows(name)
    s = summary(rs)
    control = sum(r["control_tokens"] for r in rs)
    extra = 100 * program_writer_tokens("results/gencache/gpt-5-mini.synthesis.jsonl") / control
    return [
        "| setting | failed pairs | wrong writes | missed writes | wrong reads | earned | gross |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| program writing not counted | {s['failed']}/33 | {s['wrong_writes']} | {s['missed_writes']} | "
        f"{s['wrong_reads']} | {pct(s['earned'])} | {pct(s['gross'])} |",
        f"| program writing counted | {s['failed']}/33 | {s['wrong_writes']} | {s['missed_writes']} | "
        f"{s['wrong_reads']} | {pct(s['earned'] - extra)} | {pct(s['gross'] - extra)} |",
    ]


def table12():
    out = ["| family | " + " | ".join(TABLE1) + " |", "|---|" + "---:|" * len(TABLE1)]
    for name in FAMILIES:
        cells = []
        for run in TABLE1.values():
            rs = family(rows(run), name)
            failed = sum(r["gate"] == "FAIL" for r in rs)
            mark = f" ({failed})" if failed else ""
            cells.append(f"{pooled(rs):.0f}{mark}")
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    out.append("")
    out.append("A number in parentheses is the count of domains with a wrong write on that family.")
    return out


def table13():
    out = ["| system | " + " | ".join(CONDITIONS) + " |", "|---|" + "---:|" * len(CONDITIONS)]
    for system, run in TABLE1.items():
        cells = []
        for members in CONDITIONS.values():
            rs = [r for r in rows(run) if r["family"].split("/")[1] in members]
            failed = sum(r["gate"] == "FAIL" for r in rs)
            cells.append(f"{pooled(rs):.0f}%" + (f" ({failed})" if failed else ""))
        out.append(f"| {system} | " + " | ".join(cells) + " |")
    return out


def table14():
    out = ["| system | " + " | ".join(DOMAINS) + " |", "|---|" + "---:|" * len(DOMAINS)]
    for system, run in TABLE1.items():
        out.append(
            f"| {system} | " + " | ".join(pct(pooled(domain(rows(run), d))) for d in DOMAINS) + " |"
        )
    return out


def table15():
    """Pairs binned by their family's turns per run without a cache, as in Table 3."""
    control = rows("control")
    bins = {}
    for name in FAMILIES:
        turns = round(turns_per_run(family(control, name)), 1)
        bins.setdefault(turns, set()).update(r["family"] for r in family(control, name))
    edges = sorted(bins)
    out = [
        "| turns per run | " + " | ".join(f"{e:.1f}" for e in edges) + " |",
        "|---|" + "---:|" * len(edges),
        "| family-domain pairs | " + " | ".join(str(len(bins[e])) for e in edges) + " |",
    ]
    for system, run in TABLE1.items():
        rs = rows(run)
        cells = [f"{pooled([r for r in rs if r['family'] in bins[e]]):.1f}" for e in edges]
        out.append(f"| {system} | " + " | ".join(cells) + " |")
    return out


TABLES = {
    "1": ("Seven systems at the settings of Table 1", table1),
    "3": ("The 11 families", table3),
    "5": (
        "LexCache by threshold",
        lambda: sweep("threshold", [(v, f"lexcache/threshold-{v}") for v in ("0.7", "0.8", "0.9")]),
    ),
    "6": (
        "GPTCache by similarity threshold",
        lambda: sweep(
            "threshold", [(v, f"gptcache/threshold-{v}") for v in ("0.80", "0.85", "0.90")]
        ),
    ),
    "7": (
        "vCache by error bound",
        lambda: sweep(
            "delta", [(v, f"vcache/delta-{v}") for v in ("0.005", "0.01", "0.02", "0.05")]
        ),
    ),
    "8": (
        "W5H2 by risk tolerance",
        lambda: sweep("alpha", [(v, f"w5h2/alpha-{v}") for v in ("0.05", "0.10", "0.20", "0.30")]),
    ),
    "9": (
        "AgentReuse by embedder and gamma, and four variants",
        lambda: sweep(
            "setting",
            [
                *[
                    (f"m3e-small, gamma {g}", f"agentreuse/m3e-small-gamma-{g}")
                    for g in ("0.75", "0.80", "0.85", "0.90", "0.95")
                ],
                *[
                    (f"MiniLM, gamma {g}", f"agentreuse/minilm-gamma-{g}")
                    for g in ("0.75", "0.80", "0.85", "0.90", "0.95")
                ],
                ("opening turn only", "agentreuse/opening-turn-only"),
                ("parameters left in the embedding", "agentreuse/parameters-left-in"),
                ("matched against all stored conversations", "agentreuse/matched-against-all"),
                (
                    "refusals labelled as the action they mention",
                    "agentreuse/refusals-labelled-by-action",
                ),
            ],
        ),
    ),
    "10": (
        "AgenticCache by K, with result fields, and two ablation arms",
        lambda: sweep(
            "setting",
            [
                *[(f"K = {k}", f"agenticcache/k{k}") for k in ("1", "2", "3", "5", "10")],
                *[
                    (f"K = {k}, result fields as metadata", f"agenticcache/fields-k{k}")
                    for k in ("1", "2", "3", "5", "10")
                ],
                ("K = 1, checks update counts only", "agenticcache/updates-only"),
                ("K = 1, checks replace only", "agenticcache/replacement-only"),
            ],
        ),
    ),
    "11": ("GenCache with and without the program writer's tokens", table11),
    "12": ("Earned savings per family, three domains pooled, in percent", table12),
    "13": ("Earned savings per condition", table13),
    "14": ("Earned savings per domain", table14),
    "15": ("Earned savings by turns per run without a cache", table15),
}


def main(wanted):
    for number in wanted or TABLES:
        title, make = TABLES[number]
        print(f"### Table {number}: {title}\n")
        print("\n".join(make()))
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
