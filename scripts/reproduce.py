"""Re-runs configurations the paper reports and compares them with the published result files.

    python scripts/reproduce.py --list
    python scripts/reproduce.py lexcache/threshold-0.8 agenticcache/k1
    python scripts/reproduce.py --system lexcache          every configuration of one system
    python scripts/reproduce.py --compare runs/lexcache/threshold-0.8.json

Each configuration runs in its own process with its settings in the environment, writes
runs/<name>.json, and is then compared with results/<name>.json pair by pair.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper import BY_NAME, RUNS  # noqa: E402

# Fields that must agree for a run to reproduce a published one.
COMPARED = (
    "gate",
    "wrong_writes",
    "wrong_reads",
    "wrong_answers",
    "missed_writes",
    "control_tokens",
    "system_tokens",
    "served_turns",
)

# Systems whose runs are not exactly repeatable: vCache draws its explore decisions at
# random, and GenCache's programs are written by a model.
VARIABLE = {"vcache", "gencache"}


def compare(new_path, published_path):
    """Rows of a new result file that differ from the published one."""
    new = {r["family"]: r for r in json.loads(Path(new_path).read_text())["rows"]}
    old = {r["family"]: r for r in json.loads(Path(published_path).read_text())["rows"]}
    differences = []
    for pair, row in sorted(new.items()):
        if pair not in old:
            continue
        for field in COMPARED:
            a, b = row[field], old[pair][field]
            same = abs(a - b) < 1e-6 if isinstance(a, float) or isinstance(b, float) else a == b
            if not same:
                differences.append(f"{pair} {field}: {a} here, {b} published")
    return differences


def reproduce(run, out_dir):
    out = out_dir / f"{run.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    environment = {k: v for k, v in os.environ.items() if not k.startswith("GROUNDHOG_")}
    environment.update(run.settings)
    if run.system == "gencache":
        environment["GROUNDHOG_GENCACHE_LOG"] = str(out.with_suffix(".synthesis.jsonl"))
    command = [
        sys.executable,
        "-m",
        "groundhog.run",
        "--system",
        run.system,
        "--label",
        run.name,
        "--out",
        str(out),
    ]
    print(f"\n=== {run.name}: {' '.join(f'{k}={v}' for k, v in run.settings.items())}", flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    differences = compare(out, ROOT / run.path)
    if not differences:
        print(f"  {run.name}: identical to the published result")
    elif run.system in VARIABLE:
        print(f"  {run.name}: {len(differences)} differences, expected for {run.system}")
    else:
        print(f"  {run.name}: {len(differences)} differences from the published result")
        for line in differences[:10]:
            print(f"    {line}")
    return not differences or run.system in VARIABLE


def main():
    parser = argparse.ArgumentParser(prog="python scripts/reproduce.py")
    parser.add_argument("names", nargs="*", help="configurations, as --list prints them")
    parser.add_argument("--list", action="store_true", help="print every configuration")
    parser.add_argument("--system", help="every configuration of one system")
    parser.add_argument(
        "--compare",
        metavar="FILE",
        help="compare a result file under runs/ with its published counterpart",
    )
    parser.add_argument("--out", default="runs", help="where new result files go")
    args = parser.parse_args()

    if args.list:
        for run in RUNS:
            tables = ", ".join(sorted(run.tables, key=int))
            print(f"{run.name:45} {run.system:13} table {tables}")
        return
    if args.compare:
        name = str(Path(args.compare).relative_to(args.out).with_suffix(""))
        differences = compare(args.compare, ROOT / BY_NAME[name].path)
        print("\n".join(differences) or "identical to the published result")
        sys.exit(1 if differences and BY_NAME[name].system not in VARIABLE else 0)

    chosen = [BY_NAME[n] for n in args.names] if args.names else []
    if args.system:
        chosen += [run for run in RUNS if run.system == args.system]
    if not chosen:
        parser.error("name a configuration, or --system, or --list")
    results = [reproduce(run, ROOT / args.out) for run in chosen]
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
