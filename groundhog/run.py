"""Runs a system on the benchmark and writes a result file.

For each family-domain pair the harness makes two passes over the same instances: one
straight to the model, which is the baseline and the benchmark's self-check, and one
through the system under test, started fresh for that pair.

    python -m groundhog.run --system lexcache --out runs/lexcache.json
"""

import argparse
import ast
import collections
import hashlib
import importlib.util
import json
import os
import platform
import re
import secrets
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import data, spec
from .agent import run as drive
from .conditions import FAMILIES
from .model import Model
from .score import score, table, verify_control
from .systems import SYSTEMS
from .world import DOMAINS

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent
MODELS = ROOT / "models"


def as_effect(pair):
    name, args = pair
    return (name, tuple(tuple(a) for a in args))


def load_control(path):
    """A baseline pass saved by an earlier run, so several systems share one."""
    if not path or not os.path.exists(path):
        return {}
    saved = json.load(open(path))
    for family in saved.values():
        for run in family["runs"]:
            run["effects"] = [as_effect(e) for e in run["effects"]]
    return saved


def control_digest(saved):
    if not saved:
        return None
    body = json.dumps(
        {k: [r["effects"] for r in v["runs"]] for k, v in sorted(saved.items())},
        default=list,
        sort_keys=True,
    )
    return hashlib.sha256(body.encode()).hexdigest()[:16]


def serve(server):
    threading.Thread(target=server.serve_forever, daemon=True).start()


def resolve(system):
    """`py:module:Class` for a named system, the name itself otherwise."""
    return SYSTEMS.get(system) or system


def starter(system, model_port, pinned=None):
    """Starts the system under test afresh for each family-domain pair, so nothing one
    pair taught it carries into the next."""
    upstream = f"http://127.0.0.1:{model_port}"
    if system == "control":
        return lambda label: (upstream, lambda: None)
    if system.startswith("cmd:") and "CELL" not in system:
        print(
            "  note: the command names no CELL, so any files it keeps "
            "are shared across family-domain pairs"
        )

    def start(label):
        port = pinned or free_port()
        target = f"http://127.0.0.1:{port}"
        if system.startswith("py:"):
            module_name, _, class_name = system[3:].rpartition(":")
            proxy = getattr(importlib.import_module(module_name), class_name)(port, upstream)
            serve(proxy)

            def stop():
                proxy.shutdown()
                proxy.server_close()

            return target, stop
        if system.startswith("cmd:"):
            # MODEL_PORT, SYSTEM_PORT and CELL in the command are replaced in one pass.
            swaps = {
                "MODEL_PORT": str(model_port),
                "SYSTEM_PORT": str(port),
                "CELL": label.replace("/", "-"),
            }
            pattern = re.compile("|".join(rf"\b{re.escape(k)}\b" for k in swaps))
            command = pattern.sub(lambda m: swaps[m.group(0)], system[4:])
            # A file this pair's command would write that already exists is a warm start
            # nobody asked for, so the run refuses rather than overwrite it.
            words = shlex.split(command)
            writes = {
                words[i + 1] for i, word in enumerate(words[:-1]) if word in ("--out", "--trust")
            }
            cell = swaps["CELL"]
            existing = [word for word in writes if cell in word and os.path.isfile(word)]
            if existing:
                raise SystemExit(
                    f"{label}: the command would write over {', '.join(sorted(set(existing)))}, "
                    "left by an earlier run. Move them aside or name a fresh directory."
                )
            child = subprocess.Popen(words, env=child_environment(os.environ))
            if not wait_for(port):
                child.kill()
                raise SystemExit(f"nothing listening on {port}")
            if child.poll() is not None:
                raise SystemExit(
                    f"{label}: the command exited ({child.returncode}) and "
                    f"something else is listening on {port}"
                )

            def stop():
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()

            return target, stop
        raise SystemExit(f"unknown system {system!r}")

    return start


def free_port():
    """A free port below the range the kernel hands to outgoing connections."""
    for _ in range(1000):
        port = 20000 + secrets.randbelow(12000)
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise SystemExit("no free port between 20000 and 32000")


def wait_for(port, seconds=20):
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            return True
        except OSError:
            time.sleep(0.1)
    return False


def evaluate(
    start,
    model,
    families,
    pace,
    wire="anthropic",
    model_name="stub",
    domains=DOMAINS,
    limit=None,
    control_in=None,
    control_out=None,
):
    """Score every family-domain pair. Cost is read from the model's meter, never from
    what the system reports."""
    closed_at = [0.0]
    rows = []
    upstream = f"http://127.0.0.1:{model.server_address[1]}"
    for domain, family in [(d, f) for d in domains for f in families]:
        model.domain = domain
        batch = data.instances(domain, family)[:limit]
        label = f"{domain}/{family}"
        shared = (control_in or {}).get(label)
        if shared:
            control = shared["runs"]
            problems = shared["problems"]
        else:
            quiesced(model)
            leaked = model.spent - closed_at[0] if closed_at[0] else 0.0
            if leaked > 0:
                print(
                    f"  [note] {leaked:.0f} tokens arrived after the previous pair closed;"
                    " a system outlived the harness stopping it"
                )
            # The baseline pays for its own prompt-cache writes, as each configuration does.
            model.new_phase()
            model.baseline = secrets.token_hex(8)
            strays_before = model.stray_calls
            stray_before = model.stray
            control_mark = model.spent
            control = []
            for inst in batch:
                model.era = inst.get("era", 0)
                before = model.spent
                outcome = drive(
                    upstream,
                    inst["opening"],
                    era=model.era,
                    wire=wire,
                    model=model_name,
                    domain=domain,
                    max_turns=inst.get("max_turns", 6),
                    mark=model.baseline,
                )
                outcome["metered"] = model.spent - before
                control.append(outcome)
                time.sleep(pace)
            problems = verify_control(label, batch, control)
            if control_out is not None:
                control_out[label] = {"runs": control, "problems": problems}
            quiesced(model)
            attributed(control, model.spent - control_mark)
            strayed = model.stray - stray_before
            if strayed > 0:
                print(
                    f"  [note] {model.stray_calls - strays_before} call(s) and "
                    f"{strayed:.0f} token(s) arrived during this baseline from something "
                    f"other than the harness, and are no part of it"
                )
            model.baseline = None
        control_cost = sum(r["metered"] for r in control)
        model.era = 0
        if problems:
            raise SystemExit(
                "benchmark self-check failed; nothing was scored:\n  " + "\n  ".join(problems[:5])
            )
        # Marked before the system starts, so whatever it spends starting up is its own.
        model.new_phase()
        calls_before = model.calls
        asked_mark = len(model.asked)
        system_mark = model.spent
        target, stop = start(label)
        system = []
        for inst in batch:
            model.era = inst.get("era", 0)
            before = model.spent
            outcome = drive(
                target,
                inst["opening"],
                era=model.era,
                wire=wire,
                model=model_name,
                domain=domain,
                max_turns=inst.get("max_turns", 6),
            )
            outcome["metered"] = model.spent - before
            system.append(outcome)
            time.sleep(pace)
        quiesced(model)
        unfree(system, model.asked[asked_mark:])
        attributed(system, model.spent - system_mark)
        system_cost = sum(r["metered"] for r in system)
        closed_at[0] = model.spent
        # Served turns from the meter: requests the agent made that the model never saw.
        served_metered = sum(r["requests"] for r in system) - (model.calls - calls_before)
        model.era = 0
        rows.append(score(label, batch, control, system, control_cost, system_cost, served_metered))
        stop()
    return rows


def unfree(runs, received):
    """A turn the agent got at zero cost is free only if the model was not asked that turn
    anyway. Each copy the model received is matched to a turn by digest and time; a served
    turn the model answered regardless counts as checked, not free."""
    sends = collections.defaultdict(list)
    for r, run in enumerate(runs):
        run["checked"] = 0
        for i, (turn, at) in enumerate(zip(run["asked"], run["sent"])):
            sends[turn].append((at, r, i))
    copies = collections.defaultdict(list)
    for turn, at in received:
        copies[turn].append(at)
    for turn, times in copies.items():
        times.sort()
        mine = sorted(sends.get(turn, ()))
        taken = [False] * len(times)
        for at, r, i in mine:
            if runs[r]["free"][i]:
                continue
            k = next((k for k, t in enumerate(times) if not taken[k] and t >= at), None)
            if k is not None:
                taken[k] = True
        for k, t in enumerate(times):
            if taken[k]:
                continue
            free = [(at, r, i) for at, r, i in mine if runs[r]["free"][i]]
            before = [f for f in free if f[0] <= t]
            pick = before[-1] if before else (free[0] if free else None)
            if pick is not None:
                runs[pick[1]]["free"][pick[2]] = False
                runs[pick[1]]["checked"] += 1
    for run in runs:
        run["served"] = sum(run["free"])


def quiesced(model, settle=0.75, limit=20.0):
    """Wait until the meter has been still for a moment, so traffic a system sends on its
    own behalf between runs is charged to it."""
    deadline = time.monotonic() + limit
    last = model.spent
    while time.monotonic() < deadline:
        time.sleep(settle)
        if model.spent == last:
            return
        last = model.spent


def attributed(runs, phase_total):
    """Spread what the meter charged between runs over the runs, in proportion to their cost."""
    windows = sum(r["metered"] for r in runs)
    residual = phase_total - windows
    if not runs or abs(residual) < 1e-9:
        return
    if windows <= 0:
        share = residual / len(runs)
        for run in runs:
            run["metered"] += share
        return
    for run in runs:
        run["metered"] += residual * (run["metered"] / windows)


def module_closure(name):
    """The source files of a groundhog module and every groundhog module it imports."""
    found = {}
    todo = [name]
    while todo:
        module = todo.pop()
        if module in found or not module.startswith("groundhog"):
            continue
        try:
            module_spec = importlib.util.find_spec(module)
        except (ImportError, ValueError):
            continue
        if not module_spec or not (module_spec.origin or "").endswith(".py"):
            continue
        found[module] = module_spec.origin
        package = module if module_spec.submodule_search_locations else module.rpartition(".")[0]
        tree = ast.parse(Path(module_spec.origin).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                base = (
                    importlib.util.resolve_name("." * node.level + (node.module or ""), package)
                    if node.level
                    else node.module
                )
                todo.append(base)
                todo.extend(f"{base}.{alias.name}" for alias in node.names)
            elif isinstance(node, ast.Import):
                todo.extend(alias.name for alias in node.names)
    covered = {str(PACKAGE / name) for name in spec.DEFINES}
    return sorted(path for path in found.values() if path not in covered)


def system_sources(system):
    """The files a system's behaviour comes from, outside what the spec digest covers."""
    if system == "control":
        return []
    if system.startswith("py:"):
        return module_closure(system[3:].rpartition(":")[0])
    words = shlex.split(system[4:])
    while words and (words[0] == "env" or re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", words[0])):
        words = words[1:]
    if not words:
        return []
    if os.path.basename(words[0]).startswith("python") and len(words) > 2 and words[1] == "-m":
        return module_closure(words[2])
    program = words[0] if os.path.sep in words[0] else shutil.which(words[0])
    return [program] if program and os.path.isfile(program) else []


def system_weights(sources):
    """Trained files under models/ that a system module declares in WEIGHTS."""
    out = []
    for source in sources:
        if not source.endswith(".py") or not source.startswith(str(PACKAGE)):
            continue
        tree = ast.parse(Path(source).read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "WEIGHTS" for t in node.targets
            ):
                for name in ast.literal_eval(node.value):
                    path = MODELS / name
                    if path.is_dir():
                        out += sorted(str(p) for p in path.rglob("*") if p.is_file())
                    elif path.is_file():
                        out.append(str(path))
    return sorted(out)


def system_digest(system):
    """A digest of a system's own code and trained weights."""
    parts = hashlib.sha256()
    sources = system_sources(system)
    for path in sources + system_weights(sources):
        inside = os.path.abspath(path).startswith(str(ROOT) + os.sep)
        parts.update((os.path.relpath(path, ROOT) if inside else os.path.basename(path)).encode())
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                parts.update(block)
    return parts.hexdigest()[:16]


# Thread counts change float reductions in a model's forward pass, so they are recorded.
PROCESS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONHASHSEED")


def libraries_loaded():
    """Every installed package this process imported, with its version."""
    out = {"python": platform.python_version()}
    for name, module in sorted(sys.modules.items()):
        if "." in name or name.startswith("_"):
            continue
        where = getattr(module, "__file__", None) or ""
        version = getattr(module, "__version__", None)
        if isinstance(version, str) and ("site-packages" in where or "dist-packages" in where):
            out[name] = version
    return out


SETTINGS = ("GROUNDHOG_",)
SECRETS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
PLUMBING = ("PATH", "HOME", "TMPDIR", "USER", "SHELL", "TERM", "LANG", "PWD")


def child_environment(environ):
    """What a cmd: system starts with: shell plumbing, GROUNDHOG_ settings and credentials."""
    return {
        name: value
        for name, value in environ.items()
        if name in PLUMBING
        or name.startswith("LC_")
        or name.startswith(SETTINGS)
        or any(mark in name.upper() for mark in SECRETS)
    }


def settings_of(environ):
    """The GROUNDHOG_ settings a run used. A name that looks like a credential is recorded
    with its value withheld."""
    return {
        name: "(withheld: the name looks like a credential)"
        if any(mark in name.upper() for mark in SECRETS)
        else value
        for name, value in sorted(environ.items())
        if name.startswith(SETTINGS)
    }


def build():
    parser = argparse.ArgumentParser(prog="python -m groundhog.run")
    parser.add_argument(
        "--system",
        default="control",
        help=f"one of {', '.join(['control', *SYSTEMS])}; py:<module>:<Class>; "
        "or cmd:<shell command> for a proxy in another process",
    )
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--domains", default=",".join(DOMAINS))
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="the first N instances of each family-domain pair, for a quick check; "
        "drift, quiet and standing change at instance 50",
    )
    parser.add_argument("--wire", default="anthropic", choices=["anthropic", "openai"])
    parser.add_argument(
        "--pace", type=float, default=0.05, help="seconds between instances, recorded in the result"
    )
    parser.add_argument(
        "--control",
        default=None,
        help="a baseline file to share across systems: written if absent, reused if present",
    )
    parser.add_argument("--ports", default="auto", help="model,system ports; auto picks free ones")
    parser.add_argument("--label", default=None, help="a short name for the result")
    parser.add_argument("--out", default=None, help="where to write the result file")
    return parser


def main():
    args = build().parse_args()
    system = resolve(args.system)
    if system != "control" and not system.startswith(("py:", "cmd:")):
        raise SystemExit(f"unknown system {args.system!r}")
    if args.out and os.path.exists(args.out):
        print(f"  note: {args.out} exists and is replaced only if this run finishes")
    if args.ports == "auto":
        model_port, system_port = free_port(), free_port()
    else:
        model_port, system_port = (int(p) for p in args.ports.split(","))
    model = Model(model_port)
    serve(model)
    start = starter(system, model_port, system_port if args.ports != "auto" else None)

    reuse = load_control(args.control)
    fresh = {} if (args.control and not reuse) else None
    stamped = spec.stamp()
    system_before = system_digest(system)
    rows = evaluate(
        start,
        model,
        args.families.split(","),
        args.pace,
        wire=args.wire,
        domains=tuple(args.domains.split(",")),
        limit=args.limit,
        control_in=reuse,
        control_out=fresh,
    )
    if fresh:
        with open(args.control, "w") as f:
            json.dump(fresh, f, default=list)
        reuse = load_control(args.control)
    if system_digest(system) != system_before:
        raise SystemExit("the system's code changed during the run; nothing is written")
    if spec.stamp() != stamped:
        raise SystemExit("the benchmark changed during the run; nothing is written")
    meta = {
        **stamped,
        "dataset": data.digest(),
        "control_digest": control_digest(reuse),
        "label": args.label or args.system,
        "system": system,
        "model": "stub",
        "wire": args.wire,
        "pace_seconds": args.pace,
        "instances_per_pair": args.limit or 100,
        "domains": args.domains,
        "meter": {"unit": "tokens"},
        "environment": settings_of(os.environ),
        "system_digest": system_before,
        "process": {name: os.environ[name] for name in PROCESS if name in os.environ},
        "libraries": libraries_loaded(),
    }
    print(f"\n  system: {system}  wire: {args.wire}  pace: {args.pace}s\n")
    print(table(rows))
    gated = [r["family"] for r in rows if r["gate"] == "FAIL"]
    if gated:
        print(f"\n  wrong writes on: {', '.join(gated)}. Savings there do not count.")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        beside = f"{args.out}.partial"
        with open(beside, "w") as f:
            json.dump({"meta": meta, "rows": rows}, f, indent=1)
        os.replace(beside, args.out)


if __name__ == "__main__":
    main()
