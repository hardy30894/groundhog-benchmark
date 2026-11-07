"""AgenticCache, a plan cache, written from its paper; the released cache keys on its own
simulator's state and cannot run outside it.

It caches transitions, pairs of consecutive plan steps, where a plan step is one assistant
turn that calls a tool. Each transition records the circumstances it was seen in as ranges.
When the agent has just executed a step, transitions whose ranges do not hold now are
discarded, the rest are scored by how often each was observed times the importance of the
step it leads to, and the best is served. After each served step the model is asked what it
would have done, and its answer is read K steps later: a match raises the transition's
counts; a mismatch lowers them, stores the model's transition and, if the model's step can
still be executed, replaces the running plan from there.

Settings, all optional:
    GROUNDHOG_AGENTICCACHE_K          steps until a check comes back; 1 (our agent has no latency)
    GROUNDHOG_AGENTICCACHE_JITTER     1 to draw each check's delay from 1 to 2K-1 steps; 1
    GROUNDHOG_AGENTICCACHE_ARM        full, updates (checks only update counts), replacement
                                      (checks only replace the step), none, or static
    GROUNDHOG_AGENTICCACHE_METADATA   counters, or fields to admit result fields to the state
    GROUNDHOG_AGENTICCACHE_BINDER     provenance or type: how a step's arguments are bound
    GROUNDHOG_AGENTICCACHE_WALK       ranked or argmax: how a candidate step is chosen
    GROUNDHOG_AGENTICCACHE_SEED       seed for the check delays and for breaking ties; v0
    GROUNDHOG_AGENTICCACHE_LOG        a file to log every decision to
"""

import json
import os
import random
import re
import threading
from fractions import Fraction
from functools import lru_cache

from ..model import emit_anthropic, emit_openai, text_of
from ..score import EDGE
from ..server import LocalServer
from .proxy import ProxyHandler, forward, opening_of, refuse_unknown_knobs, zero_usage

START = ("start",)
END = ("end",)
UNBOUND = ("unbound",)

ARMS = ("full", "updates", "replacement", "none", "static")
METADATA = ("counters", "fields")
BINDERS = ("provenance", "type")
WALKS = ("ranked", "argmax")


def settings(environ):
    refuse_unknown_knobs(
        environ,
        "GROUNDHOG_AGENTICCACHE_",
        ("K", "JITTER", "ARM", "METADATA", "BINDER", "WALK", "SEED", "LOG"),
    )
    raw = environ.get("GROUNDHOG_AGENTICCACHE_K", "1")
    k = int(raw) if raw.isdigit() else 0
    if k < 1:
        raise SystemExit(
            f"GROUNDHOG_AGENTICCACHE_K is a number of plan steps, at least 1; got {raw!r}. "
            "No checks at all is GROUNDHOG_AGENTICCACHE_ARM=none."
        )
    chosen = {
        "jitter": environ.get("GROUNDHOG_AGENTICCACHE_JITTER", "1"),
        "arm": environ.get("GROUNDHOG_AGENTICCACHE_ARM", "full"),
        "metadata": environ.get("GROUNDHOG_AGENTICCACHE_METADATA", "counters"),
        "binder": environ.get("GROUNDHOG_AGENTICCACHE_BINDER", "provenance"),
        "walk": environ.get("GROUNDHOG_AGENTICCACHE_WALK", "ranked"),
    }
    allowed = {
        "jitter": ("0", "1"),
        "arm": ARMS,
        "metadata": METADATA,
        "binder": BINDERS,
        "walk": WALKS,
    }
    for name, value in chosen.items():
        if value not in allowed[name]:
            raise SystemExit(
                f"GROUNDHOG_AGENTICCACHE_{name.upper()} is one of {', '.join(allowed[name])}; "
                f"got {value!r}"
            )
    return {
        "k": k,
        "jitter": chosen["jitter"] == "1",
        "arm": chosen["arm"],
        "metadata": chosen["metadata"],
        "binder": chosen["binder"],
        "walk": chosen["walk"],
        "seed": environ.get("GROUNDHOG_AGENTICCACHE_SEED", "v0"),
        "log": environ.get("GROUNDHOG_AGENTICCACHE_LOG", ""),
    }


class View:
    def __init__(self, openai, opening, turns, result, tools, tokens=None):
        self.openai = openai
        self.opening = opening
        self.tokens = (
            tokens
            if tokens is not None
            else [word.strip(EDGE) for word in opening.split() if word.strip(EDGE)]
        )
        self.turns = turns
        self.t = len(turns)
        self.result = result
        self.tools = tools

    def at(self, i):
        return View(
            self.openai, self.opening, self.turns[:i], self.turns[i][1], self.tools, self.tokens
        )

    def required(self, tool):
        declared = self.tools.get(tool)
        return None if declared is None else declared[0]

    def kind(self, tool, arg):
        declared = self.tools.get(tool)
        return declared[2].get(arg) if declared and len(declared) > 2 else None

    def reads(self, tool):
        declared = self.tools.get(tool)
        return bool(declared and declared[1])


def parsed(text):
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def calls_in(message, openai):
    if openai:
        return [
            (c["function"]["name"], json.loads(c["function"].get("arguments") or "{}"))
            for c in message.get("tool_calls") or []
        ]
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [
        (b["name"], b.get("input") or {})
        for b in content
        if isinstance(b, dict) and b.get("type") == "tool_use"
    ]


def read(body, path):
    openai = path.startswith("/v1/chat/completions")
    tools = {}
    for tool in body.get("tools") or []:
        declared = tool["function"] if isinstance(tool.get("function"), dict) else tool
        schema = declared.get("parameters") or declared.get("input_schema") or {}
        hint = (declared.get("annotations") or {}).get("readOnlyHint")
        kinds = {a: (p or {}).get("type") for a, p in (schema.get("properties") or {}).items()}
        tools[declared["name"]] = (tuple(schema.get("required") or ()), hint is True, kinds)
    turns, result = [], None
    for message in body.get("messages") or []:
        role = message.get("role")
        if role == "assistant":
            turns.append((calls_in(message, openai), result))
        elif role == "tool":
            result = parsed(message.get("content"))
        elif role == "user" and isinstance(message.get("content"), list):
            blocks = [
                b
                for b in message["content"]
                if isinstance(b, dict) and b.get("type") == "tool_result"
            ]
            if blocks:
                result = parsed(text_of(blocks[-1].get("content", "")))
    return View(openai, opening_of(body), turns, result, tools)


def calls_of(out):
    if "choices" in out:
        return calls_in(out["choices"][0].get("message") or {}, True)
    return calls_in(out, False)


@lru_cache(maxsize=None)
def compiled(shape):
    return re.compile(shape)


def shape(token):
    return "".join(
        r"\d+" if part.isdigit() else re.escape(part) for part in re.split(r"(\d+)", token) if part
    )


def source_of(value, view):
    if not isinstance(value, str):
        return UNBOUND
    if view.result is not None:
        for field, seen in view.result.items():
            if isinstance(seen, str) and seen == value:
                return ("result", field)
    if value in view.tokens:
        return ("request", shape(value))
    return UNBOUND


def resolve(source, view):
    if source[0] == "result":
        value = (view.result or {}).get(source[1])
        return value if isinstance(value, str) else None
    if source[0] == "request":
        pattern = compiled(source[1])
        return next((token for token in view.tokens if pattern.fullmatch(token)), None)
    return None


def fits(value, kind):
    number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "string":
        return isinstance(value, str)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return number
    if kind == "boolean":
        return isinstance(value, bool)
    if kind is not None:
        return False
    return isinstance(value, str) or (isinstance(value, int) and not isinstance(value, bool))


def pick(view, kind=None):
    if view.result is not None:
        values = [v for v in view.result.values() if fits(v, kind)]
        if values:
            return values[-1]
    if kind not in (None, "string"):
        return None
    return next((token for token in view.tokens if any(c.isdigit() for c in token)), None)


def identity(call, view, binder):
    name, args = call
    if binder == "type":
        return ("call", name, ())
    required = view.required(name)
    if required is None:
        required = tuple(args)
    return ("call", name, tuple((arg, source_of(args.get(arg), view)) for arg in sorted(required)))


def bind(plan, view, binder):
    required = view.required(plan[1])
    if required is None:
        return None
    if binder == "type":
        args = {arg: pick(view, view.kind(plan[1], arg)) for arg in required}
        return None if any(v is None for v in args.values()) else args
    sources = dict(plan[2])
    args = {}
    for arg in required:
        value = resolve(sources.get(arg, UNBOUND), view)
        if value is None:
            return None
        args[arg] = value
    return args


def same(call, wanted, required):
    return call[0] == wanted[0] and all(call[1].get(a) == wanted[1].get(a) for a in required)


def metadata(view, mode):
    made = [call for calls, _ in view.turns for call in calls]
    meta = {
        "step": view.t,
        "calls": len(made),
        "writes": sum(1 for name, _ in made if not view.reads(name)),
        "rejected": int(view.t > 0 and view.result is None),
    }
    if mode == "fields" and view.result is not None:
        for field, value in view.result.items():
            if isinstance(value, (bool, int)):
                meta["last." + field] = int(value)
    return meta


def named(plan):
    if plan == START:
        return "START"
    if plan == END:
        return "END"
    if plan[0] == "multi":
        return " + ".join(named(p) for p in plan[1])
    _, tool, args = plan
    if not args:
        return tool
    return tool + "(" + ", ".join(f"{arg}<-{':'.join(source)}" for arg, source in args) + ")"


def frozen(value):
    return tuple(frozen(v) for v in value) if isinstance(value, list) else value


class Entry:
    __slots__ = ("count", "ranges")

    def __init__(self, count=0, ranges=None):
        self.count = count
        self.ranges = ranges or {}

    def admits(self, meta):
        return all(lo <= meta[f] <= hi for f, (lo, hi) in self.ranges.items() if f in meta)

    def widen(self, meta):
        for field, value in meta.items():
            lo, hi = self.ranges.get(field, (value, value))
            self.ranges[field] = (min(lo, value), max(hi, value))


class Table:
    def __init__(self):
        self.transitions = {}
        self.importance = {}
        self.changed = False

    def feasible(self, prev, meta):
        return [
            plan
            for plan, entry in self.transitions.get(prev, {}).items()
            if entry.count > 0 and entry.admits(meta)
        ]

    def appeared(self, plans):
        for plan in plans:
            self.importance.setdefault(plan, [0, 0])[1] += 1
            self.changed = True

    def score(self, prev, plan):
        confirmed, candidate = self.importance.get(plan, (0, 0))
        if not candidate:
            return Fraction(0)
        return Fraction(self.transitions[prev][plan].count * confirmed, candidate)

    def rank(self, prev, plans, rng):
        scored = sorted(plans, key=lambda p: -self.score(prev, p))
        ranked, i = [], 0
        while i < len(scored):
            j = i
            while j < len(scored) and self.score(prev, scored[j]) == self.score(prev, scored[i]):
                j += 1
            group = scored[i:j]
            if len(group) > 1:
                rng.shuffle(group)
            ranked += group
            i = j
        return ranked

    def credit(self, prev, plan, meta):
        entry = self.transitions.setdefault(prev, {}).setdefault(plan, Entry())
        entry.count += 1
        entry.widen(meta)
        self.importance.setdefault(plan, [0, 0])[0] += 1
        self.changed = True

    def confirm(self, prev, plan, meta):
        self.credit(prev, plan, meta)
        counts = self.importance[plan]
        counts[1] = max(counts[1], counts[0])

    def learn(self, prev, plan, meta, feasible):
        self.credit(prev, plan, meta)
        if plan not in feasible:
            self.importance[plan][1] += 1

    def debit(self, prev, plan):
        entry = self.transitions.get(prev, {}).get(plan)
        if entry is not None:
            entry.count = max(0, entry.count - 1)
        counts = self.importance.get(plan)
        if counts is not None:
            counts[0] = max(0, counts[0] - 1)
        self.changed = True

    def correct(self, prev, wrong, right, meta, feasible):
        self.debit(prev, wrong)
        self.learn(prev, right, meta, feasible)

    def dump(self):
        return {
            "transitions": [
                [prev, plan, entry.count, {f: list(r) for f, r in entry.ranges.items()}]
                for prev, row in self.transitions.items()
                for plan, entry in row.items()
            ],
            "importance": [[plan, n, c] for plan, (n, c) in self.importance.items()],
        }

    @classmethod
    def load(cls, saved):
        table = cls()
        for prev, plan, count, ranges in saved["transitions"]:
            table.transitions.setdefault(frozen(prev), {})[frozen(plan)] = Entry(
                count, {f: tuple(r) for f, r in ranges.items()}
            )
        for plan, confirmed, candidate in saved["importance"]:
            table.importance[frozen(plan)] = [confirmed, candidate]
        return table


class Step:
    __slots__ = ("plan", "calls", "meta")

    def __init__(self, plan, calls, meta):
        self.plan = plan
        self.calls = calls
        self.meta = meta


class Conversation:
    __slots__ = ("serial", "ledger")

    def __init__(self, serial):
        self.serial = serial
        self.ledger = []


class Check:
    __slots__ = ("conv", "t", "prev", "plan", "meta", "feasible", "view", "answer", "remaining")

    def __init__(self, conv, t, prev, plan, meta, feasible, view, answer, remaining):
        self.conv = conv
        self.t = t
        self.prev = prev
        self.plan = plan
        self.meta = meta
        self.feasible = feasible
        self.view = view
        self.answer = answer
        self.remaining = remaining


def read_table(path, binder, mode):
    try:
        with open(path) as f:
            saved = json.load(f)
    except (OSError, ValueError) as error:
        raise SystemExit(f"--load {path}: {error}. Refusing to start cold under a warm label.")
    if saved.get("binder") != binder or saved.get("metadata") != mode:
        raise SystemExit(
            f"--load {path} was saved under binder={saved.get('binder')} "
            f"metadata={saved.get('metadata')}, and this run is binder={binder} metadata={mode}. "
            "Its plans and ranges mean something else here."
        )
    return Table.load(saved)


class AgenticCache(LocalServer):
    def __init__(self, port, upstream, load=None, save=None, **knobs):
        config = settings(os.environ)
        config.update(knobs)
        self.k = config["k"]
        self.jitter = config["jitter"]
        self.arm = config["arm"]
        self.metadata = config["metadata"]
        self.binder = config["binder"]
        self.walk = config["walk"]
        if self.arm == "static" and not load:
            raise SystemExit(
                "ARM=static freezes the table, so with nothing loaded it serves nothing and "
                "measures nothing. It is the warm arm: give it --load."
            )
        self.table = read_table(load, self.binder, self.metadata) if load else Table()
        super().__init__(("127.0.0.1", port), Handler)
        self.upstream = upstream
        self.save_to = save
        self.checks = self.arm in ("full", "updates", "replacement")
        self.counts = self.arm in ("full", "updates")
        self.replaces = self.arm in ("full", "replacement")
        self.frozen = self.arm == "static"
        self.ties = random.Random(f"{config['seed']}:ties")
        self.cadence = random.Random(f"{config['seed']}:cadence")
        self.lock = threading.Lock()
        self.serial = 0
        self.current = None
        self.conversations = {}
        self.check = None
        self.stats = dict.fromkeys(
            (
                "hits",
                "misses",
                "deferrals",
                "replacements",
                "checks",
                "self_checks",
                "confirmations",
                "corrections",
            ),
            0,
        )
        self.decisions = []
        self.side_log = open(config["log"], "a") if config["log"] else None

    def server_close(self):
        if self.side_log:
            self.side_log.close()
            self.side_log = None
        super().server_close()

    def delay(self):
        if self.k == 1 or not self.jitter:
            return self.k
        return self.cadence.randint(1, 2 * self.k - 1)

    def stepped(self):
        if self.check is not None:
            self.check.remaining -= 1

    def identity_of(self, calls, view):
        if not calls:
            return END
        plans = tuple(identity(call, view, self.binder) for call in calls)
        return plans[0] if len(plans) == 1 else ("multi", plans)

    def conversation(self, view):
        if view.t == 0 or self.current is None:
            self.serial += 1
            self.current = Conversation(self.serial)
            kept = {self.serial: self.current}
            if self.check is not None:
                kept[self.check.conv] = self.conversations[self.check.conv]
            self.conversations = kept
        conv = self.current
        ledger = conv.ledger
        agreed = 0
        while agreed < min(len(ledger), view.t) and ledger[agreed].calls == view.turns[agreed][0]:
            agreed += 1
        del ledger[agreed:]
        for i in range(agreed, view.t):
            past = view.at(i)
            calls = view.turns[i][0]
            ledger.append(Step(self.identity_of(calls, past), calls, metadata(past, self.metadata)))
        return conv

    def land(self, view, conv, record):
        check = self.check
        if check is None or check.remaining > 0:
            return None
        self.check = None
        calls = calls_of(check.answer)
        answer = self.identity_of(calls, check.view)
        ledger = self.conversations[check.conv].ledger
        wanted = calls[0] if len(calls) == 1 else None
        found = None
        if wanted is not None:
            required = check.view.required(wanted[0])
            if required is None:
                required = tuple(wanted[1])
            found = next(
                (
                    j
                    for j in range(check.t, len(ledger))
                    if any(same(call, wanted, required) for call in ledger[j].calls)
                ),
                None,
            )
        landed = record["landed"] = {
            "covered": [check.conv, check.t],
            "served": named(check.plan),
            "answer": named(answer),
        }
        if found is not None:
            self.stats["confirmations"] += 1
            landed["verdict"] = "confirmed"
            if self.counts:
                before = ledger[found - 1].plan if found else START
                self.table.confirm(before, ledger[found].plan, ledger[found].meta)
            return None
        self.stats["corrections"] += 1
        landed["verdict"] = "corrected"
        if self.counts:
            self.table.correct(check.prev, check.plan, answer, check.meta, check.feasible)
        if not (self.replaces and wanted is not None and check.conv == conv.serial):
            return None
        required = view.required(wanted[0])
        if required is None or any(wanted[1].get(arg) is None for arg in required):
            return None
        return (wanted[0], {arg: wanted[1][arg] for arg in required}), answer

    def decide(self, raw, body, path):
        view = read(body, path)
        conv = self.conversation(view)
        meta = metadata(view, self.metadata)
        record = {"conversation": conv.serial, "t": view.t, "meta": meta}
        replaced = self.land(view, conv, record)
        if replaced is not None:
            call, plan = replaced
            payload = self.emitted(body, view, call)
            conv.ledger.append(Step(plan, [call], meta))
            self.stats["replacements"] += 1
            self.stepped()
            record.update(decision="replaced", plan=named(plan), call=[call[0], call[1]])
            return self.done(record, payload)

        prev = conv.ledger[-1].plan if view.t else START
        feasible = self.table.feasible(prev, meta)
        if not self.frozen:
            self.table.appeared(feasible)
        ranked = self.table.rank(prev, feasible, self.ties)
        record["prev"] = named(prev)
        record["candidates"] = [[named(p), float(self.table.score(prev, p))] for p in ranked]
        chosen, deferred = None, False
        for plan in ranked:
            if plan == END:
                deferred = True
                break
            if plan[0] != "call":
                if self.walk == "argmax":
                    break
                continue
            args = bind(plan, view, self.binder)
            if args is not None:
                chosen = plan, (plan[1], args)
                break
            if self.walk == "argmax":
                break

        if chosen is not None:
            plan, call = chosen
            payload = self.emitted(body, view, call)
            conv.ledger.append(Step(plan, [call], meta))
            self.stats["hits"] += 1
            record.update(decision="served", plan=named(plan), call=[call[0], call[1]])
            if self.check is None and self.checks:
                answer = forward(self.upstream, raw, path)
                delay = self.delay()
                self.check = Check(
                    conv.serial, view.t, prev, plan, meta, set(feasible), view, answer, delay
                )
                self.stats["checks"] += 1
                record["check"] = delay
            self.stepped()
            return self.done(record, payload)

        out = forward(self.upstream, raw, path)
        calls = calls_of(out)
        answer = self.identity_of(calls, view)
        if deferred:
            self.stats["deferrals"] += 1
            if self.check is None and self.counts:
                self.stats["self_checks"] += 1
                if answer == END:
                    self.table.credit(prev, END, meta)
                else:
                    self.table.correct(prev, END, answer, meta, feasible)
                record["self_check"] = "confirmed" if answer == END else "corrected"
        else:
            self.stats["misses"] += 1
            if not self.frozen:
                self.table.learn(prev, answer, meta, feasible)
        if calls:
            conv.ledger.append(Step(answer, calls, meta))
            self.stepped()
        record.update(decision="deferred" if deferred else "miss", answer=named(answer))
        return self.done(record, json.dumps(out).encode())

    def emitted(self, body, view, call):
        decision = {"kind": "call", "name": call[0], "args": call[1]}
        out = emit_openai(decision, body) if view.openai else emit_anthropic(decision, body)
        zero_usage(out)
        return json.dumps(out).encode()

    def done(self, record, payload):
        record["in_flight"] = self.check is not None
        self.decisions.append(record)
        if self.side_log:
            self.side_log.write(json.dumps(record) + "\n")
            self.side_log.flush()
        if self.save_to and self.table.changed:
            beside = f"{self.save_to}.partial"
            with open(beside, "w") as f:
                json.dump(dict(self.table.dump(), binder=self.binder, metadata=self.metadata), f)
            os.replace(beside, self.save_to)
            self.table.changed = False
        return payload


class Handler(ProxyHandler):
    def do_POST(self):
        raw = self.rfile.read(int(self.headers["Content-Length"]))
        body = json.loads(raw)
        with self.server.lock:
            payload = self.server.decide(raw, body, self.path)
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
