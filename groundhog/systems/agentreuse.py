"""AgentReuse, a plan cache, written from Algorithm 1 of its paper (no code is released).

A reader (bert-base-uncased with an intent head and a slot-tagging head, standing in for
the paper's bert-base-chinese) reads the opening request's intent and its parameters. The
parameters are blanked out, the rest is embedded (m3e-small, the paper's model), and the
request is matched by cosine similarity against stored conversations with the same
intent. A match at or above gamma replays that conversation's plan step by step, with the
new request's parameters filled in. A request read as undefined, such as a refusal, is
never matched. The plan holds tool calls only; the final reply always comes from the model.

Settings, all optional:
    GROUNDHOG_AGENTREUSE_GAMMA       match threshold; 0.75
    GROUNDHOG_AGENTREUSE_EMBEDDER    moka-ai/m3e-small or sentence-transformers/all-MiniLM-L6-v2
    GROUNDHOG_AGENTREUSE_LABELS      outcome, or semantic to label a refusal by the act it names
    GROUNDHOG_AGENTREUSE_ABLATION    withargs (parameters left in) or oneintent (search all)
    GROUNDHOG_AGENTREUSE_TURN0       1 to serve the opening turn only
    GROUNDHOG_AGENTREUSE_TRUNCATE    head or tail: which end of a long request is kept
    GROUNDHOG_AGENTREUSE_LOG         a file to log every conversation's outcome to
"""

import hashlib
import json
import os
import re
import secrets
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np

from .. import data
from ..model import call_id
from ..score import EDGE
from ..world import DOMAINS
from .proxy import ProxyHandler, forward, opening_of, refuse_unknown_knobs, zero_usage

GAMMA = float(os.environ.get("GROUNDHOG_AGENTREUSE_GAMMA", "0.75"))
EMBEDDER = os.environ.get("GROUNDHOG_AGENTREUSE_EMBEDDER", "moka-ai/m3e-small")
LABELS = os.environ.get("GROUNDHOG_AGENTREUSE_LABELS", "outcome")
TRUNCATE = os.environ.get("GROUNDHOG_AGENTREUSE_TRUNCATE", "head")
ABLATION = os.environ.get("GROUNDHOG_AGENTREUSE_ABLATION", "")
TURN0 = os.environ.get("GROUNDHOG_AGENTREUSE_TURN0", "")
LOG = os.environ.get("GROUNDHOG_AGENTREUSE_LOG", "")
RUN = secrets.token_hex(6)

READER = "bert-base-uncased"
REVISIONS = {
    "bert-base-uncased": "86b5e0934494bd15c9632b12f734a8a67f723594",
    "moka-ai/m3e-small": "44c696631b2a8c200220aaaad5f987f096e986df",
    "sentence-transformers/all-MiniLM-L6-v2": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
}
WINDOW = 512

# The training sets in data/train/: one the reader learns from, one it is checked on.
TRAINING = "agentreuse"
CHECK = "agentreuse-check"

UNDEFINED = "undefined"

MODELS = str(Path(__file__).resolve().parents[2] / "models")
TRAINED = {"outcome": "agentreuse", "semantic": "agentreuse-semantic"}
WEIGHTS = ("agentreuse", "agentreuse-semantic")


def verb(tool):
    return tool.split("_")[0]


def intent_of(inst, labels="outcome"):
    if inst["effects"]:
        return verb(inst["effects"][0][0])
    if labels == "semantic":
        forbidden = inst.get("forbidden") or []
        if forbidden:
            return verb(forbidden[0][0])
    return UNDEFINED


def tokens(text):
    for match in re.finditer(r"\S+", text):
        word = match.group(0)
        core = word.strip(EDGE)
        if core:
            start = match.start() + len(word) - len(word.lstrip(EDGE))
            yield start, start + len(core), core


def bounded(haystack, needle, start=0):
    while needle:
        at = haystack.find(needle, start)
        if at < 0:
            return -1
        end = at + len(needle)
        if (at == 0 or not haystack[at - 1].isalnum()) and (
            end == len(haystack) or not haystack[end].isalnum()
        ):
            return at
        start = at + 1
    return -1


def flows(token, value):
    return bounded(value.lower(), token.lower()) >= 0


def shape(token):
    return re.sub(r"\d", "0", token.lower())


def drawn(name):
    """A training set as (family, instance) pairs."""
    return data.training(name)


_TYPES = {}


def types(domain):
    if not _TYPES:
        for name in DOMAINS:
            _TYPES[name] = ({}, [])
        for _, inst in drawn(TRAINING):
            seen, order = _TYPES[inst["domain"]]
            for _, _, core in tokens(inst["opening"]):
                for _, args in inst["effects"]:
                    for key, value in args:
                        if isinstance(value, str) and flows(core, value):
                            seen.setdefault(key, set()).add(shape(core))
                            if key not in order:
                                order.append(key)
    return _TYPES[domain]


def typed_spans(opening, effects, domain):
    seen, order = types(domain)
    spans = []
    for start, end, core in tokens(opening):
        kind = next(
            (
                key
                for _, args in effects
                for key, value in args
                if isinstance(value, str) and flows(core, value)
            ),
            None,
        )
        if kind is None:
            kind = next((key for key in order if shape(core) in seen[key]), None)
        if kind:
            spans.append((start, end, kind))
    return spans


def stripped(text, spans):
    out, at = [], 0
    for start, end, _ in spans:
        out.append(text[at:start])
        at = end
    out.append(text[at:])
    return " ".join("".join(out).split())


def labelled(batch, labels="outcome"):
    out = []
    for family, inst in batch:
        out.append(
            {
                "family": family,
                "domain": inst["domain"],
                "text": inst["opening"],
                "refusal": not inst["effects"],
                "intent": intent_of(inst, labels),
                "spans": typed_spans(inst["opening"], inst["effects"], inst["domain"]),
            }
        )
    said = {}
    for example in out:
        said.setdefault(stripped(example["text"], example["spans"]), []).append(example["intent"])
    for example in out:
        votes = said[stripped(example["text"], example["spans"])]
        example["intent"] = max(dict.fromkeys(votes), key=votes.count)
    return out


def examples(seed, labels="outcome"):
    return labelled(drawn(seed), labels)


def bio(text, spans):
    starts = {(start, end): kind for start, end, kind in spans}
    return [f"B-{starts[(a, b)]}" if (a, b) in starts else "O" for a, b, _ in tokens(text)]


def training_digest(labels):
    parts = hashlib.sha256()
    for example in examples(TRAINING, labels):
        parts.update(
            json.dumps(
                [example["text"], example["intent"], bio(example["text"], example["spans"])]
            ).encode()
        )
    return parts.hexdigest()[:16]


def tag_names(kinds):
    return ["O"] + [f"{edge}-{kind}" for kind in kinds for edge in ("B", "I")]


def decode(offsets, tags):
    spans, open_kind = [], None
    for (a, b), tag in zip(offsets, tags):
        if a == b:
            continue
        if tag == "O":
            open_kind = None
            continue
        edge, kind = tag.split("-", 1)
        if edge == "I" and open_kind == kind:
            spans[-1] = (spans[-1][0], b, kind)
        else:
            spans.append((a, b, kind))
            open_kind = kind
    return spans


def joint(encoder, intents, tags):
    import torch

    class Joint(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.drop = torch.nn.Dropout(encoder.config.hidden_dropout_prob)
            self.intent = torch.nn.Linear(encoder.config.hidden_size, intents)
            self.slot = torch.nn.Linear(encoder.config.hidden_size, tags)

        def forward(self, ids, mask):
            out = self.encoder(input_ids=ids, attention_mask=mask)
            return (
                self.intent(self.drop(out.pooler_output)),
                self.slot(self.drop(out.last_hidden_state)),
            )

    return Joint()


def trained(where, labels):
    path = os.path.join(where, "meta.json")
    weights = [os.path.join(where, "heads.pt"), os.path.join(where, "encoder", "config.json")]
    if not os.path.exists(path) or not all(os.path.exists(w) for w in weights):
        flag = "" if labels == "outcome" else f" --labels {labels}"
        raise SystemExit(
            f"this system's reader is not in this checkout (its weights are not committed).\n"
            f"  python scripts/train_agentreuse.py{flag}\n"
            f"builds it, seeded; it fine-tunes bert-base, so give it a machine with 16 GB."
        )
    with open(path) as f:
        meta = json.load(f)
    if meta.get("smoke"):
        raise SystemExit(f"{where} holds a --smoke reader, trained on sixteen examples")
    if meta.get("training_digest") != _digest(labels):
        raise SystemExit(
            f"{where} was trained on labelled set {meta.get('training_digest')}, and the "
            f"training set gives {_digest(labels)} now; retrain it."
        )
    return meta


_DIGESTS = {}


def _digest(labels):
    if labels not in _DIGESTS:
        _DIGESTS[labels] = training_digest(labels)
    return _DIGESTS[labels]


class Reader:
    def __init__(self, labels=LABELS, side=TRUNCATE):
        import torch
        from transformers import BertModel, BertTokenizerFast

        where = os.path.join(MODELS, TRAINED[labels])
        meta = trained(where, labels)
        self.torch = torch
        self.intents = meta["intents"]
        self.tags = meta["tags"]
        self.tokenizer = BertTokenizerFast.from_pretrained(where)
        self.tokenizer.truncation_side = "left" if side == "tail" else "right"
        self.model = joint(
            BertModel.from_pretrained(os.path.join(where, "encoder")),
            len(self.intents),
            len(self.tags),
        )
        heads = torch.load(os.path.join(where, "heads.pt"), map_location="cpu")
        self.model.intent.load_state_dict(heads["intent"])
        self.model.slot.load_state_dict(heads["slot"])
        self.model.eval()

    def read(self, text):
        got = self.tokenizer(
            text,
            truncation=True,
            max_length=WINDOW,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        with self.torch.no_grad():
            intent, slots = self.model(got["input_ids"], got["attention_mask"])
        tags = [self.tags[i] for i in slots[0].argmax(-1).tolist()]
        return (
            self.intents[int(intent[0].argmax())],
            decode(got["offset_mapping"][0].tolist(), tags),
        )


_READERS = {}
_EMBEDDERS = {}


def load_reader(labels=LABELS, side=TRUNCATE):
    if (labels, side) not in _READERS:
        _READERS[labels, side] = Reader(labels, side)
    return _READERS[labels, side]


def load_embedder(name=EMBEDDER, side=TRUNCATE):
    if (name, side) not in _EMBEDDERS:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(name, device="cpu", revision=REVISIONS.get(name))
        model.max_seq_length = WINDOW
        model.tokenizer.truncation_side = "left" if side == "tail" else "right"

        def embed(text):
            return model.encode([text], normalize_embeddings=True, show_progress_bar=False)[
                0
            ].astype(np.float32)

        _EMBEDDERS[name, side] = embed
    return _EMBEDDERS[name, side]


class Index:
    def __init__(self):
        self.entries = []
        self.rows = {}

    def add(self, intent, vector, plan):
        self.entries.append((intent, np.asarray(vector, dtype=np.float32), plan))
        self.rows.setdefault(intent, []).append(len(self.entries) - 1)

    def search(self, intent, vector, everywhere=False):
        rows = list(range(len(self.entries))) if everywhere else self.rows.get(intent, [])
        if not rows:
            return None, None
        scores = np.stack([self.entries[r][1] for r in rows]) @ np.asarray(vector, np.float32)
        best = int(np.argmax(scores))
        return float(scores[best]), rows[best]


def dialect_of(path):
    return "openai" if path.startswith("/v1/chat/completions") else "anthropic"


def parsed(content):
    if isinstance(content, list):
        content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
    try:
        value = json.loads(content)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def trajectory(body, dialect):
    messages = [m for m in body["messages"] if m.get("role") != "system"]
    turns = []
    for i, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        if dialect == "openai":
            calls = [
                (c["function"]["name"], json.loads(c["function"]["arguments"] or "{}"))
                for c in message.get("tool_calls") or []
            ]
            text = bool(message.get("content"))
            results = []
            for later in messages[i + 1 :]:
                if later.get("role") != "tool":
                    break
                results.append(parsed(later.get("content")))
            raw = message
        else:
            content = message.get("content")
            blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
            calls = [
                (b["name"], b.get("input") or {})
                for b in blocks
                if isinstance(b, dict) and b.get("type") == "tool_use"
            ]
            text = any(
                isinstance(b, dict) and b.get("type") == "text" and b.get("text") for b in blocks
            )
            after = messages[i + 1] if i + 1 < len(messages) else {}
            results = (
                [
                    parsed(b.get("content"))
                    for b in after.get("content") or []
                    if isinstance(b, dict) and b.get("type") == "tool_result"
                ]
                if after.get("role") == "user" and isinstance(after.get("content"), list)
                else []
            )
            raw = blocks
        turns.append({"calls": calls, "text": text, "raw": raw, "results": results})
    return turns


def calls_in(out):
    if "choices" in out:
        return bool(((out["choices"] or [{}])[0].get("message") or {}).get("tool_calls"))
    return any(isinstance(b, dict) and b.get("type") == "tool_use" for b in out.get("content", []))


PROSE = {"prose": True}


def reference(para, index):
    kind = para[index][0]
    return ("param", kind, sum(1 for other, _ in para[:index] if other == kind))


def link(value, para):
    low, pieces, at = value.lower(), [], 0
    while True:
        best = None
        for i, (_, text) in enumerate(para):
            where = bounded(low, text.lower(), at)
            if where >= 0 and (
                best is None or where < best[0] or (where == best[0] and len(text) > best[2])
            ):
                best = (where, i, len(text))
        if best is None:
            break
        where, i, length = best
        if where > at:
            pieces.append(value[at:where])
        pieces.append(reference(para, i))
        at = where + length
    if not pieces:
        return None
    if at < len(value):
        pieces.append(value[at:])
    return pieces


def provenance(value, para, earlier):
    if value is not None and not isinstance(value, (dict, list)):
        for step in range(len(earlier) - 1, -1, -1):
            for call, result in enumerate(earlier[step]):
                for field, got in (result or {}).items():
                    if type(got) is type(value) and got == value:
                        return ("dep", step, call, field)
    if isinstance(value, str):
        pieces = link(value, para)
        if pieces is not None:
            return (
                pieces[0]
                if len(pieces) == 1 and isinstance(pieces[0], tuple)
                else ("template", pieces)
            )
    return ("literal", value)


def build(para, turns, dialect):
    steps, earlier = [], []
    for turn in turns:
        if turn["text"] or not turn["calls"]:
            steps.append(PROSE)
        else:
            steps.append(
                {
                    "dialect": dialect,
                    "raw": json.loads(json.dumps(turn["raw"])),
                    "calls": [
                        {
                            "name": name,
                            "args": [
                                (key, provenance(value, para, earlier))
                                for key, value in args.items()
                            ],
                        }
                        for name, args in turn["calls"]
                    ],
                }
            )
        earlier.append(turn["results"])
    steps.append(PROSE)
    return {"steps": steps}


class Unbound(Exception):
    pass


def resolve(how, para, results):
    if how[0] == "literal":
        return json.loads(json.dumps(how[1]))
    if how[0] == "param":
        _, kind, k = how
        values = [text for other, text in para if other == kind]
        if k >= len(values):
            raise Unbound("missing parameter")
        return values[k]
    if how[0] == "template":
        return "".join(p if isinstance(p, str) else resolve(p, para, results) for p in how[1])
    _, step, call, field = how
    result = results[step][call] if step < len(results) and call < len(results[step]) else None
    if not isinstance(result, dict) or result.get(field) is None:
        raise Unbound("unresolved dependency")
    return result[field]


def bind(step, para, results):
    return [
        (call["name"], {key: resolve(how, para, results) for key, how in call["args"]})
        for call in step["calls"]
    ]


def served(step, calls, dialect, body, mint):
    raw = json.loads(json.dumps(step["raw"]))
    if dialect == "openai":
        for made, (name, args) in zip(raw["tool_calls"], calls):
            made["id"] = mint(len(made["id"]))
            made["function"]["name"] = name
            made["function"]["arguments"] = json.dumps(args)
        out = {
            "model": body.get("model", "stub"),
            "object": "chat.completion",
            "choices": [{"index": 0, "message": raw, "finish_reason": "tool_calls"}],
        }
    else:
        blocks = [b for b in raw if b.get("type") == "tool_use"]
        for block, (name, args) in zip(blocks, calls):
            block["id"] = mint(len(block["id"]))
            block["name"] = name
            block["input"] = args
        out = {"model": body.get("model", "stub"), "stop_reason": "tool_use", "content": blocks}
    zero_usage(out)
    return out


class Conversation:
    def __init__(self, outcome, intent, para):
        self.outcome = outcome
        self.mode = {"hit": "replay", "miss": "record"}.get(outcome, "model")
        self.intent = intent
        self.para = para
        self.vector = None
        self.plan = None
        self.score = None
        self.entry = None
        self.served = 0
        self.reason = None if outcome == "hit" else outcome
        self.ids = 0

    def mint(self, dialect):
        def fresh(width):
            got = call_id(dialect, self.ids % 100)
            self.ids += 1
            return (got + "0" * width)[:width]

        return fresh


class AgentReuse(ThreadingHTTPServer):
    def __init__(self, port, upstream, reader=None, embedder=None):
        refuse_unknown_knobs(
            os.environ,
            "GROUNDHOG_AGENTREUSE_",
            ("EMBEDDER", "GAMMA", "LABELS", "TRUNCATE", "ABLATION", "TURN0", "LOG"),
        )
        for name, value, allowed in (
            ("LABELS", LABELS, TRAINED),
            ("TRUNCATE", TRUNCATE, ("head", "tail")),
            ("ABLATION", ABLATION, ("", "withargs", "oneintent")),
            ("TURN0", TURN0, ("", "0", "1")),
        ):
            if value not in allowed:
                raise SystemExit(
                    f"GROUNDHOG_AGENTREUSE_{name}={value!r} is not one of {list(allowed)}"
                )
        super().__init__(("127.0.0.1", port), Handler)
        self.upstream = upstream
        self.reader = reader or load_reader()
        self.embed = embedder or load_embedder()
        self.gamma = GAMMA
        self.ablation = ABLATION
        self.turn0 = TURN0 == "1"
        self.log = LOG
        self.index = Index()
        self.conversations = {}
        self.lock = threading.Lock()
        self.hits = self.misses = self.undefined = self.stored = self.served = 0
        self.forwarded = {}

    def begin(self, opening, body, dialect):
        intent, spans = self.reader.read(opening)
        para = [(kind, opening[start:end]) for start, end, kind in spans]
        vector = None
        if intent != UNDEFINED:
            vector = self.embed(
                opening if self.ablation == "withargs" else stripped(opening, spans)
            )
        with self.lock:
            self.close(opening, finished=False)
            if intent == UNDEFINED:
                self.undefined += 1
                self.conversations[opening] = Conversation("undefined", intent, para)
                return None
            score, entry = self.index.search(
                intent, vector, everywhere=self.ablation == "oneintent"
            )
            hit = score is not None and score >= self.gamma
            talk = Conversation("hit" if hit else "miss", intent, para)
            talk.vector, talk.score, talk.entry = vector, score, entry
            self.conversations[opening] = talk
            if not hit:
                self.misses += 1
                return None
            self.hits += 1
            talk.plan = self.index.entries[entry][2]
            return self.step(talk, body, dialect, 0)

    def resume(self, opening, body, dialect, turn):
        with self.lock:
            talk = self.conversations.get(opening)
            if talk is None or talk.mode != "replay":
                return None
            return self.step(talk, body, dialect, turn)

    def step(self, talk, body, dialect, turn):
        steps = talk.plan["steps"]
        if turn > 0 and self.turn0:
            return self.handover(talk, "turn-0 ablation")
        if turn >= len(steps) or steps[turn] is PROSE:
            return self.handover(talk, "prose step")
        try:
            calls = bind(steps[turn], talk.para, [t["results"] for t in trajectory(body, dialect)])
        except Unbound as why:
            return self.handover(talk, str(why))
        talk.served += 1
        self.served += 1
        return served(steps[turn], calls, dialect, body, talk.mint(dialect))

    def handover(self, talk, reason):
        talk.mode, talk.reason = "model", reason
        self.forwarded[reason] = self.forwarded.get(reason, 0) + 1
        return None

    def passed(self, opening, body, dialect, out):
        if calls_in(out):
            return
        with self.lock:
            talk = self.conversations.get(opening)
            if talk is None:
                return
            if talk.mode == "record":
                self.index.add(
                    talk.intent, talk.vector, build(talk.para, trajectory(body, dialect), dialect)
                )
                self.stored += 1
            self.close(opening, finished=True)

    def close(self, opening, finished):
        talk = self.conversations.pop(opening, None)
        if talk is None or not self.log:
            return
        with open(self.log, "a") as f:
            f.write(
                json.dumps(
                    {
                        "run": RUN,
                        "opening": hashlib.sha256(opening.encode()).hexdigest()[:16],
                        "intent": talk.intent,
                        "outcome": talk.outcome,
                        "score": None if talk.score is None else round(talk.score, 4),
                        "entry": talk.entry,
                        "served": talk.served,
                        "forwarded": talk.reason,
                        "finished": finished,
                    }
                )
                + "\n"
            )

    def server_close(self):
        with self.lock:
            for opening in list(self.conversations):
                self.close(opening, finished=False)
        super().server_close()


class Handler(ProxyHandler):
    def do_POST(self):
        raw = self.rfile.read(int(self.headers["Content-Length"]))
        body = json.loads(raw)
        server = self.server
        dialect = dialect_of(self.path)
        opening = opening_of(body)
        turn = sum(1 for m in body["messages"] if m.get("role") == "assistant")
        if turn == 0:
            out = server.begin(opening, body, dialect)
        else:
            out = server.resume(opening, body, dialect, turn)
        if out is None:
            out = forward(server.upstream, raw, self.path)
            server.passed(opening, body, dialect, out)
        self.reply(out)
