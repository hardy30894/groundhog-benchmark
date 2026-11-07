"""W5H2, an intent cache: the request is reduced to a canonical class and the class is the key.

Two tiers decide the class. The first is an exact match on the request with its entities
masked, against the training requests. The second is the authors' few-shot classifier
(SetFit over all-MiniLM-L6-v2, from github.com/nabaos/w5h2-intent-cache), trained by
scripts/train_w5h2.py on 8 examples per class. It answers only when its confidence clears
the threshold certified at risk tolerance alpha on 24 held-out examples per class, using
the paper's bound (Learn Then Test with an empirical Bernstein bound, delta = 0.10).
A request neither tier is sure of goes to the model. A hit replays the stored reply with
the new request's entity substituted in; later turns are forwarded.

    GROUNDHOG_W5H2_ALPHA      risk tolerance: 0.05, 0.10, 0.20 or 0.30; 0.10 by default
    GROUNDHOG_W5H2_UPSTREAM   the authors' checkout; third_party/w5h2-intent-cache by default
"""

import json
import os
import re
import sys
from http.server import ThreadingHTTPServer
from pathlib import Path

from .. import data
from .proxy import ProxyHandler, forward, opening_of, refuse_unknown_knobs, zero_usage

MODELS = Path("models").resolve() / "w5h2"
SHIMS = Path(__file__).resolve().parent / "shims"
UPSTREAM = os.environ.get(
    "GROUNDHOG_W5H2_UPSTREAM", str(Path("third_party/w5h2-intent-cache").resolve())
)
WEIGHTS = ("w5h2",)

SHOTS = 8
CALIBRATION = 24
DRAWS = 60
ALPHA = 0.10
ALPHAS = (0.05, 0.10, 0.20, 0.30)
DELTA = 0.10

WHO = re.compile(
    r"(ord-\d+|tbl_\d+|team\d+|(?:mod|module|helper)_\d+\.py|\S+@\S+|\b\d{3,}\b)", re.I
)


def entity(value):
    return bool(WHO.search(value)) or any(c.isdigit() for c in value)


def label(effects, domain):
    """A request's class: its first call's tool and any argument that is a category rather
    than an entity. A request that should make no call is classed by its domain."""
    if not effects:
        return ("none", domain)
    tool, args = effects[0]
    parts = [tool]
    for _, value in args:
        if isinstance(value, str) and not entity(value):
            parts.append(value)
    return tuple(parts)


def swap(out, fresh):
    """Put the new request's entity into a stored reply's tool arguments."""
    for block in out.get("content", []):
        if isinstance(block, dict) and block.get("type") == "tool_use":
            for key, value in block.get("input", {}).items():
                if isinstance(value, str):
                    block["input"][key] = WHO.sub(fresh, value)
    for choice in out.get("choices", []):
        for call in choice.get("message", {}).get("tool_calls") or []:
            args = json.loads(call["function"]["arguments"] or "{}")
            for key, value in args.items():
                if isinstance(value, str):
                    args[key] = WHO.sub(fresh, value)
            call["function"]["arguments"] = json.dumps(args)


def fingerprint(text):
    """The request with every entity masked, lowercased and whitespace-normalised."""
    return " ".join(WHO.sub("<id>", text.lower()).split())


def pool():
    """Every training request, grouped by class."""
    got = {}
    for _, inst in data.training("w5h2"):
        got.setdefault(label(inst["effects"], inst["domain"]), []).append(inst["opening"])
    return got


def shots():
    """8 examples per class to learn from and 24 per class to calibrate on, taken round
    robin across the families that produce the class."""
    pools = {}
    for family, inst in data.training("w5h2"):
        key = label(inst["effects"], inst["domain"])
        pools.setdefault(key, {}).setdefault(family, []).append(inst["opening"])
    by_class, held = {}, {}
    for key, families in pools.items():
        taken = []
        for index in range(DRAWS):
            for family in sorted(families):
                if len(taken) == SHOTS + CALIBRATION:
                    break
                if index < len(families[family]):
                    taken.append(families[family][index])
            if len(taken) == SHOTS + CALIBRATION:
                break
        by_class[key] = taken[:SHOTS]
        held[key] = taken[SHOTS:]
    return by_class, held


def threshold(confidences, correct, alpha, delta=DELTA, grid=100):
    """The lowest confidence threshold at which the bound certifies the rate of wrong
    answers at or below alpha, scanning down from 1.0."""
    import numpy as np

    sure = np.asarray(confidences)
    right = np.asarray(correct, dtype=bool)
    n = len(sure)
    log_term = np.log(3.0 / delta)
    chosen = 1.0
    for step in range(grid, -1, -1):
        tau = step / grid
        loss = ((sure >= tau) & ~right).astype(float)
        variance = loss.var(ddof=1) if n > 1 else 0.0
        correction = np.sqrt(2.0 * variance * log_term / n) + 3.0 * log_term / n
        if loss.mean() + correction > alpha:
            break
        chosen = tau
    return chosen


def upstream_on_path():
    for path in (str(SHIMS), UPSTREAM):
        if path not in sys.path:
            sys.path.insert(0, path)


class Classifier:
    """The fingerprint tier, then the authors' SetFit classifier with a certified threshold."""

    def __init__(self):
        upstream_on_path()
        from hierarchical import get_predictions_with_confidence
        from setfit import SetFitModel

        if not (MODELS / "calibration.json").exists():
            raise SystemExit("W5H2's classifier is not trained yet: python scripts/train_w5h2.py")
        self.confidence_of = get_predictions_with_confidence
        self.model = SetFitModel.from_pretrained(str(MODELS / "setfit"))
        self.calibration = json.loads((MODELS / "calibration.json").read_text())
        seen = {}
        for key, texts in pool().items():
            for text in texts:
                seen.setdefault(fingerprint(text), set()).add(key)
        self.prints = {p: next(iter(k)) for p, k in seen.items() if len(k) == 1}

    def decide(self, text, alpha):
        got = self.prints.get(fingerprint(text))
        if got is not None:
            return got, 0
        preds, confidence, _ = self.confidence_of(self.model, [text])
        if confidence[0] >= self.calibration["thresholds"][f"{alpha:g}"]:
            return tuple(json.loads(preds[0])), 1
        return None, None


class W5H2(ThreadingHTTPServer):
    def __init__(self, port, upstream):
        super().__init__(("127.0.0.1", port), Handler)
        refuse_unknown_knobs(os.environ, "GROUNDHOG_W5H2_", ("ALPHA", "UPSTREAM"))
        self.upstream = upstream
        self.classifier = Classifier()
        self.alpha = float(os.environ.get("GROUNDHOG_W5H2_ALPHA", ALPHA))
        if f"{self.alpha:g}" not in self.classifier.calibration["thresholds"]:
            raise SystemExit(f"no threshold was calibrated for alpha {self.alpha:g}")
        self.cache = {}
        self.served_by = {}

    def canonical(self, text):
        key, tier = self.classifier.decide(text, self.alpha)
        if key is not None:
            self.served_by[tier] = self.served_by.get(tier, 0) + 1
        return key


class Handler(ProxyHandler):
    def do_POST(self):
        raw = self.rfile.read(int(self.headers["Content-Length"]))
        body = json.loads(raw)
        turns = sum(1 for m in body["messages"] if m.get("role") == "assistant")
        if turns:
            self.reply(forward(self.server.upstream, raw, self.path))
            return
        opening = opening_of(body)
        key = self.server.canonical(opening)
        if key is None:
            self.reply(forward(self.server.upstream, raw, self.path))
            return
        hit = self.server.cache.get(key)
        if hit is not None:
            out = json.loads(json.dumps(hit))
            wanted = WHO.search(opening)
            if wanted:
                swap(out, wanted.group(1))
            zero_usage(out)
            self.reply(out)
            return
        out = forward(self.server.upstream, raw, self.path)
        self.server.cache[key] = out
        self.reply(out)
