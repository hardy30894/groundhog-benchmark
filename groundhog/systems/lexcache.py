"""LexCache, the lexical baseline.

It masks the identifiers in the opening request, represents what is left by its set of
word bigrams, and serves the stored reply of the most similar earlier request when the
overlap reaches the threshold, with the new request's identifier substituted in. Later
turns of a conversation are forwarded.

    GROUNDHOG_LEXCACHE_THRESHOLD   overlap needed to serve, in (0, 1]; 0.8 by default
"""

import json
import os
import re

from ..server import LocalServer
from .proxy import ProxyHandler, forward, opening_of, refuse_unknown_knobs, zero_usage

IDENTIFIER = re.compile(
    r"\b(ord-\d+|tbl_\d+|team\d+|(?:mod|module|helper)_\d+\.py|\S+@\S+)\b", re.I
)


def threshold_of(environ):
    refuse_unknown_knobs(environ, "GROUNDHOG_LEXCACHE_", ("THRESHOLD",))
    raw = environ.get("GROUNDHOG_LEXCACHE_THRESHOLD", "0.8")
    try:
        threshold = float(raw)
    except ValueError:
        threshold = -1.0
    if not 0.0 < threshold <= 1.0:
        raise SystemExit(f"GROUNDHOG_LEXCACHE_THRESHOLD is a similarity in (0, 1]; got {raw!r}")
    return threshold


def shape_of(text):
    """The request's word bigrams, with every identifier replaced by one placeholder."""
    slotted = IDENTIFIER.sub(" <id> ", text.lower())
    words = re.findall(r"<id>|[a-z0-9']+", slotted)
    return {tuple(words[i : i + 2]) for i in range(max(1, len(words) - 1))}


def overlap(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / (len(a) * len(b)) ** 0.5


def nearest(shape, cache, threshold):
    """The stored reply most similar to this shape, if any reaches the threshold."""
    best, score = None, threshold
    for cached_shape, cached_out in cache:
        similarity = overlap(shape, cached_shape)
        if similarity > score or (best is None and similarity >= score):
            best, score = cached_out, similarity
    return best


def substitute(out, fresh):
    """Put the new request's identifier into a stored reply's tool arguments."""
    for block in out.get("content", []):
        if isinstance(block, dict) and block.get("type") == "tool_use":
            for key, value in block.get("input", {}).items():
                if isinstance(value, str) and IDENTIFIER.match(value):
                    block["input"][key] = fresh.upper() if value.isupper() else fresh
    for choice in out.get("choices", []):
        for call in choice.get("message", {}).get("tool_calls") or []:
            args = json.loads(call["function"]["arguments"] or "{}")
            for key, value in args.items():
                if isinstance(value, str) and IDENTIFIER.match(value):
                    args[key] = fresh.upper() if value.isupper() else fresh
            call["function"]["arguments"] = json.dumps(args)


class LexCache(LocalServer):
    def __init__(self, port, upstream):
        super().__init__(("127.0.0.1", port), Handler)
        self.upstream = upstream
        self.cache = []
        self.threshold = threshold_of(os.environ)


class Handler(ProxyHandler):
    def do_POST(self):
        raw = self.rfile.read(int(self.headers["Content-Length"]))
        body = json.loads(raw)
        opening = opening_of(body)
        turns = sum(1 for m in body["messages"] if m.get("role") == "assistant")
        if turns == 0:
            shape = shape_of(opening)
            wanted = IDENTIFIER.search(opening)
            cached_out = nearest(shape, self.server.cache, self.server.threshold)
            if cached_out is not None:
                out = json.loads(json.dumps(cached_out))
                if wanted:
                    substitute(out, wanted.group(1))
                zero_usage(out)
                self.reply(out)
                return
        out = forward(self.server.upstream, raw, self.path)
        if turns == 0:
            self.server.cache.append((shape_of(opening), out))
        self.reply(out)
