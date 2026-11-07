"""GPTCache, run from its released package (gptcache 0.1.44).

It embeds the opening request with the package's ONNX model, finds the stored request
nearest in cosine similarity, and replays that request's reply word for word when the
similarity reaches the threshold. Later turns are forwarded.

The opening request is embedded rather than the whole prompt: every prompt starts with
about 1,900 characters of system text and tool schemas, which fill the model's 512-token
window and truncate the request itself away.

    GROUNDHOG_GPTCACHE_THRESHOLD   cosine similarity needed to serve; 0.85 by default
"""

import json
import os

from ..server import LocalServer
from .embedding import cosine, onnx_embedder
from .proxy import ProxyHandler, forward, opening_of, refuse_unknown_knobs, zero_usage


class GPTCache(LocalServer):
    def __init__(self, port, upstream):
        super().__init__(("127.0.0.1", port), Handler)
        refuse_unknown_knobs(os.environ, "GROUNDHOG_GPTCACHE_", ("THRESHOLD",))
        self.threshold = float(os.environ.get("GROUNDHOG_GPTCACHE_THRESHOLD", "0.85"))
        self.upstream = upstream
        self.embed = onnx_embedder()
        self.cache = []


class Handler(ProxyHandler):
    def do_POST(self):
        raw = self.rfile.read(int(self.headers["Content-Length"]))
        body = json.loads(raw)
        turns = sum(1 for m in body["messages"] if m.get("role") == "assistant")
        if turns == 0:
            vector = self.server.embed(opening_of(body))
            best = None
            score = 0.0
            for cached_vector, cached_out in self.server.cache:
                sim = cosine(vector, cached_vector)
                if sim > score:
                    best, score = cached_out, sim
            if best is not None and score >= self.server.threshold:
                out = json.loads(json.dumps(best))
                zero_usage(out)
                self.reply(out)
                return
            out = forward(self.server.upstream, raw, self.path)
            self.server.cache.append((vector, out))
            self.reply(out)
            return
        self.reply(forward(self.server.upstream, raw, self.path))
