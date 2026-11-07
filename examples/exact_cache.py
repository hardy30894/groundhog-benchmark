"""An exact-match cache, written as a Python class the runner starts for each pair.

    groundhog --system py:examples.exact_cache:ExactCache --limit 10

It answers the opening turn of a conversation it has seen before, word for word, with the
reply the model gave then, and passes every other request to the model.
"""

import json
from http.server import ThreadingHTTPServer

from groundhog.systems.proxy import ProxyHandler, forward, opening_of, zero_usage


class Handler(ProxyHandler):
    def do_POST(self):
        raw = self.rfile.read(int(self.headers["Content-Length"]))
        body = json.loads(raw)
        first_turn = not any(m["role"] == "assistant" for m in body["messages"])
        key = opening_of(body)
        if first_turn and key in self.server.stored:
            out = json.loads(json.dumps(self.server.stored[key]))
            zero_usage(out)
            return self.reply(out)
        out = forward(self.server.upstream, raw, self.path)
        if first_turn:
            self.server.stored[key] = out
        self.reply(out)


class ExactCache(ThreadingHTTPServer):
    def __init__(self, port, upstream):
        super().__init__(("127.0.0.1", port), Handler)
        self.upstream = upstream
        self.stored = {}
