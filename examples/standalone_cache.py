"""The same exact-match cache as a standalone HTTP server, using only the standard library.

    groundhog --system "cmd:python examples/standalone_cache.py SYSTEM_PORT http://127.0.0.1:MODEL_PORT"

This is the template for a cache in any language: listen on the port you are given, read each
request the agent sends, and either answer it yourself or pass it unchanged to the model's
address. A reply you answer yourself must report zero token usage.

It serves with socketserver's ThreadingTCPServer rather than http.server's HTTPServer, which
looks up the host name before it listens, and that lookup takes half a minute on some machines.
"""

import json
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler
from socketserver import ThreadingTCPServer

PORT, MODEL = int(sys.argv[1]), sys.argv[2]
stored = {}


def ask_model(path, raw):
    request = urllib.request.Request(
        MODEL + path, data=raw, headers={"content-type": "application/json"}
    )
    return json.load(urllib.request.urlopen(request, timeout=180))


def mark_served(reply):
    if "choices" in reply:
        reply["usage"] = {"prompt_tokens": 0, "completion_tokens": 0}
    else:
        reply["usage"] = {"input_tokens": 0, "output_tokens": 0}
    return reply


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        raw = self.rfile.read(int(self.headers["Content-Length"]))
        body = json.loads(raw)
        messages = [m for m in body["messages"] if m["role"] != "system"]
        first_turn = len(messages) == 1
        key = json.dumps(messages[0], sort_keys=True)
        if first_turn and key in stored:
            reply = mark_served(json.loads(json.dumps(stored[key])))
        else:
            reply = ask_model(self.path, raw)
            if first_turn:
                stored[key] = reply
        payload = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


ThreadingTCPServer(("127.0.0.1", PORT), Handler).serve_forever()
