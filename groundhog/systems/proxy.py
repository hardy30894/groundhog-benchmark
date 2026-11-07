"""What every cache system shares: it is an HTTP proxy between the agent and the model."""

import json
import urllib.request
from http.server import BaseHTTPRequestHandler

TIMEOUT = 180


def opening_of(body):
    """The text of the first user message: the request the conversation opened with."""
    for message in body["messages"]:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        return content or ""
    return ""


def forward(upstream, raw, path):
    """Send the agent's request on to the model unchanged and return its reply. Never retried."""
    req = urllib.request.Request(
        f"{upstream}{path}",
        data=raw,
        headers={"content-type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=TIMEOUT))


def strip_ids(value):
    """A reply with its wire ids removed, for comparing two replies."""
    if isinstance(value, dict):
        return {k: strip_ids(v) for k, v in value.items() if k != "id"}
    if isinstance(value, list):
        return [strip_ids(v) for v in value]
    return value


def zero_usage(out):
    """Mark a reply as served from the cache: the model was not asked, so it cost nothing."""
    if "choices" in out:
        out["usage"] = {"prompt_tokens": 0, "completion_tokens": 0}
    else:
        out["usage"] = {"input_tokens": 0, "output_tokens": 0}


def refuse_unknown_knobs(environ, prefix, known):
    """Stop on a misspelt setting rather than run without it."""
    unknown = sorted(k for k in environ if k.startswith(prefix) and k[len(prefix) :] not in known)
    if unknown:
        raise SystemExit(
            f"{', '.join(unknown)}: not a setting this system reads; it reads "
            f"{', '.join(prefix + k for k in sorted(known))}"
        )


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def reply(self, out):
        payload = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
