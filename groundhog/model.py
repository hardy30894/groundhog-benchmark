"""The model under test and the token meter.

The model is a written policy that does what a competent agent would do, one tool call at
a time, behind two wire formats (Anthropic messages and OpenAI chat completions). Every
token is counted here, at the model, so a system's own account of its cost is never used.
"""

import hashlib
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler

from .domains import ASIDES, DOMAIN
from .server import LocalServer
from .world import STANDING_ERAS, SURVEY, SURVEY_LIMIT


def tokens(payload):
    """One token per four characters of the serialised payload, at least one."""
    return max(1, len(json.dumps(payload)) // 4)


# Prompt-cache pricing: a cached prefix costs 1.25 times its tokens when written and 0.10
# times when read back.
CACHE_WRITE = 1.25
CACHE_READ = 0.10


def cacheable(body):
    """The prefix a request marked for caching (system text and tools) and its tokens."""
    tools_declared = body.get("tools") or []
    asked = any(isinstance(t, dict) and t.get("cache_control") for t in tools_declared)
    if not asked:
        return None, 0
    prefix = {"system": body.get("system"), "tools": tools_declared}
    return json.dumps(prefix, sort_keys=True), tokens(prefix)


def charge(body, out_tokens, seen):
    """The priced tokens of one turn, and whether its prefix was written or read."""
    whole = tokens(body)
    key, prefix = cacheable(body)
    if key is None or prefix == 0:
        return whole + out_tokens, None
    fresh = max(whole - prefix, 0)
    if key in seen:
        return fresh + prefix * CACHE_READ + out_tokens, ("read", prefix)
    seen.add(key)
    return fresh + prefix * CACHE_WRITE + out_tokens, ("written", prefix)


def route_asked(d, text):
    ask = d["route_ask"].split("{")[0].strip().lower()
    return bool(ask) and ask in text


def aside(text):
    return ASIDES[sum(ord(c) for c in text) % len(ASIDES)]


class Policy:
    """The model's decision for the next turn, given the conversation so far."""

    def reply(self, turns, opening, last_result, era, domain="support"):
        d = DOMAIN[domain]
        task = opening.rsplit("</system-reminder>", 1)[-1]
        text = task.lower()

        if turns >= 1:
            parsed = None
            try:
                parsed = json.loads(last_result)
            except (ValueError, TypeError):
                pass
            # Inside a batch procedure the next step is whatever the last result names.
            if isinstance(parsed, dict) and ("next" in parsed or "held" in parsed):
                if parsed.get("held"):
                    return self.call(d["batch_write"], {d["batch_write_key"]: parsed["item"]})
                if parsed.get("next"):
                    return self.call(d["read"], {d["read_key"]: parsed["next"]})
                walked = re.search(r"\bbatch (\w+)", text)
                return self.answer(
                    f"procedure complete for batch {walked.group(1)}"
                    if walked
                    else "procedure complete"
                )
            # A survey count decides between the operator's two actions.
            _, survey_key, field = SURVEY[domain]
            if isinstance(parsed, dict) and field in parsed:
                over = parsed[field] > SURVEY_LIMIT
                tool, key, _ = d["quiet_high"] if over else d["quiet_low"]
                return self.call(tool, {key: parsed[survey_key]})
            first, _, second, second_key = d["chain"]
            if turns == 1 and isinstance(parsed, dict):
                handed = parsed.get(
                    {"support": "account_id", "code": "path", "data": "table"}[domain]
                )
                if handed and "find the" in text:
                    return self.call(second, {second_key: handed})
            # A reply that explains a choice the model made is composed, not repeated.
            if route_asked(d, text):
                return self.answer(f"result: {last_result}, {aside(text)}")
            return self.answer(f"result: {last_result}")

        batch = re.search(r"\bbatch (\w+)", text)
        if batch and "sweep" in text:
            return self.call("list_batch", {"batch_id": batch.group(1)})

        ident = re.search(self.pattern(domain), text)
        route_ask = d["route_ask"].split("{")[0].strip().lower()

        if text.startswith("review ") and ident:
            # The standing rule is held by the model and stated in no request.
            severe = era == STANDING_ERAS[1]
            tool, key, _ = d["quiet_high"] if severe else d["quiet_low"]
            return self.call(tool, {key: ident.group(0)})

        if text.startswith("survey ") and ident:
            tool, survey_key, _ = SURVEY[domain]
            return self.call(tool, {survey_key: ident.group(0)})

        if "canonicalize" in text and ident:
            return self.call(d["read"], {d["read_key"]: ident.group(0).upper()})

        if route_ask and route_ask in text:
            bucket = "general"
            for name, things in d["buckets"].items():
                if any(thing in text for thing in things):
                    bucket = name
                    break
            return self.call(d["route"], {d["route_key"]: bucket, "note": "triaged"})

        first, first_key, _, _ = d["chain"]
        chain_id = re.search(r"(?:named|for|owned by) (\S+@\S+|\S+)", text)
        if "find the" in text and chain_id:
            return self.call(first, {first_key: chain_id.group(1).rstrip("?.,")})

        if re.search(r"\b(do not|don't|dont|never)\b", text):
            return self.answer(f"leaving it alone, {aside(text)}")
        if ident and d["act"] in text:
            return self.call(d["write"], {d["read_key"]: ident.group(0)})
        if ident:
            args = {d["read_key"]: ident.group(0)}
            if era == 1:
                args["format"] = "v2"
            return self.call(d["read"], args)
        return self.answer("nothing to do")

    def pattern(self, domain):
        return {
            "support": r"ord-\d+",
            "code": r"mod_\d+\.py",
            "data": r"tbl_\d+",
        }[domain]

    def call(self, name, args):
        return {"kind": "call", "name": name, "args": args}

    def answer(self, text):
        return {"kind": "answer", "text": text}


def text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            b.get("text", b.get("content", "")) if isinstance(b, dict) else str(b) for b in content
        )
    return str(content)


def view_anthropic(body):
    messages = body["messages"]
    turns = sum(1 for m in messages if m["role"] == "assistant")
    opening = text_of(messages[0]["content"])
    last = messages[-1]["content"]
    last_result = ""
    if isinstance(last, list):
        for block in last:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                last_result = text_of(block.get("content", ""))
    return turns, opening, last_result


def view_openai(body):
    messages = [m for m in body["messages"] if m.get("role") != "system"]
    turns = sum(1 for m in messages if m.get("role") == "assistant")
    opening = text_of(messages[0].get("content")) if messages else ""
    last_result = ""
    if messages and messages[-1].get("role") == "tool":
        last_result = str(messages[-1].get("content") or "")
    return turns, opening, last_result


def call_id(dialect, step=0):
    """A tool-call id at the length providers mint, so a system minting its own pays nothing extra."""
    body = f"{step:02d}" + "q90qw90lq917835lq9x7z"
    return f"toolu_01{body}" if dialect == "anthropic" else f"call_01{body}"


def emit_anthropic(decision, body):
    if decision["kind"] == "call":
        content = [
            {
                "type": "tool_use",
                "id": call_id("anthropic"),
                "name": decision["name"],
                "input": decision["args"],
            }
        ]
        stop = "tool_use"
    else:
        content = [{"type": "text", "text": decision["text"]}]
        stop = "end_turn"
    return {"model": body.get("model", "stub"), "stop_reason": stop, "content": content}


def emit_openai(decision, body):
    if decision["kind"] == "call":
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id("openai"),
                    "type": "function",
                    "function": {
                        "name": decision["name"],
                        "arguments": json.dumps(decision["args"]),
                    },
                }
            ],
        }
        finish = "tool_calls"
    else:
        message = {"role": "assistant", "content": decision["text"]}
        finish = "stop"
    return {
        "model": body.get("model", "stub"),
        "object": "chat.completion",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
    }


def conversation_digest(body):
    """Identifies the turn a request asks about, however a system re-serialised it."""
    turn = {
        "system": body.get("system"),
        "messages": body.get("messages"),
        "tools": body.get("tools"),
    }
    return hashlib.sha256(
        json.dumps(turn, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


class Model(LocalServer):
    """The model as an HTTP server, with the meter every cost is read from."""

    def __init__(self, port=0):
        super().__init__(("127.0.0.1", port), Handler)
        self.policy = Policy()
        self.era = 0
        self.domain = "support"
        self.spent = 0
        self.calls = 0
        # While set, only requests carrying this mark count toward the pass being
        # measured. Anything else is stray traffic, reported and kept apart.
        self.baseline = None
        self.stray = 0
        self.stray_calls = 0
        self.cached = set()
        self.lock = threading.Lock()
        # Every turn the model answered in the current pass, with when it was metered.
        self.asked = []

    def new_phase(self):
        """Forget cached prefixes: a configuration about to be measured has paid for none."""
        self.cached = set()

    def respond(self, path, body):
        """The model's reply to one request, and the tokens it is charged."""
        if path.startswith("/v1/chat/completions"):
            turns, opening, last = view_openai(body)
            out = emit_openai(self.policy.reply(turns, opening, last, self.era, self.domain), body)
            produced = tokens(out["choices"])
            spent, cached = charge(body, produced, self.cached)
            out["usage"] = {
                "prompt_tokens": tokens(body),
                "completion_tokens": produced,
            }
            if cached:
                _, prefix = cacheable(body)
                out["usage"]["prompt_tokens_details"] = {"cached_tokens": prefix}
            return out, spent
        turns, opening, last = view_anthropic(body)
        out = emit_anthropic(self.policy.reply(turns, opening, last, self.era, self.domain), body)
        produced = tokens(out["content"])
        spent, cached = charge(body, produced, self.cached)
        whole = tokens(body)
        _, prefix = cacheable(body)
        out["usage"] = {
            "input_tokens": whole - prefix if cached else whole,
            "output_tokens": produced,
        }
        if cached and cached[0] == "written":
            out["usage"]["cache_creation_input_tokens"] = prefix
        elif cached:
            out["usage"]["cache_read_input_tokens"] = prefix
        return out, spent


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        raw = self.rfile.read(int(self.headers["Content-Length"]))
        body = json.loads(raw)
        out, spent = self.server.respond(self.path, body)
        self.meter(spent, conversation_digest(body))
        self.reply(200, out)

    def meter(self, spent, asked=None):
        with self.server.lock:
            mine = self.server.baseline is None or (
                self.headers.get("x-groundhog-pass") == self.server.baseline
            )
            if mine:
                self.server.spent += spent
                self.server.calls += 1
                if asked is not None:
                    self.server.asked.append((asked, time.monotonic_ns()))
            else:
                self.server.stray += spent
                self.server.stray_calls += 1

    def reply(self, status, out):
        payload = json.dumps(out).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        if self.close_connection:
            self.send_header("connection", "close")
        self.end_headers()
        self.wfile.write(payload)
