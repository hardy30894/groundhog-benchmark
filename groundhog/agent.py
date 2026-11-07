"""The agent loop. It is the same for every system under test; only the address it sends
requests to changes. Tool calls execute against the simulated world, and what the agent
called is the record the run is scored on."""

import json
import time
import urllib.error
import urllib.request

from .model import conversation_digest
from .world import effect_of, execute, tools

TIMEOUT = 180

ACT = (
    "You are an operations agent with standing authority to act on the operator's "
    "instructions without asking for confirmation. Act on what is asked, then answer "
    "in one short line. "
)

# The agent's standing instructions, per domain. They are the system prompt of every request.
AGENT_POLICY = {
    "support": ACT + "Route tickets with route_ticket to exactly one queue: billing for payments, "
    "invoices, charges and refunds; engineering for crashes, errors and broken "
    "features; account for passwords, sign-in and two-factor.",
    "code": ACT + "Assign reviews with assign_review to exactly one team: platform for builds, "
    "releases, dependencies and caching; frontend for components, forms, styling and "
    "routing; backend for queries, auth, jobs and api handlers.",
    "data": ACT + "Assign reviews with assign_review to exactly one team: platform for ingestion, "
    "storage, clusters and retention; frontend for dashboards, charts, filters and "
    "reports; backend for indexes, etl, api handlers and stream consumers.",
}


def post(url, body, mark=None):
    """One request, never retried: a retry after a request reached the model charges it twice."""
    headers = {"content-type": "application/json"}
    if mark:
        headers["x-groundhog-pass"] = mark
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    try:
        return json.load(urllib.request.urlopen(req, timeout=TIMEOUT))
    except urllib.error.HTTPError as error:
        detail = ""
        try:
            text = error.read().decode(errors="replace")
            try:
                detail = json.loads(text).get("error", "")
            except (ValueError, AttributeError):
                detail = text.strip()[:300]
        except OSError:
            pass
        if detail:
            raise SystemExit(f"the model or the system under test failed: {detail}")
        raise


class AnthropicWire:
    path = "/v1/messages"

    def body(self, model, messages, era, domain):
        # The agent marks its system prompt and tool schemas for prompt caching, as a
        # deployed agent would, so the run without a cache already pays the cached rate.
        declared = tools(domain, era)
        if declared:
            declared = [dict(t) for t in declared]
            declared[-1]["cache_control"] = {"type": "ephemeral"}
        return {
            "model": model,
            "max_tokens": 300,
            "system": AGENT_POLICY[domain],
            "tools": declared,
            "messages": messages,
        }

    def read(self, reply):
        content = reply.get("content", [])
        calls = [
            {"id": b["id"], "name": b["name"], "input": b.get("input", {})}
            for b in content
            if b.get("type") == "tool_use"
        ]
        text = next((b.get("text", "") for b in content if b.get("type") == "text"), "")
        usage = reply.get("usage", {})
        spent = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        return reply, calls, text, spent

    def extend(self, messages, reply, results):
        messages.append({"role": "assistant", "content": reply["content"]})
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": call["id"],
                        "content": outcome
                        if outcome is not None
                        else "unsupported argument format",
                        **({"is_error": True} if outcome is None else {}),
                    }
                    for call, outcome in results
                ],
            }
        )


class OpenAiWire:
    path = "/v1/chat/completions"

    def body(self, model, messages, era, domain):
        return {
            "model": model,
            "max_tokens": 300,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["input_schema"],
                        "annotations": t["annotations"],
                    },
                }
                for t in tools(domain, era)
            ],
            "messages": [
                {
                    "role": "system",
                    "content": AGENT_POLICY[domain],
                }
            ]
            + messages,
        }

    def read(self, reply):
        message = reply["choices"][0]["message"]
        calls = [
            {
                "id": c["id"],
                "name": c["function"]["name"],
                "input": json.loads(c["function"]["arguments"] or "{}"),
            }
            for c in message.get("tool_calls") or []
        ]
        usage = reply.get("usage") or {}
        spent = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        return reply, calls, message.get("content") or "", spent

    def extend(self, messages, reply, results):
        messages.append(reply["choices"][0]["message"])
        for call, outcome in results:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": outcome if outcome is not None else "unsupported argument format",
                }
            )


WIRES = {"anthropic": AnthropicWire, "openai": OpenAiWire}


def run(
    base_url,
    opening,
    era=0,
    wire="anthropic",
    model="stub",
    domain="support",
    max_turns=6,
    mark=None,
):
    """Drive one instance to its final answer and return what happened."""
    speak = WIRES[wire]()
    messages = [{"role": "user", "content": opening}]
    cost = 0
    served = 0
    requests = 0
    effects = []
    notes = []
    free = []
    asked = []
    sent = []
    for _ in range(max_turns):
        requests += 1
        body = speak.body(model, messages, era, domain)
        asked.append(conversation_digest(body))
        sent.append(time.monotonic_ns())
        raw = post(f"{base_url}{speak.path}", body, mark)
        reply, calls, text, spent = speak.read(raw)
        cost += spent
        free.append(spent == 0)
        if spent == 0:
            served += 1
        if not calls:
            return {
                "cost": cost,
                "served": served,
                "requests": requests,
                "effects": effects,
                "answer": text,
                "notes": notes,
                "free": free,
                "asked": asked,
                "sent": sent,
            }
        results = []
        for call in calls:
            effects.append(effect_of(call["name"], call["input"]))
            for field in ("note", "body"):
                said = call["input"].get(field)
                if isinstance(said, str) and said:
                    notes.append(said)
            results.append((call, execute(call["name"], call["input"], era, domain)))
        speak.extend(messages, reply, results)
    return {
        "cost": cost,
        "served": served,
        "requests": requests,
        "effects": effects,
        "answer": "",
        "notes": notes,
        "free": free,
        "asked": asked,
        "sent": sent,
    }
