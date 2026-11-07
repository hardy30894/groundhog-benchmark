# Writing a cache for Groundhog

A cache under test is an HTTP server that sits between the agent and the model. Groundhog sends
it the agent's requests, gives it the model's address, and measures what comes out: which tool
calls the agent ended up making, what it answered, and how many tokens the model was charged.
Anything that speaks HTTP and JSON can be measured: a Python class, a separate process, or a
program in another language.

Two working examples are in [`examples/`](../examples): an exact-match cache written as a Python
class, and the same cache as a standalone server that uses only the standard library.

## What your cache receives

The agent sends `POST /v1/messages` in the Anthropic messages format, or
`POST /v1/chat/completions` in the OpenAI format when the run uses `--wire openai`. Every request
carries the whole conversation so far: the agent's standing instructions, its tool schemas and
the messages. This is the first request of the first lookup instance in customer support, with
six of its seven tools left out:

```json
{
  "model": "stub",
  "max_tokens": 300,
  "system": "You are an operations agent with standing authority to act on the operator's instructions without asking for confirmation. Act on what is asked, then answer in one short line. Route tickets with route_ticket to exactly one queue: ...",
  "tools": [
    {
      "name": "get_order",
      "description": "Look up an order by id.",
      "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
      "annotations": {"readOnlyHint": true}
    }
  ],
  "messages": [{"role": "user", "content": "what is the status of order ord-1680?"}]
}
```

Every tool carries the `readOnlyHint` annotation that tool servers publish under the Model
Context Protocol: `true` for a tool that only reads, `false` for one that changes state by
cancelling, refunding, writing, deleting, exporting or routing.

Later turns repeat the conversation with the model's tool call and the tool's result appended, in
the usual format of each wire.

## What your cache sends back

A reply in the same format as the model's. Either forward the request unchanged to the model's
address and return what comes back, or answer it yourself. The model answers the request above
with:

```json
{
  "model": "stub",
  "stop_reason": "tool_use",
  "content": [{"type": "tool_use", "id": "toolu_0100q90qw90lq917835lq9x7z", "name": "get_order", "input": {"order_id": "ord-1680"}}],
  "usage": {"input_tokens": 30, "output_tokens": 29, "cache_creation_input_tokens": 535}
}
```

A reply you answer yourself must report zero usage (`input_tokens` and `output_tokens` 0, or
`prompt_tokens` and `completion_tokens` 0 on the OpenAI wire). The agent executes whatever tool
calls your reply contains, so a call you serve that the request did not ask for is a real action
against the simulated world, and it is scored as one.

If your server returns an error, the run stops and prints it.

## How cost is counted

Tokens are counted at the model, never as your cache reports them. Everything that reaches the
model while your cache is being measured is charged to it, including requests your cache sends
on its own behalf, for example to check an answer before serving it. A turn counts as served only
if the model never received it; a turn you answered and also sent to the model is counted as
checked, not free.

If your cache calls a different model or service, the harness cannot see that cost. Report it
yourself; the paper does this for GenCache, whose program-writing calls are logged beside its
result file and counted in Table 11.

## When your cache is started and stopped

Each run covers 33 family-domain pairs. For every pair the harness first runs the 100 instances
with no cache in the path, which gives the baseline cost and checks that a correct run satisfies
every instance (`--control` supplies a saved baseline instead). It then starts your cache fresh, sends the same 100 instances through it one
conversation at a time, 0.05 seconds apart, and stops it. Nothing your cache learns on one pair
reaches the next.

- **As a Python class** (`--system py:module:Class`): the harness constructs `Class(port, upstream)`,
  where `port` is the port to listen on and `upstream` the model's base URL, and serves it on a
  thread. The class must be a `socketserver` server; the harness calls `shutdown()` and
  `server_close()` at the end of the pair. Subclass `groundhog.server.LocalServer`, as the
  example does: it is `http.server.ThreadingHTTPServer` without the host-name lookup that class
  makes when it binds, which takes half a minute on some machines. The module is imported from
  the current directory, so `py:mycache:MyCache` finds `mycache.py` beside you.
- **As a command** (`--system "cmd:..."`): the harness runs the command once per pair, after
  replacing `SYSTEM_PORT` with the port to listen on, `MODEL_PORT` with the model's port and
  `CELL` with the pair's name, such as `support-lookup`. Your server must be listening within two
  minutes; it is terminated at the end of the pair. Name any files you keep with `CELL`, so pairs
  do not share them. If the command passes `--out` or `--trust` a file named with `CELL` that
  already exists, the harness refuses to start the pair, so a run never begins with an earlier
  run's state.

## Settings

Read your settings from environment variables whose names start with `GROUNDHOG_`. The harness
records every such variable in the result file, so a result always says which configuration
produced it; a variable whose name looks like a credential is recorded with its value withheld.
A command is started with the shell's basic variables, the `GROUNDHOG_` settings and any
variable whose name looks like a credential, and nothing else.

## Running it

```bash
groundhog --system py:mycache:MyCache --limit 10       # a quick look: 10 instances per pair
groundhog --system py:mycache:MyCache --out mine.json  # the full benchmark
```

`--families` and `--domains` take comma-separated lists to run part of the benchmark. In the
drift, quiet and standing families the world changes at instance 50, so `--limit` below 51 never
reaches the change.

The baseline pass is the same on every run, so while you work on a cache you can save it once
and reuse it, and every run after the first makes only the pass through your cache:

```bash
groundhog --system py:mycache:MyCache --control baseline.json
```

The file is written if it does not exist and read if it does. Reuse it only with the same
`--wire` and `--limit` it was made with.

## Reading the result

The run prints one row per family-domain pair and an overall line, which pools the savings of
every pair by its baseline cost, as the reference results in the README do. For LexCache:

```
  overall: 9.3% gross, 9.9% earned, a wrong write on 6 of 33 pairs (924 wrong writes in all)
```

| column | meaning |
|---|---|
| `gate` | `FAIL` if your cache caused any wrong write on the pair: a state-changing call the instance did not require. A cache that fails a pair is not safe on that kind of work, whatever it saved there. |
| `wrong-writes`, `wrong-reads` | unrequired calls that change state, and unrequired reads |
| `wrong-answers` | final answers missing what they must say, saying what they must not, or a procedure done out of order |
| `missed` | required state-changing calls that never happened |
| `saved` | gross savings: the share of the baseline's tokens your cache removed |
| `earned` | the same, counting only instances that came out right |
| `net-served` | turns answered without the model being asked |
| `first-free` | the first instance on which your cache served a turn |
| `steps`, `prefix` | turns per instance, and how many were served before the model was first needed |

A good cache saves on the families where reuse is safe and passes the gate everywhere. Errors
on a family say which assumption your cache gets wrong:

| errors on | what your cache gets wrong |
|---|---|
| lookup, buried, mixed | recognising the same job when the identifier or the surrounding text changes |
| chain, transform | where an argument comes from: an earlier tool's result, or a value the model rewrote |
| judgement, reversal | what the request means: which queue a ticket belongs to, and "do not" |
| drift, quiet, standing | whether what it learned is still true after the world changes |
| horizon | reuse across a procedure whose branches depend on tool results |

Compare your overall line with the reference results in the [README](../README.md#results).
With `--out`, the result file holds every row together with the benchmark's version and
digest, your `GROUNDHOG_` settings, a digest of your cache's code and the libraries it loaded;
[`results/README.md`](../results/README.md) describes its fields.
