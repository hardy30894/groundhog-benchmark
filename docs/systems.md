# The seven systems

Every system sits between the agent and the model as an HTTP proxy and receives the agent's
requests unchanged. On each request it either answers from what it stored, at no cost to
the model, or forwards the request and stores what comes back. LexCache, GPTCache, vCache and
W5H2 act on the opening turn of a conversation and forward every later turn; AgentReuse,
AgenticCache and GenCache can answer later turns as well.

Each system is started fresh for every family-domain pair, so nothing it learned on one pair
carries into the next. Settings are environment variables; the defaults are the settings
Table 1 of the paper reports, and every module's docstring lists its settings.

| system | where the code comes from | reported setting |
|---|---|---|
| LexCache | written for this benchmark | threshold 0.8 |
| GPTCache | the `gptcache` package, 0.1.44 | similarity 0.85 |
| vCache | the authors' library, unmodified | delta 0.01 |
| W5H2 | the authors' classifier code, plus the paper | alpha 0.10 |
| AgentReuse | the paper; no code is released | gamma 0.75, m3e-small |
| AgenticCache | the paper; the released cache needs its simulator | K = 1 |
| GenCache | the paper | gpt-5-mini as program writer |

## LexCache

[`systems/lexcache.py`](../groundhog/systems/lexcache.py). A lexical baseline written for
this benchmark. It masks the identifiers in the opening request, represents the rest by its
set of word bigrams, and serves the stored reply of the most similar earlier request when the
overlap (the shared bigrams over the geometric mean of the two set sizes) reaches the
threshold, with the new request's identifier substituted in. Table 5 reports thresholds 0.7,
0.8 and 0.9.

## GPTCache

[`systems/gptcache.py`](../groundhog/systems/gptcache.py). It uses the embedding model the
`gptcache` package ships, paraphrase-albert-small-v2 in ONNX form, with a cosine search over
stored requests written here, and replays the nearest stored reply word for word when the
similarity reaches the threshold. It embeds the opening request rather than the whole
prompt: every prompt starts with about 1,900 characters of system text and tool schemas,
which fill the model's 512-token window and truncate the request away. The package's own
`similarity_threshold` is on a different scale (a distance score), so its default of 0.8 is
not comparable with the cosine threshold here. Table 6 reports 0.80, 0.85 and 0.90.

## vCache

[`systems/vcache.py`](../groundhog/systems/vcache.py). The authors' library
(github.com/vcache-project/vCache at commit 50694ee) makes every decision:
`VerifiedDecisionPolicy` learns a threshold for each cached prompt and decides whether to
serve it or ask the model. The adapter supplies what `VCacheConfig` asks a user for: an
inference engine that forwards to the model under test, an embedding engine around GPTCache's
ONNX model, and the library's string comparison as the similarity evaluator, applied to
replies with wire ids and usage removed. The decision to explore is a random draw the library
does not seed, so two runs differ slightly. Table 7 reports delta 0.005, 0.01, 0.02 and 0.05.

## W5H2

[`systems/w5h2.py`](../groundhog/systems/w5h2.py). An intent cache: the request is reduced
to a canonical class (its first call's tool, and any argument that names a category rather
than an entity, 25 classes in all) and the class is the key. Two tiers assign the class. The
first is an exact match on the request with its entities masked, against the training
requests. The second is the authors' few-shot classifier, `hierarchical.train_setfit` from
github.com/nabaos/w5h2-intent-cache at commit 835c6d7: SetFit over all-MiniLM-L6-v2 on 8
examples per class, 1 epoch, batch 32, 10 contrastive iterations. Confidence is the
probability of the most likely class, as the authors' `get_predictions_with_confidence`
reports it. The threshold is the lowest at which the paper's bound, Learn Then Test with an
empirical Bernstein bound at delta = 0.10, certifies the rate of wrong hits at tolerance
alpha, computed on 24 held-out examples per class. The paper's other tiers, a fine-tuned
BERT model and a hosted language model, are not included, so a request neither tier is sure
of goes to the model under test. Table 8 reports alpha 0.05, 0.10, 0.20 and 0.30.

The release imports a module it does not ship; [`systems/shims/`](../groundhog/systems/shims)
provides it from the release's own `w5h2_taxonomy.py`.

## AgentReuse

[`systems/agentreuse.py`](../groundhog/systems/agentreuse.py). Written from Algorithm 1 of
the paper. It stores plans, the sequence of tool calls an earlier conversation made, and
replays them step by step. A reader classifies the opening request's intent (14 intents, and
undefined for a request that should lead to no action) and tags its parameters; the paper's
reader is bert-base-chinese, so this one is bert-base-uncased with an intent head and a
slot-tagging head, fine-tuned for 3 epochs at 5e-5 and batch 32 on 2,400 requests. The
parameters are blanked out, the rest is embedded with the paper's model, m3e-small, and
stored conversations with the same intent are searched by cosine similarity. A match at
gamma = 0.75 or above replays the plan with the new parameters filled in. A request read as
undefined is never matched. The paper asks the model to annotate which plan step depends on
which; a proxy cannot add a prompt, so dependencies are inferred from the recorded values:
an argument equal to a field of an earlier step's result is taken from the new result on
replay. Table 9 varies gamma and the embedding model and reports four variants: serving the
opening turn only, parameters left in the embedding, matching against every stored
conversation, and refusals labelled with the action they mention.

## AgenticCache

[`systems/agenticcache.py`](../groundhog/systems/agenticcache.py). Written from the paper;
the released cache keys on its simulator's state and has no transition cache that takes a
plan step from outside it. It caches transitions between consecutive plan steps, where a
plan step is one assistant turn that calls a tool, with the circumstances each was seen in
recorded as ranges. Candidates whose ranges do not hold are discarded, the rest are scored
by how often each was observed times the importance of the step it leads to, and the best is
served. The model is asked what it would have done, and its answer is read K steps later: a
match raises the transition's counts, and a mismatch lowers them, stores the model's
transition and replaces the running plan if it still can. In the paper K is the model's
latency; the agent here has none, so the reported row uses K = 1, where each check costs the
call it was meant to save. For K above 1 the delay of each check is drawn between 1 and
2K - 1 steps. The cache starts empty, where the paper's headline configuration preloads
transitions from earlier episodes. Table 10 varies K, admits result fields to the recorded
circumstances, and runs two arms of the paper's ablation.

## GenCache

[`systems/gencache.py`](../groundhog/systems/gencache.py). Written from the paper. Requests
are clustered by the similarity of their prompts and of their replies (T_p = 0.8,
T_r = 0.75). When a cluster holds nu = 4 examples, a model writes a program that maps any
request of that shape to its reply; the program is run on the examples and kept if at least
gamma = 50% come out right, with up to rho = 30 attempts; and a pattern is written that
decides whether a later request belongs to the cluster. The code-writing and validation
prompts are the paper's appendix prompts; the paper gives no prompt for the pattern, so that
one is ours. The paper's program writer is GPT-4o; this one is gpt-5-mini at low reasoning
effort, through OpenRouter. Each program runs in a separate process with an empty
environment and is stopped after 10 seconds. A request that arrives while its cluster's
program is being written waits up to 150 seconds for it, where the paper answers it from the
model at once. Because a model writes the programs, two runs differ. The program writer's
calls are logged beside the result file; Table 11 counts their tokens against the savings.

## Models and their licenses

| model | used by | license |
|---|---|---|
| paraphrase-albert-small-v2 (ONNX, shipped by gptcache) | GPTCache, vCache, GenCache | Apache 2.0 |
| all-MiniLM-L6-v2 | W5H2; AgentReuse in Table 9 | Apache 2.0 |
| bert-base-uncased | AgentReuse | Apache 2.0 |
| moka-ai/m3e-small | AgentReuse | per its model card |
| gpt-5-mini, through OpenRouter | GenCache | OpenAI's terms |
