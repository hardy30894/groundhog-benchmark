# Results

One file for each of the 42 configurations the paper reports, named by system and setting.
[`scripts/paper.py`](../scripts/paper.py) maps each file to the tables it appears in and to the
settings that reproduce it, and [`TABLES.md`](TABLES.md) holds the paper's tables recomputed
from these files by [`scripts/tables.py`](../scripts/tables.py).
`gencache/gpt-5-mini.synthesis.jsonl` logs every call GenCache made to the model that writes its
programs, with its tokens and price; Table 11 counts those tokens.

## Format

Each file is a JSON object with `meta`, describing the run, and `rows`, one per family-domain
pair, in the form `domain/family`.

| row field | meaning |
|---|---|
| `gate` | `FAIL` if the system made any wrong write on the pair, `pass` otherwise |
| `wrong_writes`, `wrong_reads` | unrequired state-changing and read-only calls |
| `wrong_answers` | instances whose final answer missed a required string, contained an excluded one, or did a procedure out of order |
| `missed_writes` | required state-changing calls that never happened |
| `control_tokens`, `system_tokens` | tokens the model was charged without and with the cache |
| `saved`, `earned` | gross and earned savings, as fractions of `control_tokens` |
| `served_turns` | model turns the cache answered without the model being asked |
| `checked_turns` | turns the cache answered and then asked the model about anyway |
| `first_saving_at_run` | the first instance on which the cache served a turn |
| `steps_per_run`, `served_prefix` | turns per instance, and turns served before the first one the model was asked |
| `stale_notes`, `notes_changed` | free-text notes naming a record the run did not touch, and notes that differ from the baseline's; reported, not scored |

Pooled savings weight each pair by its `control_tokens`.

## Version 1.33 names

These files were produced by version 1.33 of the harness, so their `meta` uses that version's
names for systems and settings:

| `meta.system` in these files | system | settings renamed since |
|---|---|---|
| `fuzzy` | `lexcache` | `GROUNDHOG_TRIGRAM_THRESHOLD` is `GROUNDHOG_LEXCACHE_THRESHOLD`; `GROUNDHOG_TRIGRAM_UNIT=words` is the only behaviour kept |
| `py:groundhog.thirdparty:GptCacheOnnx` | `gptcache` | `GROUNDHOG_SIMILARITY` is `GROUNDHOG_GPTCACHE_THRESHOLD` |
| `py:groundhog.official_vcache:OfficialVCache` | `vcache` | none |
| `py:groundhog.official_w5h2:OfficialW5H2` | `w5h2` | `GROUNDHOG_W5H2_UPSTREAM` now defaults to `third_party/w5h2-intent-cache` |
| `py:groundhog.agentreuse:AgentReuse` | `agentreuse` | none |
| `py:groundhog.agenticcache:AgenticCache` | `agenticcache` | none |
| `py:groundhog.gencache:GenCache` | `gencache` | none |

`GROUNDHOG_RETRIES=0` appears in every file; the harness now never retries a request, which is
what that setting did.

Except for GenCache's, these files also contain rows for two families, garnish and migrated,
that were measured in version 1.33 and are not part of the benchmark. The tables and the tests
use only the 33 benchmark pairs.
