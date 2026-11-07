# Datasheet

Following the questions of Gebru et al., *Datasheets for Datasets* (2021).

## Motivation

**Why was the dataset created?** To measure what a cache in front of a tool-calling LLM
agent saves, and what it does that nobody asked for while saving it. Earlier evaluations of
caches score a hit against a paraphrase class, an intent label or a duplicate label, on text
where a wrong hit returns a wrong answer and nothing else. None of them contains a tool call
whose effect persists, a world that changes between storing a response and reusing it, or a
procedure long enough that most of its cost comes after the first call.

## Composition

**What does an instance represent?** A request to a tool-using agent in English, together
with a specification of a correct run: the calls it must make, calls the request rules out,
the strings its final answer must and must not contain, and a hidden world state the
environment reads. The [README](../README.md#the-dataset) documents every field.

**How many instances are there?** 3,300: 11 families in 3 domains (customer support, code
and data), 100 instances each. The families fall into 5 conditions: identity (lookup,
buried, mixed), values (chain, transform), meaning (judgement, reversal), world (drift,
quiet, standing) and depth (horizon).

**Is there a train, validation and test split?** The 3,300 instances are a test set. Two of
the evaluated systems train a classifier, and their training sets are in `data/train/`:
2,400 requests each for W5H2 and for AgentReuse, and 2,400 more on which AgentReuse's reader
is checked. They come from the same templates as the benchmark with separate random draws,
covering the 11 benchmark families and three further request shapes that are not part of the
benchmark: writes carrying a free-text note, reads of a record that moves to an archive
partway through, and short read requests behind a long preamble that changes between
requests. Because some families can phrase a request in only a few ways, some benchmark
requests equal a training request word for word. Of the 3,300, 19 equal one of W5H2's 200
few-shot examples and 105 equal one of AgentReuse's 2,400 training requests; 13 and 53 of
them come from the data domain's chain family, whose request has only 90 possible forms.

**Does the dataset contain personal or offensive content?** No. Every request, identifier,
order, file, table and email address is produced from a template and refers to no real
person, organisation or system. The email addresses use the reserved domain example.com.

**Is anything missing or approximated?** The model under test is a written policy, not a
language model, so every run is reproducible and every difference between a run with a cache
and one without is due to the cache. Requests are shorter than a deployed agent's: about
1,900 characters of system text and tool schemas, the user's sentence and a few short tool
results.

## Collection process

**How was the data produced?** From request templates written by the authors, with
identifiers, subjects and hidden world states filled in by a seeded random draw. No part of
the dataset comes from a language model. Each template was run through the agent without a cache to confirm that a correct run
satisfies its specification, and the tests repeat that check for all 3,300 instances.

**Who labelled it?** No one. Each instance's specification was produced together with its
request, from the same template.

## Uses

**What has it been used for?** Measuring seven caching systems, in the paper this repository
accompanies: LexCache, GPTCache, vCache, W5H2, AgentReuse, AgenticCache and GenCache.

**What should it not be used for?** A system that passes all 33 family-domain pairs has
passed these probes and no others; the result is not a safety certificate. Systems trained
on the families should say so when they report results, since the families are public.

## Distribution

The dataset, the harness, the seven implementations and every result file are released
under the Apache 2.0 license. Some systems need third-party models and code, which keep
their own licenses and are not redistributed here.

## Maintenance

The dataset is versioned with the benchmark: every result file records the benchmark's
version and a digest of the dataset and the harness, so a result can only be compared with
results produced under the same digest. Issues and corrections go through this repository's
issue tracker.
