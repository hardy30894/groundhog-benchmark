"""Trains AgentReuse's reader: bert-base-uncased with an intent head and a slot-tagging head.

    python scripts/train_agentreuse.py                    the reader Table 1 uses
    python scripts/train_agentreuse.py --labels semantic  the reader of Table 9's last row

It learns from data/train/agentreuse.jsonl and is checked on data/train/agentreuse-check.jsonl:
3 epochs, AdamW at 5e-5, batch 32, linear warm-up over the first 10% of steps, seed 0, on CPU.
Fine-tuning bert-base takes about an hour on 16 cores and needs 16 GB of memory.
Writes models/agentreuse/ (or models/agentreuse-semantic/).
"""

import argparse
import collections
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from groundhog.systems.agentreuse import (  # noqa: E402
    CHECK,
    MODELS,
    READER,
    REVISIONS,
    TRAINED,
    TRAINING,
    WINDOW,
    decode,
    examples,
    joint,
    tag_names,
    training_digest,
)

SEED = 0
DEVICE = "cpu"
LR = 5e-5
BATCH = 32
EPOCHS = 3
WARMUP = 0.1
CLIP = 1.0
DECAY = 0.01
# Most tokens in one forward pass; a batch larger than this is run in pieces.
BUDGET = 2048


def say(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def aligned(offsets, spans, tags):
    """Slot labels for each token: B- or I- of the span it falls in, O outside, -100 for
    special tokens."""
    labels, inside = [], None
    for a, b in offsets:
        if a == b:
            labels.append(-100)
            continue
        span = next((s for s in spans if s[0] <= a and b <= s[1]), None)
        if span is None:
            labels.append(tags.index("O"))
        else:
            labels.append(tags.index(("I-" if span == inside else "B-") + span[2]))
        inside = span
    return labels


def views(tokenizer, example, tags):
    """A request too long for the window is trained on twice: its head and its tail."""
    whole = len(tokenizer(example["text"])["input_ids"])
    out = []
    for side in ("right", "left") if whole > WINDOW else ("right",):
        tokenizer.truncation_side = side
        got = tokenizer(
            example["text"], truncation=True, max_length=WINDOW, return_offsets_mapping=True
        )
        out.append((got["input_ids"], aligned(got["offset_mapping"], example["spans"], tags)))
    tokenizer.truncation_side = "right"
    return out


def padded(rows, fill):
    width = max(len(r) for r in rows)
    return [r + [fill] * (width - len(r)) for r in rows]


def pieces(rows):
    out, current = [], []
    for row in sorted(rows, key=lambda r: len(r[0])):
        if current and len(row[0]) * (len(current) + 1) > BUDGET:
            out.append(current)
            current = []
        current.append(row)
    return out + [current] if current else out


def read_all(model, tokenizer, examples, intents, tags, side):
    import torch

    tokenizer.truncation_side = "left" if side == "tail" else "right"
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(examples), 64):
            texts = [e["text"] for e in examples[start : start + 64]]
            got = tokenizer(
                texts,
                truncation=True,
                max_length=WINDOW,
                padding=True,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            intent, slots = model(got["input_ids"], got["attention_mask"])
            for row in range(len(texts)):
                labels = [tags[i] for i in slots[row].argmax(-1).tolist()]
                out.append(
                    (
                        intents[int(intent[row].argmax())],
                        decode(got["offset_mapping"][row].tolist(), labels),
                    )
                )
    tokenizer.truncation_side = "right"
    return out


def scored(examples, predictions):
    right = sum(p[0] == e["intent"] for e, p in zip(examples, predictions))
    gold = {(i, *s) for i, e in enumerate(examples) for s in e["spans"]}
    said = {(i, *s) for i, p in enumerate(predictions) for s in p[1]}
    both = len(gold & said)
    precision = both / len(said) if said else 0.0
    recall = both / len(gold) if gold else 0.0
    return {
        "intent_accuracy": round(right / len(examples), 4),
        "slot_precision": round(precision, 4),
        "slot_recall": round(recall, 4),
        "slot_f1": round(2 * precision * recall / (precision + recall), 4) if both else 0.0,
    }


def held_out(model, tokenizer, train, check, intents, tags):
    seen = frozenset(e["text"] for e in train)
    held = {}
    for side in ("head", "tail"):
        held[side] = scored(check, read_all(model, tokenizer, check, intents, tags, side))
        held[side]["check_texts_also_in_training"] = sum(e["text"] in seen for e in check)
        say(
            f"check set, {side} kept: intent accuracy {held[side]['intent_accuracy']:.1%}, "
            f"slot F1 {held[side]['slot_f1']:.3f}"
        )
    return held


def environment():
    import torch
    import transformers

    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "threads": torch.get_num_threads(),
    }


def main():
    parser = argparse.ArgumentParser(prog="python scripts/train_agentreuse.py")
    parser.add_argument(
        "--labels",
        choices=("outcome", "semantic"),
        default="outcome",
        help="semantic labels a refusal with the action it mentions",
    )
    args = parser.parse_args()

    import numpy as np
    import torch
    from transformers import BertModel, BertTokenizerFast, get_linear_schedule_with_warmup

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    train = examples(TRAINING, args.labels)
    check = examples(CHECK, args.labels)
    digest = training_digest(args.labels)
    say(f"{len(train)} training and {len(check)} check requests; digest {digest}")
    intents = sorted({e["intent"] for e in train})
    kinds = sorted({s[2] for e in train for s in e["spans"]})
    tags = tag_names(kinds)
    out = os.path.join(MODELS, TRAINED[args.labels])
    os.makedirs(out, exist_ok=True)

    tokenizer = BertTokenizerFast.from_pretrained(READER, revision=REVISIONS[READER])
    data = [
        (ids, slots, intents.index(e["intent"]))
        for e in train
        for ids, slots in views(tokenizer, e, tags)
    ]
    say(f"{len(intents)} intents, {len(kinds)} slot types, {len(data)} training views")

    model = joint(
        BertModel.from_pretrained(READER, revision=REVISIONS[READER]), len(intents), len(tags)
    ).to(DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=DECAY)
    steps = EPOCHS * math.ceil(len(data) / BATCH)
    schedule = get_linear_schedule_with_warmup(optimiser, int(WARMUP * steps), steps)
    summed = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    order = random.Random(SEED)
    seconds = []
    for epoch in range(EPOCHS):
        model.train()
        began = time.time()
        shuffled = list(range(len(data)))
        order.shuffle(shuffled)
        total = 0.0
        for k in range(0, len(shuffled), BATCH):
            rows = [data[i] for i in shuffled[k : k + BATCH]]
            labelled = sum(1 for r in rows for tag in r[1] if tag != -100)
            total_batch = 0.0
            for piece in pieces(rows):
                ids = torch.tensor(padded([r[0] for r in piece], tokenizer.pad_token_id))
                mask = (ids != tokenizer.pad_token_id).long()
                slot_gold = torch.tensor(padded([r[1] for r in piece], -100))
                intent_gold = torch.tensor([r[2] for r in piece])
                intent, slots = model(ids, mask)
                loss = (
                    summed(intent, intent_gold) / len(rows)
                    + summed(slots.reshape(-1, len(tags)), slot_gold.reshape(-1)) / labelled
                )
                loss.backward()
                total_batch += loss.item()
            loss = torch.tensor(total_batch)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimiser.step()
            schedule.step()
            optimiser.zero_grad()
            total += loss.item() * len(rows)
        seconds.append(round(time.time() - began, 1))
        say(f"epoch {epoch + 1}/{EPOCHS}: {seconds[-1]:.0f}s, mean loss {total / len(data):.4f}")

    held = held_out(model, tokenizer, train, check, intents, tags)

    tokenizer.save_pretrained(out)
    model.encoder.save_pretrained(os.path.join(out, "encoder"))
    torch.save(
        {"intent": model.intent.state_dict(), "slot": model.slot.state_dict()},
        os.path.join(out, "heads.pt"),
    )
    meta = {
        "labels": args.labels,
        "training_digest": digest,
        "reader": READER,
        "reader_revision": REVISIONS[READER],
        "window": WINDOW,
        "intents": intents,
        "slot_types": kinds,
        "tags": tags,
        "training_set": TRAINING,
        "check_set": CHECK,
        "sizes": {
            "train": len(train),
            "train_views": len(data),
            "check": len(check),
            "train_by_intent": dict(
                sorted(collections.Counter(e["intent"] for e in train).items())
            ),
        },
        "hyperparameters": {
            "optimiser": "AdamW",
            "lr": LR,
            "weight_decay": DECAY,
            "batch": BATCH,
            "epochs": EPOCHS,
            "warmup": WARMUP,
            "schedule": "linear",
            "clip": CLIP,
            "seed": SEED,
            "device": DEVICE,
            "loss": "intent CE + slot CE, equal weights",
        },
        "seconds_per_epoch": seconds,
        "check": held,
        "environment": environment(),
    }
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    say(f"written to {out}")


if __name__ == "__main__":
    main()
