"""Trains W5H2's classifier with its authors' code and certifies its thresholds.

    git clone https://github.com/nabaos/w5h2-intent-cache third_party/w5h2-intent-cache
    git -C third_party/w5h2-intent-cache checkout 835c6d75810324741fd681afb4ce943aad235aca
    python scripts/train_w5h2.py

The classifier is the authors' hierarchical.train_setfit: SetFit on all-MiniLM-L6-v2, 8
examples per class, 1 epoch, batch 32, 10 contrastive iterations, seed 42, the paper's
settings. The 24 held-out examples per class set the threshold at each risk tolerance.
Writes models/w5h2/: the SetFit model, calibration.json and meta.json.
"""

import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from groundhog.data import DATA  # noqa: E402
from groundhog.systems.w5h2 import ALPHAS, MODELS, UPSTREAM, shots, threshold, upstream_on_path  # noqa: E402

SEED = 42


def say(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def main():
    import numpy as np
    import torch

    upstream_on_path()
    from hierarchical import get_predictions_with_confidence, train_setfit

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    MODELS.mkdir(parents=True, exist_ok=True)

    learned, held = shots()
    classes = list(learned)
    name = {c: json.dumps(list(c)) for c in classes}
    train_texts = [t for c in classes for t in learned[c]]
    train_labels = [name[c] for c in classes for _ in learned[c]]
    held_texts = [t for c in classes for t in held[c]]
    held_labels = [name[c] for c in classes for _ in held[c]]
    say(f"{len(classes)} classes, {len(train_texts)} shots, {len(held_texts)} held out")

    started = time.time()
    model = train_setfit(train_texts, train_labels, seed=SEED)
    say(f"trained in {time.time() - started:.0f}s")
    model.save_pretrained(str(MODELS / "setfit"))

    preds, confidence, _ = get_predictions_with_confidence(model, held_texts)
    correct = [p == t for p, t in zip(preds, held_labels)]
    calibration = {
        "training_set": hashlib.sha256((DATA / "train" / "w5h2.jsonl").read_bytes()).hexdigest()[
            :16
        ],
        "classes": [list(c) for c in classes],
        "held_out": len(held_texts),
        "held_out_accuracy": sum(correct) / len(correct),
        "thresholds": {f"{a:g}": threshold(confidence, correct, a) for a in ALPHAS},
    }
    (MODELS / "calibration.json").write_text(json.dumps(calibration, indent=1))
    say(
        f"held-out accuracy {calibration['held_out_accuracy']:.1%}, thresholds {calibration['thresholds']}"
    )

    commit = subprocess.run(
        ["git", "-C", UPSTREAM, "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    meta = {
        "upstream": "github.com/nabaos/w5h2-intent-cache",
        "commit": commit,
        "training": "hierarchical.train_setfit, seed 42, 8 shots per class, 1 epoch, batch 32, 10 iterations",
    }
    (MODELS / "meta.json").write_text(json.dumps(meta, indent=1))
    say(f"written to {MODELS}")


if __name__ == "__main__":
    main()
