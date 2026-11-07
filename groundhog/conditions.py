"""The five conditions a cache must get right before it may serve, and their families."""

CONDITIONS = {
    "identity": ("lookup", "buried", "mixed"),
    "values": ("chain", "transform"),
    "meaning": ("judgement", "reversal"),
    "world": ("drift", "quiet", "standing"),
    "depth": ("horizon",),
}

QUESTIONS = {
    "identity": "is this the same job",
    "values": "where does the value come from",
    "meaning": "what does the request mean",
    "world": "is what was learned still true",
    "depth": "how far does reuse survive",
}

FAMILIES = tuple(family for members in CONDITIONS.values() for family in members)


def condition_of(family):
    for condition, members in CONDITIONS.items():
        if family in members:
            return condition
    raise KeyError(family)
