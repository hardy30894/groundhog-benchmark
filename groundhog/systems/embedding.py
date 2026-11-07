"""GPTCache's ONNX embedding model (paraphrase-albert-small-v2), shared by the systems that
embed requests, and cosine similarity."""

import math

# The model's window. Longer input is truncated to it.
WINDOW = 512


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def onnx_embedder():
    """GPTCache's own embedder, with truncation to the model's window switched on."""
    from transformers import PreTrainedTokenizerBase

    def encode_plus(tokenizer, text, **kwargs):
        kwargs.setdefault("return_token_type_ids", True)
        kwargs.setdefault("truncation", True)
        kwargs.setdefault("max_length", WINDOW)
        return tokenizer(text, **kwargs)

    PreTrainedTokenizerBase.encode_plus = encode_plus
    from gptcache.embedding import Onnx

    return Onnx().to_embeddings
