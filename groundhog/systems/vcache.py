"""vCache, run from its authors' library (github.com/vcache-project/vCache).

The library makes every decision: VerifiedDecisionPolicy learns a threshold for each
cached prompt, decides whether to serve or ask the model, and labels its observations with
the similarity evaluator it is configured with. This adapter supplies what VCacheConfig
asks a user for: an inference engine that forwards to the model under test, an embedding
engine around GPTCache's ONNX model, and the library's string comparison, applied to
replies with wire ids and usage removed. The library is installed unmodified; its licence
(CC BY-NC-ND 3.0) does not allow distributing a modified copy.

The opening turn of a conversation is served or forwarded; later turns are forwarded.

    GROUNDHOG_VCACHE_DELTA   the error bound; 0.01 by default
"""

import json
import os
import threading

from ..server import LocalServer
from .embedding import onnx_embedder
from .proxy import ProxyHandler, forward, opening_of, refuse_unknown_knobs, strip_ids, zero_usage


def decision_of(out):
    """A reply as vCache compares it: its content without ids or usage."""
    if "choices" in out:
        body = [c.get("message") for c in out["choices"]]
    else:
        body = out.get("content")
    return json.dumps(strip_ids(body), sort_keys=True)


def engines():
    from vcache.inference_engine.inference_engine import InferenceEngine
    from vcache.vcache_core.cache.embedding_engine.embedding_engine import EmbeddingEngine

    class ModelUnderTest(InferenceEngine):
        """Answers vCache's inference call by forwarding the request being handled."""

        def __init__(self, server):
            self.server = server
            self.local = threading.local()

        def create(self, prompt, system_prompt=None):
            raw, path = self.local.pending
            out = forward(self.server.upstream, raw, path)
            self.local.answered = out
            key = decision_of(out)
            self.server.replies.setdefault(key, out)
            return key

    class OnnxEmbedding(EmbeddingEngine):
        def __init__(self):
            self.embed = onnx_embedder()

        def get_embedding(self, text):
            return [float(x) for x in self.embed(text)]

    return ModelUnderTest, OnnxEmbedding


class VCache(LocalServer):
    def __init__(self, port, upstream):
        from vcache import VCache as Library
        from vcache import VCacheConfig, VerifiedDecisionPolicy
        from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.strategies.in_memory import (
            InMemoryEmbeddingMetadataStorage,
        )
        from vcache.vcache_core.cache.embedding_store.vector_db.strategies.hnsw_lib import (
            HNSWLibVectorDB,
        )
        from vcache.vcache_core.similarity_evaluator.strategies.string_comparison import (
            StringComparisonSimilarityEvaluator,
        )

        super().__init__(("127.0.0.1", port), Handler)
        refuse_unknown_knobs(os.environ, "GROUNDHOG_VCACHE_", ("DELTA",))
        delta = float(os.environ.get("GROUNDHOG_VCACHE_DELTA", "0.01"))
        self.upstream = upstream
        self.replies = {}
        ModelUnderTest, OnnxEmbedding = engines()
        self.engine = ModelUnderTest(self)
        config = VCacheConfig(
            inference_engine=self.engine,
            embedding_engine=OnnxEmbedding(),
            vector_db=HNSWLibVectorDB(),
            embedding_metadata_storage=InMemoryEmbeddingMetadataStorage(),
            similarity_evaluator=StringComparisonSimilarityEvaluator(),
        )
        self.policy = VerifiedDecisionPolicy(delta=delta)
        self.vcache = Library(config=config, policy=self.policy)
        self.lock = threading.Lock()

    def shutdown(self):
        self.policy.shutdown()
        super().shutdown()


class Handler(ProxyHandler):
    def do_POST(self):
        raw = self.rfile.read(int(self.headers["Content-Length"]))
        body = json.loads(raw)
        turns = sum(1 for m in body["messages"] if m.get("role") == "assistant")
        server = self.server
        if turns != 0:
            self.reply(forward(server.upstream, raw, self.path))
            return
        with server.lock:
            server.engine.local.pending = (raw, self.path)
            server.engine.local.answered = None
            hit, key, _, _ = server.vcache.infer_with_cache_info(opening_of(body))
            if hit:
                out = json.loads(json.dumps(server.replies[key]))
                zero_usage(out)
            else:
                out = server.engine.local.answered
        self.reply(out)
