"""A deterministic stand-in for a real embedding model: words in the same concept group embed alike.

It lets tests assert semantic behaviour ("login" finds authenticate_user although they share no word)
without downloading or running a model. Select it with CODEGRAPH_EMBED=tests.fake_embed:FakeEmbedder.
"""
import re

import numpy as np

CONCEPTS = [
    {"login", "signin", "authenticate", "auth", "credentials", "session"},
    {"pay", "charge", "billing", "invoice", "checkout", "payment"},
    {"parse", "decode", "deserialize", "read", "load"},
    {"send", "mail", "email", "notify", "message"},
]
HASH_DIMS = 16


class FakeEmbedder:
    name = "fake:concepts-1"
    calls = 0  # texts embedded so far, across instances: lets tests assert what was (not) re-embedded

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(len(CONCEPTS) + HASH_DIMS, dtype=np.float32)
        for w in re.findall(r"[a-z]+", text.lower()):
            hit = [i for i, c in enumerate(CONCEPTS) if w in c]
            for i in hit:
                v[i] += 1.0
            if not hit:
                v[len(CONCEPTS) + sum(map(ord, w)) % HASH_DIMS] += 0.3
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_passages(self, texts):
        FakeEmbedder.calls += len(texts)
        return np.vstack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)
