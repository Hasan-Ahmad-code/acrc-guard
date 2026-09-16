"""Vector index over a list of passages (FAISS when available, NumPy otherwise)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None


@dataclass
class Passage:
    id: str
    text: str
    title: str = ""


class Retriever:
    def __init__(self, embedder, passages: list[Passage]):
        if not passages:
            raise ValueError("Corpus is empty")
        self.embedder = embedder
        self.passages = passages
        texts = [p.text for p in passages]
        self.embedder.fit(texts)
        self.matrix = embedder.encode(texts)
        if faiss is not None:
            self.index = faiss.IndexFlatIP(self.matrix.shape[1])
            self.index.add(self.matrix)
        else:
            self.index = None

    def search(self, query: str, k: int) -> list[tuple[Passage, float]]:
        k = min(k, len(self.passages))
        q = self.embedder.encode([query])
        if self.index is not None:
            scores, ids = self.index.search(q, k)
            pairs = zip(ids[0], scores[0])
        else:
            sims = self.matrix @ q[0]
            top = np.argsort(-sims)[:k]
            pairs = zip(top, sims[top])
        return [(self.passages[i], float(s)) for i, s in pairs if i >= 0]

    def pairwise(self, passages: list[Passage]) -> np.ndarray:
        vecs = self.embedder.encode([p.text for p in passages])
        return vecs @ vecs.T
