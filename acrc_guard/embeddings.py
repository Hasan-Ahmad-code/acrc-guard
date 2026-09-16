"""Embedding backends. All return L2-normalised float32 matrices."""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def _l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype="float32")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


class SentenceTransformerEmbedder:
    """Dense embeddings (default: all-MiniLM-L6-v2)."""

    dup_threshold = 0.85

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.model = SentenceTransformer(model_name)

    def fit(self, texts: list[str]) -> "SentenceTransformerEmbedder":
        return self  # pretrained, nothing to fit

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, batch_size=64, show_progress_bar=False)
        return _l2(vecs)


class TfidfEmbedder:
    """Offline sparse fallback. Needs no model download; useful for CI and quick tests."""

    name = "tfidf"
    dup_threshold = 0.6

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        self._fitted = False

    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        self.vectorizer.fit(texts)
        self._fitted = True
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder.fit() must be called on the corpus first")
        return _l2(self.vectorizer.transform(texts).toarray())


def load_embedder(name: str):
    if name == "tfidf":
        return TfidfEmbedder()
    try:
        return SentenceTransformerEmbedder(name)
    except Exception as exc:  # missing package or no internet
        log.warning("Could not load '%s' (%s). Falling back to TF-IDF.", name, exc)
        return TfidfEmbedder()
