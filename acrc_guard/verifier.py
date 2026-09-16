"""Claim-level verification: is every sentence of the answer entailed by the evidence?"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .generator import ABSTAIN
from .text import content_tokens, split_sentences

log = logging.getLogger(__name__)


@dataclass
class Claim:
    text: str
    supported: bool
    score: float
    evidence_id: str | None


class NLIVerifier:
    """Cross-encoder NLI model (default: cross-encoder/nli-deberta-v3-base)."""

    def __init__(self, model_name: str, threshold: float):
        from sentence_transformers import CrossEncoder

        self.name = model_name
        self.threshold = threshold
        self.model = CrossEncoder(model_name)
        config = getattr(self.model, "config", None) or self.model.model.config
        labels = {v.lower(): int(k) for k, v in config.id2label.items()}
        self.entail_idx = labels.get("entailment", 1)

    def support(self, claim: str, evidence: list[str]) -> np.ndarray:
        logits = np.asarray(self.model.predict([(e, claim) for e in evidence]))
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        return probs[:, self.entail_idx]


class LexicalVerifier:
    """Offline fallback: share of the claim's content words found in a passage."""

    name = "lexical"

    def __init__(self, threshold: float):
        self.threshold = max(threshold, 0.6)

    def support(self, claim: str, evidence: list[str]) -> np.ndarray:
        c = set(content_tokens(claim))
        if not c:
            return np.ones(len(evidence))
        return np.array([len(c & set(content_tokens(e))) / len(c) for e in evidence])


class Verifier:
    def __init__(self, backend):
        self.backend = backend

    def verify(self, answer: str, evidence: list[tuple[str, str]]) -> list[Claim]:
        """evidence: list of (passage_id, text)."""
        if answer.strip() == ABSTAIN:
            return []
        claims = split_sentences(answer) or [answer]
        if not evidence:
            return [Claim(c, False, 0.0, None) for c in claims]
        ids, texts = zip(*evidence)
        out = []
        for c in claims:
            scores = self.backend.support(c, list(texts))
            best = int(np.argmax(scores))
            s = float(scores[best])
            out.append(Claim(c, s >= self.backend.threshold, round(s, 3), ids[best]))
        return out


def load_verifier(cfg) -> Verifier:
    if cfg.verifier == "lexical":
        return Verifier(LexicalVerifier(cfg.support_threshold))
    try:
        return Verifier(NLIVerifier(cfg.verifier, cfg.support_threshold))
    except Exception as exc:
        log.warning("Could not load '%s' (%s). Falling back to lexical verifier.", cfg.verifier, exc)
        return Verifier(LexicalVerifier(cfg.support_threshold))