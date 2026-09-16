"""Vanilla RAG baseline and the ACRC-Guard pipeline.

Heavy components (embedder, generator, verifier) are loaded once via `Components`
and shared by both pipelines so comparisons are fair and cheap.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from .config import GuardConfig
from .defense import PoisonFilter, RelianceController, ScoredPassage
from .embeddings import load_embedder
from .generator import load_generator
from .retriever import Passage, Retriever
from .verifier import Claim, load_verifier


@dataclass
class RAGResult:
    question: str
    answer: str
    mode: str
    quality: float | None
    passages: list[dict] = field(default_factory=list)  # every retrieved passage + scores
    used_ids: list[str] = field(default_factory=list)  # passages given to the generator
    claims: list[Claim] = field(default_factory=list)
    latency_s: float = 0.0

    @property
    def unsupported_rate(self) -> float | None:
        if not self.claims:
            return None
        return sum(not c.supported for c in self.claims) / len(self.claims)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["unsupported_rate"] = self.unsupported_rate
        return d


class Components:
    def __init__(self, config: GuardConfig, passages: list[Passage]):
        self.config = config
        self.embedder = load_embedder(config.embedder)
        self.retriever = Retriever(self.embedder, passages)
        self.generator = load_generator(config)
        self.verifier = load_verifier(config) if config.verify else None

    def reindex(self, passages: list[Passage]) -> None:
        """Rebuild the index (e.g. after injecting poison) without reloading models."""
        self.retriever = Retriever(self.embedder, passages)


class VanillaRAG:
    """Retrieve top-k -> generate -> (verify for evaluation only)."""

    def __init__(self, components: Components):
        self.c = components

    def run(self, question: str) -> RAGResult:
        t0 = time.perf_counter()
        cfg = self.c.config
        hits = self.c.retriever.search(question, cfg.top_k)
        contexts = [p.text for p, _ in hits]
        answer = self.c.generator.generate(question, contexts, "vanilla")
        claims = self.c.verifier.verify(answer, [(p.id, p.text) for p, _ in hits], question) if self.c.verifier else []
        return RAGResult(
            question=question,
            answer=answer,
            mode="vanilla",
            quality=None,
            passages=[{"id": p.id, "text": p.text, "retrieval_score": round(s, 4)} for p, s in hits],
            used_ids=[p.id for p, _ in hits],
            claims=claims,
            latency_s=round(time.perf_counter() - t0, 4),
        )


class GuardedRAG:
    """Retrieve -> filter poison/noise -> adaptive reliance -> generate -> verify claims."""

    def __init__(self, components: Components):
        self.c = components
        self.reliance = RelianceController(components.config)

    def run(self, question: str) -> RAGResult:
        t0 = time.perf_counter()
        cfg = self.c.config
        retriever = self.c.retriever
        pfilter = PoisonFilter(cfg, self.c.embedder.dup_threshold)

        # Adaptive re-retrieval: if poison floods the candidates, widen the search
        # (up to 2 times) so genuine evidence ranked lower can still be recovered.
        k = cfg.retrieve_k
        for _ in range(3):
            hits = retriever.search(question, k)
            scored: list[ScoredPassage] = pfilter.score(question, hits, retriever.pairwise([p for p, _ in hits]))
            kept = [sp for sp in scored if not sp.flagged][: cfg.top_k]
            n_flagged = sum(sp.flagged for sp in scored)
            if len(kept) >= cfg.top_k or n_flagged == 0 or k >= len(retriever.passages):
                break
            k *= 3

        quality, mode = self.reliance.assess(question, kept, len(scored), n_flagged)
        contexts = [] if mode == "parametric" else [sp.passage.text for sp in kept]
        answer = self.c.generator.generate(question, contexts, mode)

        claims: list[Claim] = []
        if self.c.verifier:
            claims = self.c.verifier.verify(answer, [(sp.passage.id, sp.passage.text) for sp in kept], question)

        return RAGResult(
            question=question,
            answer=answer,
            mode=mode,
            quality=quality,
            passages=[
                {
                    "id": sp.passage.id,
                    "text": sp.passage.text,
                    "retrieval_score": round(sp.retrieval_score, 4),
                    "query_copy": sp.query_copy,
                    "cluster": sp.cluster,
                    "suspicion": sp.suspicion,
                    "flagged": sp.flagged,
                }
                for sp in scored
            ],
            used_ids=[sp.passage.id for sp in kept] if contexts else [],
            claims=claims,
            latency_s=round(time.perf_counter() - t0, 4),
        )
