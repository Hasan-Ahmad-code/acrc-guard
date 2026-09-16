"""Defense layer: (1) poison/noise filter, (2) adaptive context reliance controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import GuardConfig
from .retriever import Passage
from .text import coverage, query_copy_score


@dataclass
class ScoredPassage:
    passage: Passage
    retrieval_score: float
    query_copy: float = 0.0
    cluster: float = 0.0
    suspicion: float = 0.0
    flagged: bool = False


class PoisonFilter:
    """Scores every retrieved passage with two attack signals.

    * query_copy - poisoned passages copy the question verbatim to win retrieval;
      genuine evidence paraphrases it. Measured as the longest verbatim run of the
      question found in the passage, divided by question length.
    * cluster    - attackers inject several templated passages that are near-duplicates
      of each other. Measured as the share of `min_cluster_size` neighbours whose
      cosine similarity exceeds the duplicate threshold.
    """

    def __init__(self, config: GuardConfig, dup_threshold: float):
        self.cfg = config
        self.dup_threshold = config.dup_threshold or dup_threshold

    def score(self, query: str, hits: list[tuple[Passage, float]], sim: np.ndarray) -> list[ScoredPassage]:
        out = []
        for i, (p, s) in enumerate(hits):
            neighbours = int((sim[i] >= self.dup_threshold).sum()) - 1  # exclude self
            cluster = min(1.0, neighbours / max(1, self.cfg.min_cluster_size))
            qc = query_copy_score(query, p.text)
            suspicion = self.cfg.query_copy_weight * qc + self.cfg.cluster_weight * cluster
            out.append(
                ScoredPassage(
                    passage=p,
                    retrieval_score=s,
                    query_copy=round(qc, 3),
                    cluster=round(cluster, 3),
                    suspicion=round(suspicion, 3),
                    flagged=suspicion >= self.cfg.suspicion_threshold,
                )
            )
        return out


class RelianceController:
    """Adaptive Context Reliance Control (ACRC).

    Estimates how much the surviving context can be trusted and picks a mode:
      * "context"    - evidence covers the question well -> answer strictly from context
      * "hybrid"     - partial coverage -> combine context with model knowledge
      * "parametric" - evidence is weak or mostly filtered -> ignore context
    """

    def __init__(self, config: GuardConfig):
        self.cfg = config

    def assess(self, query: str, kept: list[ScoredPassage], n_retrieved: int, n_flagged: int) -> tuple[float, str]:
        if not kept:
            return 0.0, "parametric"
        best_cov = max(coverage(query, sp.passage.text) for sp in kept[:3])
        clean_ratio = 1 - n_flagged / max(1, n_retrieved)
        quality = best_cov * (0.7 + 0.3 * clean_ratio)
        if quality >= self.cfg.high_quality:
            mode = "context"
        elif quality >= self.cfg.low_quality:
            mode = "hybrid"
        else:
            mode = "parametric"
        return round(quality, 3), mode
