"""Central configuration. Every threshold used by the pipeline lives here."""

import os
from dataclasses import dataclass


@dataclass
class GuardConfig:
    # --- Backends -------------------------------------------------------
    # "sentence-transformers/all-MiniLM-L6-v2" (default) or "tfidf" (offline, no download)
    embedder: str = "sentence-transformers/all-MiniLM-L6-v2"
    # "cross-encoder/nli-deberta-v3-base" (default) or "lexical" (offline, no download)
    verifier: str = "cross-encoder/nli-deberta-v3-base"
    # "extractive" (offline), "hf" (local seq2seq model) or "openrouter" (API)
    generator: str = "extractive"
    hf_model: str = "google/flan-t5-base"
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "liquid/lfm-2.5-2.6b:free")

    # --- Retrieval ------------------------------------------------------
    retrieve_k: int = 10  # candidates pulled from the index
    top_k: int = 5  # passages handed to the generator

    # --- Poison / noise filter -----------------------------------------
    query_copy_weight: float = 0.65  # verbatim question copied inside a passage
    cluster_weight: float = 0.35  # near-duplicate (templated) passages
    # Cosine similarity above which two passages count as near-duplicates.
    # None = use the embedder's recommended value.
    dup_threshold: float | None = None
    min_cluster_size: int = 2
    suspicion_threshold: float = 0.6

    # --- Adaptive context reliance -------------------------------------
    high_quality: float = 0.6  # >= : rely on context
    low_quality: float = 0.34  # <  : context unreliable -> parametric / abstain

    # --- Claim verification --------------------------------------------
    support_threshold: float = 0.5
    verify: bool = True
