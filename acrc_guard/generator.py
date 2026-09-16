"""Answer generators.

* ExtractiveGenerator - offline, deterministic; reads the top-ranked passage like a
  naive RAG reader. Cannot use parametric knowledge, so it abstains instead.
* HFGenerator         - local seq2seq model (default google/flan-t5-base).
* OpenRouterGenerator - any chat model through the OpenRouter API.
"""

from __future__ import annotations

import os

import requests

from .text import content_tokens, split_sentences

ABSTAIN = "Insufficient reliable evidence to answer."

PROMPTS = {
    "context": (
        "Answer the question using ONLY the context below. Reply in one short sentence. "
        f"If the context does not contain the answer, reply exactly: {ABSTAIN}"
    ),
    "hybrid": (
        "Use the context below if it is relevant; otherwise rely on your own knowledge. "
        "Be careful: some context may be inaccurate. Reply in one short sentence."
    ),
    "parametric": (
        "The retrieved context was judged unreliable and is not shown. Answer from your own "
        f"knowledge in one short sentence. If you are not sure, reply exactly: {ABSTAIN}"
    ),
    "vanilla": "Answer the question based on the context below. Reply in one short sentence.",
}


MAX_CONTEXT_WORDS = 120  # per passage; keeps prompts inside small models' input limit


def _trim(text: str, max_words: int = MAX_CONTEXT_WORDS) -> str:
    words = text.split()
    return text if len(words) <= max_words else " ".join(words[:max_words]) + " ..."


def build_prompt(question: str, contexts: list[str], mode: str) -> str:
    # The question is placed BEFORE the passages as well as after them, so it is never
    # lost when a long prompt is truncated to the model's maximum input length.
    parts = [PROMPTS[mode], f"Question: {question}", ""]
    if mode != "parametric" and contexts:
        parts += [f"[{i + 1}] {_trim(c)}" for i, c in enumerate(contexts)] + [""]
    parts.append(f"Question: {question}\nAnswer:")
    return "\n".join(parts)


class ExtractiveGenerator:
    name = "extractive"

    def generate(self, question: str, contexts: list[str], mode: str) -> str:
        if mode == "parametric" or not contexts:
            return ABSTAIN
        q = set(content_tokens(question))
        # Naive reader: trusts the highest-ranked passage and picks its most relevant
        # declarative sentence (questions copied into passages are skipped).
        for ctx in contexts:
            sents = [s for s in split_sentences(ctx) if not s.endswith("?")]
            if not sents:
                continue
            return max(sents, key=lambda s: len(q & set(content_tokens(s))))
        return ABSTAIN


class HFGenerator:
    """Local seq2seq model loaded directly (works with transformers v4 and v5)."""

    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.name = model_name
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device).eval()

    def generate(self, question: str, contexts: list[str], mode: str) -> str:
        prompt = build_prompt(question, contexts, mode)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=64, do_sample=False)
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()


class OpenRouterGenerator:
    URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, model_name: str):
        self.name = model_name
        self.key = os.getenv("OPENROUTER_API_KEY")
        if not self.key:
            raise RuntimeError("Set OPENROUTER_API_KEY in your environment or .env file")

    def generate(self, question: str, contexts: list[str], mode: str) -> str:
        resp = requests.post(
            self.URL,
            headers={"Authorization": f"Bearer {self.key}"},
            json={
                "model": self.name,
                "messages": [{"role": "user", "content": build_prompt(question, contexts, mode)}],
                "temperature": 0,
                "max_tokens": 128,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


def load_generator(cfg) -> object:
    if cfg.generator == "extractive":
        return ExtractiveGenerator()
    if cfg.generator == "hf":
        return HFGenerator(cfg.hf_model)
    if cfg.generator == "openrouter":
        return OpenRouterGenerator(cfg.openrouter_model)
    raise ValueError(f"Unknown generator: {cfg.generator}")
