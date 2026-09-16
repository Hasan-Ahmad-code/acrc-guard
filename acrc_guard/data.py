"""Dataset and document loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .retriever import Passage
from .text import chunk_text


@dataclass
class QAItem:
    id: str
    question: str
    answers: list[str]
    target: str  # attacker's wrong answer


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_dataset(data_dir: str | Path) -> tuple[list[Passage], list[QAItem]]:
    """Expects `corpus.jsonl` ({id, text, title?}) and `qa.jsonl` ({id, question, answers, target})."""
    data_dir = Path(data_dir)
    corpus = [Passage(id=str(r["id"]), text=r["text"], title=r.get("title", "")) for r in read_jsonl(data_dir / "corpus.jsonl")]
    qa = [QAItem(str(r["id"]), r["question"], list(r["answers"]), r["target"]) for r in read_jsonl(data_dir / "qa.jsonl")]
    return corpus, qa


def read_document(name: str, raw: bytes) -> str:
    if name.lower().endswith(".pdf"):
        import io

        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(raw)).pages)
    return raw.decode("utf-8", errors="ignore")


def documents_to_passages(docs: dict[str, str], max_words: int = 120) -> list[Passage]:
    passages = []
    for name, text in docs.items():
        for i, chunk in enumerate(chunk_text(text, max_words=max_words)):
            passages.append(Passage(id=f"{name}::{i}", text=chunk, title=name))
    return passages
