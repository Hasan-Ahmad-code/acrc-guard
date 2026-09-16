"""Small, dependency-free text utilities."""

import re
from difflib import SequenceMatcher

STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "by", "with", "from", "and", "or",
    "is", "are", "was", "were", "be", "been", "being", "it", "its", "this", "that", "these",
    "those", "as", "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "do", "does", "did", "has", "have", "had", "not", "no", "yes", "into", "than", "then",
    "there", "their", "they", "he", "she", "his", "her", "we", "you", "i", "can", "could",
    "would", "should", "will", "about", "also", "known", "called",
}

_TOKEN = re.compile(r"[a-z0-9]+")
_SENT = re.compile(r"(?<=[.!?])\s+")


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"(?<=\d),(?=\d)", "", text)  # 12,500 -> 12500
    return " ".join(_TOKEN.findall(text))


def tokens(text: str) -> list[str]:
    return normalize(text).split()


def content_tokens(text: str) -> list[str]:
    return [t for t in tokens(text) if t not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text.strip()) if s.strip()]


def contains_answer(response: str, answers: list[str]) -> bool:
    """Token-level containment, so 'au' does not match inside 'because'."""
    resp = f" {normalize(response)} "
    return any(f" {normalize(a)} " in resp for a in answers if normalize(a))


def query_copy_score(query: str, passage: str) -> float:
    """Share of the query covered by its longest verbatim run inside the passage."""
    q, p = tokens(query), tokens(passage)
    if not q or not p:
        return 0.0
    m = SequenceMatcher(None, q, p, autojunk=False).find_longest_match(0, len(q), 0, len(p))
    score = m.size / len(q)
    # Genuine evidence often restates the tail of a question ("... is the capital of X")
    # but rarely copies its interrogative opening ("What is ..."). Runs that do not start
    # at the first question token are therefore down-weighted.
    return score if m.a == 0 else 0.5 * score


def coverage(query: str, text: str) -> float:
    """Share of the query's content words that appear in the text."""
    q = set(content_tokens(query))
    if not q:
        return 0.0
    return len(q & set(content_tokens(text))) / len(q)


def chunk_text(text: str, max_words: int = 120, overlap: int = 20) -> list[str]:
    """Sentence-aware chunking for user documents."""
    chunks, current = [], []
    for sent in split_sentences(text):
        words = sent.split()
        if current and len(current) + len(words) > max_words:
            chunks.append(" ".join(current))
            current = current[-overlap:] if overlap else []
        current.extend(words)
    if current:
        chunks.append(" ".join(current))
    return chunks
