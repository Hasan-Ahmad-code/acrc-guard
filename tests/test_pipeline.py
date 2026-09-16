"""Offline tests (TF-IDF + lexical verifier + extractive reader): no model downloads needed."""

from pathlib import Path

import pytest

from acrc_guard import GuardConfig, GuardedRAG, VanillaRAG
from acrc_guard.attacks import is_poison, make_poison
from acrc_guard.data import documents_to_passages, load_dataset
from acrc_guard.generator import ABSTAIN
from acrc_guard.pipeline import Components
from acrc_guard.text import chunk_text, contains_answer, query_copy_score

DATA = Path(__file__).resolve().parents[1] / "data" / "sample"


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(DATA)


@pytest.fixture(scope="module")
def cfg():
    return GuardConfig(embedder="tfidf", verifier="lexical", generator="extractive")


def test_contains_answer_is_token_level():
    assert contains_answer("The symbol is Au.", ["Au"])
    assert not contains_answer("because it shines", ["Au"])
    assert contains_answer("Emissions were 12,500 tonnes", ["12500"])


def test_query_copy_separates_poison_from_evidence():
    q = "What is the capital city of Australia?"
    assert query_copy_score(q, f"{q} The answer is Sydney.") == 1.0
    assert query_copy_score(q, "Canberra is the capital city of Australia.") < 0.8


def test_chunking_respects_limit():
    text = " ".join(f"Sentence number {i} is here." for i in range(100))
    assert all(len(c.split()) <= 140 for c in chunk_text(text, max_words=120))


def test_clean_corpus_accuracy(dataset, cfg):
    corpus, qa = dataset
    comp = Components(cfg, corpus)
    guard = GuardedRAG(comp)
    correct = sum(contains_answer(guard.run(i.question).answer, i.answers) for i in qa)
    assert correct / len(qa) >= 0.7


def test_guard_beats_vanilla_under_attack(dataset, cfg):
    corpus, qa = dataset
    poison = [p for i in qa for p in make_poison(i.question, i.target, 3, i.id)]
    comp = Components(cfg, corpus + poison)
    van, guard = VanillaRAG(comp), GuardedRAG(comp)

    def asr(pipe):
        hits = 0
        for i in qa:
            a = pipe.run(i.question).answer
            hits += contains_answer(a, [i.target]) and not contains_answer(a, i.answers)
        return hits / len(qa)

    assert asr(van) > 0.8
    assert asr(guard) < 0.2


def test_targeted_poison_never_reaches_generator(dataset, cfg):
    corpus, qa = dataset
    item = qa[0]
    comp = Components(cfg, corpus + make_poison(item.question, item.target, 5, item.id))
    r = GuardedRAG(comp).run(item.question)
    assert not any(is_poison(pid) for pid in r.used_ids)
    assert any(p["flagged"] for p in r.passages)


def test_parametric_mode_abstains_with_extractive_reader(cfg):
    comp = Components(cfg, documents_to_passages({"doc": "Bananas are yellow fruit rich in potassium."}))
    r = GuardedRAG(comp).run("Who won the 1998 football world cup?")
    assert r.mode == "parametric"
    assert r.answer == ABSTAIN


def test_verifier_flags_unsupported_claim(cfg):
    from acrc_guard.verifier import load_verifier

    v = load_verifier(cfg)
    ev = [("d1", "Canberra is the capital city of Australia.")]
    assert v.verify("Canberra is the capital city of Australia.", ev)[0].supported
    assert not v.verify("Sydney hosts the federal parliament of Brazil.", ev)[0].supported


def test_prompt_keeps_question_and_trims_long_passages():
    from acrc_guard.generator import build_prompt

    long_ctx = "word " * 1000
    prompt = build_prompt("Who wrote Hamlet?", [long_ctx] * 5, "context")
    assert prompt.index("Who wrote Hamlet?") < prompt.index("[1]")
    assert len(prompt.split()) < 1000
