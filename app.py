"""Interactive demo: Vanilla RAG vs ACRC-Guard side by side.

    streamlit run app.py
"""

import hashlib
from pathlib import Path

import streamlit as st

from acrc_guard import GuardConfig, GuardedRAG, VanillaRAG
from acrc_guard.attacks import is_poison, make_poison
from acrc_guard.data import documents_to_passages, load_dataset, read_document
from acrc_guard.pipeline import Components

SAMPLE_DIR = Path(__file__).parent / "data" / "sample"

st.set_page_config(page_title="ACRC-Guard", page_icon="🛡️", layout="wide")
st.title("🛡️ ACRC-Guard")
st.caption("Poisoning-robust, evidence-grounded RAG · compare a vanilla pipeline with the guarded one")

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Backends")
    embedder = st.selectbox("Embedder", ["sentence-transformers/all-MiniLM-L6-v2", "tfidf"])
    verifier = st.selectbox("Claim verifier", ["cross-encoder/nli-deberta-v3-base", "lexical"])
    generator = st.selectbox("Generator", ["extractive", "hf", "openrouter"],
                             help="openrouter needs OPENROUTER_API_KEY in .env")
    st.header("Knowledge base")
    source = st.radio("Documents", ["Sample corpus", "Upload files"])
    uploads = st.file_uploader("TXT / MD / PDF", type=["txt", "md", "pdf"], accept_multiple_files=True) \
        if source == "Upload files" else None
    st.header("Attack simulation")
    n_poison = st.slider("Poisoned passages to inject", 0, 5, 3)
    target = st.text_input("Attacker's target (wrong) answer", "Sydney")


def base_passages():
    if source == "Sample corpus":
        return load_dataset(SAMPLE_DIR)[0]
    if not uploads:
        return []
    return documents_to_passages({f.name: read_document(f.name, f.getvalue()) for f in uploads})


@st.cache_resource(show_spinner="Loading models and building the index...")
def get_components(cfg_key: tuple, corpus_key: str, _passages):
    emb, ver, gen = cfg_key
    return Components(GuardConfig(embedder=emb, verifier=ver, generator=gen), _passages)


question = st.text_input("Ask a question", "What is the capital city of Australia?")
run = st.button("Run both pipelines", type="primary")

if run:
    passages = base_passages()
    if not passages:
        st.warning("Upload at least one document or switch to the sample corpus.")
        st.stop()
    poison = make_poison(question, target, n_poison, qid="demo") if n_poison and target else []
    corpus = passages + poison
    key = hashlib.md5("||".join(p.text for p in corpus).encode()).hexdigest()
    try:
        comp = get_components((embedder, verifier, generator), key, corpus)
    except Exception as exc:
        st.error(f"Could not load backends: {exc}")
        st.stop()

    cols = st.columns(2)
    for col, (label, cls) in zip(cols, [("Vanilla RAG", VanillaRAG), ("ACRC-Guard", GuardedRAG)]):
        with col:
            st.subheader(label)
            r = cls(comp).run(question)
            st.markdown(f"**Answer:** {r.answer}")
            meta = f"mode `{r.mode}` · latency {r.latency_s * 1000:.0f} ms"
            if r.quality is not None:
                meta += f" · context quality {r.quality:.2f}"
            st.caption(meta)
            if r.claims:
                st.markdown("**Claim verification**")
                for c in r.claims:
                    icon = "✅" if c.supported else "⚠️"
                    st.markdown(f"{icon} {c.text}  \n<small>support {c.score:.2f} · evidence `{c.evidence_id}`</small>",
                                unsafe_allow_html=True)
            poisoned_used = sum(is_poison(i) for i in r.used_ids)
            (st.error if poisoned_used else st.success)(f"Poisoned passages given to the generator: {poisoned_used}")
            with st.expander(f"Retrieved passages ({len(r.passages)})"):
                for p in r.passages:
                    tag = "🚩 flagged" if p.get("flagged") else ("🧪 poison" if is_poison(p["id"]) else "")
                    extra = f" · suspicion {p['suspicion']}" if "suspicion" in p else ""
                    st.markdown(f"`{p['id']}` score {p['retrieval_score']}{extra} {tag}")
                    st.write(p["text"])
