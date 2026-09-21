"""
PTCL Packages Assistant — app.py (Part 1 placeholder)
Developer: Areeba Imran

This is a MINIMAL placeholder only. The full Streamlit chat UI (RAG
retrieval, Groq LLM chat, voice input/output, etc.) will be built in
Part 2, on top of the artifacts produced by:

    scripts/build_knowledge_base.py

This placeholder simply checks whether the knowledge base has been
built, and if so, reports basic stats so you can confirm Part 1 worked
before moving on to Part 2.

Run:
    streamlit run app.py
"""

import pickle
import json
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks.pkl"
METADATA_PATH = PROJECT_ROOT / "data" / "metadata.pkl"
MANIFEST_PATH = PROJECT_ROOT / "data" / "documents_manifest.json"
FAISS_INDEX_PATH = PROJECT_ROOT / "faiss_index" / "index.faiss"

st.set_page_config(page_title="PTCL Packages Assistant", page_icon="📶")

st.title("📶 PTCL Packages Assistant")
st.caption("Part 1 placeholder — knowledge base status check")

st.markdown(
    "This is a temporary placeholder screen. The full chat assistant "
    "(retrieval + LLM answers) will be added in **Part 2**."
)

st.divider()
st.subheader("Knowledge base status")

artifacts_exist = (
    CHUNKS_PATH.exists() and METADATA_PATH.exists() and FAISS_INDEX_PATH.exists()
)

if not artifacts_exist:
    st.error(
        "Knowledge base has not been built yet.\n\n"
        "Run this first:\n\n"
        "`python scripts/build_knowledge_base.py`"
    )
else:
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    manifest = {}
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    st.success("Knowledge base found and loaded successfully.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total chunks", len(chunks))
    col2.metric("Documents indexed", len(manifest.get("documents", [])))
    col3.metric("Embedding dimension", manifest.get("embedding_dimension", "—"))

    with st.expander("Source documents"):
        for doc in manifest.get("documents", []):
            st.write(f"- {doc.get('filename')} ({doc.get('size_bytes', 0):,} bytes)")

    with st.expander("Sample chunk (first chunk + its metadata)"):
        if chunks:
            st.text(chunks[0][:500] + ("..." if len(chunks[0]) > 500 else ""))
            st.json(metadata[0])

st.divider()
st.info("Chat interface, retrieval, and LLM responses arrive in Part 2.")
