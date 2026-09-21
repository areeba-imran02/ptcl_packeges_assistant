import os
import json
import pickle
import hashlib
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from gtts import gTTS
from io import BytesIO
from streamlit_mic_recorder import mic_recorder
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "data"
FAISS_DIR = PROJECT_DIR / "faiss_index"

CHUNKS_FILE = DATA_DIR / "chunks.pkl"
METADATA_FILE = DATA_DIR / "metadata.pkl"
MANIFEST_FILE = DATA_DIR / "documents_manifest.json"
FAISS_FILE = FAISS_DIR / "index.faiss"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# =========================================================
# WHISPER SPEECH-TO-TEXT
# =========================================================

WHISPER_MODEL_SIZE = "small"


@st.cache_resource(show_spinner=False)
def load_whisper_model():
    """
    Load Faster-Whisper once and reuse it during the
    Streamlit session/application lifetime.
    """
    return WhisperModel(
        WHISPER_MODEL_SIZE,
        device="cpu",
        compute_type="int8",
    )


def transcribe_audio(audio_bytes):
    """
    Convert recorded microphone audio bytes into text.

    Returns:
        str: Clean transcript.
    """
    if not audio_bytes:
        return ""

    import tempfile

    temp_audio_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".webm",
            delete=False,
        ) as temp_audio:

            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name

        model = load_whisper_model()

        segments, _ = model.transcribe(
            temp_audio_path,
            beam_size=5,
            vad_filter=True,
        )

        transcript_parts = []

        for segment in segments:
            text = segment.text.strip()

            if text:
                transcript_parts.append(text)

        return " ".join(transcript_parts).strip()

    finally:
        if temp_audio_path:

            try:
                Path(temp_audio_path).unlink(
                    missing_ok=True
                )
            except Exception:
                pass


# =========================================================
# RETRIEVAL CONFIGURATION
# =========================================================

TOP_K = 5
SIMILARITY_THRESHOLD = 0.35


# =========================================================
# QUERY RETRIEVAL
# =========================================================

def retrieve_relevant_chunks(
    query: str,
    knowledge_base: dict,
    top_k: int = TOP_K,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
):
    """
    Convert only the user's query into an embedding and retrieve
    the most relevant pre-embedded knowledge-base chunks from FAISS.
    """

    query = query.strip()

    if not query:
        return []

    embedding_model = knowledge_base["embedding_model"]
    index = knowledge_base["index"]
    chunks = knowledge_base["chunks"]
    metadata = knowledge_base["metadata"]

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32",
    )

    scores, indices = index.search(
        query_embedding,
        top_k,
    )

    results = []

    for score, index_position in zip(
        scores[0],
        indices[0],
    ):
        if index_position < 0:
            continue

        similarity = float(score)

        if similarity < similarity_threshold:
            continue

        chunk_text = chunks[index_position]
        chunk_metadata = metadata[index_position]

        results.append(
            {
                "text": chunk_text,
                "metadata": chunk_metadata,
                "similarity": similarity,
            }
        )

    return results


def format_retrieved_context(results):
    """
    Convert retrieved chunks into a compact context block
    for the Groq model.
    """

    if not results:
        return ""

    context_parts = []

    for number, result in enumerate(results, start=1):

        metadata = result["metadata"]

        source_file = metadata.get(
            "source_file",
            "Unknown source",
        )

        page_number = metadata.get(
            "page_number",
            "N/A",
        )

        context_parts.append(
            f"""
SOURCE {number}
Document: {source_file}
Page: {page_number}
Similarity: {result["similarity"]:.3f}

Content:
{result["text"]}
""".strip()
        )

    return "\n\n".join(context_parts)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="PTCL Packages Assistant",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# PROFESSIONAL UI STYLING
# =========================================================

st.markdown(
    """
    <style>
        .stApp {
            background: #f7f8fc;
        }

        [data-testid="stSidebar"] {
            background: #111827;
        }

        [data-testid="stSidebar"] * {
            color: #f9fafb;
        }

        .main-header {
            padding: 1.2rem 0 0.4rem 0;
        }

        .main-title {
            font-size: 2rem;
            font-weight: 700;
            color: #172033;
            margin-bottom: 0.2rem;
        }

        .main-subtitle {
            font-size: 1rem;
            color: #667085;
            margin-bottom: 1.4rem;
        }

        .developer-text {
            font-size: 0.78rem;
            color: #98a2b3;
            margin-top: 0.2rem;
        }

        div[data-testid="stButton"] > button {
            border-radius: 8px;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# KNOWLEDGE-BASE LOADING
# =========================================================

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def load_faiss_index():
    if not FAISS_FILE.exists():
        raise FileNotFoundError(
            "FAISS index was not found. Please build the knowledge base first."
        )
    return faiss.read_index(str(FAISS_FILE))


@st.cache_data(show_spinner=False)
def load_chunks():
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            "chunks.pkl was not found. Please build the knowledge base first."
        )
    with open(CHUNKS_FILE, "rb") as file:
        return pickle.load(file)


@st.cache_data(show_spinner=False)
def load_metadata():
    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            "metadata.pkl was not found. Please build the knowledge base first."
        )
    with open(METADATA_FILE, "rb") as file:
        return pickle.load(file)


@st.cache_data(show_spinner=False)
def load_manifest():
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            "documents_manifest.json was not found. Please build the knowledge base first."
        )
    with open(MANIFEST_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def initialize_knowledge_base():
    index = load_faiss_index()
    chunks = load_chunks()
    metadata = load_metadata()
    manifest = load_manifest()
    embedding_model = load_embedding_model()

    if index.ntotal != len(chunks):
        raise RuntimeError("FAISS vector count does not match the number of chunks.")

    if len(chunks) != len(metadata):
        raise RuntimeError("Chunk count does not match metadata count.")

    return {
        "index": index,
        "chunks": chunks,
        "metadata": metadata,
        "manifest": manifest,
        "embedding_model": embedding_model,
    }


# =========================================================
# GROQ CONFIGURATION
# =========================================================

GROQ_MODEL = "openai/gpt-oss-120b"


def get_groq_api_key():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return api_key.strip()


@st.cache_resource(show_spinner=False)
def get_groq_client():
    api_key = get_groq_api_key()
    if not api_key:
        return None
    from groq import Groq
    return Groq(api_key=api_key)


# =========================================================
# GROUNDED RAG SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are PTCL Packages Assistant.

Answer questions ONLY from the supplied PTCL knowledge-base
context.

STRICT RULES:
1. Never invent PTCL package names.
2. Never invent prices.
3. Never invent validity.
4. Never invent internet data.
5. Never invent minutes.
6. Never invent SMS allowances.
7. Never invent activation codes.
8. Never invent deactivation codes.
9. Never use outside knowledge when the required information is not present in the supplied context.
10. Never present guesses as facts.
11. If the requested information is unavailable, clearly say:
    "This information is not available in the current PTCL knowledge base."
12. Answer in the user's language where reasonably possible: English, Urdu, or Roman Urdu.
13. Keep answers concise, useful, and professional.
14. For unrelated questions, explain that this assistant is designed for the provided PTCL knowledge base.
15. Mention source document/page when useful and when that information exists in the supplied context.

PACKAGE FORMAT:
When applicable, use:
Package Name:
Price:
Validity:
Internet:
Minutes:
SMS:
Activation:
Deactivation:

If a field is missing, write:
Not specified in the knowledge base.

CONTEXT:
{context}
"""


def generate_grounded_answer(query: str, retrieved_results: list):
    query = query.strip()

    if not query:
        return "Please enter a question about PTCL packages or services."

    if not retrieved_results:
        return "This information is not available in the current PTCL knowledge base."

    client = get_groq_client()

    if client is None:
        return "Groq API key is not configured. Please add GROQ_API_KEY to the environment before using the assistant."

    context = format_retrieved_context(retrieved_results)
    prompt = SYSTEM_PROMPT.format(context=context)

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
            temperature=0.1,
            max_tokens=700,
        )

        answer = response.choices[0].message.content

        if not answer or not answer.strip():
            return "I could not generate an answer from the available PTCL knowledge base."

        return answer.strip()

    except Exception:
        return "I’m unable to process the request right now. Please try again shortly."


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_query" not in st.session_state:
    st.session_state.last_query = ""


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="main-header">
        <div class="main-title">PTCL Packages Assistant</div>
        <div class="main-subtitle">
            Your intelligent guide to PTCL packages and services.
        </div>
        <div class="developer-text">
            Developed by Areeba Imran
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR MANIFEST UTILITIES
# =========================================================

def _calculate_document_sha256(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            sha256.update(block)
    return sha256.hexdigest()


def _build_current_document_manifest():
    knowledge_base_dir = PROJECT_DIR / "knowledge_base"
    supported_extensions = {".pdf", ".txt", ".docx"}
    documents = []

    if not knowledge_base_dir.exists():
        return documents

    for file_path in sorted(knowledge_base_dir.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in supported_extensions:
            continue
        stat = file_path.stat()
        documents.append(
            {
                "filename": file_path.name,
                "relative_path": str(file_path.relative_to(knowledge_base_dir)),
                "file_size": stat.st_size,
                "modified_timestamp": stat.st_mtime,
                "sha256": _calculate_document_sha256(file_path),
            }
        )
    return documents


def _load_saved_document_manifest():
    if not MANIFEST_FILE.exists():
        return None
    try:
        with open(MANIFEST_FILE, "r", encoding="utf-8") as file_handle:
            return json.load(file_handle)
    except Exception:
        return None


def _normalize_manifest_documents(documents):
    normalized = []
    if not isinstance(documents, list):
        return normalized
    for document in documents:
        if not isinstance(document, dict):
            continue
        filename = document.get("filename")
        if not filename:
            continue
        normalized.append(
            {
                "filename": str(filename),
                "relative_path": str(document.get("relative_path", filename)),
                "file_size": document.get("file_size"),
                "modified_timestamp": document.get("modified_timestamp"),
                "sha256": document.get("sha256"),
            }
        )
    return sorted(normalized, key=lambda item: (item["relative_path"].lower(), item["filename"].lower()))


def compare_knowledge_base_manifests():
    current_documents = _normalize_manifest_documents(_build_current_document_manifest())
    saved_manifest = _load_saved_document_manifest()

    if saved_manifest is None:
        return "missing_saved_manifest", current_documents, []

    saved_documents = _normalize_manifest_documents(saved_manifest.get("documents", []))
    status = "unchanged" if current_documents == saved_documents else "changed"
    return status, current_documents, saved_documents


def get_knowledge_base_status():
    try:
        status, current_documents, saved_documents = compare_knowledge_base_manifests()
        return {
            "status": status,
            "current_count": len(current_documents),
            "saved_count": len(saved_documents),
        }
    except Exception as error:
        return {
            "status": "error",
            "current_count": 0,
            "saved_count": 0,
            "error": str(error),
        }


def rebuild_knowledge_base():
    import subprocess
    import sys

    build_script = PROJECT_DIR / "scripts" / "build_knowledge_base.py"
    if not build_script.exists():
        return False, "Knowledge-base build script was not found."

    try:
        result = subprocess.run(
            [sys.executable, str(build_script)],
            cwd=str(PROJECT_DIR),
            capture_output=True,
            text=True,
            timeout=1800,
        )

        if result.returncode != 0:
            return False, result.stderr.strip() or result.stdout.strip() or "Rebuild failed."

        for cache_func in [load_faiss_index, load_chunks, load_metadata, load_manifest, load_embedding_model]:
            try:
                cache_func.clear()
            except Exception:
                pass

        return True, result.stdout.strip() or "Knowledge base rebuilt successfully."
    except subprocess.TimeoutExpired:
        return False, "Knowledge-base rebuild timed out."
    except Exception:
        return False, "Knowledge-base rebuild could not be completed."


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("## PTCL Assistant")
    st.markdown("---")
    st.markdown("### Knowledge Base")

    required_files = [CHUNKS_FILE, METADATA_FILE, MANIFEST_FILE, FAISS_FILE]
    kb_ready = all(file.exists() for file in required_files)

    kb_status_info = get_knowledge_base_status()
    kb_status = kb_status_info["status"]
    current_doc_count = kb_status_info["current_count"]

    if kb_ready and kb_status == "unchanged":
        st.success("Knowledge base ready")
        st.caption(f"{current_doc_count} source documents verified.")
    elif kb_ready and kb_status == "changed":
        st.warning("Knowledge base documents have changed. Rebuild required.")
    elif kb_status == "missing_saved_manifest":
        st.warning("Knowledge-base manifest missing. Rebuild required.")
    elif kb_status == "error":
        st.warning("Knowledge-base status could not be verified.")
    else:
        st.error("Knowledge base incomplete")

    if st.button("Rebuild Knowledge Base", use_container_width=True, type="secondary"):
        with st.spinner("Rebuilding knowledge base..."):
            rebuild_success, rebuild_message = rebuild_knowledge_base()
        if rebuild_success:
            st.success("Knowledge base rebuilt successfully.")
            st.rerun()
        else:
            st.error(f"Rebuild failed: {rebuild_message}")

    st.markdown("---")
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_query = ""
        st.rerun()

    st.markdown("---")
    st.markdown("### Developer")
    st.caption("Designed & Developed by Areeba Imran<br>© 2026 Areeba Imran. All rights reserved.", unsafe_allow_html=True)


# =========================================================
# MAIN LAYOUT & AUDIO HELPERS
# =========================================================

main_column, info_column = st.columns([2.2, 1], gap="large")


def render_voice_input():
    st.markdown('<div class="section-label">VOICE INPUT</div>', unsafe_allow_html=True)
    recording = mic_recorder(
        start_prompt="Start recording",
        stop_prompt="Stop recording",
        just_once=True,
        use_container_width=True,
        key="ptcl_voice_recorder",
    )
    if not recording:
        return None
    return recording.get("bytes")


def detect_tts_language(text):
    if not text:
        return "en"
    urdu_chars = sum(1 for char in text if "\u0600" <= char <= "\u06FF")
    return "ur" if urdu_chars >= 2 else "en"


def generate_tts_audio(text):
    if not text or not text.strip():
        return None
    try:
        language = detect_tts_language(text)
        audio_buffer = BytesIO()
        tts = gTTS(text=text.strip(), lang=language, slow=False)
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer.read()
    except Exception:
        return None


# =========================================================
# ASSISTANT CHAT CONTAINER
# =========================================================

with main_column:
    st.markdown("### Ask about PTCL packages")
    st.caption("Type your question or use the microphone below.")

    if not kb_ready:
        st.warning("The knowledge base is not ready. Please build it before using the assistant.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("audio"):
                st.audio(message["audio"], format="audio/mp3")
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("Sources", expanded=False):
                    for source in message["sources"]:
                        st.markdown(f"**Document:** {source.get('source_file', 'Unknown')}")
                        st.markdown(f"**Page:** {source.get('page_number', 'N/A')}")
                        st.markdown(f"**Similarity:** {source.get('similarity', 0.0):.4f}")
                        st.divider()

    query = st.chat_input("Ask about PTCL packages, internet, minutes, SMS, validity...")
    voice_audio = render_voice_input()
    voice_query = None

    if voice_audio:
        with st.spinner("Transcribing your voice..."):
            try:
                voice_query = transcribe_audio(voice_audio)
            except Exception:
                st.error("Voice transcription failed. Please try recording again.")

        if voice_query:
            st.info(f"Voice transcript: {voice_query}")

    active_query = voice_query if voice_query else query

    if active_query and active_query.strip():
        active_query = active_query.strip()
        st.session_state.messages.append({"role": "user", "content": active_query})

        with st.chat_message("user"):
            st.markdown(active_query)

        with st.chat_message("assistant"):
            if not kb_ready:
                st.error("The PTCL knowledge base is not ready.")
            else:
                try:
                    with st.spinner("Searching knowledge base..."):
                        knowledge_base = initialize_knowledge_base()
                        retrieved_results = retrieve_relevant_chunks(
                            query=active_query,
                            knowledge_base=knowledge_base,
                        )
                        answer = generate_grounded_answer(
                            query=active_query,
                            retrieved_results=retrieved_results,
                        )
                        audio_bytes = generate_tts_audio(answer)
                        sources_list = [
                            {
                                "source_file": res["metadata"].get("source_file", "Unknown"),
                                "page_number": res["metadata"].get("page_number", "N/A"),
                                "similarity": res["similarity"],
                            }
                            for res in retrieved_results
                        ]

                    st.markdown(answer)

                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3")

                    if sources_list:
                        with st.expander("Sources", expanded=False):
                            for source in sources_list:
                                st.markdown(f"**Document:** {source['source_file']}")
                                st.markdown(f"**Page:** {source['page_number']}")
                                st.markdown(f"**Similarity:** {source['similarity']:.4f}")
                                st.divider()

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "audio": audio_bytes,
                            "sources": sources_list,
                        }
                    )
                except Exception as e:
                    st.error(f"An error occurred while processing your request: {e}")
