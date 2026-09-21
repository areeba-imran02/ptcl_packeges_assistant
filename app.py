import os
import json
import pickle
import hashlib
import re
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
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="PTCL Assistant",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# PROFESSIONAL UI STYLING & COLOR PALETTE
# =========================================================

st.markdown(
    """
    <style>
        /* Color Palette variables & Global app styles */
        .stApp {
            background-color: #07152F;
            color: #F8FAFF;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        [data-testid="stSidebar"] {
            background-color: #102A56;
            border-right: 1px solid rgba(167, 139, 250, 0.15);
        }

        [data-testid="stSidebar"] * {
            color: #F8FAFF !important;
        }

        .main-header {
            padding: 1.5rem 0 0.5rem 0;
            border-bottom: 1px solid rgba(167, 139, 250, 0.15);
            margin-bottom: 1.5rem;
        }

        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #F8FAFF;
            letter-spacing: -0.5px;
            margin-bottom: 0.3rem;
        }

        .main-subtitle {
            font-size: 1.05rem;
            color: #A78BFA;
            margin-bottom: 0.6rem;
        }

        .developer-text {
            font-size: 0.8rem;
            color: #8EF0B0;
            font-weight: 500;
        }

        .welcome-card {
            background: linear-gradient(135deg, #102A56 0%, #07152F 100%);
            border: 1px solid rgba(107, 77, 255, 0.3);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 8px 32px rgba(7, 21, 47, 0.4);
        }

        .suggestion-btn {
            background-color: #102A56;
            border: 1px solid #6C4DFF;
            color: #F8FAFF;
            border-radius: 10px;
            padding: 0.6rem 1rem;
            font-size: 0.9rem;
            transition: all 0.2s ease;
            cursor: pointer;
            text-align: left;
            margin-bottom: 0.5rem;
        }

        .suggestion-btn:hover {
            background-color: #6C4DFF;
            color: #F8FAFF;
            border-shadow: 0 0 12px rgba(108, 77, 255, 0.5);
        }

        div[data-testid="stChatMessage"] {
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 0.8rem;
            border: 1px solid rgba(167, 139, 250, 0.1);
        }

        div[data-testid="stChatMessage"][data-testid*="user"] {
            background-color: #102A56;
        }

        div[data-testid="stChatMessage"][data-testid*="assistant"] {
            background-color: rgba(16, 42, 86, 0.6);
            border-left: 4px solid #6C4DFF;
        }

        div[data-testid="stButton"] > button {
            border-radius: 8px;
            font-weight: 600;
            background-color: #6C4DFF;
            color: #F8FAFF;
            border: none;
            transition: all 0.2s;
        }

        div[data-testid="stButton"] > button:hover {
            background-color: #7c5cff;
            box-shadow: 0 0 10px rgba(108, 77, 255, 0.4);
        }

        .stTextInput input {
            background-color: #102A56 !important;
            color: #F8FAFF !important;
            border: 1px solid rgba(167, 139, 250, 0.3) !important;
            border-radius: 8px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


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
                Path(temp_audio_path).unlink(missing_ok=True)
            except Exception:
                pass


# =========================================================
# LANGUAGE DETECTION & UTILITIES
# =========================================================

def detect_query_language(text):
    """
    Robust query language detection supporting Urdu script,
    Roman Urdu, and English while ignoring standard PTCL terms.
    """
    if not text:
        return "en"

    # Check for Urdu script characters
    urdu_chars = sum(1 for char in text if "\u0600" <= char <= "\u06FF")
    if urdu_chars >= 2:
        return "ur"

    # Check for Roman Urdu patterns / keywords
    roman_urdu_keywords = [
        "kon", "kya", "hain", "hai", "kese", "batao", "mujhe", 
        "packages", "wala", "aur", "ki", "ka", "ke", "mein", 
        "se", "karnay", "konsa", "konsay", "bataen", "hain"
    ]
    lower_text = text.lower()
    words = lower_text.split()
    match_count = sum(1 for w in words if w in roman_urdu_keywords)
    
    if match_count >= 1 or any(k in lower_text for k in ["kon kon", "kya hai", "bataen", "packages kon"]):
        return "roman_ur"

    return "en"


def clean_text_for_tts(text):
    """
    Clean markdown symbols, bullets, emojis, URLs, and formatting
    from the response before sending it to TTS.
    """
    if not text:
        return ""
    # Remove markdown headers, bold, italics, lists
    cleaned = re.sub(r'[\#\*\_\-\`\~\[\]\(\)]', ' ', text)
    # Remove URLs
    cleaned = re.sub(r'http\S+', '', cleaned)
    # Condense multiple whitespaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def format_source_name(filename):
    """Convert technical filename into a clean user-friendly document name."""
    if not filename:
        return "PTCL Knowledge Base"
    name = Path(filename).stem
    cleaned = re.sub(r'^[0-9]+[_]*', '', name)
    cleaned = cleaned.replace('_', ' ').title()
    if not cleaned.lower().startswith("ptcl"):
        return f"PTCL {cleaned}"
    return cleaned


# =========================================================
# RETRIEVAL CONFIGURATION
# =========================================================

TOP_K = 5
SIMILARITY_THRESHOLD = 0.30


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

    for score, index_position in zip(scores[0], indices[0]):
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
        source_file = metadata.get("source_file", "Unknown source")
        page_number = metadata.get("page_number", "N/A")

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
# KNOWLEDGE-BASE LOADING
# =========================================================

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def load_faiss_index():
    if not FAISS_FILE.exists():
        raise FileNotFoundError("FAISS index was not found. Please build the knowledge base first.")
    return faiss.read_index(str(FAISS_FILE))


@st.cache_data(show_spinner=False)
def load_chunks():
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError("chunks.pkl was not found. Please build the knowledge base first.")
    with open(CHUNKS_FILE, "rb") as file:
        return pickle.load(file)


@st.cache_data(show_spinner=False)
def load_metadata():
    if not METADATA_FILE.exists():
        raise FileNotFoundError("metadata.pkl was not found. Please build the knowledge base first.")
    with open(METADATA_FILE, "rb") as file:
        return pickle.load(file)


@st.cache_data(show_spinner=False)
def load_manifest():
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError("documents_manifest.json was not found. Please build the knowledge base first.")
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
You are PTCL Packages Assistant, an expert guide for PTCL packages and services.
Answer questions strictly and accurately from the supplied PTCL knowledge-base context.

STRICT RULES:
1. Never invent PTCL package names, prices, speeds, validity, internet data, minutes, SMS allowances, activation codes, or deactivation codes.
2. Never use outside knowledge when required information is not present in the supplied context.
3. If the information genuinely cannot be found in the context, respond politely in the SAME language as the user's query:
   - For English queries: "I couldn't find that specific information in my current PTCL knowledge base. Please try asking about PTCL internet packages, Flash Fiber, voice/mobile packages, Shoq TV, Speed Bolt-On, Quad Play, or advance packages."
   - For Urdu queries: "مجھے موجودہ PTCL معلومات میں اس سوال کی مخصوص تفصیل نہیں ملی۔ آپ PTCL انٹرنیٹ پیکیجز، Flash Fiber، Voice/Mobile Packages، Shoq TV، Speed Bolt-On، Quad Play یا Advance Packages کے بارے میں پوچھ سکتے ہیں۔"
   - For Roman Urdu queries: "Mujhe mojooda PTCL knowledge base mein is sawal ki tafseel nahi mili. Aap PTCL internet packages, Flash Fiber, voice/mobile packages, Shoq TV, Speed Bolt-On, Quad Play ya advance packages ke baray mein pooch sakte hain."
4. Match the user's query language strictly:
   - If user asks in English, answer in clear, professional English.
   - If user asks in Urdu script, answer in natural Urdu script.
   - If user asks in Roman Urdu, answer naturally in Roman Urdu.
5. Keep answers concise, useful, and professional. Use bullet points or structured lists when appropriate. Do not dump entire text verbatim.

CONTEXT:
{context}
"""


def generate_grounded_answer(query: str, retrieved_results: list):
    query = query.strip()
    query_lang = detect_query_language(query)

    if not query:
        if query_lang == "ur":
            return "براہ کرم PTCL پیکیجز یا خدمات کے بارے میں کوئی سوال درج کریں۔"
        elif query_lang == "roman_ur":
            return "Barah-e-karam PTCL packages ya services ke baray mein koi sawal darj karen."
        return "Please enter a question about PTCL packages or services."

    if not retrieved_results:
        if query_lang == "ur":
            return "مجھے موجودہ PTCL معلومات میں اس سوال کی مخصوص تفصیل نہیں ملی۔ آپ PTCL انٹرنیٹ پیکیجز، Flash Fiber، Voice/Mobile Packages، Shoq TV، Speed Bolt-On، Quad Play یا Advance Packages کے بارے میں پوچھ سکتے ہیں۔"
        elif query_lang == "roman_ur":
            return "Mujhe mojooda PTCL knowledge base mein is sawal ki tafseel nahi mili. Aap PTCL internet packages, Flash Fiber, voice/mobile packages, Shoq TV, Speed Bolt-On, Quad Play ya advance packages ke baray mein pooch sakte hain."
        return "I couldn't find that specific information in my current PTCL knowledge base. Please try asking about PTCL internet packages, Flash Fiber, voice/mobile packages, Shoq TV, Speed Bolt-On, Quad Play, or advance packages."

    client = get_groq_client()

    if client is None:
        return "Sorry, I couldn't process that request right now. Please check API key configuration or try again shortly."

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
            if query_lang == "ur":
                return "معذرت، میں دستیاب PTCL ڈیٹا بیس سے جواب تخلیق کرنے سے قاصر رہا۔"
            elif query_lang == "roman_ur":
                return "Maazrat, mein dastiyab PTCL database se jawab takhleeq karne se qasir raha."
            return "I couldn't generate an answer from the available PTCL knowledge base."

        return answer.strip()

    except Exception:
        return "Sorry, I couldn't process that request right now. Please try again."


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
        <div class="main-title">PTCL Assistant</div>
        <div class="main-subtitle">
            Your smart assistant for PTCL packages, internet, voice, mobile and services.
        </div>
        <div class="developer-text">
            Developed by Areeba Imran
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# MANIFEST & REBUILD HELPERS
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
    st.markdown("### PTCL Assistant")
    st.markdown("Ask about PTCL packages and services.")
    st.markdown("---")
    
    st.markdown("### 📚 Quick Categories")
    st.markdown("• Internet Packages\n• Flash Fiber\n• Voice & Mobile\n• Shoq TV\n• Speed Bolt-On\n• Quad Play\n• Advance Packages")
    
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
        st.warning("Knowledge base documents changed. Rebuild required.")
    elif kb_status == "missing_saved_manifest":
        st.warning("Knowledge-base manifest missing. Rebuild required.")
    elif kb_status == "error":
        st.warning("Knowledge-base status unverified.")
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

main_column, info_column = st.columns([2.4, 1], gap="large")


def render_voice_input():
    st.markdown("**Voice Input**", unsafe_allow_html=True)
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
        clean_speech = clean_text_for_tts(text)
        if not clean_speech:
            return None
        language = detect_tts_language(text)
        audio_buffer = BytesIO()
        tts = gTTS(text=clean_speech, lang=language, slow=False)
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer.read()
    except Exception:
        return None


# =========================================================
# ASSISTANT CHAT CONTAINER & WELCOME SCREEN
# =========================================================

with main_column:
    # --------------------------------------------------------
    # Welcome Screen if no chat history
    # --------------------------------------------------------
    if not st.session_state.messages:
        st.markdown(
            """
            <div class="welcome-card">
                <h3>Welcome to PTCL Assistant</h3>
                <p style="color: #A78BFA; margin-bottom: 1rem;">
                    Ask me about PTCL internet packages, Flash Fiber, voice & mobile packages, Shoq TV, Speed Bolt-On, Quad Play and more.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Suggested Questions:")
        col_s1, col_s2 = st.columns(2)
        
        suggested_clicked = None
        with col_s1:
            if st.button("What internet packages are available?", use_container_width=True):
                suggested_clicked = "What internet packages are available?"
            if st.button("Tell me about Flash Fiber.", use_container_width=True):
                suggested_clicked = "Tell me about Flash Fiber."
        with col_s2:
            if st.button("What are the PTCL voice and mobile packages?", use_container_width=True):
                suggested_clicked = "What are the PTCL voice and mobile packages?"
            if st.button("انٹرنیٹ کے کون کون سے پیکیجز ہیں؟", use_container_width=True):
                suggested_clicked = "انٹرنیٹ کے کون کون سے پیکیجز ہیں؟"

        if suggested_clicked:
            st.session_state.last_query = suggested_clicked

    # --------------------------------------------------------
    # Display existing conversation
    # --------------------------------------------------------
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("audio"):
                st.audio(message["audio"], format="audio/mp3")
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("Sources", expanded=False):
                    for source in message["sources"]:
                        st.markdown(f"**Document:** {source.get('source_file', 'PTCL Knowledge Base')}")
                        st.markdown(f"**Page:** {source.get('page_number', 'N/A')}")
                        st.markdown(f"**Similarity:** {source.get('similarity', 0.0):.4f}")
                        st.divider()

    # --------------------------------------------------------
    # Input Area
    # --------------------------------------------------------
    query = st.chat_input("Ask about PTCL packages, internet, voice, minutes, SMS, validity...")
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

    # Determine active query from suggestion, voice input, or chat input
    active_query = st.session_state.pop("last_query", None)
    if voice_query:
        active_query = voice_query
    elif query:
        active_query = query

    if active_query and active_query.strip():
        active_query = active_query.strip()
        st.session_state.messages.append({"role": "user", "content": active_query})

        with st.chat_message("user"):
            st.markdown(active_query)

        with st.chat_message("assistant"):
            if not kb_ready:
                st.error("The PTCL knowledge base is not ready. Please build the knowledge base first.")
            else:
                try:
                    with st.spinner("Searching PTCL knowledge base..."):
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
                                "source_file": format_source_name(res["metadata"].get("source_file", "PTCL Knowledge Base")),
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
                except Exception:
                    err_msg = "Sorry, I couldn't process that request right now. Please try again."
                    st.error(err_msg)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": err_msg,
                            "audio": None,
                            "sources": [],
                        }
                    )
