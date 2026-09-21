import os
import json
import pickle
import hashlib
import re
import tempfile
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
# UI STYLING
# Palette (ocean blue + signal teal + soft foam text):
#   --abyss   #04101F   page background
#   --deep    #0A2140   panels / sidebar
#   --tide    #123763   raised surfaces
#   --signal  #2DD4BF   primary accent (teal)
#   --azure   #3B82F6   secondary accent (blue)
#   --foam    #EAF4FF   main text
#   --mist    #93A9C7   muted text
# =========================================================

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=Noto+Naskh+Arabic:wght@400;600&display=swap');

        :root {
            --abyss: #04101F;
            --deep: #0A2140;
            --tide: #123763;
            --signal: #2DD4BF;
            --azure: #3B82F6;
            --foam: #EAF4FF;
            --mist: #93A9C7;
            --line: rgba(147, 169, 199, 0.18);
        }

        /* ---------- Base ---------- */
        .stApp {
            background:
                radial-gradient(900px 500px at 85% -10%, rgba(45, 212, 191, 0.14), transparent 60%),
                radial-gradient(900px 600px at -10% 10%, rgba(59, 130, 246, 0.16), transparent 60%),
                linear-gradient(180deg, #04101F 0%, #071A33 100%);
            color: var(--foam);
            font-family: 'Outfit', 'Noto Naskh Arabic', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        header[data-testid="stHeader"] { background: transparent; }

        #MainMenu, footer { visibility: hidden; }

        .block-container, [data-testid="stMainBlockContainer"] {
            max-width: 1280px;
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }

        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li {
            color: var(--foam);
            font-family: 'Outfit', 'Noto Naskh Arabic', sans-serif;
            line-height: 1.65;
        }

        .rtl-block {
            direction: rtl;
            text-align: right;
            font-family: 'Noto Naskh Arabic', 'Outfit', sans-serif;
            font-size: 1.05rem;
            line-height: 2;
        }

        /* ---------- Sidebar ---------- */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0A2140 0%, #061428 100%);
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] .stMarkdown { color: var(--foam); }

        .brand {
            display: flex; align-items: center; gap: 0.75rem;
            padding: 0.4rem 0 1rem 0;
        }
        .brand-logo {
            width: 46px; height: 46px; border-radius: 14px;
            display: grid; place-items: center; font-size: 1.5rem;
            background: linear-gradient(135deg, var(--signal), var(--azure));
            box-shadow: 0 6px 20px rgba(45, 212, 191, 0.3);
        }
        .brand-name { font-weight: 700; font-size: 1.15rem; color: var(--foam); line-height: 1.1; }
        .brand-tag { font-size: 0.78rem; color: var(--mist); }

        .side-title {
            font-weight: 600; font-size: 0.95rem; color: var(--foam);
            margin: 0.9rem 0 0.5rem 0;
        }

        .pill-wrap { display: flex; flex-wrap: wrap; gap: 0.4rem; }
        .pill {
            font-size: 0.78rem; padding: 0.28rem 0.7rem; border-radius: 999px;
            background: rgba(59, 130, 246, 0.14);
            border: 1px solid rgba(59, 130, 246, 0.35);
            color: #CFE2FF;
        }

        .status-chip {
            display: inline-flex; align-items: center; gap: 0.5rem;
            padding: 0.4rem 0.8rem; border-radius: 10px; font-size: 0.85rem;
            border: 1px solid var(--line);
        }
        .status-ok   { background: rgba(45, 212, 191, 0.12); color: #7FF0DF; border-color: rgba(45, 212, 191, 0.4); }
        .status-warn { background: rgba(251, 191, 36, 0.12); color: #FCD980; border-color: rgba(251, 191, 36, 0.4); }
        .status-bad  { background: rgba(248, 113, 113, 0.12); color: #FCA5A5; border-color: rgba(248, 113, 113, 0.4); }
        .dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; }

        .dev-credit { font-size: 0.78rem; color: var(--mist); line-height: 1.5; }

        /* ---------- Hero ---------- */
        .hero {
            position: relative; overflow: hidden;
            border-radius: 22px; padding: 1.9rem 2rem 1.7rem 2rem; margin-bottom: 1.3rem;
            background: linear-gradient(120deg, #0B2A52 0%, #0A2140 55%, #0B3B4A 100%);
            border: 1px solid var(--line);
        }
        .hero::after {
            content: ""; position: absolute; right: -90px; top: -90px;
            width: 340px; height: 340px; border-radius: 50%;
            background: repeating-radial-gradient(circle at center,
                rgba(45, 212, 191, 0.0) 0, rgba(45, 212, 191, 0.0) 22px,
                rgba(45, 212, 191, 0.22) 23px, rgba(45, 212, 191, 0.0) 25px);
            pointer-events: none;
        }
        .hero-title {
            font-size: 2.3rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1.1;
            color: var(--foam); margin: 0 0 0.5rem 0;
        }
        .hero-sub { font-size: 1.02rem; color: #B9CCE6; max-width: 560px; margin: 0 0 1rem 0; }
        .hero-badges { display: flex; flex-wrap: wrap; gap: 0.5rem; position: relative; z-index: 1; }
        .badge {
            font-size: 0.8rem; padding: 0.3rem 0.75rem; border-radius: 999px;
            background: rgba(4, 16, 31, 0.55); border: 1px solid var(--line); color: #CFE2FF;
        }

        /* ---------- Chat ---------- */
        div[data-testid="stChatMessage"] {
            border-radius: 18px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.9rem;
            border: 1px solid var(--line);
            background: rgba(10, 33, 64, 0.65);
        }
        div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
        div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.30), rgba(18, 55, 99, 0.60));
            border-color: rgba(59, 130, 246, 0.45);
        }
        /* Assistant answers: light "paper" card so they stand apart from the dark UI */
        div[data-testid="stChatMessage"]:has(:is([data-testid="stChatMessageAvatarAssistant"], [data-testid="chatAvatarIcon-assistant"])) {
            background: linear-gradient(135deg, #F8FCFF 0%, #E4F2FF 100%);
            border: 1px solid rgba(45, 212, 191, 0.7);
            border-left: 6px solid #14B8A6;
            box-shadow: 0 10px 30px rgba(45, 212, 191, 0.20);
        }
        div[data-testid="stChatMessage"]:has(:is([data-testid="stChatMessageAvatarAssistant"], [data-testid="chatAvatarIcon-assistant"]))
            :is([data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] *, [data-testid="stSpinner"] *, [data-testid="stCaptionContainer"] *) {
            color: #0B1F3A !important;
        }
        div[data-testid="stChatMessage"]:has(:is([data-testid="stChatMessageAvatarAssistant"], [data-testid="chatAvatarIcon-assistant"]))
            [data-testid="stMarkdownContainer"] strong { color: #0F5F5A !important; }

        /* ---------- Buttons ---------- */
        .stButton > button, div[data-testid="stButton"] > button {
            border-radius: 12px;
            font-weight: 500;
            font-family: 'Outfit', 'Noto Naskh Arabic', sans-serif;
            background: rgba(18, 55, 99, 0.55);
            color: var(--foam);
            border: 1px solid rgba(59, 130, 246, 0.35);
            padding: 0.55rem 0.9rem;
            transition: border-color 0.15s ease, background 0.15s ease;
        }
        .stButton > button:hover, div[data-testid="stButton"] > button:hover {
            background: rgba(45, 212, 191, 0.14);
            border-color: var(--signal);
            color: #FFFFFF;
        }
        .stButton > button:focus-visible { outline: 2px solid var(--signal); outline-offset: 2px; }

        /* ---------- Chat input ---------- */
        [data-testid="stChatInput"] {
            border-radius: 16px;
            background: #0A2140 !important;
            border: 1px solid rgba(45, 212, 191, 0.55) !important;
            box-shadow: 0 0 18px rgba(45, 212, 191, 0.18);
        }
        [data-testid="stChatInput"]:focus-within {
            border-color: var(--signal) !important;
            box-shadow: 0 0 0 1px var(--signal), 0 0 22px rgba(45, 212, 191, 0.35);
        }
        [data-testid="stChatInput"] * {
            background-color: transparent !important;
            border-color: transparent !important;
            box-shadow: none !important;
            color: var(--foam) !important;
            -webkit-text-fill-color: var(--foam);
            font-family: 'Outfit', 'Noto Naskh Arabic', sans-serif;
        }
        [data-testid="stChatInput"] textarea::placeholder {
            color: var(--mist) !important;
            -webkit-text-fill-color: var(--mist);
            opacity: 1;
        }
        [data-testid="stChatInput"] [data-testid="stChatInputSubmitButton"] {
            background-color: var(--signal) !important;
            border-radius: 10px;
        }
        [data-testid="stChatInput"] [data-testid="stChatInputSubmitButton"] * {
            color: #04101F !important;
            -webkit-text-fill-color: #04101F;
        }

        /* ---------- Chatbot panel: glowing border on its 4 sides ---------- */
        .st-key-chat_panel {
            border: 2px solid rgba(45, 212, 191, 0.75) !important;
            border-radius: 24px !important;
            background: rgba(4, 16, 31, 0.50);
            box-shadow:
                0 0 16px rgba(45, 212, 191, 0.50),
                0 0 44px rgba(59, 130, 246, 0.32),
                inset 0 0 30px rgba(45, 212, 191, 0.10);
            animation: panelGlow 4.5s ease-in-out infinite;
        }
        @keyframes panelGlow {
            0%, 100% {
                box-shadow: 0 0 12px rgba(45, 212, 191, 0.35), 0 0 34px rgba(59, 130, 246, 0.22),
                            inset 0 0 24px rgba(45, 212, 191, 0.08);
            }
            50% {
                box-shadow: 0 0 20px rgba(45, 212, 191, 0.65), 0 0 56px rgba(59, 130, 246, 0.40),
                            inset 0 0 34px rgba(45, 212, 191, 0.14);
            }
        }
        @media (prefers-reduced-motion: reduce) { .st-key-chat_panel { animation: none; } }

        .composer-note {
            font-size: 0.85rem; color: var(--mist);
            margin: 0.5rem 0 0.4rem 0; padding-top: 0.7rem;
            border-top: 1px solid var(--line);
        }

        /* ---------- Suggested-questions dropdown (light shade, always visible) ---------- */
        .st-key-sq_dropdown [data-testid="stExpander"],
        [data-testid="stMain"] [data-testid="stExpander"] {
            background: linear-gradient(160deg, #F8FCFF 0%, #E3F1FF 100%) !important;
            border: 1.5px solid rgba(45, 212, 191, 0.75) !important;
            border-radius: 18px !important;
            box-shadow: 0 8px 26px rgba(45, 212, 191, 0.20);
        }
        .st-key-sq_dropdown [data-testid="stExpander"] details,
        [data-testid="stMain"] [data-testid="stExpander"] details {
            background: transparent !important;
            border: none !important;
        }
        .st-key-sq_dropdown [data-testid="stExpander"] summary,
        .st-key-sq_dropdown [data-testid="stExpander"] summary *,
        [data-testid="stMain"] [data-testid="stExpander"] summary,
        [data-testid="stMain"] [data-testid="stExpander"] summary * {
            color: #0B1F3A !important;
            font-weight: 600;
        }

        audio { width: 100%; border-radius: 12px; margin-top: 0.4rem; }

        /* ---------- Footer ---------- */
        .app-footer {
            margin-top: 2.2rem; padding: 1.1rem 0 0.4rem 0;
            border-top: 1px solid var(--line);
            text-align: center; color: var(--mist); font-size: 0.82rem; line-height: 1.7;
        }
        .app-footer b { color: var(--foam); font-weight: 600; }
        .app-footer .note { font-size: 0.76rem; opacity: 0.85; }

        @media (max-width: 640px) {
            .hero { padding: 1.4rem 1.2rem; }
            .hero-title { font-size: 1.8rem; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# WHISPER SPEECH-TO-TEXT & VALIDATION
# =========================================================

WHISPER_MODEL_SIZE = "small"
WHISPER_HINT = "PTCL, Flash Fiber, Shoq TV, Speed Bolt-On, Quad Play, internet package, minutes, SMS."

# Whisper often mislabels spoken Urdu as one of these. For Pakistani users
# they are almost always speaking Urdu/Punjabi, so we re-run as Urdu.
URDU_FAMILY = {"hi", "pa", "ar", "fa", "ps", "sd"}


@st.cache_resource(show_spinner=False)
def load_whisper_model():
    return WhisperModel(
        WHISPER_MODEL_SIZE,
        device="cpu",
        compute_type="int8",
    )


def transcribe_audio(audio_bytes):
    """
    Convert recorded audio into text. The spoken language is auto-detected,
    with a correction for Urdu being mis-detected as Hindi/Arabic/Persian etc.
    """
    if not audio_bytes:
        return ""

    temp_audio_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name

        model = load_whisper_model()

        segments, info = model.transcribe(
            temp_audio_path,
            beam_size=5,
            vad_filter=True,
            initial_prompt=WHISPER_HINT,
            condition_on_previous_text=False,
        )

        detected = getattr(info, "language", None)
        probability = getattr(info, "language_probability", 1.0) or 0.0

        if detected != "ur" and detected != "en" and (
            detected in URDU_FAMILY or probability < 0.55
        ):
            segments, info = model.transcribe(
                temp_audio_path,
                language="ur",
                beam_size=5,
                vad_filter=True,
                condition_on_previous_text=False,
            )

        parts = [segment.text.strip() for segment in segments if segment.text.strip()]
        return " ".join(parts).strip()

    finally:
        if temp_audio_path:
            try:
                Path(temp_audio_path).unlink(missing_ok=True)
            except Exception:
                pass


WHISPER_JUNK_PHRASES = (
    "thanks for watching",
    "thank you for watching",
    "subscribe",
    "amara.org",
    "like and subscribe",
)


def is_valid_transcript(text):
    """Reject empty text, hallucinated boilerplate and symbol noise."""
    if not text:
        return False
    clean = text.strip()
    if len(clean) < 2:
        return False
    lowered = clean.lower()
    if any(phrase in lowered for phrase in WHISPER_JUNK_PHRASES):
        return False

    def allowed(ch):
        code = ord(ch)
        return (
            code < 0x250
            or 0x0600 <= code <= 0x06FF
            or 0x0750 <= code <= 0x077F
            or 0xFB50 <= code <= 0xFDFF
            or 0xFE70 <= code <= 0xFEFF
            or 0x0900 <= code <= 0x097F
            or 0x0A00 <= code <= 0x0A7F
            or ch.isalpha()
        )

    odd = sum(1 for c in clean if not allowed(c))
    symbols = len(re.findall(r"[^\w\s]", clean))
    if odd > 3 or symbols > len(clean) * 0.4:
        return False
    return True


# =========================================================
# LANGUAGE UTILITIES
# =========================================================

def is_rtl_text(text):
    """True when most letters are Arabic-script (Urdu, Punjabi Shahmukhi, Arabic, ...)."""
    if not text:
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    rtl = sum(1 for c in letters if "\u0600" <= c <= "\u06FF" or "\u0750" <= c <= "\u077F")
    return rtl / len(letters) > 0.5


def detect_query_language(text):
    """Cheap heuristic used only as a fallback when the LLM analysis fails."""
    if not text:
        return "en"

    urdu_chars = sum(1 for char in text if "\u0600" <= char <= "\u06FF")
    if urdu_chars >= 2:
        return "ur"

    roman_urdu_keywords = {
        "kon", "kya", "hain", "hai", "kese", "kaise", "batao", "bataen", "bataye",
        "mujhe", "wala", "aur", "ki", "ka", "ke", "mein", "main", "se", "karnay",
        "konsa", "konsay", "kitna", "kitne", "chahiye", "hota", "hoti", "ko", "ap", "aap",
    }
    words = re.findall(r"[a-z]+", text.lower())
    if sum(1 for w in words if w in roman_urdu_keywords) >= 1:
        return "roman_ur"
    return "en"


def normalize_language(name):
    """Turn a free-form language name into a stable key such as 'roman_urdu'."""
    if not name:
        return "english"
    key = str(name).strip().lower().replace("-", " ").replace("_", " ")
    if "roman" in key or "urdu (latin" in key or "hinglish" in key:
        return "roman_urdu"
    key = re.sub(r"\(.*?\)", "", key).strip()
    if not key:
        return "english"
    return key.replace(" ", "_")


def language_label(key):
    labels = {
        "english": "English",
        "urdu": "Urdu (written in Urdu/Arabic script)",
        "roman_urdu": "Roman Urdu (Urdu written with English/Latin letters only)",
        "punjabi": "Punjabi (in the same script the user used)",
    }
    return labels.get(key, key.replace("_", " ").title())


# gTTS voice per language (Arabic-script regional languages use the Urdu voice).
TTS_LANG_MAP = {
    "english": "en",
    "urdu": "ur",
    "roman_urdu": "ur",
    "punjabi": "ur",
    "pashto": "ur",
    "sindhi": "ur",
    "hindi": "hi",
    "arabic": "ar",
    "french": "fr",
    "spanish": "es",
    "german": "de",
    "turkish": "tr",
    "chinese": "zh-CN",
}


def clean_text_for_tts(text):
    """Strip markdown, emojis, URLs and bullets so speech sounds natural."""
    if not text:
        return ""
    cleaned = re.sub(r"http\S+", " ", text)
    cleaned = re.sub(r"[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F]", " ", cleaned)
    cleaned = re.sub(r"(?m)^\s*[-*•]\s+", "", cleaned)
    cleaned = re.sub(r"[#*_`~\[\]()|>]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def truncate_for_speech(text, limit=700):
    if len(text) <= limit:
        return text
    cut = text[:limit]
    last_stop = max(cut.rfind("."), cut.rfind("۔"), cut.rfind("!"), cut.rfind("?"))
    if last_stop > limit * 0.5:
        return cut[: last_stop + 1]
    return cut


def sanitize_answer(text):
    """Safety net: never show source/document references to the user."""
    if not text:
        return text
    text = re.sub(r"(?im)^\s*[-*•]?\s*(sources?|references?)\s*\d*\s*:.*$", "", text)
    text = re.sub(r"\(\s*(?:source|document)\s*\d+\s*\)", "", text, flags=re.I)
    text = re.sub(r"\b(?:source|document)\s*\d+\b", "", text, flags=re.I)
    text = re.sub(r"[\w\-.]+\.(?:pdf|docx?|txt)\b", "", text, flags=re.I)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# =========================================================
# RETRIEVAL CONFIGURATION
# =========================================================

TOP_K = 6
SIMILARITY_THRESHOLD = 0.30


def retrieve_relevant_chunks(
    query: str,
    knowledge_base: dict,
    top_k: int = TOP_K,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
):
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
    query_embedding = np.asarray(query_embedding, dtype="float32")

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, index_position in zip(scores[0], indices[0]):
        if index_position < 0:
            continue
        similarity = float(score)
        if similarity < similarity_threshold:
            continue
        results.append(
            {
                "text": chunks[index_position],
                "metadata": metadata[index_position],
                "similarity": similarity,
            }
        )
    return results


def format_retrieved_context(results):
    """
    Build the context for the model. Document names and page numbers are
    deliberately left out so the assistant can never repeat them.
    """
    if not results:
        return ""
    return "\n\n---\n\n".join(result["text"].strip() for result in results)


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


def call_llm(messages, max_tokens=1500, temperature=0.1):
    """Single place for Groq calls. Uses low reasoning effort for speed when supported."""
    client = get_groq_client()
    if client is None:
        raise RuntimeError("Groq client is not configured.")

    kwargs = dict(
        model=GROQ_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    try:
        response = client.chat.completions.create(reasoning_effort="low", **kwargs)
    except Exception:
        response = client.chat.completions.create(**kwargs)
    return (response.choices[0].message.content or "").strip()


# =========================================================
# QUERY UNDERSTANDING (any language -> English search query)
# =========================================================
# The embedding model is English-only, so Urdu / Roman Urdu / other
# languages are first interpreted into a clean English search query.
# The final answer is still written in the user's own language.

def format_history(history, limit=4, chars=300):
    lines = []
    for item in history[-limit:]:
        who = "User" if item["role"] == "user" else "Assistant"
        lines.append(f"{who}: {item['content'][:chars]}")
    return "\n".join(lines)


UNDERSTAND_PROMPT = """You analyse messages for a PTCL (Pakistan Telecommunication Company Limited) customer assistant.
The message may be in ANY language (English, Urdu script, Roman Urdu, Punjabi, Pashto, Sindhi, Hindi, Arabic, mixed...).
It may come from speech recognition, so spelling can be imperfect: use common sense.

Return ONLY a JSON object, no other text:
{"language": "<language the user wrote/spoke, in English, e.g. English, Urdu, Roman Urdu, Punjabi, Hindi, Arabic, French>",
 "english_query": "<a clear standalone English search query about PTCL; keep product names and numbers; resolve follow-ups using the conversation>",
 "intent": "<greeting | thanks | question>"}

Rules:
- "Roman Urdu" means Urdu written in Latin letters (e.g. "internet ke packages kon se hain").
- Use intent "greeting" for hello/salam/hi only, "thanks" for thank-you only, otherwise "question".

Recent conversation (may be empty):
{history}

User message:
{query}
"""


def understand_query(query, history):
    fallback_key = {"ur": "urdu", "roman_ur": "roman_urdu", "en": "english"}[detect_query_language(query)]
    fallback = {"language": fallback_key, "english_query": query, "intent": "question"}

    if get_groq_client() is None:
        return fallback

    try:
        prompt = UNDERSTAND_PROMPT.replace("{history}", format_history(history) or "(none)").replace("{query}", query)
        raw = call_llm([{"role": "user", "content": prompt}], max_tokens=600, temperature=0)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(match.group(0))

        intent = str(data.get("intent", "question")).strip().lower()
        if intent not in {"greeting", "thanks", "question"}:
            intent = "question"

        english_query = str(data.get("english_query") or query).strip() or query
        return {
            "language": normalize_language(data.get("language")),
            "english_query": english_query,
            "intent": intent,
        }
    except Exception:
        return fallback


# =========================================================
# ANSWER GENERATION
# =========================================================

SYSTEM_PROMPT = """
You are PTCL Assistant, a friendly and professional virtual assistant for PTCL packages and services.
Answer strictly and accurately from the PTCL information provided below.

RULES:
1. Never invent PTCL package names, prices, speeds, validity, data, minutes, SMS allowances, activation or deactivation codes.
2. Never use outside knowledge when the required information is not in the provided information.
3. If the information is missing, politely say you could not find that specific detail, and suggest asking about PTCL internet packages, Flash Fiber, voice/mobile packages, Shoq TV, Speed Bolt-On, Quad Play or advance packages. Say this in the user's language.
4. LANGUAGE: reply ONLY in this language: {language}.
   - Urdu: use natural Urdu script (never Hindi/Devanagari).
   - Roman Urdu: use Latin letters only, natural everyday Urdu.
   - Keep PTCL product names, prices (Rs.), speeds (Mbps) and codes exactly as given.
5. NEVER mention documents, files, file names, sources, page numbers, links to documents, "context" or "knowledge base". Just answer naturally as PTCL Assistant.
6. Be concise and helpful. Use short bullet points for lists of packages, and put key numbers (price, speed, validity) up front. Do not dump raw text.
7. Use the earlier conversation only to understand follow-up questions.

PTCL INFORMATION:
{context}
"""

NOT_FOUND_MESSAGES = {
    "english": "I couldn't find that specific information right now. Please try asking about PTCL internet packages, Flash Fiber, voice/mobile packages, Shoq TV, Speed Bolt-On, Quad Play, or advance packages.",
    "urdu": "مجھے اس سوال کی مخصوص تفصیل ابھی نہیں ملی۔ آپ PTCL انٹرنیٹ پیکیجز، Flash Fiber، Voice/Mobile Packages، Shoq TV، Speed Bolt-On، Quad Play یا Advance Packages کے بارے میں پوچھ سکتے ہیں۔",
    "roman_urdu": "Mujhe is sawal ki makhsoos tafseel abhi nahi mili. Aap PTCL internet packages, Flash Fiber, voice/mobile packages, Shoq TV, Speed Bolt-On, Quad Play ya advance packages ke baray mein pooch sakte hain.",
}

GREETING_MESSAGES = {
    "english": "Hello! I'm your PTCL Assistant. Ask me about internet, Flash Fiber, voice and mobile packages, Shoq TV and more.",
    "urdu": "السلام علیکم! میں آپ کا PTCL اسسٹنٹ ہوں۔ انٹرنیٹ، Flash Fiber، وائس اور موبائل پیکیجز، Shoq TV اور دیگر خدمات کے بارے میں پوچھیں۔",
    "roman_urdu": "Assalam o Alaikum! Main aap ka PTCL Assistant hoon. Internet, Flash Fiber, voice aur mobile packages, Shoq TV aur dusri services ke baray mein poochiye.",
}

GENERIC_ERROR = "Sorry, I couldn't process that request right now. Please try again."


def generate_smalltalk(query, language):
    """Reply to greetings / thanks without touching the knowledge base."""
    if get_groq_client() is not None:
        try:
            system = (
                "You are PTCL Assistant. The user sent a greeting or a thank-you. "
                f"Reply warmly in 1-2 short sentences in {language_label(language)}, "
                "and invite them to ask about PTCL internet, Flash Fiber, voice/mobile packages, "
                "Shoq TV, Speed Bolt-On or Quad Play. Do not state any package details."
            )
            answer = call_llm(
                [{"role": "system", "content": system}, {"role": "user", "content": query}],
                max_tokens=400,
                temperature=0.4,
            )
            if answer:
                return sanitize_answer(answer)
        except Exception:
            pass
    return GREETING_MESSAGES.get(language, GREETING_MESSAGES["english"])


def generate_grounded_answer(query, language, retrieved_results, history):
    query = query.strip()

    if not retrieved_results and language in NOT_FOUND_MESSAGES:
        return NOT_FOUND_MESSAGES[language]

    client = get_groq_client()
    if client is None:
        return GENERIC_ERROR

    context = format_retrieved_context(retrieved_results) or "(No relevant information was found.)"
    prompt = SYSTEM_PROMPT.replace("{language}", language_label(language)).replace("{context}", context)

    messages = [{"role": "system", "content": prompt}]
    for item in history[-4:]:
        messages.append({"role": item["role"], "content": item["content"][:600]})
    messages.append({"role": "user", "content": query})

    try:
        answer = call_llm(messages, max_tokens=1500, temperature=0.1)
        answer = sanitize_answer(answer)
        if not answer:
            return NOT_FOUND_MESSAGES.get(language, NOT_FOUND_MESSAGES["english"])
        return answer
    except Exception:
        return GENERIC_ERROR


# =========================================================
# TEXT-TO-SPEECH
# =========================================================

def to_urdu_speech_text(text):
    """
    gTTS's Urdu voice cannot read Latin letters properly, so Roman Urdu (and
    English brand words inside Urdu) are rewritten in Urdu script for speaking.
    """
    if get_groq_client() is None:
        return ""
    try:
        system = (
            "Rewrite the user's text in natural Urdu script so a text-to-speech engine can read it aloud. "
            "If it is Roman Urdu, transliterate it to Urdu script. Write English words, brand names and units "
            "phonetically in Urdu script (e.g. PTCL -> پی ٹی سی ایل, Mbps -> ایم بی پی ایس). Keep numbers as digits. "
            "Output only the rewritten text."
        )
        return call_llm(
            [{"role": "system", "content": system}, {"role": "user", "content": text}],
            max_tokens=1200,
            temperature=0,
        )
    except Exception:
        return ""


def prepare_speech_text(answer, language):
    text = truncate_for_speech(clean_text_for_tts(answer))
    if not text:
        return ""
    if language == "roman_urdu":
        return to_urdu_speech_text(text)
    if language == "urdu" and re.search(r"[A-Za-z]", text):
        return to_urdu_speech_text(text) or text
    return text


def generate_tts_audio(text, language):
    """Return MP3 bytes spoken in the same language as the answer (or None)."""
    lang_code = TTS_LANG_MAP.get(language)
    if not text or not lang_code:
        return None
    try:
        speech = prepare_speech_text(text, language)
        if not speech:
            return None
        buffer = BytesIO()
        gTTS(text=speech, lang=lang_code, slow=False).write_to_fp(buffer)
        buffer.seek(0)
        return buffer.read()
    except Exception:
        return None


def render_audio(audio_bytes, autoplay=False):
    if not audio_bytes:
        return
    if autoplay:
        try:
            st.audio(audio_bytes, format="audio/mp3", autoplay=True)
            return
        except TypeError:
            pass  # older Streamlit without autoplay support
    st.audio(audio_bytes, format="audio/mp3")


def render_text(text):
    """Render answer text, switching to right-to-left layout for Urdu/Arabic-script text."""
    if is_rtl_text(text):
        st.markdown(f'<div class="rtl-block" dir="rtl">\n\n{text}\n\n</div>', unsafe_allow_html=True)
    else:
        st.markdown(text)


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


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


@st.cache_data(ttl=120, show_spinner=False)
def get_knowledge_base_status():
    """Cached so documents are not re-hashed on every Streamlit rerun."""
    try:
        status, _current, _saved = compare_knowledge_base_manifests()
        return {"status": status}
    except Exception:
        return {"status": "error"}


def rebuild_knowledge_base():
    import subprocess
    import sys

    build_script = PROJECT_DIR / "scripts" / "build_knowledge_base.py"
    if not build_script.exists():
        return False, "Build script was not found."

    try:
        result = subprocess.run(
            [sys.executable, str(build_script)],
            cwd=str(PROJECT_DIR),
            capture_output=True,
            text=True,
            timeout=1800,
        )

        if result.returncode != 0:
            return False, "Rebuild failed. Please check the build script logs."

        for cache_func in [
            load_faiss_index, load_chunks, load_metadata,
            load_manifest, load_embedding_model, get_knowledge_base_status,
        ]:
            try:
                cache_func.clear()
            except Exception:
                pass

        return True, "Knowledge base rebuilt successfully."
    except subprocess.TimeoutExpired:
        return False, "Rebuild timed out."
    except Exception:
        return False, "Rebuild could not be completed."


# =========================================================
# SIDEBAR
# =========================================================

SUGGESTIONS = [
    "What internet packages are available?",
    "Tell me about Flash Fiber.",
    "What are the PTCL voice and mobile packages?",
    "انٹرنیٹ کے کون کون سے پیکیجز ہیں؟",
    "Shoq TV ke baray mein bataen",
    "Speed Bolt-On kya hai?",
    "What is Quad Play?",
    "Advance packages kon kon se hain?",
    "What is the cheapest internet package?",
    "Flash Fiber ki speed aur price kya hai?",
    "Voice packages mein kitne minutes milte hain?",
    "Which packages include free SMS?",
    "How do I activate or deactivate a package?",
    "Package ki validity kitni hoti hai?",
    "فلیش فائبر کے بارے میں بتائیں",
    "مجھے وائس پیکیجز کی تفصیل بتائیں",
    # ---- 10 new questions ----
    "Which internet package is best for home use?",
    "What is the difference between Flash Fiber and Speed Bolt-On?",
    "Quad Play mein kya kya milta hai?",
    "Shoq TV mein kitne channels milte hain?",
    "Internet package ki monthly price kitni hai?",
    "How can I upgrade my internet package?",
    "PTCL to PTCL free calling hai kya?",
    "Advance package ki validity aur price bataen",
    "کم قیمت انٹرنیٹ پیکیج کون سا ہے؟",
    "کواڈ پلے میں کیا کیا ملتا ہے؟",
]

# (background, hover background, accent stripe) - light shades, text stays dark.
SUGGESTION_COLORS = [
    ("#CCFBF1", "#99F6E4", "#0D9488"),  # teal
    ("#DBEAFE", "#BFDBFE", "#2563EB"),  # blue
    ("#EDE9FE", "#DDD6FE", "#7C3AED"),  # violet
    ("#FCE7F3", "#FBCFE8", "#DB2777"),  # pink
    ("#FFEDD5", "#FED7AA", "#EA580C"),  # orange
    ("#DCFCE7", "#BBF7D0", "#16A34A"),  # green
    ("#CFFAFE", "#A5F3FC", "#0891B2"),  # cyan
    ("#FEF9C3", "#FEF08A", "#CA8A04"),  # yellow
]


def build_suggestion_css():
    rules = []
    for i in range(len(SUGGESTIONS)):
        bg, hover, accent = SUGGESTION_COLORS[i % len(SUGGESTION_COLORS)]
        rules.append(
            f".st-key-sq_{i} button {{"
            f" background: {bg} !important;"
            f" border: 1px solid {hover} !important;"
            f" border-left: 6px solid {accent} !important;"
            f" color: #0B1F3A !important; height: auto; min-height: 2.6rem;"
            f" white-space: normal; justify-content: flex-start; }}"
            f" .st-key-sq_{i} button p {{ color: #0B1F3A !important; text-align: left;"
            f" font-weight: 500; unicode-bidi: plaintext; }}"
            f" .st-key-sq_{i} button:hover {{ background: {hover} !important;"
            f" border-color: {accent} !important; }}"
        )
    return "<style>" + "\n".join(rules) + "</style>"


st.markdown(build_suggestion_css(), unsafe_allow_html=True)

required_files = [CHUNKS_FILE, METADATA_FILE, MANIFEST_FILE, FAISS_FILE]
kb_ready = all(file.exists() for file in required_files)

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-logo">📡</div>
            <div>
                <div class="brand-name">PTCL Assistant</div>
                <div class="brand-tag">Packages &amp; services, answered</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="side-title">Topics you can ask about</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="pill-wrap">
            <span class="pill">Internet</span>
            <span class="pill">Flash Fiber</span>
            <span class="pill">Voice &amp; Mobile</span>
            <span class="pill">Shoq TV</span>
            <span class="pill">Speed Bolt-On</span>
            <span class="pill">Quad Play</span>
            <span class="pill">Advance Packages</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pop("pending_query", None)
        st.rerun()

    with st.expander("Admin tools"):
        if st.button("Rebuild knowledge base", use_container_width=True):
            with st.spinner("Rebuilding..."):
                rebuild_success, rebuild_message = rebuild_knowledge_base()
            if rebuild_success:
                st.success(rebuild_message)
                st.rerun()
            else:
                st.error(rebuild_message)

    st.markdown("---")
    st.markdown(
        '<div class="dev-credit">Designed &amp; developed by<br><b>Areeba Imran</b></div>',
        unsafe_allow_html=True,
    )


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">PTCL Assistant</div>
        <div class="hero-sub">
            Ask about internet, Flash Fiber, voice and mobile packages, TV and more —
            type or speak in the language you're comfortable with.
        </div>
        <div class="hero-badges">
            <span class="badge">🎙️ Voice enabled</span>
            <span class="badge">🌐 Urdu · English · Roman Urdu</span>
            <span class="badge">⚡ Instant answers</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LAYOUT: [ chatbot panel (chat + type + mic) | suggestions on the right ]
# =========================================================

def keyed_container(key, **kwargs):
    """st.container with a CSS-hook key (falls back on older Streamlit)."""
    try:
        return st.container(key=key, **kwargs)
    except TypeError:
        return st.container(**kwargs)


def scroll_container(height):
    try:
        return st.container(height=height, border=False)
    except TypeError:
        return st.container()


def render_message(message):
    avatar = "📡" if message["role"] == "assistant" else "🙋"
    with st.chat_message(message["role"], avatar=avatar):
        render_text(message["content"])
        if message["role"] == "user" and message.get("voice"):
            st.caption("🎙️ Voice message")
        if message["role"] == "assistant":
            render_audio(message.get("audio"))


pending_query = st.session_state.pop("pending_query", None)

chat_col, side_col = st.columns([3.1, 1.25], gap="large")

# ---- right side: light dropdown, always open, one tap = direct answer ----
with side_col:
    with keyed_container("sq_dropdown"):
        with st.expander("💡 Suggested questions", expanded=True):
            with scroll_container(540):
                for position, suggestion in enumerate(SUGGESTIONS):
                    if st.button(suggestion, use_container_width=True, key=f"sq_{position}"):
                        st.session_state.pending_query = suggestion
                        st.rerun()

# ---- left side: the chatbot panel with the glowing border ----
with chat_col:
    with keyed_container("chat_panel", border=True):
        chat_area = st.container()
        composer = st.container()

voice_query = None
with composer:
    st.markdown(
        '<div class="composer-note">Type or speak in any language — replies come in the same language.</div>',
        unsafe_allow_html=True,
    )
    try:
        type_col, mic_col = st.columns([3.6, 1.4], vertical_alignment="center")
    except TypeError:
        type_col, mic_col = st.columns([3.6, 1.4])

    with type_col:
        text_query = st.chat_input("Ask about PTCL packages — English, اردو or Roman Urdu")

    with mic_col:
        recording = mic_recorder(
            start_prompt="🎙️ Speak",
            stop_prompt="⏹️ Stop & send",
            just_once=True,
            use_container_width=True,
            key="ptcl_voice_recorder",
        )
    voice_audio = recording.get("bytes") if recording else None

    if voice_audio:
        with st.spinner("Listening to your voice..."):
            try:
                heard = transcribe_audio(voice_audio)
                if is_valid_transcript(heard):
                    voice_query = heard
                else:
                    st.warning(
                        "I couldn't hear that clearly. Please speak closer to the mic and try again. / "
                        "آواز صاف نہیں تھی، براہ کرم دوبارہ کوشش کریں۔"
                    )
            except Exception:
                st.error("Voice transcription failed. Please try recording again.")

active_query = None
from_voice = False
if voice_query:
    active_query, from_voice = voice_query, True
elif text_query and text_query.strip():
    active_query = text_query.strip()
elif pending_query and pending_query.strip():
    active_query = pending_query.strip()


# =========================================================
# CHAT
# =========================================================

with chat_area:
    for message in st.session_state.messages:
        render_message(message)

    if active_query:
        history = list(st.session_state.messages)[-6:]
        user_message = {"role": "user", "content": active_query, "voice": from_voice}
        st.session_state.messages.append(user_message)
        render_message(user_message)

        with st.chat_message("assistant", avatar="📡"):
            if not kb_ready:
                error_text = "The assistant is not ready yet. Please build the knowledge base first."
                st.error(error_text)
                st.session_state.messages.append({"role": "assistant", "content": error_text, "audio": None})
            else:
                try:
                    with st.spinner("Thinking..."):
                        understanding = understand_query(active_query, history)
                        language = understanding["language"]

                        if understanding["intent"] in ("greeting", "thanks"):
                            answer = generate_smalltalk(active_query, language)
                        else:
                            knowledge_base = initialize_knowledge_base()
                            retrieved_results = retrieve_relevant_chunks(
                                query=understanding["english_query"],
                                knowledge_base=knowledge_base,
                            )
                            if not retrieved_results and understanding["english_query"] != active_query:
                                retrieved_results = retrieve_relevant_chunks(
                                    query=active_query,
                                    knowledge_base=knowledge_base,
                                )
                            answer = generate_grounded_answer(
                                query=active_query,
                                language=language,
                                retrieved_results=retrieved_results,
                                history=history,
                            )

                        audio_bytes = (
                            generate_tts_audio(answer, language)
                            if answer and answer != GENERIC_ERROR
                            else None
                        )

                    render_text(answer)
                    render_audio(audio_bytes, autoplay=from_voice)

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "audio": audio_bytes}
                    )
                except Exception:
                    st.error(GENERIC_ERROR)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": GENERIC_ERROR, "audio": None}
                    )


# =========================================================
# FOOTER (very end of the page)
# =========================================================

st.markdown(
    """
    <div class="app-footer">
        <div>© 2026 <b>PTCL Assistant</b> · Designed &amp; developed by <b>Areeba Imran</b>. All rights reserved.</div>
        <div class="note">Answers are based on the PTCL package information available to this assistant.
        For the latest offers and official confirmation, please contact PTCL directly.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
