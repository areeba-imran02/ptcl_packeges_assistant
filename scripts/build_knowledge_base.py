from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import tempfile
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from faster_whisper import WhisperModel
from groq import Groq
from sentence_transformers import SentenceTransformer

# ============================================================
# Page configuration (must be the first Streamlit call)
# ============================================================
st.set_page_config(
    page_title="PTCL Packages Assistant",
    page_icon="📡",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ============================================================
# Settings
# ============================================================
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
FAISS_DIR = PROJECT_DIR / "faiss_index"

CHUNKS_PATH = DATA_DIR / "chunks.pkl"
METADATA_PATH = DATA_DIR / "metadata.pkl"  # loaded but never shown to the user
FAISS_INDEX_PATH = FAISS_DIR / "index.faiss"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Bigger model = much better Urdu / Roman Urdu answers.
# If you hit Groq rate limits, change CHAT_MODEL to "llama-3.1-8b-instant".
CHAT_MODEL = os.environ.get("CHAT_MODEL", "llama-3.3-70b-versatile")
ROUTER_MODEL = os.environ.get("ROUTER_MODEL", "llama-3.1-8b-instant")

PTCL_HELPLINE = "1218"
TOP_K = 4

WHISPER_HINT = "PTCL, Flash Fiber, EVO, broadband, internet package, landline. پی ٹی سی ایل انٹرنیٹ پیکیج"

QUICK_QUESTIONS = [
    "PTCL Flash Fiber packages batao",
    "Sab se sasta internet package konsa hai?",
    "Student packages kaunse hain?",
    "Do you have unlimited internet packages?",
    "لینڈ لائن پیکیج کی تفصیل بتائیں",
]

# ============================================================
# Styling
# ============================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Noto+Nastaliq+Urdu:wght@400;600&display=swap');

:root {
  --ink: #0A2540;
  --fiber: #1B6FE0;
  --signal: #12B76A;
  --mist: #EDF3FA;
  --paper: #FFFFFF;
  --line: #D6E2F1;
  --muted: #5B6E86;
}

html, body, [class*="css"], .stApp {
  font-family: 'Plus Jakarta Sans', 'Segoe UI', sans-serif;
}

.stApp {
  background:
    radial-gradient(1100px 500px at 100% -10%, rgba(27,111,224,.10), transparent 60%),
    radial-gradient(900px 500px at -10% 110%, rgba(18,183,106,.10), transparent 60%),
    var(--mist);
  color: var(--ink);
}

#MainMenu, header[data-testid="stHeader"], footer { visibility: hidden; height: 0; }
.block-container { padding-top: 1.4rem; padding-bottom: 6rem; max-width: 820px; }

/* ---------- Hero ---------- */
.hero {
  position: relative;
  overflow: hidden;
  border-radius: 22px;
  padding: 30px 30px 26px 30px;
  color: #fff;
  background:
    repeating-linear-gradient(115deg, rgba(255,255,255,.05) 0 2px, transparent 2px 22px),
    linear-gradient(120deg, #0A2540 0%, #0F4C9A 55%, #12B76A 130%);
  box-shadow: 0 18px 40px -22px rgba(10,37,64,.65);
}
.hero h1 {
  margin: 0 0 8px 0;
  font-size: 2.05rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  color: #fff;
  line-height: 1.15;
}
.hero p { margin: 0; color: rgba(255,255,255,.86); font-size: 1rem; line-height: 1.6; max-width: 34rem; }
.hero .langs { margin-top: 16px; display: flex; gap: 8px; flex-wrap: wrap; }
.hero .chip {
  background: rgba(255,255,255,.14);
  border: 1px solid rgba(255,255,255,.28);
  padding: 4px 12px;
  border-radius: 999px;
  font-size: .82rem;
  font-weight: 600;
}

/* ---------- Welcome cards ---------- */
.tips { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 18px 0 6px 0; }
.tip {
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 16px 16px 14px 16px;
}
.tip:nth-child(1) { border-top: 4px solid var(--fiber); }
.tip:nth-child(2) { border-top: 4px solid var(--signal); }
.tip:nth-child(3) { border-top: 4px solid var(--ink); }
.tip b { display: block; color: var(--ink); font-size: .98rem; margin-bottom: 4px; }
.tip span { color: var(--muted); font-size: .86rem; line-height: 1.5; }
@media (max-width: 640px) { .tips { grid-template-columns: 1fr; } }

/* ---------- Chat bubbles ---------- */
div[data-testid="stChatMessage"] {
  border-radius: 18px;
  padding: 14px 18px;
  margin-bottom: 12px;
  border: 1px solid var(--line);
  background: var(--paper);
  color: var(--ink);
  box-shadow: 0 6px 18px -14px rgba(10,37,64,.45);
}
div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] li { color: var(--ink); line-height: 1.7; }
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background: linear-gradient(135deg, #E6F0FF 0%, #F2F7FF 100%);
  border-color: #BFD6F7;
}
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
  border-left: 5px solid var(--signal);
}
.rtl { direction: rtl; text-align: right; font-family: 'Noto Nastaliq Urdu', 'Jameel Noori Nastaleeq', serif; line-height: 2.3; }

/* ---------- Voice panel ---------- */
div[data-testid="stExpander"] {
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: 16px;
  margin-top: 18px;
}
div[data-testid="stExpander"] summary p { color: var(--ink); font-weight: 700; }

/* ---------- Chat input ---------- */
div[data-testid="stChatInput"] { border-radius: 16px; border: 2px solid var(--fiber); background: #fff; }
div[data-testid="stChatInput"] textarea { color: var(--ink); }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0A2540 0%, #0D3A75 100%);
}
section[data-testid="stSidebar"] * { color: #EAF2FF; }
section[data-testid="stSidebar"] h3 { color: #fff; font-weight: 800; letter-spacing: -0.01em; }
section[data-testid="stSidebar"] .stButton > button {
  width: 100%;
  text-align: left;
  background: rgba(255,255,255,.08);
  border: 1px solid rgba(255,255,255,.20);
  color: #fff;
  border-radius: 12px;
  padding: 8px 12px;
  font-size: .88rem;
}
section[data-testid="stSidebar"] .stButton > button:hover {
  background: rgba(18,183,106,.28);
  border-color: var(--signal);
}
.side-note { font-size: .82rem; line-height: 1.6; color: #BFD3EE; }

/* ---------- Footer ---------- */
.app-footer {
  margin-top: 34px;
  border-radius: 20px;
  padding: 22px 24px;
  background: var(--ink);
  color: #C9D8EE;
  display: grid;
  grid-template-columns: 1.3fr 1fr 1.3fr;
  gap: 20px;
  font-size: .84rem;
  line-height: 1.6;
}
.app-footer b { color: #fff; display: block; margin-bottom: 4px; font-size: .95rem; }
.app-footer .bar { grid-column: 1 / -1; border-top: 1px solid rgba(255,255,255,.14); padding-top: 12px; color: #8FA8C9; font-size: .78rem; }
@media (max-width: 640px) { .app-footer { grid-template-columns: 1fr; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ============================================================
# Resources
# ============================================================
@st.cache_resource(show_spinner="Assistant tayyar ho raha hai...")
def load_resources():
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    index = faiss.read_index(str(FAISS_INDEX_PATH)) if FAISS_INDEX_PATH.exists() else None

    chunks, metadata = [], []
    if CHUNKS_PATH.exists():
        with CHUNKS_PATH.open("rb") as f:
            chunks = pickle.load(f)
    if METADATA_PATH.exists():
        with METADATA_PATH.open("rb") as f:
            metadata = pickle.load(f)
    return embed_model, index, chunks, metadata


@st.cache_resource(show_spinner="Voice model load ho raha hai...")
def load_whisper_model():
    return WhisperModel("small", device="cpu", compute_type="int8")


def get_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    if key:
        return key
    try:  # st.secrets raises if no secrets.toml exists
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return ""


embed_model, faiss_index, chunks, _metadata = load_resources()
whisper_model = load_whisper_model()

_api_key = get_api_key()
if not _api_key:
    st.error("GROQ_API_KEY nahi mili. Isay environment variable ya .streamlit/secrets.toml mein set karein.")
    st.stop()
groq_client = Groq(api_key=_api_key)

# ============================================================
# Text helpers
# ============================================================
ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")
URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.I)
FILE_RE = re.compile(r"[\w\-.]+\.(?:pdf|docx?|txt|csv|xlsx?|pptx?|md|json)\b", re.I)
SOURCE_LINE_RE = re.compile(r"^\s*\(?\s*(?:sources?|references?|document)\s*[:\-].*$", re.I | re.M)


def has_urdu_script(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    return sum(bool(ARABIC_RE.match(c)) for c in letters) / len(letters) > 0.4


def scrub(text: str) -> str:
    """Remove links, file names and 'source' lines so nothing internal is ever shown."""
    text = URL_RE.sub("", text)
    text = FILE_RE.sub("", text)
    text = SOURCE_LINE_RE.sub("", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def render_text(text: str) -> None:
    if has_urdu_script(text):
        st.markdown(f'<div class="rtl">\n\n{text}\n\n</div>', unsafe_allow_html=True)
    else:
        st.markdown(text)


def recent_history(n: int = 6) -> list[dict]:
    return [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-n:]]


# ============================================================
# LLM helpers
# ============================================================
def call_llm(model: str, messages: list[dict], temperature: float, max_tokens: int, json_mode: bool = False) -> str:
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    completion = groq_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    return completion.choices[0].message.content


def analyze_query(query: str, history: list[dict], voice_lang: str | None) -> tuple[str, str]:
    """
    Understand ANY language the user used.
    Returns (reply_language, english_search_query).
    The knowledge base is English, so the search query is always turned into English.
    """
    hist_txt = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in history[-4:]) or "(none)"
    voice_txt = f"The message came from voice transcription (detected language code: {voice_lang})." if voice_lang else "The message was typed."

    prompt = f"""You prepare customer messages for a PTCL (Pakistan Telecommunication Company Limited) packages assistant.
The message can be English, Urdu script, Roman Urdu (Urdu written in English letters), Punjabi, mixed languages, or a voice transcript containing recognition mistakes.

{voice_txt}

Recent conversation:
{hist_txt}

User message:
{query}

Return ONLY a JSON object with exactly these keys:
- "reply_language": the language AND script the reply must use. One of "English", "Urdu script", "Roman Urdu", or another language name.
- "search_query": a short English search query that captures what the user wants (internet packages, Flash Fiber, landline, prices, speeds, etc.). Fix obvious speech-recognition mistakes, resolve follow-up questions using the conversation, keep product names like Flash Fiber, EVO, Smart TV unchanged."""

    reply_lang, search_query = "", ""
    try:
        raw = call_llm(ROUTER_MODEL, [{"role": "user", "content": prompt}], 0.0, 200, json_mode=True)
        data = json.loads(raw)
        reply_lang = str(data.get("reply_language") or "").strip()
        search_query = str(data.get("search_query") or "").strip()
    except Exception:
        pass

    if has_urdu_script(query):
        reply_lang = "Urdu script"
    return (reply_lang or "the same language and script as the user's message"), (search_query or query)


def search_knowledge_base(queries: list[str], top_k: int = TOP_K) -> list[str]:
    if faiss_index is None or not chunks:
        return []

    best: dict[int, float] = {}
    for q in dict.fromkeys(q for q in queries if q.strip()):
        vec = embed_model.encode([q], convert_to_numpy=True, normalize_embeddings=True)
        vec = np.asarray(vec, dtype=np.float32)
        scores, indices = faiss_index.search(vec, top_k)
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1 and idx < len(chunks):
                best[int(idx)] = max(best.get(int(idx), -1e9), float(score))

    top = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [scrub(str(chunks[i])) for i, _ in top]


def build_system_prompt(reply_language: str) -> str:
    return f"""You are "PTCL Packages Assistant", a friendly and professional customer-support assistant for PTCL internet, Flash Fiber, EVO, landline and related packages in Pakistan.

LANGUAGE RULES (very important)
- Reply language for this message: {reply_language}.
- English question -> English answer. Urdu script question -> Urdu script answer. Roman Urdu question -> Roman Urdu answer (Urdu in English letters). Any other language -> that language if you can, otherwise English.
- Keep package names, prices (Rs.), speeds (Mbps) and data amounts exactly as given.
- The user may be speaking by voice, so the text can contain small recognition mistakes (for example "flash fever" for "Flash Fiber"). Interpret the intent sensibly instead of pointing out mistakes.

ANSWER RULES
- Use ONLY the facts in the information block provided. Never invent prices, speeds, dates or offers.
- If the exact answer is not available, say so politely in the user's language, share the closest matching packages if any, and suggest calling the PTCL helpline {PTCL_HELPLINE}.
- Greetings, thanks and small talk: reply warmly in one or two lines and offer help with packages.
- Unrelated questions: politely bring the conversation back to PTCL packages and services.
- NEVER mention "context", "knowledge base", "documents", "sources", file names, page numbers or links. Speak as if you know this yourself.
- Be clear and concise. Use short bullet points for package lists (name in bold, speed, price, key benefit). Ask at most one follow-up question, only if it really helps."""


def generate_response(query: str, info_chunks: list[str], reply_language: str, history: list[dict]) -> str:
    info_text = "\n\n".join(info_chunks) if info_chunks else "(no matching information found)"
    user_prompt = f"""Information:
{info_text}

Customer message:
{query}

Answer in: {reply_language}"""

    messages = [{"role": "system", "content": build_system_prompt(reply_language)}]
    messages += history
    messages.append({"role": "user", "content": user_prompt})

    try:
        return scrub(call_llm(CHAT_MODEL, messages, 0.3, 900))
    except Exception:
        return "⚠️ Maazrat, abhi jawab tayyar nahi ho saka. Baraye meherbani thori dair baad dobara koshish karein."


# ============================================================
# Voice helpers
# ============================================================
def transcribe_audio(audio_bytes: bytes, suffix: str = ".wav") -> tuple[str, str]:
    """Auto-detect the spoken language and transcribe. Returns (text, language_code)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        path = tmp.name

    def run(language: str | None):
        segments, info = whisper_model.transcribe(
            path,
            beam_size=5,
            language=language,
            vad_filter=True,
            condition_on_previous_text=False,
            initial_prompt=WHISPER_HINT,
        )
        return " ".join(s.text.strip() for s in segments).strip(), info.language

    try:
        text, lang = run(None)
        # Whisper often labels spoken Urdu as Hindi; force Urdu in that case.
        if lang == "hi":
            text, lang = run("ur")
        return text, lang
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def read_new_audio(file_obj, suffix: str):
    """Return the audio bytes only if this recording has not been processed yet."""
    if file_obj is None:
        return None
    data = file_obj.getvalue()
    sig = hashlib.md5(data).hexdigest()
    if sig == st.session_state.last_audio_sig:
        return None
    st.session_state.last_audio_sig = sig
    return data


# ============================================================
# Session state
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_audio_sig" not in st.session_state:
    st.session_state.last_audio_sig = ""

# ============================================================
# Sidebar
# ============================================================
pending: dict | None = None

with st.sidebar:
    st.markdown("### 📡 PTCL Assistant")
    st.markdown(
        '<div class="side-note">Internet, Flash Fiber aur landline packages ke baray mein poochein. '
        "English, اردو ya Roman Urdu, jis mein aap comfortable hon.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("&nbsp;")
    st.markdown("**Jaldi poochein**")
    for i, q in enumerate(QUICK_QUESTIONS):
        if st.button(q, key=f"quick_{i}"):
            pending = {"text": q, "voice": False, "lang": None}

    st.markdown("&nbsp;")
    if st.button("🗑️ Chat saaf karein", key="clear_chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("&nbsp;")
    st.markdown(
        f'<div class="side-note">Madad chahiye? PTCL helpline <b>{PTCL_HELPLINE}</b></div>',
        unsafe_allow_html=True,
    )

# ============================================================
# Header
# ============================================================
st.markdown(
    """
<div class="hero">
<h1>PTCL Packages Assistant</h1>
<p>Apna sawal likhein ya bol kar poochein. Assistant aap ki zaban samajh kar usi zaban mein jawab dega.</p>
<div class="langs"><span class="chip">English</span><span class="chip">اردو</span><span class="chip">Roman Urdu</span></div>
</div>
""",
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.markdown(
        """
<div class="tips">
<div class="tip"><b>Likh kar poochein</b><span>Neeche box mein sawal type karein, jaise "Flash Fiber packages batao".</span></div>
<div class="tip"><b>Bol kar poochein</b><span>Mic dabayein aur apni zaban mein bolein. Audio file bhi upload ho sakti hai.</span></div>
<div class="tip"><b>Har zaban mein jawab</b><span>Jis zaban ya lehjay mein poochein gay, jawab usi mein milega.</span></div>
</div>
""",
        unsafe_allow_html=True,
    )

# ============================================================
# Voice input
# ============================================================
with st.expander("🎙️ Voice se poochein (mic ya audio file)", expanded=False):
    audio_data, audio_suffix = None, ".wav"

    if hasattr(st, "audio_input"):
        mic = st.audio_input("Mic dabayein aur bolein", key="mic_input")
        audio_data = read_new_audio(mic, ".wav")
    else:
        st.info("Mic recording ke liye `pip install -U streamlit` (version 1.40 ya naya) chalayein.")

    upload = st.file_uploader("Ya audio file upload karein", type=["wav", "mp3", "m4a", "ogg", "webm"], key="audio_upload")
    if audio_data is None and upload is not None:
        audio_data = read_new_audio(upload, "")
        audio_suffix = Path(upload.name).suffix or ".wav"

    if audio_data is not None:
        with st.spinner("🔄 Aap ki awaaz sun raha hoon..."):
            transcript, detected_lang = transcribe_audio(audio_data, audio_suffix)
        if transcript:
            pending = {"text": transcript, "voice": True, "lang": detected_lang}
        else:
            st.warning("Awaaz samajh nahi aayi. Baraye meherbani dobara saaf awaaz mein bolein.")

# ============================================================
# Chat history
# ============================================================
for message in st.session_state.messages:
    avatar = "🧑" if message["role"] == "user" else "📡"
    with st.chat_message(message["role"], avatar=avatar):
        if message.get("voice"):
            st.caption("🎙️ Voice message")
        render_text(message["content"])

# ============================================================
# New input (typed text has priority over voice / quick buttons)
# ============================================================
typed = st.chat_input("Yahan apna sawal likhein... (English / اردو / Roman Urdu)")
if typed:
    pending = {"text": typed.strip(), "voice": False, "lang": None}

if pending and pending["text"]:
    history = recent_history()

    with st.chat_message("user", avatar="🧑"):
        if pending["voice"]:
            st.caption("🎙️ Voice message")
        render_text(pending["text"])
    st.session_state.messages.append({"role": "user", "content": pending["text"], "voice": pending["voice"]})

    with st.chat_message("assistant", avatar="📡"):
        with st.spinner("✍️ Jawab tayyar ho raha hai..."):
            reply_lang, search_q = analyze_query(pending["text"], history, pending["lang"])
            info = search_knowledge_base([search_q, pending["text"]])
            answer = generate_response(pending["text"], info, reply_lang, history)
        render_text(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ============================================================
# Footer
# ============================================================
st.markdown(
    f"""
<div class="app-footer">
<div><b>PTCL Packages Assistant</b>Internet, Flash Fiber aur landline packages ki maloomat, aap ki apni zaban mein.</div>
<div><b>Zabanein</b>English<br>اردو<br>Roman Urdu</div>
<div><b>Zaroori note</b>Packages aur qeematein tabdeel ho sakti hain. Aakhri tasdeeq ke liye PTCL helpline {PTCL_HELPLINE} par rabta karein.</div>
<div class="bar">© {datetime.now().year} PTCL Packages Assistant. Sab huqooq mehfooz hain.</div>
</div>
""",
    unsafe_allow_html=True,
)
