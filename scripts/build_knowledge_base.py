from __future__ import annotations

import os
import pickle
from pathlib import Path
import faiss
import numpy as np
import streamlit as st
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
from groq import Groq

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="PTCL Packages Assistant",
    page_icon="🤖",
    layout="centered"
)

# ============================================================
# Paths Configuration
# ============================================================
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
FAISS_DIR = PROJECT_DIR / "faiss_index"

CHUNKS_PATH = DATA_DIR / "chunks.pkl"
METADATA_PATH = DATA_DIR / "metadata.pkl"
FAISS_INDEX_PATH = FAISS_DIR / "index.faiss"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ============================================================
# Load Models & Data (Cached for Performance)
# ============================================================
@st.cache_resource
def load_resources():
    # Load Embedding Model
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # Load FAISS Index
    if FAISS_INDEX_PATH.exists():
        index = faiss.read_index(str(FAISS_INDEX_PATH))
    else:
        index = None

    # Load Chunks and Metadata
    chunks = []
    metadata = []
    if CHUNKS_PATH.exists():
        with CHUNKS_PATH.open("rb") as f:
            chunks = pickle.load(f)
            
    if METADATA_PATH.exists():
        with METADATA_PATH.open("rb") as f:
            metadata = pickle.load(f)
            
    return embed_model, index, chunks, metadata

@st.cache_resource
def load_whisper_model():
    # Using 'small' or 'base' for robust multi-language transcription
    return WhisperModel("small", device="cpu", compute_type="int8")

embed_model, faiss_index, chunks, metadata = load_resources()
whisper_model = load_whisper_model()

# Initialize Groq Client (Make sure GROQ_API_KEY is in your environment variables or Streamlit secrets)
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", st.secrets.get("GROQ_API_KEY", "")))

# ============================================================
# Helper Functions
# ============================================================
def search_knowledge_base(query: str, top_k: int = 3):
    if not faiss_index or not chunks:
        return []
    
    query_vector = embed_model.encode(
        [query], 
        convert_to_numpy=True, 
        normalize_embeddings=True
    )
    query_vector = np.asarray(query_vector, dtype=np.float32)
    
    distances, indices = faiss_index.search(query_vector, top_k)
    
    results = []
    for idx, score in zip(indices[0], distances[0]):
        if idx != -1 and idx < len(chunks):
            results.append({
                "chunk": chunks[idx],
                "metadata": metadata[idx],
                "score": float(score)
            })
    return results

def generate_response(user_query: str, retrieved_context: list):
    context_text = "\n\n".join([item["chunk"] for item in retrieved_context])
    
    system_prompt = """
Aap ek madadgar PTCL Packages Assistant hain. 
Aapka kaam sirf di gayi knowledge base (context) ke mutabiq PTCL ke internet, Flash Fiber, landline, aur packages ke baray mein sawalon ke jawab dena hai.

Instructions:
1. User jis zaban (English, Roman Urdu, ya Urdu script) mein sawal pooche, usay achhi tarah samajh kar usi lehje ya Roman Urdu mein jawab dein.
2. Agar sawal ka jawab knowledge base mein mojood na ho, to polite tareeqay se batayen ke yeh maloomat dastiyab nahi hain aur milte-julte packages ki list dein.
3. Kisi bhi ghair-mutaliqa (irrelevant) ya galat mishear ki gayi term ka jawab na dein balkay user ko guide karein.
"""

    user_prompt = f"""
Context from Knowledge Base:
{context_text}

User Query:
{user_query}
"""

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Aap apne pasand ka Groq model use kar sakti hain
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error generating response from Groq: {e}"

# ============================================================
# Streamlit UI
# ============================================================
st.title("🤖 PTCL Packages Assistant")
st.write("Aap PTCL packages ke baray mein likh kar ya bol kar (Voice/Audio) pooch sakte hain!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Audio Input Section using Streamlit audio recorder or file uploader
st.markdown("---")
st.subheader("🎤 Voice Input")
audio_file = st.file_uploader("Upload an audio file (.wav, .mp3, .m4a)", type=["wav", "mp3", "m4a"])

user_query = None

if audio_file is not None:
    # Save temporary audio file
    temp_audio_path = PROJECT_DIR / "temp_audio.wav"
    with open(temp_audio_path, "wb") as f:
        f.write(audio_file.getbuffer())
    
    with st.spinner("🔄 Transcribing audio (Multi-language support)..."):
        # Transcribe with multi-language auto-detect and vocabulary prompt hint
        segments, info = whisper_model.transcribe(
            str(temp_audio_path),
            beam_size=5,
            language=None,  # Auto-detects English, Urdu, or Roman Urdu
            initial_prompt="PTCL internet packages, Flash Fiber, landline, broadband, unlimited, student packages, PTCL ke packages batao"
        )
        transcript_text = " ".join([segment.text for segment in segments]).strip()
        
    if temp_audio_path.exists():
        temp_audio_path.unlink()  # Clean up temp file
        
    if transcript_text:
        st.info(f"**Voice transcript:** {transcript_text}")
        user_query = transcript_text

# Text input fallback/alternative
text_query = st.chat_input("Ya yahan type karein (e.g., PTCL Flash Fiber packages batao)...")

if text_query:
    user_query = text_query

# Process the query if available (from voice or text)
if user_query:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching PTCL knowledge base..."):
            retrieved_docs = search_knowledge_base(user_query, top_k=3)
            response = generate_response(user_query, retrieved_docs)
            st.markdown(response)
            
    st.session_state.messages.append({"role": "assistant", "content": response})
