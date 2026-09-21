import os
import json
import pickle
from pathlib import Path
import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from gtts import gTTS
import tempfile
from streamlit_mic_recorder import mic_recorder
from faster_whisper import WhisperModel

# Page configuration
st.set_page_config(
    page_title="PTCL Packages Assistant",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
KB_DIR = Path("knowledge_base")
DATA_DIR = Path("data")
FAISS_DIR = Path("faiss_index")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 1.2

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

@st.cache_resource
def load_whisper_model():
    return WhisperModel("small", device="cpu", compute_type="int8")

@st.cache_resource
def load_vector_store():
    index_path = FAISS_DIR / "index.faiss"
    chunks_path = DATA_DIR / "chunks.pkl"
    meta_path = DATA_DIR / "metadata.pkl"
    
    if not (index_path.exists() and chunks_path.exists() and meta_path.exists()):
        return None, None, None
    
    index = faiss.read_index(str(index_path))
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)
        
    return index, chunks, metadata

def check_manifest_changed():
    manifest_path = DATA_DIR / "documents_manifest.json"
    if not manifest_path.exists():
        return True
    
    with open(manifest_path, "r") as f:
        saved_manifest = json.load(f)
        
    current_files = list(KB_DIR.iterdir()) if KB_DIR.exists() else []
    current_filenames = {f.name for f in current_files if not f.is_dir()}
    
    if set(saved_manifest.keys()) != current_filenames:
        return True
        
    for f in current_files:
        if f.is_dir():
            continue
        if f.name in saved_manifest:
            if f.stat().st_size != saved_manifest[f.name]["size"] or f.stat().st_mtime != saved_manifest[f.name]["modified"]:
                return True
    return False

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("PTCL Packages Assistant")
st.markdown("Your intelligent guide to PTCL packages and services.")
st.markdown("---")

with st.sidebar:
    st.subheader("Knowledge Base Status")
    index, chunks, metadata = load_vector_store()
    kb_changed = check_manifest_changed()
    
    doc_count = len(list(KB_DIR.iterdir())) if KB_DIR.exists() else 0
    chunk_count = len(chunks) if chunks else 0
    
    st.markdown(f"**Documents:** {doc_count}")
    st.markdown(f"**Chunks:** {chunk_count}")
    
    if index is not None and not kb_changed:
        st.success("FAISS Index: Ready")
    else:
        st.warning("FAISS Index: Needs Rebuild / Missing")
        
    st.markdown("---")
    st.subheader("Actions")
    if st.button("Rebuild Knowledge Base"):
        with st.spinner("Rebuilding knowledge base..."):
            import subprocess
            result = subprocess.run(["python", "scripts/build_knowledge_base.py"], capture_output=True, text=True)
            if result.returncode == 0:
                st.success("Knowledge base rebuilt successfully!")
                st.rerun()
            else:
                st.error("Error building knowledge base.")
                
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("---")
    st.markdown("### About")
    st.markdown("PTCL Packages Assistant\nA retrieval-based educational assistant built from the provided PTCL knowledge base.")
    st.markdown("**Designed & Developed by Areeba Imran**")

api_key = os.getenv("GROQ_API_KEY")

col_main, col_info = st.columns([3, 1])

with col_info:
    st.container(border=True)
    st.markdown("#### Assistant Info")
    st.markdown("Ask questions in **English**, **Urdu**, or **Roman Urdu**.")
    st.markdown("---")
    st.markdown("#### Supported Queries")
    st.markdown("- Internet Packages\n- Call & SMS Bundles\n- Prices & Validity\n- Activation Codes")

with col_main:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("Sources"):
                    for src in message["sources"]:
                        st.markdown(f"- **Document:** {src['source_file']} (Page {src['page_number']}) — *Similarity Score:* {src['score']:.2f}")
            if "audio_bytes" in message and message["audio_bytes"]:
                st.audio(message["audio_bytes"], format="audio/mp3")

    st.markdown("#### Voice Input")
    voice_audio = mic_recorder(start_prompt="Record Question", stop_prompt="Stop Recording", key='mic')

    user_query = None
    
    if voice_audio:
        audio_bytes_data = voice_audio['bytes']
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_bytes_data)
            temp_path = temp_audio.name
            
        with st.spinner("Transcribing speech..."):
            try:
                whisper_model = load_whisper_model()
                segments, _ = whisper_model.transcribe(temp_path, beam_size=5)
                transcript = " ".join([segment.text for segment in segments])
                if transcript.strip():
                    user_query = transcript.strip()
                    st.info(f"**Transcript:** {user_query}")
            except Exception as e:
                st.error("Could not process audio recording.")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    typed_query = st.chat_input("Ask your question about PTCL packages...")
    if typed_query:
        user_query = typed_query

    if user_query:
        if not api_key:
            st.error("Groq API key is not configured. Please add GROQ_API_KEY to the environment before using the assistant.")
        elif index is None or chunks is None:
            st.error("Knowledge base index is missing. Please run the knowledge base builder first.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)
                
            with st.spinner("Searching knowledge base & generating answer..."):
                embed_model = load_embedding_model()
                q_embedding = embed_model.encode([user_query], convert_to_numpy=True).astype("float32")
                
                k = min(4, len(chunks))
                distances, indices = index.search(q_embedding, k)
                
                retrieved_context = []
                sources = []
                for dist, idx in zip(distances[0], indices[0]):
                    if dist <= SIMILARITY_THRESHOLD:
                        retrieved_context.append(chunks[idx])
                        meta = metadata[idx].copy()
                        meta["score"] = float(dist)
                        sources.append(meta)
                
                answer = ""
                if not retrieved_context:
                    answer = "I couldn't find this information in the available PTCL knowledge base."
                else:
                    context_str = "\n\n".join(retrieved_context)
                    system_prompt = (
                        "You are a professional PTCL Packages & Services Knowledge Assistant.\n"
                        "Use ONLY the retrieved knowledge-base context provided below to answer the user's question.\n"
                        "Do not use outside knowledge or invent packages/prices.\n"
                        "Respond in the same language or script used by the user (English, Urdu, or Roman Urdu)."
                    )
                    user_prompt = f"Context:\n{context_str}\n\nQuestion: {user_query}"
                    
                    try:
                        client = Groq(api_key=api_key)
                        completion = client.chat.completions.create(
                            model="openai/gpt-oss-120b",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            temperature=0.1,
                            max_tokens=1024
                        )
                        answer = completion.choices[0].message.content
                    except Exception as e:
                        answer = f"An error occurred while communicating with Groq: {str(e)}"

                audio_bytes = None
                try:
                    lang_code = "en"
                    if any(ord(char) > 127 for char in user_query):
                        lang_code = "ur"
                        
                    tts = gTTS(text=answer[:500], lang=lang_code, slow=False)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tf:
                        tts.save(tf.name)
                        tf_path = tf.name
                    with open(tf_path, "rb") as af:
                        audio_bytes = af.read()
                    os.remove(tf_path)
                except Exception:
                    st.warning("Audio could not be generated for this response.")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "audio_bytes": audio_bytes
                })
                
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    if sources:
                        with st.expander("Sources"):
                            for src in sources:
                                st.markdown(f"- **Document:** {src['source_file']} (Page {src['page_number']}) — *Similarity Score:* {src['score']:.2f}")
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.9em;'>"
    "Designed & Developed by Areeba Imran<br>© 2026 Areeba Imran. All rights reserved."
    "</div>",
    unsafe_allow_html=True
)
