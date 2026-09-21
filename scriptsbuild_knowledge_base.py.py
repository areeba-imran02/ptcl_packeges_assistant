import os
import json
import pickle
import hashlib
from pathlib import Path
import fitz  # PyMuPDF
from docx import Document
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

KB_DIR = Path("knowledge_base")
DATA_DIR = Path("data")
FAISS_DIR = Path("faiss_index")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def get_file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def extract_text_from_pdf(pdf_path):
    text_pages = []
    try:
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                text_pages.append((page_num + 1, text))
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
    return text_pages

def extract_text_from_docx(docx_path):
    text_pages = []
    try:
        doc = Document(docx_path)
        full_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        if full_text:
            text_pages.append((1, full_text))
    except Exception as e:
        print(f"Error reading DOCX {docx_path}: {e}")
    return text_pages

def extract_text_from_txt(txt_path):
    text_pages = []
    try:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            if content.strip():
                text_pages.append((1, content))
    except Exception as e:
        print(f"Error reading TXT {txt_path}: {e}")
    return text_pages

def chunk_text(text, chunk_size=600, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def build_kb():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    
    if not KB_DIR.exists() or not list(KB_DIR.iterdir()):
        print("No documents found in knowledge_base/. Please add documents first.")
        return

    manifest = {}
    all_chunks = []
    all_metadata = []

    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    for file_path in KB_DIR.iterdir():
        if file_path.is_dir():
            continue
        
        ext = file_path.suffix.lower()
        print(f"Processing: {file_path.name}")
        
        file_hash = get_file_hash(file_path)
        manifest[file_path.name] = {
            "size": file_path.stat().st_size,
            "modified": file_path.stat().st_mtime,
            "hash": file_hash
        }

        extracted = []
        if ext == ".pdf":
            extracted = extract_text_from_pdf(file_path)
        elif ext == ".docx":
            extracted = extract_text_from_docx(file_path)
        elif ext == ".txt":
            extracted = extract_text_from_txt(file_path)
        else:
            continue

        for page_num, text in extracted:
            chunks = chunk_text(text)
            for idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    "source_file": file_path.name,
                    "page_number": page_num,
                    "document_title": file_path.stem,
                    "chunk_id": f"{file_path.name}_p{page_num}_c{idx}"
                })

    if not all_chunks:
        return

    embeddings = model.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))

    faiss.write_index(index, str(FAISS_DIR / "index.faiss"))
    with open(DATA_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    with open(DATA_DIR / "metadata.pkl", "wb") as f:
        pickle.dump(all_metadata, f)
    with open(DATA_DIR / "documents_manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)

if __name__ == "__main__":
    build_kb()