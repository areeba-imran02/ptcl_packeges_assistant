
from __future__ import annotations

import hashlib
import json
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
import pymupdf
from docx import Document
from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

KNOWLEDGE_BASE_DIR = PROJECT_DIR / "knowledge_base"
DATA_DIR = PROJECT_DIR / "data"
FAISS_DIR = PROJECT_DIR / "faiss_index"

CHUNKS_PATH = DATA_DIR / "chunks.pkl"
METADATA_PATH = DATA_DIR / "metadata.pkl"
MANIFEST_PATH = DATA_DIR / "documents_manifest.json"
FAISS_INDEX_PATH = FAISS_DIR / "index.faiss"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Approximate word-based chunking.
CHUNK_SIZE_WORDS = 650
CHUNK_OVERLAP_WORDS = 100

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}


# ============================================================
# Directory Setup
# ============================================================

def ensure_directories() -> None:
    KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FAISS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Text Cleaning
# ============================================================

def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive horizontal whitespace while preserving lines.
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# PDF Extraction
# ============================================================

def extract_pdf(path: Path) -> List[Dict]:
    pages = []

    try:
        document = pymupdf.open(path)

        try:
            for page_number, page in enumerate(document, start=1):
                text = clean_text(page.get_text("text"))

                if text:
                    pages.append(
                        {
                            "text": text,
                            "page_number": page_number,
                            "document_title": path.stem,
                        }
                    )
        finally:
            document.close()

    except Exception as exc:
        print(f"⚠️ Skipping corrupted/unreadable PDF: {path.name}")
        print(f"   Reason: {exc}")

    return pages


# ============================================================
# TXT Extraction
# ============================================================

def extract_txt(path: Path) -> List[Dict]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        text = clean_text(text)

        if text:
            return [
                {
                    "text": text,
                    "page_number": None,
                    "document_title": path.stem,
                }
            ]

    except Exception as exc:
        print(f"⚠️ Could not read TXT file: {path.name}")
        print(f"   Reason: {exc}")

    return []


# ============================================================
# DOCX Extraction
# ============================================================

def extract_docx(path: Path) -> List[Dict]:
    try:
        document = Document(path)

        paragraphs = []

        for paragraph in document.paragraphs:
            text = clean_text(paragraph.text)
            if text:
                paragraphs.append(text)

        # Preserve table content in a readable text representation.
        for table in document.tables:
            for row in table.rows:
                cells = [
                    clean_text(cell.text)
                    for cell in row.cells
                ]

                row_text = " | ".join(
                    cell for cell in cells if cell
                )

                if row_text:
                    paragraphs.append(row_text)

        text = clean_text("\n".join(paragraphs))

        if text:
            return [
                {
                    "text": text,
                    "page_number": None,
                    "document_title": path.stem,
                }
            ]

    except Exception as exc:
        print(f"⚠️ Could not read DOCX file: {path.name}")
        print(f"   Reason: {exc}")

    return []


# ============================================================
# Generic Document Extraction
# ============================================================

def extract_document(path: Path) -> List[Dict]:
    extension = path.suffix.lower()

    if extension == ".pdf":
        return extract_pdf(path)

    if extension == ".txt":
        return extract_txt(path)

    if extension == ".docx":
        return extract_docx(path)

    return []


# ============================================================
# Chunking
# ============================================================

def chunk_words(
    text: str,
    chunk_size: int = CHUNK_SIZE_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS,
) -> List[str]:

    words = text.split()

    if not words:
        return []

    if overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))

        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks


# ============================================================
# File Manifest
# ============================================================

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def build_manifest(files: List[Path]) -> List[Dict]:
    manifest = []

    for path in files:
        stat = path.stat()

        manifest.append(
            {
                "filename": path.name,
                "relative_path": str(
                    path.relative_to(KNOWLEDGE_BASE_DIR)
                ),
                "file_size": stat.st_size,
                "modified_timestamp": stat.st_mtime,
                "sha256": sha256_file(path),
            }
        )

    return manifest


# ============================================================
# Main Build Process
# ============================================================

def main() -> None:
    ensure_directories()

    files = sorted(
        [
            path
            for path in KNOWLEDGE_BASE_DIR.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    )

    if not files:
        print("\n❌ No supported knowledge-base documents found.")
        print(
            "Please add PDF, TXT, or DOCX files to:"
        )
        print(KNOWLEDGE_BASE_DIR)
        return

    print("=" * 70)
    print("PTCL PACKAGES ASSISTANT — KNOWLEDGE BASE BUILD")
    print("=" * 70)
    print(f"Documents found: {len(files)}")
    print(f"Embedding model: {EMBEDDING_MODEL_NAME}")
    print()

    all_chunks = []
    all_metadata = []

    chunk_counter = 0

    for document_path in files:
        print(f"📄 Processing: {document_path.name}")

        sections = extract_document(document_path)

        if not sections:
            print("   ⚠️ No readable text found.")
            continue

        document_chunk_count = 0

        for section in sections:
            section_chunks = chunk_words(section["text"])

            for chunk_text in section_chunks:
                chunk_id = f"chunk_{chunk_counter:06d}"

                metadata = {
                    "source_file": document_path.name,
                    "page_number": section["page_number"],
                    "document_title": section["document_title"],
                    "chunk_id": chunk_id,
                }

                all_chunks.append(chunk_text)
                all_metadata.append(metadata)

                chunk_counter += 1
                document_chunk_count += 1

        print(f"   ✅ Chunks created: {document_chunk_count}")

    if not all_chunks:
        print("\n❌ No usable text chunks were created.")
        return

    print()
    print(f"✅ Total chunks: {len(all_chunks)}")
    print()
    print("🔄 Loading embedding model...")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("🔄 Generating embeddings...")

    embeddings = model.encode(
        all_chunks,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(
        index,
        str(FAISS_INDEX_PATH),
    )

    with CHUNKS_PATH.open("wb") as file:
        pickle.dump(all_chunks, file)

    with METADATA_PATH.open("wb") as file:
        pickle.dump(all_metadata, file)

    manifest = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "document_count": len(files),
        "chunk_count": len(all_chunks),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "documents": build_manifest(files),
    }

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 70)
    print("✅ KNOWLEDGE BASE BUILD COMPLETE")
    print("=" * 70)
    print(f"Chunks:   {CHUNKS_PATH}")
    print(f"Metadata: {METADATA_PATH}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"FAISS:    {FAISS_INDEX_PATH}")
    print(f"Vectors:  {index.ntotal}")
    print("=" * 70)


if __name__ == "__main__":
    main()
