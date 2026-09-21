"""
PTCL Packages Assistant — Knowledge Base Builder
Developer: Areeba Imran

This script performs the full offline preprocessing pipeline:

    Documents (PDF / DOCX / TXT)
        -> Text extraction
        -> Cleaning
        -> Chunking (with overlap, metadata-tagged)
        -> Embeddings (local, sentence-transformers)
        -> FAISS index

Run:
    python scripts/build_knowledge_base.py

Outputs (relative to project root):
    data/chunks.pkl
    data/metadata.pkl
    data/documents_manifest.json
    faiss_index/index.faiss

This script is safe to re-run. It always rebuilds from whatever documents
currently exist in knowledge_base/, so it can be wired up later to a
"Rebuild Knowledge Base" button in the Streamlit app without any changes.
"""

import os
import sys
import json
import pickle
import hashlib
import re
from pathlib import Path
from datetime import datetime, timezone

# ----------------------------------------------------------------------
# Paths (all relative to the project root, regardless of cwd)
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
DATA_DIR = PROJECT_ROOT / "data"
FAISS_DIR = PROJECT_ROOT / "faiss_index"

CHUNKS_PATH = DATA_DIR / "chunks.pkl"
METADATA_PATH = DATA_DIR / "metadata.pkl"
MANIFEST_PATH = DATA_DIR / "documents_manifest.json"
FAISS_INDEX_PATH = FAISS_DIR / "index.faiss"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking configuration (words, not strict tokens — see chunk_text())
CHUNK_SIZE_WORDS = 650      # target chunk size, within the 500-800 range
CHUNK_OVERLAP_WORDS = 100   # within the 80-120 range

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}


# ----------------------------------------------------------------------
# Utility: manifest (used for change detection / rebuild decisions)
# ----------------------------------------------------------------------
def compute_file_hash(file_path: Path, block_size: int = 65536) -> str:
    """Compute a SHA-256 hash of a file's contents."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            hasher.update(block)
    return hasher.hexdigest()


def build_manifest_entry(file_path: Path) -> dict:
    stat = file_path.stat()
    return {
        "filename": file_path.name,
        "relative_path": str(file_path.relative_to(PROJECT_ROOT)),
        "size_bytes": stat.st_size,
        "modified_time": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
        "sha256": compute_file_hash(file_path),
    }


def load_previous_manifest() -> dict:
    if MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def has_manifest_changed(old_manifest: dict, new_documents: list) -> bool:
    """
    Compare old vs. new manifest entries by filename + hash.
    Used only for informational logging here; the Streamlit app can reuse
    this same function later to decide whether a rebuild is required.
    """
    old_docs = {d["filename"]: d.get("sha256") for d in old_manifest.get("documents", [])}
    new_docs = {d["filename"]: d.get("sha256") for d in new_documents}
    return old_docs != new_docs


# ----------------------------------------------------------------------
# Text extraction
# ----------------------------------------------------------------------
def extract_pdf(file_path: Path) -> list:
    """
    Extract text from a PDF using PyMuPDF.
    Returns a list of (page_number, text) tuples, 1-indexed pages.
    """
    import fitz  # PyMuPDF

    pages = []
    with fitz.open(file_path) as doc:
        for page_index in range(len(doc)):
            page = doc[page_index]
            text = page.get_text("text")
            pages.append((page_index + 1, text))
    return pages


def extract_docx(file_path: Path) -> list:
    """
    Extract text from a DOCX file using python-docx.
    DOCX has no native page concept, so we treat the whole document as
    "page 1" — page_number metadata will simply be 1 for these chunks.
    Paragraphs and tables are both captured, since PTCL package details
    are frequently laid out in tables.
    """
    import docx

    document = docx.Document(str(file_path))
    parts = []

    for para in document.paragraphs:
        if para.text.strip():
            parts.append(para.text)

    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            row_text = " | ".join(c for c in cells if c)
            if row_text.strip():
                parts.append(row_text)

    full_text = "\n".join(parts)
    return [(1, full_text)]


def extract_txt(file_path: Path) -> list:
    """
    Extract text from a plain text file. Treated as a single page.
    Tries UTF-8 first, falls back to latin-1 to avoid crashing on
    unusual encodings.
    """
    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = file_path.read_text(encoding="latin-1")
    return [(1, text)]


def extract_text_from_file(file_path: Path) -> list:
    """
    Dispatch extraction based on file extension.
    Returns a list of (page_number, raw_text) tuples.
    Raises an exception on failure — the caller is responsible for
    catching it and continuing with other documents.
    """
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return extract_pdf(file_path)
    elif ext == ".docx":
        return extract_docx(file_path)
    elif ext == ".txt":
        return extract_txt(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


# ----------------------------------------------------------------------
# Text cleaning
# ----------------------------------------------------------------------
def clean_text(raw_text: str) -> str:
    """
    Clean extracted text while preserving meaningful package information
    such as names, prices, codes, units, dates and headings.

    This intentionally does NOT:
    - lowercase the text
    - strip digits, currency symbols, or punctuation used in codes
      (e.g. "*443#", "Rs. 50", "GB", "500MB")
    - collapse distinct lines into one another

    It DOES:
    - normalize Windows/Mac line endings
    - collapse runs of horizontal whitespace (spaces/tabs) into one space
    - drop fully empty/blank lines (keeping paragraph breaks minimal)
    - strip common PDF extraction artifacts (stray form-feed characters,
      repeated page-break dashes, null bytes)
    - trim trailing/leading whitespace per line
    """
    if not raw_text:
        return ""

    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x0c", "\n")  # form feed -> newline
    text = text.replace("\x00", "")    # null bytes

    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        # collapse repeated spaces/tabs, keep the words intact
        line = re.sub(r"[ \t]+", " ", line).strip()

        # skip pure decorative artifact lines like "----" or "____" or "...."
        if re.fullmatch(r"[-_.=~*]{3,}", line):
            continue

        if line:
            cleaned_lines.append(line)

    # Collapse 3+ consecutive blank results into a single blank line
    # (there won't be blank lines here since we dropped empties, but this
    # keeps behavior correct if that logic changes later)
    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    return cleaned_text.strip()


# ----------------------------------------------------------------------
# Chunking
# ----------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_WORDS,
               overlap: int = CHUNK_OVERLAP_WORDS) -> list:
    """
    Split text into overlapping word-based chunks.

    Splitting is done on whitespace-delimited words rather than a strict
    tokenizer, which keeps the implementation dependency-free while still
    landing in the requested 500-800 word range with 80-120 word overlap.

    Returns a list of chunk strings (order preserved).
    """
    words = text.split()
    if not words:
        return []

    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    step = max(chunk_size - overlap, 1)

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))

        if end == len(words):
            break
        start += step

    return chunks


def build_chunks_for_document(file_path: Path, pages: list, doc_title: str) -> list:
    """
    Given extracted (page_number, raw_text) pairs for one document,
    clean each page, chunk it, and attach metadata.

    Chunking is done per-page so that page_number metadata stays accurate.
    This also naturally keeps related package info (name/price/validity/
    codes) together, since that information is usually co-located within
    the same page/section of PTCL package sheets.
    """
    document_chunks = []

    for page_number, raw_text in pages:
        cleaned = clean_text(raw_text)
        if not cleaned:
            continue

        page_chunks = chunk_text(cleaned)

        for chunk_body in page_chunks:
            chunk_id = f"{file_path.stem}_p{page_number}_c{len(document_chunks)}"
            document_chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_body,
                "source_file": file_path.name,
                "document_title": doc_title,
                "page_number": page_number,
            })

    return document_chunks


# ----------------------------------------------------------------------
# Embeddings + FAISS
# ----------------------------------------------------------------------
def generate_embeddings(chunk_texts: list, model_name: str = EMBEDDING_MODEL_NAME):
    """
    Generate embeddings locally using sentence-transformers.
    Returns a numpy float32 array of shape (n_chunks, embedding_dim).
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np

    print(f"Loading embedding model: {model_name} ...")
    model = SentenceTransformer(model_name)

    print(f"Generating embeddings for {len(chunk_texts)} chunks ...")
    embeddings = model.encode(
        chunk_texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings.astype("float32")


def build_faiss_index(embeddings) -> "faiss.Index":
    """
    Build a flat L2 FAISS index from the given embeddings.
    A flat index is used deliberately: knowledge bases of this size
    (package sheets / FAQs) do not need approximate search, and a flat
    index guarantees exact, reproducible retrieval.
    """
    import faiss

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------
def discover_documents() -> list:
    """Return a sorted list of supported document paths in knowledge_base/."""
    if not KNOWLEDGE_BASE_DIR.exists():
        return []

    files = [
        p for p in sorted(KNOWLEDGE_BASE_DIR.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return files


def main():
    print("=" * 60)
    print("PTCL Packages Assistant — Knowledge Base Builder")
    print("=" * 60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)

    documents = discover_documents()

    if not documents:
        print()
        print("ERROR: No supported documents found in 'knowledge_base/'.")
        print(f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        print("Please add your PTCL package documents to that folder and re-run:")
        print("    python scripts/build_knowledge_base.py")
        sys.exit(1)

    print(f"\nFound {len(documents)} document(s) in knowledge_base/:")
    for doc in documents:
        print(f"  - {doc.name}")
    print()

    old_manifest = load_previous_manifest()

    all_chunks = []
    manifest_documents = []
    processed_count = 0
    failed_count = 0
    failed_files = []

    for file_path in documents:
        try:
            print(f"Processing: {file_path.name} ...")
            pages = extract_text_from_file(file_path)
            doc_title = file_path.stem.replace("_", " ").replace("-", " ").strip()

            doc_chunks = build_chunks_for_document(file_path, pages, doc_title)

            if not doc_chunks:
                print(f"  Warning: no extractable text found in {file_path.name}")
            else:
                all_chunks.extend(doc_chunks)
                print(f"  -> {len(doc_chunks)} chunk(s) created")

            manifest_documents.append(build_manifest_entry(file_path))
            processed_count += 1

        except Exception as exc:  # noqa: BLE001 - intentionally broad; see requirement 13
            failed_count += 1
            failed_files.append(file_path.name)
            print(f"  ERROR: failed to process '{file_path.name}': {exc}")
            continue

    if not all_chunks:
        print()
        print("ERROR: No text could be extracted from any document.")
        print("Nothing to embed or index. Aborting.")
        sys.exit(1)

    print()
    print(f"Total chunks created: {len(all_chunks)}")

    # ---- Embeddings ----
    chunk_texts = [c["text"] for c in all_chunks]
    try:
        embeddings = generate_embeddings(chunk_texts)
    except Exception as exc:  # noqa: BLE001
        print()
        print(f"ERROR: embedding generation failed: {exc}")
        sys.exit(1)

    embedding_dim = embeddings.shape[1]

    # ---- FAISS index ----
    try:
        index = build_faiss_index(embeddings)
        import faiss
        faiss.write_index(index, str(FAISS_INDEX_PATH))
    except Exception as exc:  # noqa: BLE001
        print()
        print(f"ERROR: FAISS index build/save failed: {exc}")
        sys.exit(1)

    # ---- Save chunks + metadata ----
    # chunks.pkl: list of chunk text strings, in the same order as the FAISS index
    # metadata.pkl: list of metadata dicts, same order/length as chunks.pkl
    chunk_bodies = [c["text"] for c in all_chunks]
    chunk_metadata = [
        {
            "chunk_id": c["chunk_id"],
            "source_file": c["source_file"],
            "document_title": c["document_title"],
            "page_number": c["page_number"],
        }
        for c in all_chunks
    ]

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunk_bodies, f)

    with open(METADATA_PATH, "wb") as f:
        pickle.dump(chunk_metadata, f)

    # ---- Manifest ----
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dimension": embedding_dim,
        "chunk_size_words": CHUNK_SIZE_WORDS,
        "chunk_overlap_words": CHUNK_OVERLAP_WORDS,
        "total_chunks": len(all_chunks),
        "documents": manifest_documents,
    }
    rebuild_needed_next_time_hint = has_manifest_changed(old_manifest, manifest_documents)

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # ---- Summary ----
    print()
    print("=" * 60)
    print("BUILD SUMMARY")
    print("=" * 60)
    print(f"Documents processed:  {processed_count}")
    print(f"Documents failed:     {failed_count}")
    if failed_files:
        print(f"  Failed files: {', '.join(failed_files)}")
    print(f"Total chunks:         {len(all_chunks)}")
    print(f"Embedding dimension:  {embedding_dim}")
    print(f"FAISS vectors:        {index.ntotal}")
    print(f"Index saved to:       {FAISS_INDEX_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Chunks saved to:      {CHUNKS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Metadata saved to:    {METADATA_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Manifest saved to:    {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")
    print("Index saved successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
