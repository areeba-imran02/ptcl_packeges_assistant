# PTCL Packages Assistant

Developer: **Areeba Imran**

## 1. Project Purpose

PTCL Packages Assistant is a Retrieval-Augmented Generation (RAG) tool for
answering questions about PTCL packages (internet, call, SMS, hybrid
bundles, activation/deactivation codes, terms and conditions, FAQs, etc.)
based on official PTCL knowledge-base documents.

**This is Part 1** of the project. It builds the complete offline
knowledge-base preprocessing pipeline: document extraction, cleaning,
chunking, embeddings, and FAISS indexing. Part 2 (a future step) will add
the full Streamlit chat application, Groq-based LLM responses, and voice
features on top of the artifacts produced here.

## 2. Folder Structure

```
PTCL_Packages_Assistant/
├── app.py                        # Minimal placeholder UI (Part 1 only)
├── requirements.txt
├── README.md
│
├── knowledge_base/                # <-- put your PTCL PDFs/DOCX/TXT files here
│
├── data/
│   ├── chunks.pkl                 # Generated: all text chunks
│   ├── metadata.pkl               # Generated: metadata for each chunk
│   └── documents_manifest.json    # Generated: source-document fingerprint
│
├── faiss_index/
│   └── index.faiss                # Generated: FAISS vector index
│
└── scripts/
    └── build_knowledge_base.py    # The preprocessing pipeline (this part's focus)
```

## 3. Where to Put PTCL Documents

Download the documents from the shared Google Drive folder and place them
directly inside:

```
knowledge_base/
```

Supported formats: `.pdf`, `.docx`, `.txt`.
The script automatically detects and processes every supported file found
in that folder — you never need to hard-code filenames or a document
count.

## 4. How Preprocessing Works

Running the build script performs the following pipeline, in order, for
every document in `knowledge_base/`:

1. **Text extraction** — PDFs are read with PyMuPDF (page by page), DOCX
   files with `python-docx` (paragraphs and tables), and TXT files as
   plain text.
2. **Cleaning** — removes extraction noise (repeated whitespace, blank
   lines, stray decorative separator lines) while carefully preserving
   package names, prices, codes, units (MB/GB/minutes/SMS), dates,
   validity periods, headings, and table content.
3. **Chunking** — see below.
4. **Embeddings** — see below.
5. **FAISS indexing** — see below.

If a single document fails to process, the script logs the filename and
the error, then continues with the remaining documents rather than
crashing. If `knowledge_base/` is empty, the script stops with a clear
error message asking you to add documents.

## 5. How Chunking Works

- Chunk size target: **500–800 words** (default 650).
- Overlap between consecutive chunks: **80–120 words** (default 100).
- Chunking happens **per page** (per PDF page, or the whole document for
  DOCX/TXT), so that related package details — name, price, validity,
  internet volume, minutes, SMS, activation/deactivation codes, terms —
  that appear together in the source document are kept together in the
  same chunk wherever possible.
- Every chunk carries metadata: `chunk_id`, `source_file`,
  `document_title`, `page_number`.

## 6. How Embeddings Work

- Model: `sentence-transformers/all-MiniLM-L6-v2` — a free, local
  embedding model. No paid API (OpenAI or otherwise) is used.
- Embeddings are generated **once**, during preprocessing only.
- The resulting vectors are stored in the FAISS index, not regenerated at
  app startup.

## 7. How FAISS Works

- A flat L2 FAISS index (`IndexFlatL2`) is built from the chunk
  embeddings — exact search, appropriate for the size of a package
  knowledge base.
- Saved to `faiss_index/index.faiss`, in the same order as
  `data/chunks.pkl` and `data/metadata.pkl`, so vector `i` in the index
  always corresponds to chunk `i` in both pickle files.

## 8. How to Build the Knowledge Base

Install dependencies:

```bash
pip install -r requirements.txt
```

Place your PTCL documents in `knowledge_base/`, then run:

```bash
python scripts/build_knowledge_base.py
```

On success, the project will contain:

```
data/chunks.pkl
data/metadata.pkl
data/documents_manifest.json
faiss_index/index.faiss
```

The script prints a summary, for example:

```
Documents processed: 5
Documents failed: 0
Total chunks: 142
Embedding dimension: 384
FAISS vectors: 142
Index saved successfully.
```

## 9. Rebuilding

The script is safe to re-run at any time — it always processes whatever
documents currently exist in `knowledge_base/` and overwrites the
artifacts in `data/` and `faiss_index/`. `documents_manifest.json` records
each source file's size, modified time, and SHA-256 hash, so a future
"Rebuild Knowledge Base" button in the Streamlit app (Part 2) can detect
whether the source documents have changed before deciding to rebuild.

## 10. What Part 1 Does NOT Include

By design, this part does **not** include:

- The full Streamlit chat UI
- Voice input (Whisper)
- Text-to-speech (TTS)
- Groq-based chat responses

These will be added in Part 2, built on top of the artifacts produced
here. `app.py` currently contains only a minimal placeholder that
verifies the knowledge base was built successfully.
