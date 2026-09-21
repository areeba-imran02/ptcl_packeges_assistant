# PTCL Assistant

An AI-powered PTCL customer support assistant built with Python, Streamlit, FAISS, Sentence Transformers and Groq.

## Features

- Natural-language PTCL package queries
- Retrieval-Augmented Generation (RAG)
- FAISS-based semantic search
- Grounded answers from the PTCL knowledge base
- English and Urdu voice input
- Text-to-speech responses
- PTCL package and service information
- Source-grounded responses

## Knowledge Base

The assistant currently uses PTCL reference documents covering:

- Internet Packages
- Shoq TV Digital Signup
- Shoq TV Voice & Mobile Packages
- 3 Months Advance Packages
- QuadPlay Packages
- Speed Bolt-On Voice & Mobile
- PTCL Flash Fiber

## Project Structure

PTCL_Packages_Assistant/

    app.py
    requirements.txt
    README.md
    .gitignore

    data/
        chunks.pkl
        metadata.pkl
        documents_manifest.json

    faiss_index/
        index.faiss

    knowledge_base/
        PTCL reference PDFs

    scripts/
        build_knowledge_base.py

## Technology Stack

- Python
- Streamlit
- FAISS
- Sentence Transformers
- Groq API
- Speech Recognition
- Text-to-Speech
- PDF Knowledge Base
- Retrieval-Augmented Generation

## Setup

Clone the repository:

    git clone YOUR_GITHUB_REPOSITORY_URL
    cd PTCL_Packages_Assistant

Install dependencies:

    pip install -r requirements.txt

Set the Groq API key as an environment variable.

Linux / macOS:

    export GROQ_API_KEY="your_api_key_here"

Windows PowerShell:

    $env:GROQ_API_KEY="your_api_key_here"

Run the application:

    streamlit run app.py

## Security

Do not commit API keys, passwords, tokens, .env files or Streamlit secrets to GitHub.

The application reads the Groq API key from the GROQ_API_KEY environment variable.

## RAG Pipeline

User Query
    |
    v
Query Embedding
    |
    v
FAISS Semantic Retrieval
    |
    v
Relevant PTCL Documents
    |
    v
Groq LLM
    |
    v
Grounded Answer
    |
    v
Text / Voice Response

## Disclaimer

This project is intended as an AI-powered demonstration and customer-support assistant. Information is based on the documents included in the project's knowledge base.

## Author

Areeba Imran

Built with Python, Generative AI and Retrieval-Augmented Generation.
