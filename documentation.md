# Agentic PDF/DOCX Q&A Application Documentation

## Overview
This is a Streamlit-based application that implements a RAG (Retrieval-Augmented Generation) system using:
- Ollama LLMs (default: llama3.1:8b)
- Chroma vector store
- LangChain framework

## Key Features

### File Processing
- Supports PDF, DOCX, and TXT files
- Automatic chunking with configurable size/overlap
- Persistent storage in Chroma DB

### Core Components
1. **Vector Store**: Chroma DB with:
   - Configurable collection names
   - Persistent storage
   - Cosine similarity search
2. **Embeddings**: Ollama embeddings (default: nomic-embed-text)
3. **LLM**: Ollama language models

### Auto-Indexing
- Scans `uploads/` directory on startup
- Tracks modified files using `storage/index_state.json`
- Only re-indexes changed files

## Usage Guide

### Setup
1. Install requirements:
```bash
pip install -r requirements.txt
```
2. Ensure Ollama is running with desired models

### Running the App
```bash
streamlit run app_1.py
```

### Interface Components
1. **Sidebar Settings**:
   - Model selection
   - Chunking parameters
   - Retrieval settings
2. **Main Panel**:
   - File upload
   - Indexing controls
   - Q&A interface

### API Endpoints
While `app_1.py` uses Streamlit, the core RAG functionality in `rag_pipeline.py` can also be used with the FastAPI version in `app.py`.

## Configuration
Edit `config.json` to set defaults for:
- `ollama_llm`: Default LLM model
- `ollama_embed`: Default embeddings model
- `chunk_size`: Default document chunk size
- `chunk_overlap`: Default chunk overlap
- `k`: Number of passages to retrieve
- `min_relevance`: Similarity threshold
- `persist_dir`: Chroma storage location

## Troubleshooting
1. **Slow Responses**:
   - Try smaller LLM models
   - Reduce chunk size
   - Lower `k` value
2. **Indexing Issues**:
   - Check `storage/index_state.json`
   - Use "Clear Collection" button to reset
3. **Model Errors**:
   - Verify Ollama service is running
   - Confirm model names are correct
