from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import json
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from rag_pipeline import (
    load_docs, chunk_docs, add_to_vectorstore, retrieve, answer_query_with_context,
    delete_docs_by_source,
)

app = FastAPI()

# Allow CORS for React app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- State Management for Auto-Indexing ---
STATE_FILE = os.path.join("storage", "index_state.json")

def load_index_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {} # Handle empty or corrupt file
    return {}

def save_index_state(state):
    os.makedirs("storage", exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4)

# Sidebar config
cfg_path = "config.json"
if os.path.exists(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
else:
    cfg = {
        "ollama_llm": "llama3.1:8b",
        "ollama_embed": "nomic-embed-text",
        "chunk_size": 1200,
        "chunk_overlap": 150,
        "k": 4,
        "min_relevance": 0.5,
        "persist_dir": "storage/chroma"
    }

ollama_llm = cfg["ollama_llm"]
ollama_embed = cfg["ollama_embed"]
chunk_size = int(cfg["chunk_size"])
chunk_overlap = int(cfg["chunk_overlap"])
k = int(cfg["k"])
min_relevance = float(cfg["min_relevance"])
collection_name = "default"
persist_dir = cfg["persist_dir"]

# Initialize vector store
embeddings = OllamaEmbeddings(model=ollama_embed)
vectorstore = Chroma(
    collection_name=collection_name,
    embedding_function=embeddings,
    persist_directory=persist_dir,
    collection_metadata={"hnsw:space": "cosine"}
)

# --- Auto-indexing on startup ---
if not os.path.exists("startup_scan_complete"):
    uploads_dir = "uploads"
    os.makedirs(uploads_dir, exist_ok=True)

    all_states = load_index_state()
    collection_state = all_states.get(collection_name, {})

    files_in_dir = [os.path.join(uploads_dir, f) for f in os.listdir(uploads_dir) if os.path.isfile(os.path.join(uploads_dir, f))]

    new_files = []
    modified_files = []

    for file_path in files_in_dir:
        norm_path = os.path.normpath(file_path)
        mod_time = os.path.getmtime(norm_path)

        if norm_path not in collection_state:
            new_files.append(norm_path)
        elif mod_time > collection_state[norm_path]:
            modified_files.append(norm_path)

    if new_files or modified_files:
        with open("startup_scan_complete", "w") as f:
            f.write("")

        if modified_files:
            delete_docs_by_source(vectorstore, modified_files)

        files_to_index = new_files + modified_files

        docs = load_docs(files_to_index)
        chunks = chunk_docs(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        add_to_vectorstore(vectorstore, chunks)

        for file_path in files_to_index:
            collection_state[os.path.normpath(file_path)] = os.path.getmtime(file_path)
        all_states[collection_name] = collection_state
        save_index_state(all_states)

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Process uploaded files and add to vectorstore"""
    try:
        # Save files temporarily
        file_paths = []
        for file in files:
            file_path = f"uploads/{file.filename}"
            with open(file_path, "wb") as f:
                f.write(await file.read())
            file_paths.append(file_path)
        
        # Process documents
        docs = load_docs(file_paths)
        chunks = chunk_docs(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        add_to_vectorstore(vectorstore, chunks)
        
        # Update state file
        all_states = load_index_state()
        collection_state = all_states.get(collection_name, {})
        for p in file_paths:
            norm_path = os.path.normpath(p)
            collection_state[norm_path] = os.path.getmtime(norm_path)
        all_states[collection_name] = collection_state
        save_index_state(all_states)

        return {"status": "success", "message": f"Processed {len(files)} files"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/query", response_class=PlainTextResponse)
async def query(q: str, model: str = "llama3.1:8b"):
    """Query the RAG system"""
    try:
        results, found = retrieve(vectorstore, q)
        if not found:
            return "No relevant information found"
            
        # Collect all chunks into single response
        full_response = ""
        for chunk in answer_query_with_context(q, results, model):
            full_response += chunk
            
        return full_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents")
async def delete_documents(paths: List[str]):
    """Delete documents from vectorstore"""
    try:
        delete_docs_by_source(vectorstore, paths)
        
        # Update state file
        all_states = load_index_state()
        collection_state = all_states.get(collection_name, {})
        for p in paths:
            if os.path.normpath(p) in collection_state:
                del collection_state[os.path.normpath(p)]
        all_states[collection_name] = collection_state
        save_index_state(all_states)

        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
