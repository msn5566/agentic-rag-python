from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import json
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from rag_google import (
    load_docs, chunk_docs, add_to_vectorstore, retrieve, answer_query_with_context,
    delete_docs_by_source,
)

app = FastAPI(
    title="Google Gemini RAG API",
    description="API for Retrieval-Augmented Generation using Gemini 1.5 Flash."
)

# Allow CORS for frontend apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- State Management for Auto-Indexing ---
STATE_FILE = os.path.join("storage", "index_state_google.json")

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

# --- Configuration ---
cfg_path = "config_google.json"
PLACEHOLDER_API_KEY = "your-google-api-key-here"

if os.path.exists(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
else:
    # Default config if file doesn't exist
    cfg = {
        "google_api_key": PLACEHOLDER_API_KEY,
        "google_llm": "gemini-1.5-flash-latest",
        "google_embed": "text-embedding-004",
        "chunk_size": 1200,
        "chunk_overlap": 150,
        "k": 4,
        "min_relevance": 0.5,
        "persist_dir": "storage/chroma_google"
    }

# --- Set API Key and Initialize Models ---
api_key = cfg.get("google_api_key")

print(f"DEBUG: API Key read from config: '{api_key}'") # This will show you what the script sees

if not api_key or api_key == PLACEHOLDER_API_KEY:
    raise ValueError("Google API Key not found or is a placeholder in config_google.json. Please create one at https://aistudio.google.com/app/apikey")

# --- Initialize Models and Vector Store ---
embeddings = GoogleGenerativeAIEmbeddings(model=cfg["google_embed"], google_api_key=api_key)

llm = ChatGoogleGenerativeAI(
    model=cfg["google_llm"],
    google_api_key=api_key,
)

COLLECTION_NAME = "google_rag_collection"
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=cfg["persist_dir"],
    collection_metadata={"hnsw:space": "cosine"}
)

# --- Auto-indexing on startup (simplified for API context) ---
def startup_scan():
    uploads_dir = "uploads"
    os.makedirs(uploads_dir, exist_ok=True)
    all_states = load_index_state()
    collection_state = all_states.get(COLLECTION_NAME, {})
    files_in_dir = [os.path.join(uploads_dir, f) for f in os.listdir(uploads_dir) if os.path.isfile(os.path.join(uploads_dir, f))]
    new_files, modified_files = [], []

    for file_path in files_in_dir:
        norm_path, mod_time = os.path.normpath(file_path), os.path.getmtime(file_path)
        if norm_path not in collection_state:
            new_files.append(norm_path)
        elif mod_time > collection_state[norm_path]:
            modified_files.append(norm_path)

    if new_files or modified_files:
        print(f"Startup Scan: Found {len(new_files)} new and {len(modified_files)} modified files.")
        if modified_files:
            delete_docs_by_source(vectorstore, modified_files)
        files_to_index = new_files + modified_files
        docs = load_docs(files_to_index)
        chunks = chunk_docs(docs, chunk_size=cfg["chunk_size"], chunk_overlap=cfg["chunk_overlap"])
        add_to_vectorstore(vectorstore, chunks)
        for file_path in files_to_index:
            collection_state[os.path.normpath(file_path)] = os.path.getmtime(file_path)
        all_states[COLLECTION_NAME] = collection_state
        save_index_state(all_states)
        print("Startup Scan: Indexing complete.")

startup_scan()

@app.post("/upload", summary="Upload and process documents")
async def upload_files(files: List[UploadFile] = File(...)):
    """Saves uploaded files, chunks them, and adds them to the vector store."""
    try:
        saved_paths = []
        for file in files:
            file_path = os.path.join("uploads", file.filename)
            with open(file_path, "wb") as f:
                f.write(await file.read())
            saved_paths.append(file_path)
        
        docs = load_docs(saved_paths)
        chunks = chunk_docs(docs, chunk_size=cfg["chunk_size"], chunk_overlap=cfg["chunk_overlap"])
        add_to_vectorstore(vectorstore, chunks)
        
        all_states = load_index_state()
        collection_state = all_states.get(COLLECTION_NAME, {})
        for p in saved_paths:
            norm_path = os.path.normpath(p)
            collection_state[norm_path] = os.path.getmtime(norm_path)
        all_states[COLLECTION_NAME] = collection_state
        save_index_state(all_states)

        return {"status": "success", "message": f"Processed {len(files)} files, creating {len(chunks)} chunks."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

@app.get("/query", summary="Query the RAG system")
async def query(q: str):
    """
    Performs retrieval from the vector store and generates a streaming response
    from the Gemini model based on the retrieved context.
    """
    try:
        results, found = retrieve(vectorstore, q, k=cfg["k"], min_relevance=cfg["min_relevance"])
        if not found:
            async def not_found_generator():
                yield "No relevant information found in the indexed documents to answer this question."
            return StreamingResponse(not_found_generator(), media_type="text/plain")
            
        response_generator = answer_query_with_context(q, results, llm)
        return StreamingResponse(response_generator, media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

@app.delete("/documents", summary="Delete specific documents")
async def delete_documents(paths: List[str]):
    """Deletes specified documents and their chunks from the vector store and state file."""
    try:
        delete_docs_by_source(vectorstore, paths)
        
        all_states = load_index_state()
        collection_state = all_states.get(COLLECTION_NAME, {})
        for p in paths:
            if os.path.normpath(p) in collection_state:
                del collection_state[os.path.normpath(p)]
        all_states[COLLECTION_NAME] = collection_state
        save_index_state(all_states)

        return {"status": "success", "message": f"Attempted to delete {len(paths)} documents."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)