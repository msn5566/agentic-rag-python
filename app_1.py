
import os
import json
import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from rag_pipeline import (
    load_docs, chunk_docs, add_to_vectorstore, retrieve, answer_query_with_context,
    delete_docs_by_source,
)

st.set_page_config(page_title="Agentic PDF/DOCX QA", page_icon="📄", layout="wide")

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
st.sidebar.title("⚙️ Settings")
cfg_path = "backup/config.json"
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

ollama_llm = st.sidebar.text_input("Ollama LLM model", cfg["ollama_llm"])
ollama_embed = st.sidebar.text_input("Ollama Embedding model", cfg["ollama_embed"])
chunk_size = st.sidebar.slider("Chunk size", 256, 2000, int(cfg["chunk_size"]), step=64)
chunk_overlap = st.sidebar.slider("Chunk overlap", 0, 400, int(cfg["chunk_overlap"]), step=10)
k = st.sidebar.slider("Top-K passages", 1, 10, int(cfg["k"]))
min_relevance = st.sidebar.slider("Min relevance threshold", 0.0, 1.0, float(cfg["min_relevance"]), step=0.05)
collection_name = st.sidebar.text_input("Collection name", value="default")
persist_dir = st.sidebar.text_input("Chroma persist dir", value=cfg["persist_dir"])

# --- Model & Resource Caching ---
@st.cache_resource
def get_embeddings_model(model_name: str):
    """Load and cache the embedding model."""
    return OllamaEmbeddings(model=model_name)

@st.cache_resource
def get_vector_store(_collection_name, _persist_dir, _embedding_function):
    """
    Load and cache the Chroma vector store, ensuring consistent
    configuration for cosine similarity.
    """
    vs = Chroma(
        collection_name=_collection_name,
        embedding_function=_embedding_function,
        persist_directory=_persist_dir,
        collection_metadata={"hnsw:space": "cosine"}
    )
    return vs

embeddings = get_embeddings_model(ollama_embed)
vs = get_vector_store(collection_name, persist_dir, embeddings)

# --- Auto-indexing on startup ---
if 'startup_scan_complete' not in st.session_state:
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
        with st.status("Performing automatic startup indexing...", expanded=True) as status:
            if modified_files:
                status.update(label=f"Found {len(modified_files)} modified file(s). Removing old versions...")
                delete_docs_by_source(vs, modified_files)
                st.write(f"Cleared old data for: {', '.join(os.path.basename(p) for p in modified_files)}")

            files_to_index = new_files + modified_files
            status.update(label=f"Indexing {len(files_to_index)} new/modified file(s)...")

            docs = load_docs(files_to_index)
            chunks = chunk_docs(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            add_to_vectorstore(vs, chunks)
            st.write(f"Indexed {len(chunks)} chunks from {len(files_to_index)} file(s).")

            for file_path in files_to_index:
                collection_state[os.path.normpath(file_path)] = os.path.getmtime(file_path)
            all_states[collection_name] = collection_state
            save_index_state(all_states)
            status.update(label="Startup indexing complete!", state="complete", expanded=False)

    st.session_state['startup_scan_complete'] = True

st.title("📄 Agentic PDF/DOCX Q&A (LangChain • Chroma • Llama)")

st.markdown(
    " **How it works**\n"
    " 1. Upload PDF/DOCX.\n"
    " 2. Click *Index Files* to build/update the vector store.\n"
    " 3. Ask questions. If nothing relevant is found, you'll see **No content found**."
)

uploaded_files = st.file_uploader("Upload files", type=["pdf", "docx", "txt"], accept_multiple_files=True)

col1, col2 = st.columns(2)

with col1:
    if st.button("Index Files", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one file first.")
        else:
            os.makedirs("uploads", exist_ok=True)
            saved_paths = []
            for f in uploaded_files:
                save_path = os.path.join("uploads", f.name)
                with open(save_path, "wb") as out:
                    out.write(f.read())
                saved_paths.append(save_path)
            st.session_state["last_uploaded"] = saved_paths

            with st.spinner("Loading -> Chunking -> Embedding -> Persisting to Chroma..."):
                docs = load_docs(saved_paths)
                chunks = chunk_docs(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                add_to_vectorstore(vs, chunks)

            # Update state file
            all_states = load_index_state()
            collection_state = all_states.get(collection_name, {})
            for p in saved_paths:
                norm_path = os.path.normpath(p)
                collection_state[norm_path] = os.path.getmtime(norm_path)
            all_states[collection_name] = collection_state
            save_index_state(all_states)

            st.success(f"Indexed {len(chunks)} chunks into collection '{collection_name}'.")

with col2:
    if st.button("Clear Collection", use_container_width=True):
        try:
            import chromadb
            client = chromadb.PersistentClient(path=persist_dir)
            client.delete_collection(name=collection_name)

            # Also clear state file for this collection
            all_states = load_index_state()
            if collection_name in all_states:
                del all_states[collection_name]
                save_index_state(all_states)

            st.success(f"Cleared collection '{collection_name}'.")
            # Also clear any session state related to uploads for this collection
            if 'last_uploaded' in st.session_state:
                del st.session_state['last_uploaded']
        except ValueError: # ChromaDB raises ValueError if the collection doesn't exist
            st.info(f"Collection '{collection_name}' not found or already cleared.")
        except Exception as e:
            st.error(f"An error occurred while clearing the collection: {e}")

st.divider()
query = st.text_input("Ask a question about your documents")

if query:
    with st.spinner("Retrieving relevant chunks..."):
        # Use the globally defined and cached vector store
        results, found = retrieve(vs, query, k=k, min_relevance=min_relevance)

    if not found:
        st.warning("Could not find any relevant passages in the documents to answer your question. Try lowering the 'Min relevance threshold' in the sidebar or rephrasing your question.", icon="⚠️")
    else:
        with st.spinner("Generating answer with Llama..."):
            answer = answer_query_with_context(query, results, model_name=ollama_llm)
        st.markdown("### Answer")
        st.write(answer)

        with st.expander("Show supporting chunks"):
            for i, (doc, score) in enumerate(results, start=1):
                st.markdown(f"**Chunk {i} (score={score:.2f})**")
                st.caption(str(doc.metadata))
                st.write(doc.page_content)
