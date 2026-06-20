from fastapi import FastAPI, UploadFile, File, HTTPException
import time
import math
import base64
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pydantic import BaseModel
import os
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import json
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from rag_google import (
    load_docs, chunk_docs, add_to_vectorstore, retrieve, answer_query_with_context,
    delete_docs_by_source,
)
from rag_google import get_similar_queries_from_llm  # Import the helper
from src.core.multimodal_processor import MultimodalDocumentProcessor
from src.core.input_validator import input_validator
from src.core.output_validator import output_validator
import logging                                

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
CHAT_HISTORY_DIR = os.path.join("storage", "chat_histories")


def load_index_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}  # Handle empty or corrupt file
    return {}


def save_index_state(state):
    os.makedirs("storage", exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4)

# --- Chat History Management ---

def load_chat_history(chat_id: str):
    os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
    history_file = os.path.join(CHAT_HISTORY_DIR, f"{chat_id}.json")
    if os.path.exists(history_file):
        with open(history_file, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def save_chat_history(chat_id: str, user_query: str, ai_response: str):
    os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
    history_file = os.path.join(CHAT_HISTORY_DIR, f"{chat_id}.json")
    history = load_chat_history(chat_id)
    
    history.append({"user": user_query, "ai": ai_response, "timestamp": time.time()})
    
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

# Select only relevant parts of chat history for contextual retrieval
def select_relevant_history(chat_history, query, top_k=3, threshold=0.4, max_turns=None):
    try:
        # Limit to last N turns if specified
        if isinstance(max_turns, int) and max_turns > 0:
            chat_history = chat_history[-max_turns:]
        texts = []
        for entry in chat_history:
            user = entry.get("user", "")
            ai = entry.get("ai", "")
            combined = (f"User: {user}\nAI: {ai}").strip()
            if combined:
                texts.append(combined)
        if not texts:
            return []
        q_vec = embeddings.embed_query(query)
        doc_vecs = embeddings.embed_documents(texts)

        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            if na == 0 or nb == 0:
                return 0.0
            return dot / (na * nb)

        scored = []
        for i, vec in enumerate(doc_vecs):
            sim = cosine(q_vec, vec)
            if sim >= threshold:
                scored.append((sim, texts[i]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:top_k]]
    except Exception as e:
        logger.warning(f"History selection error: {str(e)}")
        return []

# Adaptive retrieval with fallback thresholds
def retrieve_with_thresholds(vs, query: str, k: int, thresholds: list) -> tuple:
    """Try retrieval with a list of min_relevance thresholds, return on first success."""
    for t in thresholds:
        try:
            results, found = retrieve(vs, query, k=k, min_relevance=t)
            if found:
                print(f"[Query] Retrieval succeeded with min_relevance={t}.")
                return results, True
        except Exception as e:
            logger.warning(f"Retrieval error at threshold {t}: {str(e)}")
            continue
    return [], False

# --- Image retrieval helpers ---
def _guess_mime(filename: str) -> str:
    ext = (os.path.splitext(filename)[1] or '').lower()
    return {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp'
    }.get(ext, 'image/png')

def _encode_image_to_base64(image_file: str, source_path: str) -> str:
    try:
        # Prefer extracted images dir if image_file provided
        if image_file:
            extracted_path = os.path.join("storage", "extracted_images", image_file)
            if os.path.exists(extracted_path):
                with open(extracted_path, 'rb') as f:
                    b64 = base64.b64encode(f.read()).decode('utf-8')
                mime = _guess_mime(image_file)
                return f"data:{mime};base64,{b64}"
        # Fallback to original source path
        if source_path and os.path.exists(source_path):
            with open(source_path, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode('utf-8')
            mime = _guess_mime(source_path)
            return f"data:{mime};base64,{b64}"
    except Exception as e:
        logger.warning(f"Failed to encode image to base64: {str(e)}")
    return ""

def get_image_matches(query_text: str, thresholds: list, k_images: int) -> List[dict]:
    try:
        results, found = retrieve_with_thresholds(image_vectorstore, query_text, k=k_images, thresholds=thresholds)
        if not found:
            return []
        matches = []
        for doc, score in results:
            md = getattr(doc, 'metadata', {}) or {}
            image_file = md.get('image_file', '')
            source = md.get('source', '')
            page = md.get('page')
            data_uri = _encode_image_to_base64(image_file, source)
            if data_uri:
                matches.append({
                    'image_file': image_file,
                    'source': source,
                    'page': page,
                    'score': score,
                    'data_uri': data_uri
                })
        return matches
    except Exception as e:
        logger.warning(f"Image retrieval error: {str(e)}")
        return []

def build_image_attachment_chunk(query_text: str, thresholds: list) -> str:
    if not cfg.get('include_image_attachments', True):
        return ""
    k_images = cfg.get('image_k', 2)
    matches = get_image_matches(query_text, thresholds=thresholds, k_images=k_images)
    if not matches:
        return ""
    lines = ["\n---\nMatched Images:"]
    for i, m in enumerate(matches, start=1):
        # Only include the DataURI, no text metadata
        data_line = f"   DataURI: {m['data_uri']}"
        lines.append(data_line)
    lines.append("---\n")
    return "\n".join(lines)


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
        "google_embed": "models/text-embedding-004",
        "chunk_size": 1200,
        "chunk_overlap": 150,
        "k": 4,
        "min_relevance": 0.5,
        "persist_dir": "storage/chroma_google",
        # New configurable knobs
        "history_top_k": 3,
        "history_threshold": 0.4,
        "history_max_turns": 20,
        "retrieval_thresholds": [0.5, 0.45, 0.35, 0.25],
        # Image vector store
        "image_persist_dir": "storage/chroma_google_images",
        "image_k": 2,
        "image_min_relevance": 0.5,
        "include_image_attachments": True
    }

# --- Set API Key and Initialize Models ---
api_key = cfg.get("google_api_key")

print(f"DEBUG: API Key read from config: '{api_key}'")  # This will show you what the script sees

if not api_key or api_key == PLACEHOLDER_API_KEY:
    raise ValueError(
        "Google API Key not found or is a placeholder in config_google.json. Please create one at https://aistudio.google.com/app/apikey")

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

# Separate image vector store
IMAGE_COLLECTION_NAME = "google_image_collection"
image_vectorstore = Chroma(
    collection_name=IMAGE_COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=cfg.get("image_persist_dir", "storage/chroma_google_images"),
    collection_metadata={"hnsw:space": "cosine"}
)

# Configure Google Generative AI for multimodal processing
import google.generativeai as genai
genai.configure(api_key=api_key)

# Create a simple settings object for multimodal processor compatibility
class SimpleSettings:
    def __init__(self, config):
        self.google_api_key = config["google_api_key"]
        self.google_llm_model = config["google_llm"]

# Temporarily patch the settings for multimodal processor
import sys
from types import ModuleType
settings_module = ModuleType('settings')
settings_module.settings = SimpleSettings(cfg)
sys.modules['src.config.settings'] = settings_module

# Initialize multimodal processor for enhanced document processing
multimodal_processor = MultimodalDocumentProcessor()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Background Scheduler for Auto-Indexing ---
scheduler = AsyncIOScheduler()


def scan_and_index_uploads(scan_type="Scheduled"):
    """Scans the uploads directory for new/modified files and indexes them."""
    uploads_dir = "uploads"
    os.makedirs(uploads_dir, exist_ok=True)
    all_states = load_index_state()
    collection_state = all_states.get(COLLECTION_NAME, {})
    files_in_dir = [os.path.join(uploads_dir, f) for f in os.listdir(uploads_dir) if
                    os.path.isfile(os.path.join(uploads_dir, f))]
    new_files, modified_files = [], []

    for file_path in files_in_dir:
        norm_path, mod_time = os.path.normpath(file_path), os.path.getmtime(file_path)
        if norm_path not in collection_state:
            new_files.append(norm_path)
        elif mod_time > collection_state[norm_path]:
            modified_files.append(norm_path)

    if not new_files and not modified_files:
        print(f"{scan_type} Scan: No new or modified files found.")
        return
    print(f"{scan_type} Scan: Found {len(new_files)} new and {len(modified_files)} modified files.")
    files_to_index = new_files + modified_files
    
    if modified_files:
        print(f"Re-indexing {len(modified_files)} modified files...")
        delete_docs_by_source(vectorstore, modified_files)
    
    try:
        print(f"{scan_type} Scan: Loading {len(files_to_index)} file(s) with multimodal processing...")
        
        # Process documents with multimodal capabilities (images, tables, text)
        all_docs = []
        for file_path in files_to_index:
            try:
                # Try multimodal processing first
                docs = multimodal_processor.process_document(file_path)
                print(f"{scan_type} Scan: Multimodal processing completed for {file_path}: {len(docs)} chunks extracted")
                all_docs.extend(docs)
            except Exception as e:
                print(f"{scan_type} Scan: Multimodal processing failed for {file_path}, falling back to basic text: {e}")
                # Fallback to basic processing
                if file_path.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    all_docs.extend(docs)
                else:
                    # Use original load_docs for other formats
                    docs = load_docs([file_path])
                    all_docs.extend(docs)
        
        print(f"{scan_type} Scan: Loaded {len(all_docs)} documents with enhanced processing")
        
        # Split image index docs and regular docs
        image_index_docs = [d for d in all_docs if getattr(d, 'metadata', {}).get('content_type') == 'image_index']
        regular_docs = [d for d in all_docs if getattr(d, 'metadata', {}).get('content_type') != 'image_index']
        
        chunks = chunk_docs(regular_docs, chunk_size=cfg["chunk_size"], chunk_overlap=cfg["chunk_overlap"])
        print(f"{scan_type} Scan: Created {len(chunks)} chunks (text/multimodal)")
        
        add_to_vectorstore(vectorstore, chunks)
        print(f"{scan_type} Scan: Added {len(chunks)} chunks to text vector store")
        
        if image_index_docs:
            add_to_vectorstore(image_vectorstore, image_index_docs)
            print(f"{scan_type} Scan: Added {len(image_index_docs)} image index docs to image vector store")
        
        # Update state
        for file_path in files_to_index:
            collection_state[os.path.normpath(file_path)] = os.path.getmtime(file_path)
        all_states[COLLECTION_NAME] = collection_state
        save_index_state(all_states)
        
        print(f"{scan_type} Scan: Indexing complete for {len(files_to_index)} file(s).")
        
        # Verify the vector store has documents
        try:
            count = vectorstore._collection.count()
            print(f"{scan_type} Scan: Vector store now contains {count} total documents")
        except Exception as e:
            print(f"{scan_type} Scan: Could not verify document count: {e}")
            
    except Exception as e:
        print(f"{scan_type} Scan: Error during indexing: {e}")
        import traceback
        traceback.print_exc()


@app.on_event("startup")
def startup_event():
    scan_and_index_uploads(scan_type="Startup")
    scheduler.add_job(scan_and_index_uploads, 'interval', minutes=5, id="scan_job")
    scheduler.start()
    print("Scheduler started. It will scan the 'uploads' folder every 5 minutes.")


@app.on_event("shutdown")
def shutdown_event():
    scheduler.shutdown(wait=False)
    print("Scheduler shut down.")


@app.post("/upload", summary="Upload and process documents")
async def upload_files(files: List[UploadFile] = File(...)):
    """Saves uploaded files, chunks them, and adds them to the vector store."""
    try:
        # Validate uploaded files with Guardrails
        validated_files = []
        for file in files:
            validation_result = input_validator.validate_file_upload(file)
            if validation_result["is_valid"]:
                validated_files.append((file, validation_result["safe_filename"]))
            else:
                logger.warning(f"File validation failed for {file.filename}")
        
        if not validated_files:
            raise HTTPException(status_code=400, detail="No valid files to process")
        
        saved_paths = []
        for file, safe_filename in validated_files:
            file_path = os.path.join("uploads", safe_filename)
            content = await file.read()
            
            # Validate file content if it's text-based
            if file_path.endswith(('.txt', '.md')):
                try:
                    content_str = content.decode('utf-8', errors='ignore')
                    safety_result = input_validator.validate_document_content(content_str, safe_filename)
                    if not safety_result["is_safe"]:
                        logger.warning(f"Document content validation failed for {safe_filename}: {safety_result.get('error', 'Unknown error')}")
                        continue
                except Exception as e:
                    logger.warning(f"Document content validation error for {safe_filename}: {str(e)}")
                    continue
            
            with open(file_path, "wb") as f:
                f.write(content)
            saved_paths.append(file_path)

        # Process documents with multimodal capabilities (images, tables, text)
        all_docs = []
        for file_path in saved_paths:
            try:
                # Try multimodal processing first
                docs = multimodal_processor.process_document(file_path)
                logger.info(f"Multimodal processing completed for {file.filename}: {len(docs)} chunks extracted")
                all_docs.extend(docs)
            except Exception as e:
                logger.warning(f"Multimodal processing failed for {file.filename}, falling back to basic text: {e}")
                # Fallback to basic processing
                if file_path.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    all_docs.extend(docs)
                else:
                    # Use original load_docs for other formats
                    docs = load_docs([file_path])
                    all_docs.extend(docs)
        
        # Split image index docs and regular docs
        image_index_docs = [d for d in all_docs if getattr(d, 'metadata', {}).get('content_type') == 'image_index']
        regular_docs = [d for d in all_docs if getattr(d, 'metadata', {}).get('content_type') != 'image_index']

        chunks = chunk_docs(regular_docs, chunk_size=cfg["chunk_size"], chunk_overlap=cfg["chunk_overlap"])
        add_to_vectorstore(vectorstore, chunks)
        if image_index_docs:
            add_to_vectorstore(image_vectorstore, image_index_docs)

        all_states = load_index_state()
        collection_state = all_states.get(COLLECTION_NAME, {})
        for p in saved_paths:
            norm_path = os.path.normpath(p)
            collection_state[norm_path] = os.path.getmtime(norm_path)
        all_states[COLLECTION_NAME] = collection_state
        save_index_state(all_states)

        return {"status": "success", "message": f"Processed {len(files)} files with multimodal capabilities (images, tables, text), creating {len(chunks)} chunks."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


class QueryRequest(BaseModel):
    q: str
    chatId: str = None  # Add chatId to the request model


async def answer_query_with_context_validated(query: str, results: List, llm):
    """
    Generate answer with validated context and output validation using Guardrails.
    """
    # Use the original answer_query_with_context function
    response_generator = answer_query_with_context(query, results, llm)
    
    # Validate each chunk of the response
    for chunk in response_generator:
        try:
            # Validate the response chunk
            validation_result = output_validator.validate_ai_response(chunk, context={"query": query})
            
            if validation_result["is_valid"]:
                # Sanitize for safe display
                safe_chunk = output_validator.sanitize_response_for_display(validation_result["cleaned_response"])
                yield safe_chunk
            else:
                # Log the issue and skip this chunk (don't yield error message)
                logger.warning(f"Response chunk validation failed: {validation_result.get('error', 'Unknown error')}")
                # Skip unsafe chunks silently to avoid breaking the response flow
                continue
        except Exception as e:
            # Handle any validation errors gracefully
            logger.warning(f"Response validation error: {str(e)}")
            # For educational content, yield the original chunk if validation fails due to system error
            yield chunk




@app.post("/query", summary="Query the RAG system")
async def query(request: QueryRequest):
    try:
        # Validate user query with Guardrails
        try:
            query_validation = input_validator.validate_search_query(request.q)
            q = query_validation["validated_query"]
        except HTTPException as e:
            logger.warning(f"Query validation failed: {str(e.detail)}")
            raise HTTPException(status_code=400, detail="Invalid or unsafe query")
        print(f"[Query] Received and validated user query: {q} for chatId: {request.chatId}")

        # Load chat history if chatId is provided
        chat_history = []
        if request.chatId:
            chat_history = load_chat_history(request.chatId)
        
        # Step 1: Try original query first (no chat context)
        thresholds = cfg.get("retrieval_thresholds", [cfg["min_relevance"], max(0.0, cfg["min_relevance"] - 0.05), 0.35, 0.25])
        results, found = retrieve_with_thresholds(vectorstore, q, k=cfg["k"], thresholds=thresholds) 
        if found:
            print("[Query] Found results for original query.")
            
            # Validate search results with Guardrails
            try:
                # Transform results to the expected format for validation (add stable idx for mapping)
                results_for_validation = [
                    {'idx': i, 'content': doc.page_content, 'metadata': getattr(doc, 'metadata', {})}
                    for i, (doc, score) in enumerate(results)
                ]
                results_validation = output_validator.validate_search_results(results_for_validation)
                
                # Reconstruct the original format with validated content using idx mapping
                validated_idx_to_content = {
                    item['idx']: item['content']
                    for item in results_validation["filtered_results"]
                    if isinstance(item, dict) and 'idx' in item
                }
                validated_results = []
                for i, (doc, score) in enumerate(results):
                    if i in validated_idx_to_content:
                        try:
                            doc.page_content = validated_idx_to_content[i]
                        except Exception:
                            pass
                        validated_results.append((doc, score))
                # Fallback: if nothing mapped back, keep originals
                if not validated_results:
                    validated_results = results

                if results_validation["removed_count"] > 0:
                    logger.info(f"Filtered out {results_validation['removed_count']} unsafe results")
            except Exception as e:
                logger.warning(f"Search results validation error: {str(e)}")
                # Use original results if validation fails
                validated_results = results
            
            if validated_results:
                response_generator = answer_query_with_context_validated(q, validated_results, llm)
                image_chunk_text = build_image_attachment_chunk(q, thresholds)
                
                # Collect the full response to save to history
                full_response = ""
                async def history_wrapper(generator):
                    nonlocal full_response
                    async for chunk in generator:
                        full_response += chunk
                        yield chunk
                    # Append matched images if available
                    if image_chunk_text:
                        try:
                            v = output_validator.validate_ai_response(image_chunk_text, context={"query": q})
                            if v.get("is_valid", True):
                                sanitized = output_validator.sanitize_response_for_display(v.get("cleaned_response", image_chunk_text))
                                full_response += sanitized
                                yield sanitized
                            else:
                                full_response += image_chunk_text
                                yield image_chunk_text
                        except Exception:
                            full_response += image_chunk_text
                            yield image_chunk_text
                    if request.chatId:
                        save_chat_history(request.chatId, q, full_response)
                
                return StreamingResponse(history_wrapper(response_generator), media_type="text/plain")
            else:
                print("[Query] All results were filtered out for safety.")

        # Step 2: Try with selected relevant chat history (if any)
        if chat_history:
            selected = select_relevant_history(
                chat_history,
                q,
                top_k=cfg.get("history_top_k", 3),
                threshold=cfg.get("history_threshold", 0.4),
                max_turns=cfg.get("history_max_turns", 20),
            )
            if selected:
                context_block = "\n\n".join(selected)
                q_with_context = f"{context_block}\n\nUser: {q}"
                results, found = retrieve_with_thresholds(vectorstore, q_with_context, k=cfg["k"], thresholds=thresholds) 
                if found:
                    print("[Query] Found results when using selected chat context.")
                    # Validate search results with Guardrails
                    try:
                        # Transform results to the expected format for validation (add stable idx for mapping)
                        results_for_validation = [
                            {'idx': i, 'content': doc.page_content, 'metadata': getattr(doc, 'metadata', {})}
                            for i, (doc, score) in enumerate(results)
                        ]
                        results_validation = output_validator.validate_search_results(results_for_validation)
                        
                        # Reconstruct the original format with validated content using idx mapping
                        validated_idx_to_content = {
                            item['idx']: item['content']
                            for item in results_validation["filtered_results"]
                            if isinstance(item, dict) and 'idx' in item
                        }
                        validated_results = []
                        for i, (doc, score) in enumerate(results):
                            if i in validated_idx_to_content:
                                try:
                                    doc.page_content = validated_idx_to_content[i]
                                except Exception:
                                    pass
                                validated_results.append((doc, score))
                        # Fallback: if nothing mapped back, keep originals
                        if not validated_results:
                            validated_results = results

                        if results_validation["removed_count"] > 0:
                            logger.info(f"Filtered out {results_validation['removed_count']} unsafe results (history step)")
                    except Exception as e:
                        logger.warning(f"Search results validation error (history step): {str(e)}")
                        # Use original results if validation fails
                        validated_results = results
                    
                    if validated_results:
                        response_generator = answer_query_with_context_validated(q, validated_results, llm)
                        image_chunk_text = build_image_attachment_chunk(q_with_context, thresholds)
                        
                        # Collect the full response to save to history
                        full_response = ""
                        async def history_wrapper(generator):
                            nonlocal full_response
                            async for chunk in generator:
                                full_response += chunk
                                yield chunk
                            # Append matched images if available
                            if image_chunk_text:
                                try:
                                    v = output_validator.validate_ai_response(image_chunk_text, context={"query": q})
                                    if v.get("is_valid", True):
                                        sanitized = output_validator.sanitize_response_for_display(v.get("cleaned_response", image_chunk_text))
                                        full_response += sanitized
                                        yield sanitized
                                    else:
                                        full_response += image_chunk_text
                                        yield image_chunk_text
                                except Exception:
                                    full_response += image_chunk_text
                                    yield image_chunk_text
                            if request.chatId:
                                save_chat_history(request.chatId, q, full_response)
                        
                        return StreamingResponse(history_wrapper(response_generator), media_type="text/plain")
                    else:
                        print("[Query] All results were filtered out for safety (history step).")

        # Step 3: Get semantically similar queries from LLM
        print("[Query] No results found; calling LLM for similar queries...")
        similar_queries = get_similar_queries_from_llm(llm, q)

        # Step 3: Iterate through similar queries
        for idx, sim_q in enumerate(similar_queries, start=1):
            print(f"[Query] Trying similar query #{idx}: {sim_q}")
            results, found = retrieve_with_thresholds(vectorstore, sim_q, k=cfg["k"], thresholds=thresholds)
            if found:
                print(f"[Query] Found results for similar query #{idx}.")
                
                # Validate search results with Guardrails
                try:
                    # Transform results to the expected format for validation (add stable idx for mapping)
                    results_for_validation = [
                        {'idx': i, 'content': doc.page_content, 'metadata': getattr(doc, 'metadata', {})}
                        for i, (doc, score) in enumerate(results)
                    ]
                    results_validation = output_validator.validate_search_results(results_for_validation)
                    
                    # Reconstruct the original format with validated content using idx mapping
                    validated_idx_to_content = {
                        item['idx']: item['content']
                        for item in results_validation["filtered_results"]
                        if isinstance(item, dict) and 'idx' in item
                    }
                    validated_results = []
                    for i, (doc, score) in enumerate(results):
                        if i in validated_idx_to_content:
                            try:
                                doc.page_content = validated_idx_to_content[i]
                            except Exception:
                                pass
                            validated_results.append((doc, score))
                    # Fallback: if nothing mapped back, keep originals
                    if not validated_results:
                        validated_results = results

                    if results_validation["removed_count"] > 0:
                        logger.info(f"Filtered out {results_validation['removed_count']} unsafe results for similar query #{idx}")
                except Exception as e:
                    logger.warning(f"Search results validation error for similar query #{idx}: {str(e)}")
                    # Use original results if validation fails
                    validated_results = results
                
                if validated_results:
                    response_generator = answer_query_with_context_validated(sim_q, validated_results, llm)
                    image_chunk_text = build_image_attachment_chunk(sim_q, thresholds)
                    
                    # Collect the full response to save to history
                    full_response = ""
                    async def history_wrapper(generator):
                        nonlocal full_response
                        async for chunk in generator:
                            full_response += chunk
                            yield chunk
                        # Append matched images if available
                        if image_chunk_text:
                            try:
                                v = output_validator.validate_ai_response(image_chunk_text, context={"query": q})
                                if v.get("is_valid", True):
                                    sanitized = output_validator.sanitize_response_for_display(v.get("cleaned_response", image_chunk_text))
                                    full_response += sanitized
                                    yield sanitized
                                else:
                                    full_response += image_chunk_text
                                    yield image_chunk_text
                            except Exception:
                                full_response += image_chunk_text
                                yield image_chunk_text
                        if request.chatId:
                            save_chat_history(request.chatId, q, full_response)
                    
                    return StreamingResponse(history_wrapper(response_generator), media_type="text/plain")
                else:
                    print(f"[Query] All results were filtered out for safety for similar query #{idx}.")
                    continue

        # Step 4: Nothing found - still check image index and attach matches if any
        print("[Query] No relevant information found for any similar queries.")
        async def not_found_generator():
            response = "No relevant information found in the indexed documents based on your query and related concepts."
            image_chunk_text = build_image_attachment_chunk(q, thresholds)
            if request.chatId:
                # Save response plus any image attachment text
                save_chat_history(request.chatId, q, response + ("\n" + image_chunk_text if image_chunk_text else ""))
            yield response
            if image_chunk_text:
                # Validate and stream image chunk if allowed, otherwise stream raw
                try:
                    v = output_validator.validate_ai_response(image_chunk_text, context={"query": q})
                    if v.get("is_valid", True):
                        sanitized = output_validator.sanitize_response_for_display(v.get("cleaned_response", image_chunk_text))
                        yield sanitized
                    else:
                        yield image_chunk_text
                except Exception:
                    yield image_chunk_text

        return StreamingResponse(not_found_generator(), media_type="text/plain")

    except Exception as e:
        print(f"[Error] Exception during query processing: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


@app.get("/images/{filename}", summary="Serve extracted images")
async def serve_image(filename: str):
    """Serve extracted images from documents."""
    try:
        image_path = os.path.join("storage", "extracted_images", filename)
        
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="Image not found")
        
        return FileResponse(
            image_path,
            media_type="image/png",
            headers={"Content-Disposition": f"inline; filename={filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving image: {e}")


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
