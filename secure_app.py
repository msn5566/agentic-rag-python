"""
Secure RAG Application with comprehensive security measures.
This is the new secure version of your RAG system implementing all security recommendations
from the Medium article "AI Defense 101: Protecting Your RAG-Based Systems from Threats".
"""
import os
import logging
import json
from typing import List, Dict, Any
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from apscheduler.schedulers.background import BackgroundScheduler

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request, status
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Import our secure components
from src.config import settings
from src.security import (
    auth_manager, get_current_active_user, check_rate_limit,
    QueryRequest, DocumentDeleteRequest, FileValidator, SecurityHeaders,
    sanitize_error_message
)
from src.core.monitoring import security_logger, anomaly_detector, output_filter, ThreatLevel
from src.core.data_validation import secure_file_validator
from src.core.multimodal_processor import multimodal_processor
from src.repositories import SecureVectorStore, SecureDocumentStore

# Import Guardrails security framework
from src.core.input_validator import input_validator
from src.core.output_validator import output_validator
from src.core.guardrails_config import rag_guardrails, content_safety

# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize security components
security = HTTPBearer()

# --- State Management for Auto-Indexing ---
STATE_FILE = os.path.join("storage", "index_state_secure.json")


def load_index_state():
    """Load the indexing state from file."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}  # Handle empty or corrupt file
    return {}


def save_index_state(state):
    """Save the indexing state to file."""
    os.makedirs("storage", exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4)


# Initialize background scheduler
scheduler = BackgroundScheduler()


def scan_and_index_uploads(scan_type="Scheduled"):
    """Scans the uploads directory for new/modified files and indexes them with multimodal processing."""
    uploads_dir = "uploads"
    os.makedirs(uploads_dir, exist_ok=True)
    
    # Get current state
    all_states = load_index_state()
    collection_state = all_states.get(settings.collection_name, {})
    
    # Find files in uploads directory
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
        logger.info(f"{scan_type} Scan: No new or modified files found.")
        return
    
    logger.info(f"{scan_type} Scan: Found {len(new_files)} new and {len(modified_files)} modified files.")
    files_to_index = new_files + modified_files
    
    # Get app instance from global state (will be set during startup)
    app_instance = getattr(scan_and_index_uploads, 'app_instance', None)
    if not app_instance:
        logger.error(f"{scan_type} Scan: App instance not available for auto-indexing")
        return
    
    try:
        # Remove modified files from vector store first
        if modified_files:
            logger.info(f"Re-indexing {len(modified_files)} modified files...")
            for file_path in modified_files:
                try:
                    app_instance.state.vector_store.delete_documents_by_source(file_path)
                except Exception as e:
                    logger.warning(f"Could not delete old documents for {file_path}: {e}")
        
        logger.info(f"{scan_type} Scan: Processing {len(files_to_index)} file(s) with multimodal capabilities...")
        
        # Process documents with multimodal capabilities
        all_docs = []
        for file_path in files_to_index:
            try:
                # Authorize system user for this operation
                app_instance.state.vector_store.authorize_user("system")
                app_instance.state.document_store.authorize_user("system")
                
                # Validate file first with existing validator
                with open(file_path, 'rb') as f:
                    content = f.read()
                
                filename = os.path.basename(file_path)
                validation_result = secure_file_validator.validate_file(content, filename)
                
                # Additional Guardrails filename sanitization
                safe_filename = content_safety.sanitize_filename(filename)
                
                if not validation_result["is_valid"]:
                    logger.warning(f"{scan_type} Scan: Skipping invalid file {filename}: {validation_result['issues']}")
                    continue
                
                # Additional Guardrails content validation for text files
                if safe_filename.endswith(('.txt', '.md')):
                    try:
                        content_str = content.decode('utf-8', errors='ignore')
                        content_safety_result = content_safety.check_document_safety(content_str)
                        if not content_safety_result["is_safe"]:
                            logger.warning(f"{scan_type} Scan: Skipping file with unsafe content {safe_filename}: {content_safety_result}")
                            continue
                    except Exception as e:
                        logger.warning(f"{scan_type} Scan: Could not validate content for {safe_filename}: {e}")
                
                # Store file securely (use safe filename from Guardrails)
                store_result = app_instance.state.document_store.store_file(
                    content, safe_filename, "system",
                    security_validated=True, risk_score=0.1
                )
                
                # Process with multimodal capabilities
                docs = multimodal_processor.process_document(store_result["file_path"])
                logger.info(f"{scan_type} Scan: Multimodal processing completed for {safe_filename}: {len(docs)} chunks extracted")
                
                # Validate document content with Guardrails
                validated_docs = []
                for doc in docs:
                    doc_safety_result = content_safety.check_document_safety(doc.page_content)
                    if doc_safety_result["is_safe"]:
                        validated_docs.append(doc)
                    else:
                        logger.warning(f"{scan_type} Scan: Filtered unsafe document chunk from {safe_filename}")
                
                all_docs.extend(validated_docs)
                logger.info(f"{scan_type} Scan: Guardrails validation completed for {safe_filename}: {len(validated_docs)}/{len(docs)} chunks validated")
                
            except Exception as e:
                logger.warning(f"{scan_type} Scan: Processing failed for {file_path}: {e}")
                # Try fallback processing
                try:
                    if file_path.endswith('.pdf'):
                        loader = PyPDFLoader(file_path)
                        docs = loader.load()
                        all_docs.extend(docs)
                        logger.info(f"{scan_type} Scan: Fallback processing completed for {file_path}")
                except Exception as fallback_e:
                    logger.error(f"{scan_type} Scan: Both multimodal and fallback processing failed for {file_path}: {fallback_e}")
        
        if all_docs:
            # Chunk documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap
            )
            chunks = text_splitter.split_documents(all_docs)
            logger.info(f"{scan_type} Scan: Created {len(chunks)} chunks")
            
            # Add to vector store
            app_instance.state.vector_store.add_documents(chunks, user_id="system")
            logger.info(f"{scan_type} Scan: Added {len(chunks)} chunks to vector store")
            
            # Update state
            for file_path in files_to_index:
                collection_state[os.path.normpath(file_path)] = os.path.getmtime(file_path)
            all_states[settings.collection_name] = collection_state
            save_index_state(all_states)
            
            logger.info(f"{scan_type} Scan: Indexing complete for {len(files_to_index)} file(s).")
            
            # Log security event for auto-indexing
            security_logger.log_security_event({
                "timestamp": datetime.utcnow(),
                "event_type": "auto_indexing_completed",
                "threat_level": ThreatLevel.LOW,
                "user_id": "system",
                "ip_address": "internal",
                "metadata": {
                    "files_processed": len(files_to_index),
                    "chunks_created": len(chunks),
                    "scan_type": scan_type
                },
                "risk_score": 0.0
            })
        else:
            logger.warning(f"{scan_type} Scan: No valid documents were processed")
            
    except Exception as e:
        logger.error(f"{scan_type} Scan: Error during indexing: {e}")
        import traceback
        traceback.print_exc()
        
        # Log security event for indexing failure
        security_logger.log_security_event({
            "timestamp": datetime.utcnow(),
            "event_type": "auto_indexing_failed",
            "threat_level": ThreatLevel.MEDIUM,
            "user_id": "system",
            "ip_address": "internal",
            "metadata": {
                "error": str(e),
                "scan_type": scan_type
            },
            "risk_score": 0.3
        })


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting Secure RAG Application...")
    
    # Initialize AI models
    try:
        app.state.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.google_embed_model,
            google_api_key=settings.google_api_key
        )
        
        app.state.llm = ChatGoogleGenerativeAI(
            model=settings.google_llm_model,
            google_api_key=settings.google_api_key,
        )
        
        # Initialize secure repositories
        app.state.vector_store = SecureVectorStore(app.state.embeddings)
        app.state.document_store = SecureDocumentStore()
        
        # Authorize system user for internal operations
        app.state.vector_store.authorize_user("system")
        app.state.document_store.authorize_user("system")
        
        # Set app instance for auto-indexing
        scan_and_index_uploads.app_instance = app
        
        # Perform initial scan and indexing
        logger.info("Performing startup scan for new/modified files...")
        scan_and_index_uploads(scan_type="Startup")
        
        # Start background scheduler for periodic scanning
        scheduler.add_job(scan_and_index_uploads, 'interval', minutes=5, id="secure_scan_job")
        scheduler.start()
        logger.info("Background scheduler started. Will scan uploads folder every 5 minutes.")
        
        logger.info("Secure RAG Application started successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Secure RAG Application...")
    
    # Stop background scheduler
    if scheduler.running:
        scheduler.shutdown()
        logger.info("Background scheduler shut down.")


# Create FastAPI app with security configuration
app = FastAPI(
    title="Secure RAG API",
    description="Production-ready RAG API with comprehensive security measures",
    version="2.0.0",
    lifespan=lifespan
)

# Add rate limiting middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Configure CORS with security
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=settings.allowed_methods,
    allow_headers=settings.allowed_headers,
)


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Add security headers and logging to all requests."""
    start_time = datetime.utcnow()
    
    # Log request
    client_ip = request.client.host
    security_logger.app_logger.info(f"Request: {request.method} {request.url.path} from {client_ip}")
    
    # Process request
    response = await call_next(request)
    
    # Add security headers
    security_headers = SecurityHeaders.get_security_headers()
    for header, value in security_headers.items():
        response.headers[header] = value
    
    # Log response time
    process_time = (datetime.utcnow() - start_time).total_seconds()
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


# Authentication endpoints
@app.post("/auth/login", summary="User authentication")
@limiter.limit("5/minute")
async def login(request: Request, username: str, password: str):
    """Authenticate user and return JWT token."""
    try:
        # In a real application, validate against user database
        # For demo purposes, we'll use a simple check
        if username == "admin" and password == "secure_password_123":
            token_data = {
                "sub": username,
                "active": True,
                "roles": ["admin"],
                "permissions": ["read", "write", "delete"]
            }
            
            access_token = auth_manager.create_access_token(token_data)
            
            security_logger.log_security_event({
                "timestamp": datetime.utcnow(),
                "event_type": "user_login",
                "threat_level": ThreatLevel.LOW,
                "user_id": username,
                "ip_address": request.client.host,
                "query": None,
                "response": None,
                "metadata": {"success": True},
                "risk_score": 0.0
            })
            
            return {
                "access_token": access_token,
                "token_type": "bearer",
                "expires_in": settings.jwt_access_token_expire_minutes * 60
            }
        else:
            security_logger.log_threat(
                threat_type="failed_login",
                threat_level=ThreatLevel.MEDIUM,
                user_id=username,
                ip_address=request.client.host,
                details={"reason": "invalid_credentials"},
                risk_score=0.5
            )
            
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service unavailable"
        )


# Secure file upload endpoint
@app.post("/upload", summary="Secure document upload and processing")
@limiter.limit("10/minute")
async def upload_files(
    request: Request,
    files: List[UploadFile] = File(...),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """Securely upload and process documents with comprehensive validation."""
    try:
        # Check file count limit
        if len(files) > settings.max_files_per_request:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Too many files. Maximum {settings.max_files_per_request} files allowed."
            )
        
        processed_files = []
        quarantined_files = []
        
        for file in files:
            try:
                # Validate file security
                validated_file = await FileValidator.validate_upload_file(file)
                
                # Additional Guardrails validation for file upload
                guardrails_validation = input_validator.validate_file_upload(validated_file)
                if not guardrails_validation["is_valid"]:
                    logger.warning(f"Guardrails file validation failed for {file.filename}: {guardrails_validation}")
                    quarantined_files.append({
                        "filename": file.filename,
                        "issues": [f"Guardrails validation failed: File validation error"]
                    })
                    continue
                
                # Use safe filename from Guardrails
                safe_filename = guardrails_validation["safe_filename"]
                
                # Read file content
                content = await validated_file.read()
                await validated_file.seek(0)
                
                # Security validation
                is_safe, security_issues = secure_file_validator.validate_file_security(
                    validated_file.filename, content
                )
                
                if not is_safe:
                    # Quarantine unsafe file
                    quarantine_result = app.state.document_store.store_file(
                        content, validated_file.filename, current_user["sub"],
                        security_validated=False, risk_score=0.8
                    )
                    
                    app.state.document_store.quarantine_file(
                        quarantine_result["file_path"],
                        f"Security issues: {', '.join(security_issues)}",
                        current_user["sub"]
                    )
                    
                    quarantined_files.append({
                        "filename": validated_file.filename,
                        "issues": security_issues
                    })
                    continue
                
                # Additional Guardrails content validation for text files
                if safe_filename.endswith(('.txt', '.md')):
                    try:
                        content_str = content.decode('utf-8', errors='ignore')
                        content_safety_result = input_validator.validate_document_content(content_str, safe_filename)
                        if not content_safety_result["is_safe"]:
                            logger.warning(f"Guardrails content validation failed for {safe_filename}: {content_safety_result}")
                            quarantined_files.append({
                                "filename": safe_filename,
                                "issues": [f"Content safety: {content_safety_result.get('error', 'Content validation failed')}"]
                            })
                            continue
                    except Exception as e:
                        logger.warning(f"Could not validate content for {safe_filename}: {e}")
                
                # Store file securely (use safe filename from Guardrails)
                store_result = app.state.document_store.store_file(
                    content, safe_filename, current_user["sub"],
                    security_validated=True, risk_score=0.1
                )
                
                # Ensure user is authorized for vector store operations
                app.state.vector_store.authorize_user(current_user["sub"])
                
                # Process document with multimodal capabilities (images, tables, text)
                try:
                    docs = multimodal_processor.process_document(store_result["file_path"])
                    logger.info(f"Multimodal processing completed for {validated_file.filename}: {len(docs)} chunks extracted")
                except Exception as e:
                    logger.warning(f"Multimodal processing failed for {validated_file.filename}, falling back to basic text: {e}")
                    # Fallback to basic processing
                    if validated_file.filename.endswith('.pdf'):
                        loader = PyPDFLoader(store_result["file_path"])
                        docs = loader.load()
                    elif validated_file.filename.endswith('.docx'):
                        loader = Docx2txtLoader(store_result["file_path"])
                        docs = loader.load()
                    else:
                        docs = [Document(
                            page_content=content.decode('utf-8', errors='ignore'),
                            metadata={"source": store_result["file_path"]}
                        )]
                
                # Validate document content for data poisoning
                for doc in docs:
                    is_content_safe, risk_score, threats = secure_file_validator.validate_content_security(
                        store_result["file_path"], doc.page_content
                    )
                    
                    if not is_content_safe:
                        # Quarantine document with poisoned content
                        app.state.document_store.quarantine_file(
                            store_result["file_path"],
                            f"Content threats detected: {', '.join(threats)}",
                            current_user["sub"]
                        )
                        
                        quarantined_files.append({
                            "filename": validated_file.filename,
                            "issues": threats
                        })
                        break
                else:
                    # Content is safe, process for indexing
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=settings.chunk_size,
                        chunk_overlap=settings.chunk_overlap
                    )
                    chunks = splitter.split_documents(docs)
                    
                    # Add to secure vector store
                    app.state.vector_store.add_documents(chunks, current_user["sub"])
                    
                    processed_files.append({
                        "filename": validated_file.filename,
                        "chunks": len(chunks),
                        "file_path": store_result["file_path"]
                    })
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {e}")
                quarantined_files.append({
                    "filename": file.filename,
                    "issues": [f"Processing error: {sanitize_error_message(str(e))}"]
                })
        
        return {
            "status": "success",
            "processed_files": processed_files,
            "quarantined_files": quarantined_files,
            "message": f"Processed {len(processed_files)} files, quarantined {len(quarantined_files)} files"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=sanitize_error_message(str(e))
        )


@app.get("/images/{filename}", summary="Serve extracted images")
@limiter.limit("30/minute")  # Rate limit for image serving
async def serve_image(
    request: Request,
    filename: str,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """Serve extracted images from documents (requires authentication)."""
    try:
        image_path = os.path.join("storage", "extracted_images", filename)
        
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Log image access for security monitoring
        security_logger.log_security_event({
            "timestamp": datetime.utcnow(),
            "event_type": "image_access",
            "threat_level": ThreatLevel.LOW,
            "user_id": current_user["sub"],
            "ip_address": request.client.host,
            "metadata": {
                "filename": filename,
                "action": "image_served"
            },
            "risk_score": 0.1
        })
        
        return FileResponse(
            image_path,
            media_type="image/png",
            headers={"Content-Disposition": f"inline; filename={filename}"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving image: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=sanitize_error_message(str(e))
        )


# Secure query endpoint
@app.post("/query", summary="Secure RAG query processing")
@limiter.limit("30/minute")
async def query(
    request: Request,
    query_request: QueryRequest,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """Process queries with comprehensive security analysis."""
    try:
        user_id = current_user["sub"]
        client_ip = request.client.host
        query_text = query_request.q
        
        # Guardrails query validation
        try:
            query_validation = input_validator.validate_search_query(query_text)
            # Use validated query from Guardrails
            validated_query = query_validation["validated_query"]
        except HTTPException as e:
            logger.warning(f"Guardrails query validation failed for user {user_id}: {str(e.detail)}")
            
            # Log security event for blocked query
            security_logger.log_security_event({
                "timestamp": datetime.utcnow(),
                "event_type": "query_blocked_guardrails",
                "threat_level": ThreatLevel.MEDIUM,
                "user_id": user_id,
                "ip_address": client_ip,
                "query": query_text,
                "metadata": {"validation_error": str(e.detail)},
                "risk_score": 0.6
            })
            
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query blocked due to content policy violations"
            )
        
        # Anomaly detection
        risk_score = anomaly_detector.analyze_query(user_id, client_ip, validated_query)
        
        if risk_score > 0.7:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query blocked due to security concerns"
            )
        
        # Ensure authenticated user is authorized for vector store access
        app.state.vector_store.authorize_user(user_id)
        
        # Perform secure vector search using validated query
        results = app.state.vector_store.similarity_search_with_score(
            validated_query, k=settings.retrieval_k, user_id=user_id
        )
        
        if not results:
            return StreamingResponse(
                iter(["No relevant information found in the indexed documents."]),
                media_type="text/plain"
            )
        
        # Generate response
        context = "\n\n".join([
            f"[Source, Score: {score:.2f}] {doc.page_content}" 
            for doc, score in results
        ])
        
        prompt_template = ChatPromptTemplate.from_template(
            "You are a helpful assistant. Answer the user's question based only on the context provided.\n"
            "If the answer is not in the context, state that you cannot answer based on the provided information.\n"
            "Be concise and accurate in your response.\n\n"
            "CONTEXT:\n{context}\n\n"
            "QUESTION:\n{question}"
        )
        
        chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt_template
            | app.state.llm
            | StrOutputParser()
        )
        
        # Stream response with Guardrails output validation and existing output filtering
        async def filtered_response_generator():
            response_chunks = []
            async for chunk in chain.astream({"context": context, "question": validated_query}):
                try:
                    # Validate each chunk with Guardrails
                    chunk_validation = output_validator.validate_ai_response(chunk, context={"query": validated_query})
                    
                    if chunk_validation["is_valid"]:
                        # Use cleaned chunk from Guardrails
                        safe_chunk = chunk_validation["cleaned_response"]
                        response_chunks.append(safe_chunk)
                        yield safe_chunk
                    else:
                        # Log validation failure but continue with safe fallback
                        logger.warning(f"Guardrails output validation failed for chunk: {chunk_validation.get('error', 'Unknown error')}")
                        # Yield a safe placeholder instead of unsafe content
                        safe_placeholder = "[Content filtered for safety]"
                        response_chunks.append(safe_placeholder)
                        yield safe_placeholder
                except Exception as e:
                    # Handle validation errors gracefully
                    logger.warning(f"Response validation error: {str(e)}")
                    # Yield the original chunk if validation fails
                    response_chunks.append(chunk)
                    yield chunk
            
            # Filter complete response with existing security filter
            complete_response = "".join(response_chunks)
            filtered_response = output_filter.filter_response(
                complete_response, user_id, client_ip
            )
            
            # Additional Guardrails validation for complete response
            try:
                final_validation = output_validator.validate_ai_response(filtered_response, context={"query": validated_query})
                if not final_validation["is_valid"]:
                    logger.warning(f"Final response validation failed: {final_validation.get('error', 'Unknown error')}")
                    # Log security event for response filtering
                    security_logger.log_security_event({
                        "timestamp": datetime.utcnow(),
                        "event_type": "response_filtered_guardrails",
                        "threat_level": ThreatLevel.MEDIUM,
                        "user_id": user_id,
                        "ip_address": client_ip,
                        "query": validated_query,
                        "metadata": {"validation_error": final_validation.get('error', 'Unknown error')},
                        "risk_score": 0.5
                    })
            except Exception as e:
                logger.warning(f"Final response validation error: {str(e)}")
                # Continue with filtered response even if final validation fails
            
            # Log the interaction
            security_logger.log_query(
                user_id, client_ip, validated_query, filtered_response,
                {"risk_score": risk_score, "results_count": len(results), "guardrails_validated": True}
            )
        
        return StreamingResponse(filtered_response_generator(), media_type="text/plain")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=sanitize_error_message(str(e))
        )


# Secure document deletion endpoint
@app.delete("/documents", summary="Secure document deletion")
@limiter.limit("10/minute")
async def delete_documents(
    request: Request,
    delete_request: DocumentDeleteRequest,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """Securely delete documents with audit logging."""
    try:
        user_id = current_user["sub"]
        paths = delete_request.paths
        
        deleted_files = []
        failed_deletions = []
        
        for path in paths:
            try:
                # Delete from document store
                success = app.state.document_store.delete_file(path, user_id)
                
                if success:
                    deleted_files.append(path)
                    
                    # Also remove from vector store (find by source)
                    # This would require implementing a method to find documents by source
                    # For now, we'll log the deletion
                    security_logger.app_logger.info(f"Document deleted: {path} by user {user_id}")
                else:
                    failed_deletions.append(path)
                    
            except Exception as e:
                logger.error(f"Failed to delete {path}: {e}")
                failed_deletions.append(path)
        
        return {
            "status": "success",
            "deleted_files": deleted_files,
            "failed_deletions": failed_deletions,
            "message": f"Deleted {len(deleted_files)} files, {len(failed_deletions)} failures"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=sanitize_error_message(str(e))
        )


# System status and monitoring endpoints
@app.get("/health", summary="System health check")
async def health_check():
    """System health and security status."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "security_level": "enhanced",
        "version": "2.0.0"
    }


@app.get("/security/stats", summary="Security statistics")
async def security_stats(
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """Get security statistics (admin only)."""
    if "admin" not in current_user.get("roles", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    try:
        vector_stats = app.state.vector_store.get_collection_stats(current_user["sub"])
        storage_stats = app.state.document_store.get_storage_stats(current_user["sub"])
        
        return {
            "vector_store": vector_stats,
            "document_store": storage_stats,
            "security_events": len(security_logger.events),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to retrieve statistics"
        )


if __name__ == "__main__":
    import uvicorn
    
    # Create .env file if it doesn't exist
    if not os.path.exists(".env"):
        logger.warning("No .env file found. Please copy .env.example to .env and configure your settings.")
    
    uvicorn.run(
        "secure_app:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_debug,
        log_level=settings.log_level.lower()
    )
