"""
Secure vector store repository with encryption and access controls.
"""
import os
import json
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import logging
from cryptography.fernet import Fernet
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.config import settings
from src.core.monitoring import security_logger, ThreatLevel

logger = logging.getLogger(__name__)


class SecureVectorStore:
    """Secure wrapper for vector store with encryption and access controls."""
    
    def __init__(self, embeddings: GoogleGenerativeAIEmbeddings):
        self.embeddings = embeddings
        self._encryption_key = self._get_or_create_encryption_key()
        self._cipher = Fernet(self._encryption_key)
        
        # Initialize ChromaDB with secure settings
        self.vectorstore = Chroma(
            collection_name=settings.collection_name,
            embedding_function=embeddings,
            persist_directory=settings.persist_dir,
            collection_metadata={
                "hnsw:space": "cosine",
                "encrypted": True,
                "created_at": datetime.utcnow().isoformat()
            }
        )
        
        # Access control tracking
        self.access_log = []
        self.authorized_users = set()
        
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key for vector store."""
        key_file = os.path.join(settings.persist_dir, ".encryption_key")
        
        if os.path.exists(key_file):
            try:
                with open(key_file, "rb") as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Failed to read encryption key: {e}")
                # Generate new key if reading fails
        
        # Generate new encryption key
        key = Fernet.generate_key()
        
        # Ensure directory exists
        os.makedirs(settings.persist_dir, exist_ok=True)
        
        # Save key securely
        try:
            with open(key_file, "wb") as f:
                f.write(key)
            
            # Set restrictive permissions (Unix-like systems)
            if hasattr(os, 'chmod'):
                os.chmod(key_file, 0o600)
                
        except Exception as e:
            logger.error(f"Failed to save encryption key: {e}")
            
        return key
    
    def authorize_user(self, user_id: str) -> None:
        """Authorize a user for vector store access."""
        self.authorized_users.add(user_id)
        security_logger.log_security_event({
            "timestamp": datetime.utcnow(),
            "event_type": "user_authorized",
            "threat_level": ThreatLevel.LOW,
            "user_id": user_id,
            "ip_address": "internal",
            "query": None,
            "response": None,
            "metadata": {"action": "vector_store_authorization"},
            "risk_score": 0.0
        })
    
    def _check_access(self, user_id: str, operation: str) -> bool:
        """Check if user is authorized for the operation."""
        if user_id not in self.authorized_users:
            security_logger.log_threat(
                threat_type="unauthorized_vector_access",
                threat_level=ThreatLevel.HIGH,
                user_id=user_id,
                ip_address="internal",
                details={
                    "operation": operation,
                    "authorized_users": len(self.authorized_users)
                },
                risk_score=0.8
            )
            return False
        
        # Log authorized access
        self.access_log.append({
            "timestamp": datetime.utcnow(),
            "user_id": user_id,
            "operation": operation
        })
        
        return True
    
    def _encrypt_document_content(self, doc: Document) -> Document:
        """Encrypt sensitive document content."""
        # Create a copy to avoid modifying original
        encrypted_doc = Document(
            page_content=doc.page_content,
            metadata=doc.metadata.copy()
        )
        
        # Encrypt the page content
        encrypted_content = self._cipher.encrypt(doc.page_content.encode())
        encrypted_doc.metadata["encrypted_content"] = encrypted_content.decode('latin-1')
        encrypted_doc.metadata["is_encrypted"] = True
        encrypted_doc.metadata["encryption_timestamp"] = datetime.utcnow().isoformat()
        
        # Replace page content with hash for indexing
        content_hash = hashlib.sha256(doc.page_content.encode()).hexdigest()
        encrypted_doc.page_content = f"[ENCRYPTED_CONTENT_{content_hash[:16]}]"
        
        return encrypted_doc
    
    def _decrypt_document_content(self, doc: Document) -> Document:
        """Decrypt document content."""
        if not doc.metadata.get("is_encrypted", False):
            return doc
        
        try:
            encrypted_content = doc.metadata["encrypted_content"].encode('latin-1')
            decrypted_content = self._cipher.decrypt(encrypted_content).decode()
            
            # Create decrypted document
            decrypted_doc = Document(
                page_content=decrypted_content,
                metadata={k: v for k, v in doc.metadata.items() 
                         if k not in ["encrypted_content", "is_encrypted", "encryption_timestamp"]}
            )
            
            return decrypted_doc
            
        except Exception as e:
            logger.error(f"Failed to decrypt document content: {e}")
            return doc
    
    def add_documents(self, documents: List[Document], user_id: str) -> None:
        """Add documents to vector store with encryption."""
        if not self._check_access(user_id, "add_documents"):
            raise PermissionError(f"User {user_id} not authorized for vector store access")
        
        try:
            # Encrypt documents before storing
            encrypted_docs = [self._encrypt_document_content(doc) for doc in documents]
            
            # Add to vector store
            self.vectorstore.add_documents(encrypted_docs)
            
            security_logger.app_logger.info(
                f"Added {len(documents)} encrypted documents to vector store by user {user_id}"
            )
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {e}")
            security_logger.log_threat(
                threat_type="vector_store_error",
                threat_level=ThreatLevel.MEDIUM,
                user_id=user_id,
                ip_address="internal",
                details={
                    "operation": "add_documents",
                    "error": str(e),
                    "document_count": len(documents)
                },
                risk_score=0.4
            )
            raise
    
    def similarity_search_with_score(self, query: str, k: int, user_id: str) -> List[Tuple[Document, float]]:
        """Perform similarity search with access control."""
        if not self._check_access(user_id, "similarity_search"):
            raise PermissionError(f"User {user_id} not authorized for vector store access")
        
        try:
            # Perform search
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            # Decrypt results
            decrypted_results = []
            for doc, score in results:
                decrypted_doc = self._decrypt_document_content(doc)
                decrypted_results.append((decrypted_doc, score))
            
            security_logger.app_logger.info(
                f"Similarity search performed by user {user_id}, returned {len(results)} results"
            )
            
            return decrypted_results
            
        except Exception as e:
            logger.error(f"Failed to perform similarity search: {e}")
            security_logger.log_threat(
                threat_type="vector_search_error",
                threat_level=ThreatLevel.MEDIUM,
                user_id=user_id,
                ip_address="internal",
                details={
                    "operation": "similarity_search",
                    "error": str(e),
                    "query": query[:100]  # Truncate query for logging
                },
                risk_score=0.3
            )
            raise
    
    def delete_documents(self, document_ids: List[str], user_id: str) -> None:
        """Delete documents from vector store."""
        if not self._check_access(user_id, "delete_documents"):
            raise PermissionError(f"User {user_id} not authorized for vector store access")
        
        try:
            self.vectorstore.delete(ids=document_ids)
            
            security_logger.app_logger.info(
                f"Deleted {len(document_ids)} documents from vector store by user {user_id}"
            )
            
            security_logger.log_security_event({
                "timestamp": datetime.utcnow(),
                "event_type": "documents_deleted",
                "threat_level": ThreatLevel.LOW,
                "user_id": user_id,
                "ip_address": "internal",
                "query": None,
                "response": None,
                "metadata": {
                    "operation": "delete_documents",
                    "document_count": len(document_ids)
                },
                "risk_score": 0.1
            })
            
        except Exception as e:
            logger.error(f"Failed to delete documents from vector store: {e}")
            security_logger.log_threat(
                threat_type="vector_delete_error",
                threat_level=ThreatLevel.MEDIUM,
                user_id=user_id,
                ip_address="internal",
                details={
                    "operation": "delete_documents",
                    "error": str(e),
                    "document_ids": document_ids
                },
                risk_score=0.4
            )
            raise
    
    def get_collection_stats(self, user_id: str) -> Dict[str, Any]:
        """Get collection statistics."""
        if not self._check_access(user_id, "get_stats"):
            raise PermissionError(f"User {user_id} not authorized for vector store access")
        
        try:
            count = self.vectorstore._collection.count()
            
            return {
                "document_count": count,
                "collection_name": settings.collection_name,
                "encrypted": True,
                "authorized_users": len(self.authorized_users),
                "access_log_entries": len(self.access_log)
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}
    
    def get_access_log(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get access log for auditing."""
        if not self._check_access(user_id, "get_access_log"):
            raise PermissionError(f"User {user_id} not authorized for vector store access")
        
        # Return recent access log entries
        return self.access_log[-limit:]
