"""
Input validation service using Guardrails for RAG system.
Validates user inputs, file uploads, and search queries.
"""

import logging
from typing import Dict, Any, Optional, List
from fastapi import HTTPException, UploadFile
from .guardrails_config import rag_guardrails, content_safety

logger = logging.getLogger(__name__)


class InputValidationService:
    """Service for validating all user inputs using Guardrails."""
    
    def __init__(self):
        self.guardrails = rag_guardrails
        self.safety_checker = content_safety
        
    def validate_search_query(self, query: str) -> Dict[str, Any]:
        """
        Validate search query with comprehensive checks.
        
        Args:
            query: User search query
            
        Returns:
            Validation result with cleaned query
            
        Raises:
            HTTPException: If validation fails critically
        """
        if not query or not query.strip():
            raise HTTPException(status_code=400, detail="Search query cannot be empty")
        
        # Basic sanitization
        query = query.strip()
        
        # Guardrails validation
        validation_result = self.guardrails.validate_search_query(query)
        
        if not validation_result["is_valid"]:
            logger.warning(f"Query validation failed: {validation_result['error']}")
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid search query: {validation_result['error']}"
            )
        
        return {
            "original_query": query,
            "validated_query": validation_result["cleaned_query"],
            "is_safe": True,
            "validation_passed": validation_result["validation_passed"]
        }
    
    def validate_user_message(self, message: str) -> Dict[str, Any]:
        """
        Validate user message/input with safety checks.
        
        Args:
            message: User message or input
            
        Returns:
            Validation result with cleaned message
            
        Raises:
            HTTPException: If validation fails critically
        """
        if not message or not message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Basic sanitization
        message = message.strip()
        
        # Length check
        if len(message) > 10000:
            raise HTTPException(status_code=400, detail="Message too long (max 10,000 characters)")
        
        # Guardrails validation
        validation_result = self.guardrails.validate_user_input(message)
        
        if not validation_result["is_valid"]:
            logger.warning(f"Message validation failed: {validation_result['error']}")
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid message: {validation_result['error']}"
            )
        
        return {
            "original_message": message,
            "validated_message": validation_result["cleaned_input"],
            "is_safe": True,
            "validation_passed": validation_result["validation_passed"]
        }
    
    def validate_file_upload(self, file: UploadFile) -> Dict[str, Any]:
        """
        Validate uploaded file for safety and format.
        
        Args:
            file: FastAPI UploadFile object
            
        Returns:
            Validation result with file info
            
        Raises:
            HTTPException: If file validation fails
        """
        # Check file exists
        if not file or not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Sanitize filename
        safe_filename = self.safety_checker.sanitize_filename(file.filename)
        
        # Check file size (50MB limit)
        if hasattr(file, 'size') and file.size and file.size > 50 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 50MB)")
        
        # Check file extension
        allowed_extensions = {'.pdf', '.docx', '.doc', '.txt', '.md'}
        file_ext = safe_filename.lower().split('.')[-1] if '.' in safe_filename else ''
        
        if f'.{file_ext}' not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"File type not allowed. Allowed: {', '.join(allowed_extensions)}"
            )
        
        return {
            "original_filename": file.filename,
            "safe_filename": safe_filename,
            "file_extension": file_ext,
            "content_type": file.content_type,
            "is_valid": True
        }
    
    def validate_document_content(self, content: str, filename: str = "") -> Dict[str, Any]:
        """
        Validate document content for safety.
        
        Args:
            content: Document text content
            filename: Optional filename for context
            
        Returns:
            Safety validation result
        """
        if not content or not content.strip():
            return {
                "is_safe": False,
                "error": "Document content is empty",
                "risk_level": "medium"
            }
        
        # Check content safety
        safety_result = self.safety_checker.check_document_safety(content)
        
        # Additional checks for document length
        if len(content) > 1000000:  # 1MB text limit
            logger.warning(f"Large document detected: {filename} ({len(content)} chars)")
            safety_result["warnings"] = safety_result.get("warnings", [])
            safety_result["warnings"].append("Document is very large")
        
        return safety_result
    
    def validate_api_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate API parameters for safety and format.
        
        Args:
            params: Dictionary of API parameters
            
        Returns:
            Validated parameters
            
        Raises:
            HTTPException: If parameters are invalid
        """
        validated_params = {}
        
        # Validate common parameters
        if 'limit' in params:
            limit = params.get('limit', 10)
            if not isinstance(limit, int) or limit < 1 or limit > 100:
                raise HTTPException(status_code=400, detail="Limit must be between 1 and 100")
            validated_params['limit'] = limit
        
        if 'offset' in params:
            offset = params.get('offset', 0)
            if not isinstance(offset, int) or offset < 0:
                raise HTTPException(status_code=400, detail="Offset must be non-negative")
            validated_params['offset'] = offset
        
        if 'threshold' in params:
            threshold = params.get('threshold', 0.3)
            if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
                raise HTTPException(status_code=400, detail="Threshold must be between 0 and 1")
            validated_params['threshold'] = threshold
        
        return validated_params


# Global instance for easy access
input_validator = InputValidationService()
