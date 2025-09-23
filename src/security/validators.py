"""
Input validation and sanitization for RAG application.
"""
import re
import os
from typing import List, Optional
from fastapi import HTTPException, status, UploadFile
from pydantic import BaseModel, validator, Field
from src.config import settings
import logging

logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    """Validated query request model."""
    q: str = Field(..., min_length=1, max_length=1000, description="Query string")
    
    @validator('q')
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        
        # Remove potentially dangerous characters
        v = v.strip()
        
        # Check for SQL injection patterns
        sql_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
            r"(--|#|/\*|\*/)",
            r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                logger.warning(f"Potential SQL injection attempt detected: {v}")
                raise ValueError("Invalid query format")
        
        # Check for script injection
        script_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
        ]
        
        for pattern in script_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                logger.warning(f"Potential script injection attempt detected: {v}")
                raise ValueError("Invalid query format")
        
        return v


class DocumentDeleteRequest(BaseModel):
    """Validated document deletion request."""
    paths: List[str] = Field(..., min_items=1, max_items=100)
    
    @validator('paths')
    def validate_paths(cls, v):
        validated_paths = []
        for path in v:
            # Normalize path and check for directory traversal
            normalized_path = os.path.normpath(path)
            
            if ".." in normalized_path or normalized_path.startswith("/"):
                logger.warning(f"Potential directory traversal attempt: {path}")
                raise ValueError(f"Invalid path: {path}")
            
            # Ensure path is within uploads directory
            if not normalized_path.startswith("uploads"):
                raise ValueError(f"Path must be within uploads directory: {path}")
            
            validated_paths.append(normalized_path)
        
        return validated_paths


class FileValidator:
    """Validates uploaded files for security."""
    
    @staticmethod
    def validate_file_extension(filename: str) -> bool:
        """Check if file extension is allowed."""
        if not filename:
            return False
        
        file_ext = os.path.splitext(filename.lower())[1]
        return file_ext in settings.allowed_file_extensions
    
    @staticmethod
    def validate_file_size(file_size: int) -> bool:
        """Check if file size is within limits."""
        return file_size <= settings.max_file_size_bytes
    
    @staticmethod
    def validate_filename(filename: str) -> str:
        """Sanitize and validate filename."""
        if not filename:
            raise ValueError("Filename cannot be empty")
        
        # Remove directory traversal attempts
        filename = os.path.basename(filename)
        
        # Remove potentially dangerous characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Limit filename length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:250] + ext
        
        # Ensure filename is not empty after sanitization
        if not filename or filename.startswith('.'):
            raise ValueError("Invalid filename")
        
        return filename
    
    @staticmethod
    async def validate_upload_file(file: UploadFile) -> UploadFile:
        """Comprehensive file validation."""
        # Validate filename
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Filename is required"
            )
        
        # Sanitize filename
        try:
            file.filename = FileValidator.validate_filename(file.filename)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        
        # Validate file extension
        if not FileValidator.validate_file_extension(file.filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type not allowed. Allowed types: {', '.join(settings.allowed_file_extensions)}"
            )
        
        # Read file to check size
        content = await file.read()
        file_size = len(content)
        
        # Reset file pointer
        await file.seek(0)
        
        # Validate file size
        if not FileValidator.validate_file_size(file_size):
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB"
            )
        
        # Basic content validation (check for executable headers)
        if content.startswith(b'MZ') or content.startswith(b'\x7fELF'):
            logger.warning(f"Executable file upload attempt: {file.filename}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Executable files are not allowed"
            )
        
        return file


class SecurityHeaders:
    """Security headers for HTTP responses."""
    
    @staticmethod
    def get_security_headers() -> dict:
        """Get standard security headers."""
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline';",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }


def sanitize_error_message(error_msg: str) -> str:
    """Sanitize error messages to prevent information disclosure."""
    # Remove file paths
    error_msg = re.sub(r'[A-Za-z]:\\[^\\]+(?:\\[^\\]+)*', '[PATH_REMOVED]', error_msg)
    error_msg = re.sub(r'/[^/\s]+(?:/[^/\s]+)*', '[PATH_REMOVED]', error_msg)
    
    # Remove API keys or tokens
    error_msg = re.sub(r'[A-Za-z0-9]{20,}', '[TOKEN_REMOVED]', error_msg)
    
    # Remove IP addresses
    error_msg = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '[IP_REMOVED]', error_msg)
    
    return error_msg
