"""
Data validation and poisoning protection for RAG system.
"""
import os
import hashlib
import mimetypes
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
# import magic  # Temporarily disabled for Windows compatibility
import re
from datetime import datetime
import logging
from src.config import settings
from src.core.monitoring import security_logger, ThreatLevel

logger = logging.getLogger(__name__)


class DataPoisoningDetector:
    """Detects potential data poisoning attempts in uploaded documents."""
    
    def __init__(self):
        self.suspicious_patterns = [
            # Prompt injection patterns in documents
            r"ignore\s+previous\s+instructions",
            r"you\s+are\s+now\s+a\s+helpful\s+assistant\s+that",
            r"system\s*:\s*ignore",
            r"human\s*:\s*forget",
            r"assistant\s*:\s*disregard",
            
            # Data extraction attempts
            r"show\s+all\s+users",
            r"dump\s+database",
            r"list\s+all\s+passwords",
            r"reveal\s+api\s+keys",
            
            # Malicious instructions
            r"execute\s+shell\s+command",
            r"run\s+system\s+command",
            r"access\s+file\s+system",
            
            # Social engineering
            r"urgent\s+security\s+update",
            r"click\s+here\s+immediately",
            r"verify\s+your\s+account",
        ]
        
        self.trusted_sources = set()  # Can be populated with known good sources
    
    def validate_document_content(self, file_path: str, content: str) -> Tuple[bool, float, List[str]]:
        """
        Validate document content for potential poisoning.
        Returns: (is_safe, risk_score, detected_threats)
        """
        threats = []
        risk_score = 0.0
        
        # Check for suspicious patterns
        pattern_score = self._check_suspicious_patterns(content)
        risk_score += pattern_score
        if pattern_score > 0.3:
            threats.append("suspicious_patterns")
        
        # Check for excessive special characters (potential obfuscation)
        special_char_score = self._check_special_characters(content)
        risk_score += special_char_score
        if special_char_score > 0.2:
            threats.append("excessive_special_chars")
        
        # Check for potential encoding attacks
        encoding_score = self._check_encoding_attacks(content)
        risk_score += encoding_score
        if encoding_score > 0.2:
            threats.append("encoding_attack")
        
        # Check document metadata for suspicious information
        metadata_score = self._check_document_metadata(file_path)
        risk_score += metadata_score
        if metadata_score > 0.1:
            threats.append("suspicious_metadata")
        
        # Check for repetitive content (potential spam/flooding)
        repetition_score = self._check_content_repetition(content)
        risk_score += repetition_score
        if repetition_score > 0.3:
            threats.append("repetitive_content")
        
        is_safe = risk_score < 0.5  # Threshold for considering content safe
        
        if not is_safe:
            security_logger.log_threat(
                threat_type="data_poisoning_attempt",
                threat_level=ThreatLevel.HIGH if risk_score > 0.7 else ThreatLevel.MEDIUM,
                user_id="system",
                ip_address="internal",
                details={
                    "file_path": file_path,
                    "risk_score": risk_score,
                    "threats": threats,
                    "content_preview": content[:200]
                },
                risk_score=risk_score
            )
        
        return is_safe, risk_score, threats
    
    def _check_suspicious_patterns(self, content: str) -> float:
        """Check for suspicious patterns in content."""
        score = 0.0
        content_lower = content.lower()
        
        for pattern in self.suspicious_patterns:
            matches = len(re.findall(pattern, content_lower))
            if matches > 0:
                score += min(matches * 0.1, 0.3)  # Cap contribution per pattern
        
        return min(score, 1.0)
    
    def _check_special_characters(self, content: str) -> float:
        """Check for excessive special characters."""
        total_chars = len(content)
        if total_chars == 0:
            return 0.0
        
        special_chars = len(re.findall(r'[<>{}[\]|\\`~!@#$%^&*()_+=]', content))
        ratio = special_chars / total_chars
        
        # Normal documents should have < 5% special characters
        if ratio > 0.15:  # 15% threshold
            return 0.5
        elif ratio > 0.10:  # 10% threshold
            return 0.3
        elif ratio > 0.05:  # 5% threshold
            return 0.1
        
        return 0.0
    
    def _check_encoding_attacks(self, content: str) -> float:
        """Check for potential encoding-based attacks."""
        score = 0.0
        
        # Check for excessive Unicode escape sequences
        unicode_escapes = len(re.findall(r'\\u[0-9a-fA-F]{4}', content))
        if unicode_escapes > 10:
            score += 0.3
        
        # Check for HTML entities
        html_entities = len(re.findall(r'&[a-zA-Z0-9]+;', content))
        if html_entities > 20:
            score += 0.2
        
        # Check for base64-like strings (potential obfuscation)
        base64_like = len(re.findall(r'[A-Za-z0-9+/]{20,}={0,2}', content))
        if base64_like > 5:
            score += 0.2
        
        return min(score, 1.0)
    
    def _check_document_metadata(self, file_path: str) -> float:
        """Check document metadata for suspicious information."""
        score = 0.0
        
        try:
            # Check file creation/modification times
            stat = os.stat(file_path)
            now = datetime.now().timestamp()
            
            # Files created in the future are suspicious
            if stat.st_ctime > now + 3600:  # 1 hour tolerance
                score += 0.3
            
            # Very old files being uploaded might be suspicious
            age_days = (now - stat.st_ctime) / (24 * 3600)
            if age_days > 365 * 5:  # 5 years old
                score += 0.1
            
        except Exception as e:
            logger.warning(f"Could not check metadata for {file_path}: {e}")
        
        return min(score, 1.0)
    
    def _check_content_repetition(self, content: str) -> float:
        """Check for repetitive content patterns."""
        if len(content) < 100:
            return 0.0
        
        # Split into sentences and check for repetition
        sentences = re.split(r'[.!?]+', content)
        if len(sentences) < 5:
            return 0.0
        
        unique_sentences = set(s.strip().lower() for s in sentences if s.strip())
        repetition_ratio = 1 - (len(unique_sentences) / len(sentences))
        
        if repetition_ratio > 0.7:  # 70% repetition
            return 0.5
        elif repetition_ratio > 0.5:  # 50% repetition
            return 0.3
        elif repetition_ratio > 0.3:  # 30% repetition
            return 0.1
        
        return 0.0


class SecureFileValidator:
    """Enhanced file validation with security focus."""
    
    def __init__(self):
        self.data_poisoning_detector = DataPoisoningDetector()
        
        # Allowed MIME types
        self.allowed_mime_types = {
            'application/pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/msword',
            'text/plain'
        }
        
        # Maximum file sizes by type (in bytes)
        self.max_file_sizes = {
            'application/pdf': 100 * 1024 * 1024,  # 100MB for PDFs
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 50 * 1024 * 1024,  # 50MB for DOCX
            'application/msword': 50 * 1024 * 1024,  # 50MB for DOC
            'text/plain': 10 * 1024 * 1024,  # 10MB for text files
        }
    
    def validate_file_security(self, file_path: str, content: bytes) -> Tuple[bool, List[str]]:
        """
        Comprehensive security validation of uploaded file.
        Returns: (is_safe, security_issues)
        """
        issues = []
        
        # Check file type using mimetypes (magic library temporarily disabled)
        try:
            # Use mimetypes module for basic file type detection
            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type:
                # Basic fallback based on file extension
                ext = Path(file_path).suffix.lower()
                if ext == '.pdf':
                    mime_type = 'application/pdf'
                elif ext in ['.docx', '.doc']:
                    mime_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                elif ext == '.txt':
                    mime_type = 'text/plain'
                else:
                    mime_type = 'application/octet-stream'
        except Exception:
            mime_type = 'application/octet-stream'
        
        if mime_type not in self.allowed_mime_types:
            issues.append(f"Disallowed file type: {mime_type}")
        
        # Check file size
        file_size = len(content)
        max_size = self.max_file_sizes.get(mime_type, settings.max_file_size_bytes)
        if file_size > max_size:
            issues.append(f"File too large: {file_size} bytes (max: {max_size})")
        
        # Check for executable headers
        if self._has_executable_headers(content):
            issues.append("File contains executable headers")
        
        # Check for embedded scripts
        if self._has_embedded_scripts(content):
            issues.append("File contains embedded scripts")
        
        # Check for suspicious file structure
        if self._has_suspicious_structure(content, mime_type):
            issues.append("File has suspicious structure")
        
        return len(issues) == 0, issues
    
    def validate_content_security(self, file_path: str, text_content: str) -> Tuple[bool, float, List[str]]:
        """Validate extracted text content for security issues."""
        return self.data_poisoning_detector.validate_document_content(file_path, text_content)
    
    def _has_executable_headers(self, content: bytes) -> bool:
        """Check for executable file headers."""
        executable_headers = [
            b'MZ',      # Windows PE
            b'\x7fELF',  # Linux ELF
            b'\xfe\xed\xfa\xce',  # Mach-O (macOS)
            b'\xfe\xed\xfa\xcf',  # Mach-O (macOS)
            b'#!/bin/',  # Shell script
            b'#!/usr/',  # Shell script
        ]
        
        for header in executable_headers:
            if content.startswith(header):
                return True
        
        return False
    
    def _has_embedded_scripts(self, content: bytes) -> bool:
        """Check for embedded scripts in documents."""
        script_patterns = [
            b'<script',
            b'javascript:',
            b'vbscript:',
            rb'on\w+\s*=',  # Event handlers
            rb'eval\s*\(',
            rb'document\.',
            rb'window\.',
        ]
        
        content_lower = content.lower()
        for pattern in script_patterns:
            if pattern in content_lower:
                return True
        
        return False
    
    def _has_suspicious_structure(self, content: bytes, mime_type: str) -> bool:
        """Check for suspicious file structure."""
        # For PDFs, check for suspicious elements
        if mime_type == 'application/pdf':
            suspicious_pdf_elements = [
                b'/JavaScript',
                b'/JS',
                b'/Launch',
                b'/EmbeddedFile',
                b'/OpenAction',
                b'/AA',  # Additional Actions
            ]
            
            for element in suspicious_pdf_elements:
                if element in content:
                    return True
        
        # For Office documents, check for macros
        elif 'officedocument' in mime_type or mime_type == 'application/msword':
            macro_indicators = [
                b'macros',
                b'vbaProject',
                b'Microsoft Visual Basic',
                b'Sub ',
                b'Function ',
            ]
            
            for indicator in macro_indicators:
                if indicator in content:
                    return True
        
        return False


# Global instance
secure_file_validator = SecureFileValidator()
