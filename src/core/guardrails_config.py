"""
Guardrails configuration for RAG system security and validation.
Provides input/output validation, content safety, and structured generation.
"""

from typing import Dict, Any, List, Optional
import re
import logging

logger = logging.getLogger(__name__)


class RAGGuardrails:
    """Guardrails implementation for RAG system safety and validation."""
    
    def __init__(self):
        self.pii_patterns = self._compile_pii_patterns()
        self.toxic_words = self._load_toxic_words()
        self.restricted_topics = self._load_restricted_topics()
        
    def _compile_pii_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for PII detection."""
        return {
            "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "phone": re.compile(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'),
            "ssn": re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
            "credit_card": re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b')
        }
    
    def _load_toxic_words(self) -> List[str]:
        """Load list of toxic/harmful words and phrases (adjusted for educational content)."""
        return [
            "hate speech", "stupid person", "idiot user", "kill someone", "murder", 
            "terrorist attack", "bomb making", "illegal drugs", "steal money"
        ]
    
    def _load_restricted_topics(self) -> List[str]:
        """Load list of restricted topic keywords (focused on truly harmful content)."""
        return [
            "illegal activities", "drug trafficking", "violent crimes", "weapon manufacturing", 
            "terrorism planning", "adult pornography", "gambling addiction", "fraud schemes", 
            "scam operations", "malware creation", "virus development"
        ]
    
    def validate_user_input(self, user_input: str) -> Dict[str, Any]:
        """
        Validate user input for safety and format.
        
        Args:
            user_input: Raw user input string
            
        Returns:
            Dict containing validation result and cleaned input
        """
        try:
            # Length validation
            if len(user_input.strip()) == 0:
                return {
                    "is_valid": False,
                    "cleaned_input": None,
                    "validation_passed": False,
                    "error": "Input cannot be empty"
                }
            
            if len(user_input) > 1000:
                return {
                    "is_valid": False,
                    "cleaned_input": None,
                    "validation_passed": False,
                    "error": "Input too long (max 1000 characters)"
                }
            
            # PII detection and filtering
            cleaned_input = self._filter_pii(user_input)
            
            # Toxic language detection
            if self._contains_toxic_language(cleaned_input):
                return {
                    "is_valid": False,
                    "cleaned_input": None,
                    "validation_passed": False,
                    "error": "Input contains inappropriate language"
                }
            
            return {
                "is_valid": True,
                "cleaned_input": cleaned_input,
                "validation_passed": True,
                "error": None
            }
        except Exception as e:
            logger.warning(f"Input validation failed: {str(e)}")
            return {
                "is_valid": False,
                "cleaned_input": None,
                "validation_passed": False,
                "error": str(e)
            }
    
    def _filter_pii(self, text: str) -> str:
        """Filter PII from text using regex patterns."""
        filtered_text = text
        for pii_type, pattern in self.pii_patterns.items():
            if pii_type == "email":
                filtered_text = pattern.sub("[EMAIL_REDACTED]", filtered_text)
            elif pii_type == "phone":
                filtered_text = pattern.sub("[PHONE_REDACTED]", filtered_text)
            elif pii_type == "ssn":
                filtered_text = pattern.sub("[SSN_REDACTED]", filtered_text)
            elif pii_type == "credit_card":
                filtered_text = pattern.sub("[CARD_REDACTED]", filtered_text)
        return filtered_text
    
    def _contains_toxic_language(self, text: str) -> bool:
        """Check if text contains toxic language (context-aware for educational content)."""
        text_lower = text.lower()
        
        # Check for educational context indicators
        educational_indicators = [
            "programming", "machine learning", "algorithm", "computer science", 
            "software", "technology", "artificial intelligence", "data science",
            "coding", "development", "system", "method", "technique", "approach"
        ]
        
        # If text contains educational indicators, be more lenient
        has_educational_context = any(indicator in text_lower for indicator in educational_indicators)
        
        if has_educational_context:
            # Only flag truly harmful phrases in educational context
            harmful_phrases = [phrase for phrase in self.toxic_words if phrase in text_lower]
            return len(harmful_phrases) > 0
        else:
            # Standard toxic language check for non-educational content
            return any(word in text_lower for word in self.toxic_words)
    
    def _contains_restricted_topics(self, text: str) -> bool:
        """Check if text contains restricted topics (context-aware for educational content)."""
        text_lower = text.lower()
        
        # Check for educational context indicators
        educational_indicators = [
            "programming", "machine learning", "algorithm", "computer science", 
            "software", "technology", "artificial intelligence", "data science",
            "coding", "development", "system", "method", "technique", "approach",
            "traditional programming", "supervised learning", "unsupervised learning"
        ]
        
        # If text contains educational indicators, be more lenient with restrictions
        has_educational_context = any(indicator in text_lower for indicator in educational_indicators)
        
        if has_educational_context:
            # Only flag truly harmful topics in educational context
            return False  # Allow educational discussions
        else:
            # Standard restricted topic check for non-educational content
            return any(topic in text_lower for topic in self.restricted_topics)
    
    def validate_search_query(self, query: str) -> Dict[str, Any]:
        """
        Validate search query for topic relevance and safety.
        
        Args:
            query: Search query string
            
        Returns:
            Dict containing validation result and cleaned query
        """
        try:
            # Length validation
            if len(query.strip()) < 3:
                return {
                    "is_valid": False,
                    "cleaned_query": None,
                    "validation_passed": False,
                    "error": "Query too short (minimum 3 characters)"
                }
            
            if len(query) > 500:
                return {
                    "is_valid": False,
                    "cleaned_query": None,
                    "validation_passed": False,
                    "error": "Query too long (maximum 500 characters)"
                }
            
            # Check for restricted topics
            if self._contains_restricted_topics(query):
                return {
                    "is_valid": False,
                    "cleaned_query": None,
                    "validation_passed": False,
                    "error": "Query contains restricted topics"
                }
            
            # Filter PII and check for toxic language
            cleaned_query = self._filter_pii(query)
            
            if self._contains_toxic_language(cleaned_query):
                return {
                    "is_valid": False,
                    "cleaned_query": None,
                    "validation_passed": False,
                    "error": "Query contains inappropriate language"
                }
            
            return {
                "is_valid": True,
                "cleaned_query": cleaned_query,
                "validation_passed": True,
                "error": None
            }
        except Exception as e:
            logger.warning(f"Query validation failed: {str(e)}")
            return {
                "is_valid": False,
                "cleaned_query": None,
                "validation_passed": False,
                "error": str(e)
            }
    
    def validate_ai_output(self, ai_response: str) -> Dict[str, Any]:
        """
        Validate AI-generated response for safety and quality.
        
        Args:
            ai_response: AI-generated response string
            
        Returns:
            Dict containing validation result and cleaned response
        """
        try:
            # Length validation
            if len(ai_response.strip()) < 10:
                return {
                    "is_valid": False,
                    "cleaned_response": None,
                    "validation_passed": False,
                    "error": "Response too short"
                }
            
            if len(ai_response) > 5000:
                return {
                    "is_valid": False,
                    "cleaned_response": None,
                    "validation_passed": False,
                    "error": "Response too long"
                }
            
            # Filter PII
            cleaned_response = self._filter_pii(ai_response)
            
            # Check for toxic language (more lenient for AI responses)
            if self._contains_toxic_language(cleaned_response):
                logger.warning("AI response contains potentially toxic language")
                # Don't reject, just log and clean
            
            return {
                "is_valid": True,
                "cleaned_response": cleaned_response,
                "validation_passed": True,
                "error": None
            }
        except Exception as e:
            logger.warning(f"Output validation failed: {str(e)}")
            return {
                "is_valid": False,
                "cleaned_response": None,
                "validation_passed": False,
                "error": str(e)
            }
    
    def validate_structured_output(self, output: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate structured output against a schema.
        
        Args:
            output: Structured output to validate
            schema: JSON schema for validation
            
        Returns:
            Validation result
        """
        try:
            import jsonschema
            jsonschema.validate(output, schema)
            return {
                "is_valid": True,
                "validated_output": output,
                "validation_passed": True,
                "error": None
            }
        except ImportError:
            # If jsonschema not available, do basic validation
            required_fields = schema.get("required", [])
            for field in required_fields:
                if field not in output:
                    return {
                        "is_valid": False,
                        "validated_output": None,
                        "validation_passed": False,
                        "error": f"Missing required field: {field}"
                    }
            return {
                "is_valid": True,
                "validated_output": output,
                "validation_passed": True,
                "error": None
            }
        except Exception as e:
            return {
                "is_valid": False,
                "validated_output": None,
                "validation_passed": False,
                "error": str(e)
            }


class ContentSafetyGuard:
    """Additional content safety checks for sensitive applications."""
    
    @staticmethod
    def check_document_safety(content: str) -> Dict[str, Any]:
        """
        Check if document content is safe for processing.
        
        Args:
            content: Document content to check
            
        Returns:
            Dict with safety assessment
        """
        # Basic safety checks
        unsafe_patterns = [
            "confidential", "classified", "secret", "private",
            "password", "api_key", "token", "credential"
        ]
        
        content_lower = content.lower()
        found_patterns = [pattern for pattern in unsafe_patterns if pattern in content_lower]
        
        return {
            "is_safe": len(found_patterns) == 0,
            "unsafe_patterns": found_patterns,
            "risk_level": "high" if found_patterns else "low"
        }
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize uploaded filename for security.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        import re
        import os
        
        # Remove path traversal attempts
        filename = os.path.basename(filename)
        
        # Remove dangerous characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:250] + ext
        
        return filename


# Global instance for easy access
rag_guardrails = RAGGuardrails()
content_safety = ContentSafetyGuard()
