"""
Output validation service using Guardrails for RAG system.
Validates AI-generated responses and ensures safe, quality outputs.
"""

import logging
import json
from typing import Dict, Any, Optional, List
from .guardrails_config import rag_guardrails

logger = logging.getLogger(__name__)


class OutputValidationService:
    """Service for validating AI-generated outputs using Guardrails."""
    
    def __init__(self):
        self.guardrails = rag_guardrails
        
    def validate_ai_response(self, response: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Validate AI-generated response for safety and quality.
        
        Args:
            response: AI-generated response text
            context: Optional context about the query/response
            
        Returns:
            Validation result with cleaned response
        """
        if not response or not response.strip():
            return {
                "is_valid": False,
                "cleaned_response": "",
                "error": "Empty response from AI",
                "validation_passed": False
            }
        
        # Guardrails validation
        validation_result = self.guardrails.validate_ai_output(response)
        
        # Additional quality checks
        quality_score = self._assess_response_quality(response, context)
        
        return {
            "is_valid": validation_result["is_valid"] and quality_score["is_acceptable"],
            "cleaned_response": validation_result.get("cleaned_response", response),
            "validation_passed": validation_result["validation_passed"],
            "quality_score": quality_score["score"],
            "quality_issues": quality_score.get("issues", []),
            "error": validation_result.get("error")
        }
    
    def validate_search_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate search results for safety and relevance.
        
        Args:
            results: List of search result dictionaries
            
        Returns:
            Validation result with filtered results
        """
        if not results:
            return {
                "is_valid": True,
                "filtered_results": [],
                "removed_count": 0,
                "total_count": 0
            }
        
        filtered_results = []
        removed_count = 0
        
        for result in results:
            try:
                # Validate each result's content
                content = result.get('content', '')
                if content:
                    validation = self.validate_ai_response(content)
                    if validation["is_valid"]:
                        result['content'] = validation["cleaned_response"]
                        filtered_results.append(result)
                    else:
                        removed_count += 1
                        logger.warning(f"Removed unsafe search result: {validation.get('error', 'Unknown error')}")
                else:
                    filtered_results.append(result)
            except Exception as e:
                # If validation fails for any reason, include the original result
                logger.warning(f"Search result validation error: {str(e)}")
                filtered_results.append(result)
        
        return {
            "is_valid": True,
            "filtered_results": filtered_results,
            "removed_count": removed_count,
            "total_count": len(results)
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
            # Create structured guard
            structured_guard = self.guardrails.create_structured_guard(schema)
            
            # Convert to JSON string for validation
            output_json = json.dumps(output)
            result = structured_guard.validate(output_json)
            
            return {
                "is_valid": True,
                "validated_output": json.loads(result.validated_output),
                "validation_passed": result.validation_passed,
                "error": None
            }
        except Exception as e:
            logger.warning(f"Structured output validation failed: {str(e)}")
            return {
                "is_valid": False,
                "validated_output": None,
                "validation_passed": False,
                "error": str(e)
            }
    
    def _assess_response_quality(self, response: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Assess the quality of an AI response.
        
        Args:
            response: AI response text
            context: Optional context information
            
        Returns:
            Quality assessment result
        """
        issues = []
        score = 1.0
        
        # Length checks (more lenient for streaming chunks)
        if len(response.strip()) < 3:  # Only reject extremely short chunks
            issues.append("Response too short")
            score -= 0.3
        elif len(response) > 5000:
            issues.append("Response too long")
            score -= 0.2
        
        # Content quality checks
        if response.count('\n') > 50:
            issues.append("Too many line breaks")
            score -= 0.1
        
        # Repetition check
        words = response.lower().split()
        if len(words) > 0:
            unique_words = set(words)
            repetition_ratio = 1 - (len(unique_words) / len(words))
            if repetition_ratio > 0.5:
                issues.append("High repetition detected")
                score -= 0.3
        
        # Check for incomplete sentences (more lenient for streaming chunks)
        if response.endswith(('...', '..', '.')):
            pass  # Normal ending
        elif len(response.strip()) > 50 and not response.endswith(('.', '!', '?', '"', "'", ' ')):
            # Only flag as incomplete if it's a substantial chunk that doesn't end properly
            issues.append("Response appears incomplete")
            score -= 0.1  # Reduced penalty for streaming
        
        # Context relevance (if context provided)
        if context and 'query' in context:
            query_words = set(context['query'].lower().split())
            response_words = set(response.lower().split())
            overlap = len(query_words.intersection(response_words))
            if overlap == 0 and len(query_words) > 0:
                issues.append("Response may not be relevant to query")
                score -= 0.4
        
        return {
            "score": max(0.0, score),
            "is_acceptable": score >= 0.5,
            "issues": issues
        }
    
    def sanitize_response_for_display(self, response: str) -> str:
        """
        Sanitize response for safe display in web interface.
        
        Args:
            response: Raw response text
            
        Returns:
            Sanitized response text
        """
        import bleach
        
        # Allow basic formatting tags
        allowed_tags = ['p', 'br', 'strong', 'em', 'ul', 'ol', 'li', 'code', 'pre']
        allowed_attributes = {}
        
        # Clean HTML
        cleaned = bleach.clean(response, tags=allowed_tags, attributes=allowed_attributes)
        
        # Remove excessive whitespace
        import re
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
        cleaned = re.sub(r' +', ' ', cleaned)
        
        return cleaned.strip()
    
    def create_safe_error_response(self, error_type: str, user_query: str = "") -> Dict[str, Any]:
        """
        Create a safe error response that doesn't expose system details.
        
        Args:
            error_type: Type of error that occurred
            user_query: Original user query (for context)
            
        Returns:
            Safe error response
        """
        safe_responses = {
            "validation_failed": "I apologize, but I couldn't process your request safely. Please try rephrasing your question.",
            "content_filtered": "I found some relevant information, but it couldn't be displayed due to safety filters.",
            "no_results": "I couldn't find relevant information for your query. Please try different keywords.",
            "system_error": "I'm experiencing technical difficulties. Please try again in a moment.",
            "rate_limited": "Too many requests. Please wait a moment before trying again."
        }
        
        return {
            "response": safe_responses.get(error_type, safe_responses["system_error"]),
            "error_type": error_type,
            "is_error": True,
            "timestamp": None  # Will be set by the API layer
        }


# Global instance for easy access
output_validator = OutputValidationService()
