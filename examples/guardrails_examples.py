"""
Guardrails Examples for RAG System
Demonstrates how to use the implemented Guardrails for input/output validation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.guardrails_config import rag_guardrails, content_safety
from src.core.input_validator import input_validator
from src.core.output_validator import output_validator


def example_input_validation():
    """Example of input validation using Guardrails."""
    print("=== INPUT VALIDATION EXAMPLES ===\n")
    
    # Example 1: Valid search query
    print("1. Valid Search Query:")
    query = "What are the benefits of machine learning in healthcare?"
    result = input_validator.validate_search_query(query)
    print(f"Query: {query}")
    print(f"Result: {result}")
    print()
    
    # Example 2: Query with PII (should be filtered)
    print("2. Query with PII (should be filtered):")
    pii_query = "Find documents about john.doe@email.com and his research"
    try:
        result = input_validator.validate_search_query(pii_query)
        print(f"Query: {pii_query}")
        print(f"Result: {result}")
    except Exception as e:
        print(f"Query: {pii_query}")
        print(f"Validation failed: {str(e)}")
    print()
    
    # Example 3: Toxic language (should fail)
    print("3. Toxic Language (should fail):")
    toxic_query = "I hate this stupid system"
    try:
        result = input_validator.validate_search_query(toxic_query)
        print(f"Query: {toxic_query}")
        print(f"Result: {result}")
    except Exception as e:
        print(f"Query: {toxic_query}")
        print(f"Validation failed: {str(e)}")
    print()


def example_output_validation():
    """Example of output validation using Guardrails."""
    print("=== OUTPUT VALIDATION EXAMPLES ===\n")
    
    # Example 1: Valid AI response
    print("1. Valid AI Response:")
    response = "Machine learning has several benefits in healthcare, including improved diagnosis accuracy, personalized treatment plans, and efficient drug discovery processes."
    result = output_validator.validate_ai_response(response)
    print(f"Response: {response}")
    print(f"Validation Result: {result}")
    print()
    
    # Example 2: Response with PII (should be filtered)
    print("2. Response with PII (should be filtered):")
    pii_response = "The patient John Smith (SSN: 123-45-6789) shows improvement in his condition."
    result = output_validator.validate_ai_response(pii_response)
    print(f"Response: {pii_response}")
    print(f"Validation Result: {result}")
    print()
    
    # Example 3: Too short response (quality issue)
    print("3. Too Short Response (quality issue):")
    short_response = "Yes."
    result = output_validator.validate_ai_response(short_response)
    print(f"Response: {short_response}")
    print(f"Validation Result: {result}")
    print()


def example_file_validation():
    """Example of file upload validation."""
    print("=== FILE VALIDATION EXAMPLES ===\n")
    
    # Example 1: Safe filename
    print("1. Safe Filename:")
    safe_filename = "research_paper.pdf"
    sanitized = content_safety.sanitize_filename(safe_filename)
    print(f"Original: {safe_filename}")
    print(f"Sanitized: {sanitized}")
    print()
    
    # Example 2: Unsafe filename with path traversal
    print("2. Unsafe Filename (path traversal):")
    unsafe_filename = "../../etc/passwd"
    sanitized = content_safety.sanitize_filename(unsafe_filename)
    print(f"Original: {unsafe_filename}")
    print(f"Sanitized: {sanitized}")
    print()
    
    # Example 3: Document content safety check
    print("3. Document Content Safety Check:")
    safe_content = "This is a research paper about artificial intelligence applications."
    safety_result = content_safety.check_document_safety(safe_content)
    print(f"Content: {safe_content}")
    print(f"Safety Result: {safety_result}")
    print()
    
    unsafe_content = "This document contains confidential API keys and passwords."
    safety_result = content_safety.check_document_safety(unsafe_content)
    print(f"Content: {unsafe_content}")
    print(f"Safety Result: {safety_result}")
    print()


def example_structured_validation():
    """Example of structured output validation."""
    print("=== STRUCTURED OUTPUT VALIDATION ===\n")
    
    # Define a schema for search results
    search_result_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "content": {"type": "string"},
            "relevance_score": {"type": "number", "minimum": 0, "maximum": 1},
            "source": {"type": "string"}
        },
        "required": ["title", "content", "relevance_score"]
    }
    
    # Example 1: Valid structured output
    print("1. Valid Structured Output:")
    valid_output = {
        "title": "Machine Learning in Healthcare",
        "content": "This paper discusses the applications of ML in medical diagnosis.",
        "relevance_score": 0.85,
        "source": "research_paper.pdf"
    }
    
    result = rag_guardrails.validate_structured_output(valid_output, search_result_schema)
    print(f"Output: {valid_output}")
    print(f"Validation Result: {result}")
    print()
    
    # Example 2: Invalid structured output (missing required field)
    print("2. Invalid Structured Output (missing required field):")
    invalid_output = {
        "title": "Machine Learning in Healthcare",
        "content": "This paper discusses the applications of ML in medical diagnosis."
        # Missing relevance_score
    }
    
    result = rag_guardrails.validate_structured_output(invalid_output, search_result_schema)
    print(f"Output: {invalid_output}")
    print(f"Validation Result: {result}")
    print()


def example_api_parameter_validation():
    """Example of API parameter validation."""
    print("=== API PARAMETER VALIDATION ===\n")
    
    # Example 1: Valid parameters
    print("1. Valid Parameters:")
    valid_params = {"limit": 10, "offset": 0, "threshold": 0.3}
    try:
        result = input_validator.validate_api_parameters(valid_params)
        print(f"Parameters: {valid_params}")
        print(f"Validated: {result}")
    except Exception as e:
        print(f"Validation failed: {str(e)}")
    print()
    
    # Example 2: Invalid parameters
    print("2. Invalid Parameters:")
    invalid_params = {"limit": 150, "offset": -5, "threshold": 1.5}  # All invalid
    try:
        result = input_validator.validate_api_parameters(invalid_params)
        print(f"Parameters: {invalid_params}")
        print(f"Validated: {result}")
    except Exception as e:
        print(f"Parameters: {invalid_params}")
        print(f"Validation failed: {str(e)}")
    print()


if __name__ == "__main__":
    print("GUARDRAILS EXAMPLES FOR RAG SYSTEM")
    print("=" * 50)
    print()
    
    try:
        example_input_validation()
        example_output_validation()
        example_file_validation()
        example_structured_validation()
        example_api_parameter_validation()
        
        print("=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {str(e)}")
        print("Note: Make sure to install guardrails-ai first:")
        print("pip install guardrails-ai")
