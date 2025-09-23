#!/usr/bin/env python3
"""
Test script to verify improved Guardrails validation for educational content.
"""

from src.core.guardrails_config import RAGGuardrails

def test_improved_validation():
    """Test the improved Guardrails validation."""
    print("Testing improved Guardrails validation:")
    print("=" * 50)
    
    # Initialize Guardrails
    guardrails = RAGGuardrails()
    
    # Test queries that should be allowed
    test_queries = [
        "what is Traditional Programming?",
        "what is machine learning?",
        "explain supervised learning",
        "how does artificial intelligence work?",
        "what are programming algorithms?",
        "difference between supervised and unsupervised learning"
    ]
    
    print("\n1. Testing Search Query Validation:")
    print("-" * 40)
    
    for query in test_queries:
        try:
            result = guardrails.validate_search_query(query)
            print(f"Query: {query}")
            print(f"Valid: {result['is_valid']}")
            print(f"Error: {result.get('error', 'None')}")
            print()
        except Exception as e:
            print(f"Query: {query}")
            print(f"Error during validation: {str(e)}")
            print()
    
    print("\n2. Testing AI Output Validation:")
    print("-" * 40)
    
    # Test AI responses that should be allowed
    test_responses = [
        "Traditional programming involves writing explicit instructions for computers to follow. Machine learning is when computer systems perform tasks effectively without explicit instructions, relying on patterns and inference.",
        "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
        "Programming algorithms are step-by-step procedures for solving computational problems."
    ]
    
    for response in test_responses:
        try:
            result = guardrails.validate_ai_output(response)
            print(f"Response: {response[:60]}...")
            print(f"Valid: {result['is_valid']}")
            print(f"Error: {result.get('error', 'None')}")
            print()
        except Exception as e:
            print(f"Response: {response[:60]}...")
            print(f"Error during validation: {str(e)}")
            print()

if __name__ == "__main__":
    test_improved_validation()
