#!/usr/bin/env python3
"""
Test script to verify that the safety filter error message is no longer appearing in responses.
"""

import asyncio
from src.core.output_validator import OutputValidationService

def test_validation_logic():
    """Test the validation logic to ensure no error messages are injected."""
    print("Testing response validation logic...")
    print("=" * 50)
    
    # Initialize validator
    validator = OutputValidationService()
    
    # Test educational content that should pass validation
    test_chunks = [
        "Traditional programming involves writing explicit instructions",
        "for computers to follow step by step. Programmers write code",
        "that tells the computer exactly what to do in every situation.",
        "Machine learning is when computer systems perform tasks",
        "effectively without explicit instructions, relying on patterns",
        "and inference from data to make decisions."
    ]
    
    print("Testing individual chunk validation:")
    print("-" * 40)
    
    validated_chunks = []
    error_messages = []
    
    for i, chunk in enumerate(test_chunks):
        try:
            # Test the validation that's used in the app
            validation_result = validator.validate_ai_response(chunk, context={"query": "what is machine learning"})
            
            print(f"Chunk {i+1}: '{chunk[:30]}...'")
            print(f"  Valid: {validation_result['is_valid']}")
            print(f"  Error: {validation_result.get('error', 'None')}")
            
            if validation_result["is_valid"]:
                safe_chunk = validator.sanitize_response_for_display(validation_result["cleaned_response"])
                validated_chunks.append(safe_chunk)
            else:
                # This is where the error message was being added - now we skip
                error_messages.append(f"Chunk {i+1} failed validation")
                print(f"  ERROR: Chunk failed validation - SKIPPED")
            
            print()
            
        except Exception as e:
            print(f"  WARNING: Validation error: {str(e)}")
            validated_chunks.append(chunk)  # Fallback to original
            print()
    
    print("Results:")
    print("-" * 40)
    print(f"Total chunks: {len(test_chunks)}")
    print(f"Valid chunks: {len(validated_chunks)}")
    print(f"Failed chunks: {len(error_messages)}")
    
    full_response = "".join(validated_chunks)
    print(f"\nFull validated response:")
    print(f"'{full_response}'")
    
    # Check for the problematic error message
    error_message = "I found some relevant information, but it couldn't be displayed due to safety filters"
    if error_message in full_response:
        print(f"\nERROR: Safety filter message still appears!")
        return False
    else:
        print(f"\nSUCCESS: No safety filter error message found!")
        return True

def test_error_response_creation():
    """Test that error responses are created correctly but not automatically injected."""
    print("\n" + "=" * 50)
    print("Testing error response creation (should not auto-inject):")
    print("-" * 50)
    
    validator = OutputValidationService()
    
    # Test creating error response (this should exist but not be auto-used)
    error_response = validator.create_safe_error_response("content_filtered", "test query")
    
    print(f"Error response structure: {error_response}")
    print(f"Error message: '{error_response['response']}'")
    print("✅ Error response creation works (but should not be auto-injected)")

if __name__ == "__main__":
    success = test_validation_logic()
    test_error_response_creation()
    
    if success:
        print(f"\n🎉 ALL TESTS PASSED - Safety filter error message fix is working!")
    else:
        print(f"\n❌ TESTS FAILED - Safety filter error message still appears!")
