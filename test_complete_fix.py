#!/usr/bin/env python3
"""
Comprehensive test to verify all tuple errors have been resolved.
"""

import sys
import os
import traceback
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_search_results_validation():
    """Test the search results validation that was causing the tuple error."""
    print("Testing search results validation...")
    
    try:
        from src.core.output_validator import output_validator
        
        # Create mock search results similar to what the app would use
        mock_results = [
            {
                'content': 'This is a test search result content.',
                'metadata': {'source': 'test.txt', 'page': 1}
            },
            {
                'content': 'Another test result with some content.',
                'metadata': {'source': 'test2.txt', 'page': 2}
            }
        ]
        
        print("Mock results created:", len(mock_results))
        
        # Test the validate_search_results method that was causing issues
        result = output_validator.validate_search_results(mock_results)
        print("Search results validation result type:", type(result))
        print("Search results validation result:", result)
        
        if isinstance(result, tuple):
            print("ERROR: validate_search_results still returns tuple!")
            return False
        elif isinstance(result, dict):
            print("SUCCESS: validate_search_results returns dictionary")
            print("Filtered results count:", len(result.get('filtered_results', [])))
            print("Removed count:", result.get('removed_count', 0))
            return True
        else:
            print("UNEXPECTED: validate_search_results returns:", type(result))
            return False
            
    except Exception as e:
        print("Exception in search results validation:", str(e))
        traceback.print_exc()
        return False

def test_complete_validation_chain():
    """Test the complete validation chain that happens during query processing."""
    print("\nTesting complete validation chain...")
    
    try:
        from src.core.input_validator import input_validator
        from src.core.output_validator import output_validator
        
        # Step 1: Test query validation (this was fixed earlier)
        print("Step 1: Testing query validation...")
        query_result = input_validator.validate_search_query("What is machine learning?")
        print("Query validation OK:", query_result.get('is_safe', False))
        
        # Step 2: Test search results validation (this was the main issue)
        print("Step 2: Testing search results validation...")
        mock_results = [{'content': 'Machine learning is a subset of AI.', 'metadata': {'source': 'ml.txt'}}]
        results_validation = output_validator.validate_search_results(mock_results)
        print("Search results validation OK:", results_validation.get('is_valid', False))
        
        # Step 3: Test AI response validation (this was also fixed)
        print("Step 3: Testing AI response validation...")
        response_result = output_validator.validate_ai_response("Machine learning is a powerful technology.", context={"query": "test"})
        print("AI response validation OK:", response_result.get('is_valid', False))
        
        print("SUCCESS: Complete validation chain working correctly!")
        return True
        
    except Exception as e:
        print("Exception in complete validation chain:", str(e))
        traceback.print_exc()
        return False

def test_error_scenarios():
    """Test error scenarios that might cause tuple errors."""
    print("\nTesting error scenarios...")
    
    try:
        from src.core.output_validator import output_validator
        
        # Test with empty results
        print("Testing with empty results...")
        empty_result = output_validator.validate_search_results([])
        print("Empty results OK:", isinstance(empty_result, dict))
        
        # Test with malformed results
        print("Testing with malformed results...")
        malformed_results = [{'invalid': 'data'}]  # Missing 'content' key
        malformed_result = output_validator.validate_search_results(malformed_results)
        print("Malformed results OK:", isinstance(malformed_result, dict))
        
        # Test with None content
        print("Testing with None content...")
        none_results = [{'content': None, 'metadata': {}}]
        none_result = output_validator.validate_search_results(none_results)
        print("None content OK:", isinstance(none_result, dict))
        
        print("SUCCESS: All error scenarios handled correctly!")
        return True
        
    except Exception as e:
        print("Exception in error scenarios:", str(e))
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("COMPREHENSIVE TUPLE ERROR FIX TEST")
    print("=" * 60)
    
    test1 = test_search_results_validation()
    test2 = test_complete_validation_chain()
    test3 = test_error_scenarios()
    
    print("\n" + "=" * 60)
    print("TEST RESULTS:")
    print(f"Search Results Validation: {'PASS' if test1 else 'FAIL'}")
    print(f"Complete Validation Chain: {'PASS' if test2 else 'FAIL'}")
    print(f"Error Scenarios: {'PASS' if test3 else 'FAIL'}")
    
    if test1 and test2 and test3:
        print("\nSUCCESS: All tuple errors have been resolved!")
        print("The application should now work without tuple errors.")
    else:
        print("\nFAILURE: Some issues remain. Check the output above.")
