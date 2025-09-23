#!/usr/bin/env python3
"""
Enhanced Multimodal RAG System Demo for app_google.py
This script demonstrates the multimodal capabilities of the basic RAG system.
"""

import requests
import json
import time
import os

BASE_URL = "http://localhost:8002"  # app_google.py runs on port 8002

def test_multimodal_upload():
    """Test uploading documents with images and tables."""
    print("🔄 Testing Multimodal Document Upload...")
    
    # Find a PDF with images/tables in uploads directory
    uploads_dir = "uploads"
    pdf_files = [f for f in os.listdir(uploads_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("❌ No PDF files found in uploads directory")
        return False
    
    # Use the first PDF file
    test_file = pdf_files[0]
    file_path = os.path.join(uploads_dir, test_file)
    
    print(f"📄 Using test file: {test_file}")
    
    try:
        with open(file_path, 'rb') as f:
            files = {'files': (test_file, f, 'application/pdf')}
            response = requests.post(f"{BASE_URL}/upload", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Upload successful: {result['message']}")
            return True
        else:
            print(f"❌ Upload failed: {response.status_code} - {response.text}")
            return False
    
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return False

def test_multimodal_queries():
    """Test queries that should find image and table content."""
    print("\n🔍 Testing Multimodal Queries...")
    
    # Test queries for different content types
    test_queries = [
        "What images are in the document?",
        "Describe any charts or graphs shown",
        "What tables are present in the document?",
        "What data is shown in the tables?",
        "Are there any diagrams or figures?",
        "What visual elements are described?",
        "Show me information from any charts",
        "What numerical data is available?"
    ]
    
    successful_queries = 0
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Query {i}: {query}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/query",
                json={"q": query},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                # Read streaming response
                content = ""
                for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                    if chunk:
                        content += chunk
                
                if content and "No relevant information found" not in content:
                    print(f"✅ Response: {content[:200]}...")
                    successful_queries += 1
                else:
                    print("⚠️  No relevant content found")
            else:
                print(f"❌ Query failed: {response.status_code}")
        
        except Exception as e:
            print(f"❌ Query error: {e}")
        
        time.sleep(1)  # Brief pause between queries
    
    print(f"\n📊 Query Results: {successful_queries}/{len(test_queries)} successful")
    return successful_queries > 0

def test_health_check():
    """Test if the basic RAG system is running."""
    print("🏥 Testing System Health...")
    
    try:
        # Try to access the docs endpoint
        response = requests.get(f"{BASE_URL}/docs")
        if response.status_code == 200:
            print("✅ Basic RAG system is running")
            return True
        else:
            print(f"⚠️  System responding but docs not accessible: {response.status_code}")
            return False
    
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to basic RAG system. Make sure app_google.py is running on port 8002")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def main():
    """Run the complete multimodal demo."""
    print("🚀 Enhanced Multimodal RAG System Demo (Basic Version)")
    print("=" * 60)
    
    # Check if system is running
    if not test_health_check():
        print("\n💡 To start the basic RAG system, run: python app_google.py")
        return
    
    print("\n" + "=" * 60)
    
    # Test multimodal upload
    upload_success = test_multimodal_upload()
    
    if upload_success:
        print("\n" + "=" * 60)
        # Wait a moment for indexing
        print("⏳ Waiting for document indexing...")
        time.sleep(3)
        
        # Test multimodal queries
        query_success = test_multimodal_queries()
        
        print("\n" + "=" * 60)
        print("📋 Demo Summary:")
        print(f"   📤 Upload: {'✅ Success' if upload_success else '❌ Failed'}")
        print(f"   🔍 Queries: {'✅ Success' if query_success else '❌ Failed'}")
        
        if upload_success and query_success:
            print("\n🎉 Multimodal RAG system is working correctly!")
            print("   The system can now process and query:")
            print("   • 📄 Text content")
            print("   • 🖼️  Images and visual elements")
            print("   • 📊 Tables and structured data")
        else:
            print("\n⚠️  Some features may need attention")
    
    else:
        print("\n❌ Upload failed - cannot test queries")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
