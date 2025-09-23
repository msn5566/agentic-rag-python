#!/usr/bin/env python3
"""
Debug script to test multimodal processing on IJRAR1ARP035.pdf
This will help us understand why image content isn't being properly extracted and indexed.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

# Configure logging to see detailed output
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_multimodal_processing():
    """Test multimodal processing on the specific PDF file."""
    
    # Set up environment
    os.environ['GOOGLE_API_KEY'] = 'AIzaSyCBtZrShtXHQZVCyjg5YbNXe69wz4wmCx4'  # From config
    
    try:
        # Import and configure
        import google.generativeai as genai
        genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
        
        # Create settings compatibility for multimodal processor
        from types import ModuleType
        settings_module = ModuleType('settings')
        
        class SimpleSettings:
            def __init__(self):
                self.google_api_key = os.environ['GOOGLE_API_KEY']
                self.google_llm_model = "gemini-1.5-flash-latest"
        
        settings_module.settings = SimpleSettings()
        sys.modules['src.config.settings'] = settings_module
        
        # Import multimodal processor
        from src.core.multimodal_processor import MultimodalDocumentProcessor
        
        # Initialize processor
        processor = MultimodalDocumentProcessor()
        
        # Test file path
        test_file = "uploads/IJRAR1ARP035.pdf"
        
        if not os.path.exists(test_file):
            print(f"❌ Test file not found: {test_file}")
            return False
        
        print(f"Testing multimodal processing on: {test_file}")
        print("=" * 60)
        
        # Process the document
        documents = processor.process_document(test_file)
        
        print(f"Total documents extracted: {len(documents)}")
        print("=" * 60)
        
        # Analyze each document
        image_count = 0
        table_count = 0
        
        for i, doc in enumerate(documents):
            print(f"\n📝 Document {i+1}:")
            print(f"   Page: {doc.metadata.get('page', 'Unknown')}")
            print(f"   Content Type: {doc.metadata.get('content_type', 'Unknown')}")
            print(f"   Has Images: {doc.metadata.get('has_images', False)}")
            print(f"   Has Tables: {doc.metadata.get('has_tables', False)}")
            print(f"   Content Length: {len(doc.page_content)} characters")
            
            if doc.metadata.get('has_images'):
                image_count += 1
                
            if doc.metadata.get('has_tables'):
                table_count += 1
            
            # Show first 300 characters of content
            content_preview = doc.page_content[:300]
            print(f"   Content Preview: {content_preview}...")
            
            # Look for image descriptions in content
            if "[IMAGE" in doc.page_content:
                print("   ✅ Contains image descriptions!")
                # Extract and show image descriptions
                lines = doc.page_content.split('\n')
                for line in lines:
                    if "[IMAGE" in line:
                        print(f"      🖼️  {line.strip()}")
            else:
                print("   ❌ No image descriptions found")
            
            print("-" * 40)
        
        print(f"\n📊 Summary:")
        print(f"   Total Documents: {len(documents)}")
        print(f"   Documents with Images: {image_count}")
        print(f"   Documents with Tables: {table_count}")
        
        # Test specific image-related queries
        print(f"\n🔍 Testing image content search:")
        
        # Look for flowchart-related content
        flowchart_docs = []
        for doc in documents:
            if any(keyword in doc.page_content.lower() for keyword in ['flowchart', 'diagram', 'figure', 'chart', 'image']):
                flowchart_docs.append(doc)
        
        print(f"   Documents mentioning flowchart/diagram/figure/chart/image: {len(flowchart_docs)}")
        
        for i, doc in enumerate(flowchart_docs):
            print(f"   📋 Relevant Document {i+1} (Page {doc.metadata.get('page')}):")
            # Show lines containing image-related keywords
            lines = doc.page_content.split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in ['flowchart', 'diagram', 'figure', 'chart', 'image']):
                    print(f"      → {line.strip()}")
        
        if len(flowchart_docs) == 0:
            print("   ❌ No documents found with flowchart/diagram content!")
            print("   🔧 This suggests the image extraction isn't working properly.")
        
        return len(documents) > 0 and image_count > 0
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pdf_images_directly():
    """Test direct PDF image extraction using PyMuPDF."""
    print(f"\n🔧 Testing direct PDF image extraction...")
    
    try:
        import fitz  # PyMuPDF
        
        test_file = "uploads/IJRAR1ARP035.pdf"
        
        if not os.path.exists(test_file):
            print(f"❌ Test file not found: {test_file}")
            return False
        
        # Open PDF
        pdf_document = fitz.open(test_file)
        
        print(f"📄 PDF has {len(pdf_document)} pages")
        
        total_images = 0
        
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            image_list = page.get_images()
            
            print(f"   Page {page_num + 1}: {len(image_list)} images found")
            total_images += len(image_list)
            
            # Show image details
            for i, img in enumerate(image_list):
                print(f"      Image {i+1}: xref={img[0]}, width={img[2]}, height={img[3]}")
        
        pdf_document.close()
        
        print(f"📊 Total images in PDF: {total_images}")
        
        if total_images == 0:
            print("❌ No images found in PDF - this might be why image content isn't being extracted!")
        else:
            print("✅ Images found in PDF - the issue might be in the description generation")
        
        return total_images > 0
        
    except Exception as e:
        print(f"❌ Error during direct PDF testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Multimodal Processing Debug Test")
    print("=" * 60)
    
    # Test 1: Direct PDF image extraction
    has_images = test_pdf_images_directly()
    
    # Test 2: Full multimodal processing
    processing_works = test_multimodal_processing()
    
    print("\n" + "=" * 60)
    print("🎯 Debug Results:")
    print(f"   PDF contains images: {'✅ Yes' if has_images else '❌ No'}")
    print(f"   Multimodal processing works: {'✅ Yes' if processing_works else '❌ No'}")
    
    if has_images and not processing_works:
        print("\n💡 Diagnosis: Images exist but multimodal processing failed")
        print("   Possible issues:")
        print("   - Gemini Vision API configuration problem")
        print("   - Image format compatibility issue")
        print("   - Memory or processing error during description generation")
    elif not has_images:
        print("\n💡 Diagnosis: No images found in PDF")
        print("   Possible issues:")
        print("   - PDF images might be embedded as vector graphics (not extractable)")
        print("   - Images might be part of the page content (not separate objects)")
        print("   - PDF might use a format that PyMuPDF can't extract")
    elif processing_works:
        print("\n✅ Diagnosis: Everything appears to be working correctly!")
        print("   The issue might be in the query/retrieval process")
    
    print("\n🔧 Next steps:")
    if not processing_works:
        print("   1. Fix multimodal processing issues")
        print("   2. Check Gemini Vision API configuration")
        print("   3. Test with a different PDF file")
    else:
        print("   1. Check if processed content is being indexed correctly")
        print("   2. Test query retrieval system")
        print("   3. Verify vector store contains image descriptions")
