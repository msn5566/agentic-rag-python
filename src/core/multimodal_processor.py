"""
Multimodal Document Processor for handling images, tables, and text content.
Integrates with Google Gemini Vision API for comprehensive document understanding.
"""

import base64
import io
import logging
import os
import re
import uuid
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import fitz  # PyMuPDF for PDF processing
import pandas as pd
from PIL import Image
import google.generativeai as genai
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI

from ..config.settings import settings

logger = logging.getLogger(__name__)


class MultimodalDocumentProcessor:
    """Enhanced document processor that handles text, images, and tables."""
    
    def __init__(self):
        """Initialize the multimodal processor with Gemini Vision."""
        self.vision_model = genai.GenerativeModel('gemini-1.5-flash')
        self.text_model = ChatGoogleGenerativeAI(
            model=settings.google_llm_model,
            google_api_key=settings.google_api_key
        )
        
        # Configure Gemini API
        genai.configure(api_key=settings.google_api_key)
        
        # Create directories for storing extracted images
        self.images_dir = Path("storage/extracted_images")
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
    def process_document(self, file_path: str) -> List[Document]:
        """
        Process a document and extract text, images, and tables.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects with multimodal content
        """
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.pdf':
            return self._process_pdf(file_path)
        elif file_path.suffix.lower() in ['.docx', '.doc']:
            return self._process_docx(file_path)
        elif file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
            return self._process_image(file_path)
        else:
            # Fallback to text processing
            return self._process_text_file(file_path)
    
    def _process_pdf(self, file_path: Path) -> List[Document]:
        """Process PDF with text, image, and table extraction."""
        documents = []
        
        try:
            # Open PDF with PyMuPDF
            pdf_document = fitz.open(file_path)
            
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                
                # Extract text content
                text_content = page.get_text()
                
                # Extract images and vector graphics from the page
                image_descriptions = self._extract_and_describe_images(page, page_num)
                vector_descriptions = self._extract_and_describe_vector_graphics(page, page_num)
                
                # Extract and process tables
                table_content = self._extract_tables_from_page(page, page_num)
                
                # Combine all content
                combined_content = self._combine_multimodal_content(
                    text_content, image_descriptions + vector_descriptions, table_content
                )
                
                if combined_content.strip():
                    doc = Document(
                        page_content=combined_content,
                        metadata={
                            "source": str(file_path),
                            "page": page_num + 1,
                            "content_type": "multimodal",
                            "has_images": len(image_descriptions) > 0,
                            "has_vector_graphics": len(vector_descriptions) > 0,
                            "has_tables": len(table_content) > 0
                        }
                    )
                    documents.append(doc)

                # Create separate image index documents with semantic context
                if image_descriptions:
                    img_index_docs = self._create_image_index_docs(
                        image_descriptions,
                        text_content,
                        file_path,
                        page_num
                    )
                    documents.extend(img_index_docs)
            
            pdf_document.close()
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            # Fallback to basic text extraction
            return self._fallback_text_extraction(file_path)
        
        return documents
    
    def _extract_and_describe_images(self, page, page_num: int) -> List[str]:
        """Extract images from PDF page and generate descriptions using Gemini Vision."""
        image_descriptions = []
        
        try:
            # Get images from the page
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    # Extract image data
                    xref = img[0]
                    pix = fitz.Pixmap(page.parent, xref)
                    
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        # Convert to PIL Image
                        img_data = pix.tobytes("png")
                        pil_image = Image.open(io.BytesIO(img_data))
                        
                        # Save the image for serving
                        image_filename = self._save_extracted_image(pil_image, page_num, img_index, "bitmap")
                        
                        # Generate description using Gemini Vision
                        description = self._describe_image_with_gemini(pil_image, page_num, img_index)
                        
                        if description:
                            image_descriptions.append(
                                f"[IMAGE {img_index + 1} on page {page_num + 1}]: {description}\n[IMAGE_FILE: {image_filename}]"
                            )
                    
                    pix = None  # Free memory
                    
                except Exception as e:
                    logger.warning(f"Error processing image {img_index} on page {page_num}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error extracting images from page {page_num}: {e}")
        
        return image_descriptions
    
    def _extract_and_describe_vector_graphics(self, page, page_num: int) -> List[str]:
        """Extract vector graphics/drawings from PDF page and render them as images for description."""
        vector_descriptions = []
        
        try:
            # Get drawings/vector graphics from the page
            drawings = page.get_drawings()
            
            if len(drawings) > 5:  # Only process pages with significant vector content
                try:
                    # Render the page as an image to capture vector graphics
                    # Use higher resolution for better quality
                    mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
                    pix = page.get_pixmap(matrix=mat)
                    
                    # Convert to PIL Image
                    img_data = pix.tobytes("png")
                    pil_image = Image.open(io.BytesIO(img_data))
                    
                    # Save the vector graphics image for serving
                    vector_filename = self._save_extracted_image(pil_image, page_num, 0, "vector")
                    
                    # Generate description using Gemini Vision with focus on diagrams
                    description = self._describe_vector_graphics_with_gemini(pil_image, page_num, len(drawings))
                    
                    if description:
                        vector_descriptions.append(
                            f"[VECTOR GRAPHICS on page {page_num + 1}]: {description}\n[IMAGE_FILE: {vector_filename}]"
                        )
                    
                    pix = None  # Free memory
                    
                except Exception as e:
                    logger.warning(f"Error processing vector graphics on page {page_num}: {e}")
                    
        except Exception as e:
            logger.error(f"Error extracting vector graphics from page {page_num}: {e}")
        
        return vector_descriptions
    
    def _describe_vector_graphics_with_gemini(self, image: Image.Image, page_num: int, drawing_count: int) -> str:
        """Use Gemini Vision to describe vector graphics with focus on diagrams and flowcharts."""
        try:
            # Convert PIL Image to base64 for Gemini
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Create specialized prompt for vector graphics/diagrams
            prompt = f"""
            This page contains {drawing_count} vector graphics elements. Analyze this page image and provide a comprehensive description focusing on:
            
            1. DIAGRAMS AND FLOWCHARTS: Identify any flowcharts, process diagrams, system diagrams, or organizational charts
            2. VISUAL ELEMENTS: Describe shapes, arrows, connections, boxes, and their relationships
            3. TEXT IN GRAPHICS: Extract any text visible within diagrams, flowcharts, or graphic elements
            4. STRUCTURE AND FLOW: Explain the logical flow or structure shown in any diagrams
            5. TECHNICAL CONTENT: If it's a technical diagram, explain the process or system being illustrated
            
            Pay special attention to:
            - Flowchart symbols and their meanings
            - Process steps and decision points
            - System architecture or workflow representations
            - Any labels, annotations, or captions within graphics
            
            Provide a detailed description that would help someone understand the visual content and find it through search queries about flowcharts, diagrams, or processes.
            """
            
            # Generate description using Gemini Vision
            response = self.vision_model.generate_content([
                prompt,
                {"mime_type": "image/png", "data": img_base64}
            ])
            
            if response and response.text:
                description = response.text.strip()
                
                # Ensure no base64 data leaks into the description
                if 'data:image' in description or 'base64' in description.lower():
                    logger.warning(f"Removing base64 data from vector graphics description on page {page_num + 1}")
                    # Clean the description by removing base64 patterns
                    import re
                    description = re.sub(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+', '[IMAGE_DATA_REMOVED]', description)
                    description = re.sub(r'base64[,:]?[A-Za-z0-9+/=]+', '[BASE64_DATA_REMOVED]', description, flags=re.IGNORECASE)
                
                return description
            
        except Exception as e:
            logger.error(f"Error describing vector graphics with Gemini: {e}")
        
        return ""
    
    def _save_extracted_image(self, image: Image.Image, page_num: int, img_index: int, image_type: str) -> str:
        """Save extracted image to storage and return filename."""
        try:
            # Generate unique filename
            image_id = str(uuid.uuid4())[:8]
            filename = f"{image_type}_page{page_num + 1}_img{img_index + 1}_{image_id}.png"
            filepath = self.images_dir / filename
            
            # Save image
            image.save(filepath, "PNG")
            logger.info(f"Saved {image_type} image: {filename}")
            
            return filename
            
        except Exception as e:
            logger.error(f"Error saving {image_type} image: {e}")
            return ""
    
    def _describe_image_with_gemini(self, image: Image.Image, page_num: int, img_index: int) -> str:
        """Use Gemini Vision to describe an image."""
        try:
            # Convert PIL Image to base64 for Gemini API call ONLY
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Create prompt for image description - NEVER include base64 in response
            prompt = """
            Analyze this image and provide a comprehensive description that includes:
            1. What the image shows (objects, people, scenes, diagrams, charts, etc.)
            2. Any text visible in the image (OCR)
            3. Important details that would be relevant for document search
            4. If it's a chart/graph/diagram, explain the data and relationships shown
            
            Provide ONLY a clear, detailed text description. Do NOT include any base64 data, image data, or file references in your response.
            """
            
            # Generate description using Gemini Vision
            response = self.vision_model.generate_content([
                prompt,
                {"mime_type": "image/png", "data": img_base64}
            ])
            
            # Ensure no base64 data leaks into the description
            description = response.text.strip() if response.text else ""
            
            # Remove any potential base64 data that might have leaked
            if 'data:image' in description or 'base64' in description.lower():
                logger.warning(f"Removing base64 data from image description for image {img_index + 1}")
                # Clean the description by removing base64 patterns
                import re
                description = re.sub(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+', '[IMAGE_DATA_REMOVED]', description)
                description = re.sub(r'base64[,:]?[A-Za-z0-9+/=]+', '[BASE64_DATA_REMOVED]', description, flags=re.IGNORECASE)
            
            return description
            
        except Exception as e:
            logger.error(f"Error describing image with Gemini: {e}")
            return f"Image {img_index + 1} (description unavailable)"
    
    def _extract_tables_from_page(self, page, page_num: int) -> List[str]:
        """Extract and process tables from PDF page."""
        table_content = []
        
        try:
            # Find tables using PyMuPDF
            tables = page.find_tables()
            
            for table_index, table in enumerate(tables):
                try:
                    # Extract table data
                    table_data = table.extract()
                    
                    if table_data:
                        # Convert to pandas DataFrame for better processing
                        raw_cols = table_data[0]
                        cols = [str(c) if c is not None else f"col_{i+1}" for i, c in enumerate(raw_cols)]
                        df = pd.DataFrame(table_data[1:], columns=cols)
                        
                        # Generate table description using Gemini
                        table_description = self._describe_table_with_gemini(df, page_num, table_index)
                        
                        # Format table content
                        table_text = f"[TABLE {table_index + 1} on page {page_num + 1}]:\n"
                        table_text += f"Description: {table_description}\n"
                        table_text += f"Data:\n{df.to_string(index=False)}\n"
                        
                        table_content.append(table_text)
                        
                except Exception as e:
                    logger.warning(f"Error processing table {table_index} on page {page_num}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error extracting tables from page {page_num}: {e}")
        
        return table_content
    
    def _describe_table_with_gemini(self, df: pd.DataFrame, page_num: int, table_index: int) -> str:
        """Use Gemini to analyze and describe table content."""
        try:
            # Create a summary of the table for Gemini
            table_summary = f"Table with {len(df)} rows and {len(df.columns)} columns.\n"
            columns_text = ', '.join([str(c) if c is not None else '' for c in list(df.columns)])
            table_summary += f"Columns: {columns_text}\n"
            table_summary += f"Sample data:\n{df.head(3).fillna('').to_string(index=False)}"
            
            prompt = f"""
            Analyze this table data and provide a comprehensive description:
            
            {table_summary}
            
            Please describe:
            1. What type of data this table contains
            2. Key patterns or insights from the data
            3. The purpose or context of this table
            4. Any notable values or trends
            
            Provide a clear summary that would help someone understand the table's content and significance.
            """
            
            response = self.text_model.invoke(prompt)
            if hasattr(response, 'content') and response.content:
                return response.content.strip()
            elif response:
                return str(response).strip()
            else:
                return f"Table {table_index + 1} with {len(df)} rows and {len(df.columns)} columns (description unavailable)"
            
        except Exception as e:
            logger.error(f"Error describing table with Gemini: {e}")
            return f"Table {table_index + 1} with {len(df)} rows and {len(df.columns)} columns"
    
    def _sanitize_content(self, content: str) -> str:
        """Remove any base64 data or image data that might have leaked into content."""
        if not content:
            return content
            
        # Remove base64 image data patterns
        content = re.sub(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+', '[IMAGE_DATA_REMOVED]', content)
        content = re.sub(r'base64[,:]?\s*[A-Za-z0-9+/=]{20,}', '[BASE64_DATA_REMOVED]', content, flags=re.IGNORECASE)
        
        # Remove long strings that look like base64 data (20+ chars of base64 pattern)
        content = re.sub(r'[A-Za-z0-9+/=]{50,}', '[ENCODED_DATA_REMOVED]', content)
        
        return content
    
    def _combine_multimodal_content(self, text: str, images: List[str], tables: List[str]) -> str:
        """Combine text, image descriptions, and table content into a unified document."""
        combined_content = []
        
        # Add main text content (sanitized)
        if text and text.strip():
            combined_content.append("=== TEXT CONTENT ===")
            combined_content.append(self._sanitize_content(text.strip()))
        
        # Add image descriptions (filter out None values and sanitize)
        valid_images = [self._sanitize_content(str(img)) for img in images if img is not None and str(img).strip()]
        if valid_images:
            combined_content.append("\n=== IMAGES ===")
            combined_content.extend(valid_images)
        
        # Add table content (filter out None values and sanitize)
        valid_tables = [self._sanitize_content(str(table)) for table in tables if table is not None and str(table).strip()]
        if valid_tables:
            combined_content.append("\n=== TABLES ===")
            combined_content.extend(valid_tables)
        
        # Filter out any None values from combined_content before joining
        filtered_content = [item for item in combined_content if item is not None]
        final_content = "\n\n".join(filtered_content)
        
        # Final sanitization pass
        return self._sanitize_content(final_content)

    def _extract_context_snippet(self, text: str, max_chars: int = 600) -> str:
        """Extract a short context snippet, heuristically taking the first heading/paragraph."""
        if not text:
            return ""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return text[:max_chars]
        # Heuristic: use first 2-3 non-empty lines as heading/lead paragraph
        snippet = " ".join(lines[:3])
        if len(snippet) < max_chars:
            return snippet[:max_chars]
        return snippet[:max_chars]

    def _create_image_index_docs(self, image_descriptions: List[str], page_text: str, file_path: Path, page_num: int) -> List[Document]:
        """Create per-image Documents for image vector indexing, including semantic context and filename."""
        docs: List[Document] = []
        context = self._extract_context_snippet(page_text)
        for desc in image_descriptions:
            try:
                if not desc:
                    continue
                # Expect pattern like: "[IMAGE X on page Y]: <description>\n[IMAGE_FILE: filename]"
                description_text = desc
                filename = ""
                m = re.search(r"\[IMAGE_FILE:\s*([^\]]+)\]", desc)
                if m:
                    filename = m.group(1).strip()
                    # Remove the IMAGE_FILE tag from description text
                    description_text = re.sub(r"\n?\[IMAGE_FILE:[^\]]+\]", "", description_text).strip()

                content = f"Image Description: {description_text}\nContext: {context}"
                md = {
                    "source": str(file_path),
                    "page": page_num + 1,
                    "content_type": "image_index",
                    "image_file": filename or ""
                }
                docs.append(Document(page_content=content, metadata=md))
            except Exception as e:
                logger.warning(f"Failed to create image index doc on page {page_num + 1}: {e}")
                continue
        return docs
    
    def _process_docx(self, file_path: Path) -> List[Document]:
        """Process DOCX files with image and table extraction."""
        # For now, use basic text extraction
        # TODO: Implement full DOCX multimodal processing
        return self._fallback_text_extraction(file_path)
    
    def _process_image(self, file_path: Path) -> List[Document]:
        """Process standalone image files."""
        try:
            image = Image.open(file_path)
            description = self._describe_image_with_gemini(image, 0, 0)
            
            # Create an image index document for the separate image vector store
            img_index_doc = Document(
                page_content=f"IMAGE: {description}",
                metadata={
                    "source": str(file_path),
                    "page": 1,
                    "content_type": "image_index",
                    "image_file": os.path.basename(str(file_path)),
                    "has_images": True,
                    "has_tables": False
                }
            )
            
            # Also keep a simple analysis doc for the combined text index if needed
            combined_doc = Document(
                page_content=f"=== IMAGE ANALYSIS ===\n{description}",
                metadata={
                    "source": str(file_path),
                    "content_type": "multimodal",
                    "has_images": True,
                    "has_tables": False
                }
            )
            
            return [combined_doc, img_index_doc]
            
        except Exception as e:
            logger.error(f"Error processing image {file_path}: {e}")
            return []
    
    def _process_text_file(self, file_path: Path) -> List[Document]:
        """Process plain text files."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            doc = Document(
                page_content=content,
                metadata={
                    "source": str(file_path),
                    "content_type": "text",
                    "has_images": False,
                    "has_tables": False
                }
            )
            
            return [doc]
            
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {e}")
            return []
    
    def _fallback_text_extraction(self, file_path: Path) -> List[Document]:
        """Fallback to basic text extraction if multimodal processing fails."""
        try:
            if file_path.suffix.lower() == '.pdf':
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(str(file_path))
                return loader.load()
            elif file_path.suffix.lower() in ['.docx', '.doc']:
                from langchain_community.document_loaders import Docx2txtLoader
                loader = Docx2txtLoader(str(file_path))
                return loader.load()
            else:
                return self._process_text_file(file_path)
                
        except Exception as e:
            logger.error(f"Fallback text extraction failed for {file_path}: {e}")
            return []


# Global instance
multimodal_processor = MultimodalDocumentProcessor()
