# Technical Design Document: Secure RAG System with Guardrails AI Integration

## Document Information
- **Document Version**: 1.0
- **Created Date**: 2025-01-19
- **Last Updated**: 2025-01-19
- **Author**: AI Development Team
- **Project**: Secure Retrieval-Augmented Generation (RAG) System
- **Status**: Production Ready

---

## 1. Executive Summary

### 1.1 Project Overview
This document outlines the technical design for a comprehensive Secure RAG (Retrieval-Augmented Generation) system that combines advanced AI capabilities with enterprise-grade security measures. The system provides intelligent document processing, multimodal content analysis, and secure query processing while maintaining robust protection against AI-specific threats.

### 1.2 Key Features
- **Multimodal System**: Handles text, images, and complex documents with separate image indexing and retrieval.
- **Dynamic Image Attachments**: Attaches Base64-encoded images in responses based on query keywords.
- **Adaptive Retrieval**: Uses fallback thresholds to improve relevance and reduce "no content found" errors.
- **Configurable History Selection**: Filters chat history to include only relevant turns.
- **Enhanced Security**: Context-aware validation with Guardrails AI for safe multimodal processing.

### 1.3 Business Value
- **Enhanced Security**: Protection against prompt injection, data leakage, and malicious content
- **Improved User Experience**: Clean, accurate responses without false positive security blocks
- **Scalable Architecture**: Production-ready system with monitoring and logging
- **Compliance Ready**: Built-in PII detection, content filtering, and audit trails

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Client Layer                               │
├─────────────────────────────────────────────────────────────────────┤
│                       API Gateway Layer                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐      │
│  │   Standard App  │  │   Secure App    │  │   Multimodal    │      │
│  │  (app_google)   │  │  (secure_app)   │  │   Processor     │      │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘      │
├─────────────────────────────────────────────────────────────────────┤
│                       Security Layer                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐      │
│  │  Guardrails AI  │  │  Authentication │  │  Rate Limiting  │      │
│  │   Validation    │  │   & Authorization│  │   & Monitoring  │      │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘      │
├─────────────────────────────────────────────────────────────────────┤
│                      Processing Layer                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐      │
│  │   Document      │  │   Vector        │  │   Image         │      │
│  │   Loader        │  │   Embeddings    │  │   Retrieval     │      │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘      │
├─────────────────────────────────────────────────────────────────────┤
│                        Storage Layer                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐      │
│  │   ChromaDB      │  │   Document      │  │   Metadata      │      │
│  │  Text Store     │  │   Storage       │  │   & Images      │      │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Architecture

#### 2.2.1 Application Layer
- **Main App (`app_google.py`)**: RAG functionality with basic Guardrails AI integration
- **RAG Utilities (`rag_google.py`)**: Core document processing and retrieval functions

#### 2.2.2 Security Layer (Current Implementation)
- **Guardrails AI Framework**: Input/output validation, PII detection, toxic language filtering
- **Basic CORS**: Allow all origins for development
- **File Validation**: Basic file type and content validation
- **Error Handling**: Safe error message sanitization

#### 2.2.3 Processing Layer
- **Multimodal Processor**: Text, image, and document analysis
- **Document Loaders**: PDF, DOCX, and various format support
- **Vector Embeddings**: Google Generative AI embeddings
- **LLM Integration**: Google Gemini model integration

#### 2.2.5 Multimodal and Image Processing Layer
- **Multimodal Processor (`src/core/multimodal_processor.py`)**: Extracts and describes images, tables, and text from PDFs and other documents using Google Gemini Vision.
- **Separate Image Vector Store**: Dedicated ChromaDB collection for image descriptions with semantic context.
- **Dynamic Image Attachments**: Attaches Base64-encoded images to responses if the query contains keywords like "image", "screenshot", "picture", or "pic".
- **Adaptive Retrieval**: Uses configurable thresholds for both text and image retrieval to ensure relevant results.

---

## 3. Security Architecture

### 3.1 Guardrails AI Integration

#### 3.1.1 Input Validation
```python
# Context-aware validation for educational content
educational_indicators = [
    "programming", "machine learning", "algorithm", "computer science",
    "software", "technology", "artificial intelligence", "data science"
]

# Validation pipeline
def validate_user_input(input_text):
    - PII Detection and Masking
    - Toxic Language Filtering (context-aware)
    - Topic Restriction Checks
    - Length and Format Validation
```

#### 3.1.2 Output Validation
```python
# Streaming-friendly validation
def validate_ai_response(response_chunk):
    - Content Safety Checks
    - PII Redaction
    - Quality Assessment (optimized for chunks)
    - Educational Content Allowance
```

#### 3.1.3 Search Results Validation
```python
# Comprehensive result filtering
def validate_search_results(results):
    - Individual result validation
    - Content safety filtering
    - Relevance scoring
    - Safe error handling
```

### 3.2 Current Security Features Matrix

| Feature | Current Status | Implementation |
|---------|----------------|----------------|
| Guardrails AI | ✅ | Input/output validation, PII detection |
| CORS | ✅ | Allow all origins (development mode) |
| File Validation | ✅ | Basic file type and content checks |
| Error Sanitization | ✅ | Safe error message handling |
| Auto-Indexing | ✅ | Background document processing |
| Multimodal Processing | ✅ | Text, images, and tables extraction |
| Chat History | ✅ | Conversation context management |
| Adaptive Retrieval | ✅ | Fallback thresholds for better results |
| Image Attachments | ✅ | Dynamic image inclusion in responses |

---

## 4. API Design

### 4.1 Standard App Endpoints

#### 4.1.2 Enhanced Endpoints
```
GET /images/{filename}
- Description: Serve extracted images with security
- Security: Path traversal protection
- Rate Limit: 30/minute
- Authentication: None (public for extracted images)

POST /upload
- Features:
  - Multimodal processing (text, images, tables)
  - Separate image indexing
  - Base64 image extraction
  - Enhanced document chunking
```

### 4.2 Current API Endpoints

#### 4.2.1 Document Operations
```
POST /upload
- Description: Upload and process documents
- Features:
  - Multimodal processing (text, images, tables)
  - Guardrails file validation
  - Content safety checks
  - Auto-indexing with background scheduler

GET /images/{filename}
- Description: Serve extracted images
- Security: Basic path validation
- Public access for extracted images
```

#### 4.2.2 Query Processing
```
POST /query
- Description: RAG query processing
- Features:
  - Guardrails input validation
  - Adaptive retrieval with fallback thresholds
  - Chat history integration
  - Streaming response validation
  - Dynamic image attachments
  - Similar query generation via LLM
```

#### 4.2.3 Document Management
```
DELETE /documents
- Description: Delete specific documents
- Features:
  - Remove from vector store
  - Update indexing state
  - Batch deletion support
```

---

## 5. Data Flow Architecture

### 5.1 Document Upload Flow

```
1. File Upload Request
   ↓
2. Authentication Check (Secure App)
   ↓
3. Rate Limit Validation
   ↓
4. File Format Validation
   ↓
5. Guardrails Content Safety Check
   ↓
6. Multimodal Processing
   ↓
7. Document Chunking
   ↓
8. Vector Embedding Generation
   ↓
9. ChromaDB Storage
   ↓
10. Metadata Storage
    ↓
11. Response Generation
```

### 5.2 Query Processing Flow

```
1. Query Request
   ↓
2. Authentication Check (Secure App)
   ↓
3. Rate Limit Validation
   ↓
4. Guardrails Input Validation
   ↓
5. Vector Similarity Search
   ↓
6. Search Results Validation
   ↓
7. Context Assembly
   ↓
8. LLM Response Generation
   ↓
9. Streaming Response Validation
   ↓
10. Output Sanitization
    ↓
### 5.4 Query Processing with Image Attachments
```
1. Query Request
   ↓
2. Keyword Check for Image Attachments
   ↓
3. Text Retrieval (Adaptive Thresholds)
   ↓
4. Image Retrieval (Separate Vector Store)
   ↓
5. Base64 Encoding of Matched Images
   ↓
6. LLM Response Generation
   ↓
7. Streaming with Image Attachments
   ↓
8. Validation and Sanitization
```
1. Document Upload
   ↓
2. Multimodal Extraction (Text, Images, Tables)
   ↓
3. Image Description Generation (Gemini Vision)
   ↓
4. Context Snippet Creation
   ↓
5. Separate Image Index Creation
   ↓
6. Text Chunking and Embedding
   ↓
7. Dual Vector Store Storage
   ↓
8. Metadata and Image File Storage
```

### 5.5 Complete System Workflow Diagram

The following comprehensive workflow diagram illustrates the complete end-to-end system workflow, including both document upload and query processing flows with all security layers and multimodal processing.

```
                    ┌─────────────────────────────────────────────────────────────┐
                    │                    RAG SYSTEM WORKFLOW                      │
                    └─────────────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
                    ┌─────────────────────────────────────────────────────────────┐
                    │                      USER INTERACTION                       │
                    │  ┌─────────────────┐              ┌─────────────────┐       │
                    │  │  Document Upload│              │   Query Request │       │
                    │  │   (POST /upload)│              │  (POST /query)  │       │
                    │  └─────────────────┘              └─────────────────┘       │
                    └─────────────────────────────────────────────────────────────┘
                                    │                              │
                                    ▼                              ▼
                    ┌─────────────────────────────────────────────────────────────┐
                    │                    SECURITY LAYER                           │
                    │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
                    │  │ Basic CORS      │  │ Error Handling  │  │ File Type   │  │
                    │  │ (Allow All)     │  │ Sanitization    │  │ Validation  │  │
                    │  └─────────────────┘  └─────────────────┘  └─────────────┘  │
                    └─────────────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
                    ┌─────────────────────────────────────────────────────────────┐
                    │                  GUARDRAILS AI VALIDATION                   │
                    │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
                    │  │   PII Detection │  │ Toxic Language  │  │   Content   │  │
                    │  │   & Masking     │  │   Detection     │  │   Safety    │  │
                    │  └─────────────────┘  └─────────────────┘  └─────────────┘  │
                    └─────────────────────────────────────────────────────────────┘
                                    │                              │
                                    ▼                              ▼
            ┌─────────────────────────────────┐      ┌─────────────────────────────────┐
            │        DOCUMENT PROCESSING       │      │        QUERY PROCESSING         │
            │  ┌─────────────────────────────┐ │      │  ┌─────────────────────────────┐ │
            │  │    Multimodal Extraction    │ │      │  │      Vector Search          │ │
            │  │  ┌─────────┐ ┌─────────────┐│ │      │  │  ┌─────────┐ ┌─────────────┐│ │
            │  │  │  Text   │ │   Images    ││ │      │  │  │  Text   │ │   Images    ││ │
            │  │  │Extract  │ │  & Tables   ││ │      │  │  │Retrieval│ │  Retrieval  ││ │
            │  │  └─────────┘ └─────────────┘│ │      │  │  └─────────┘ └─────────────┘│ │
            │  └─────────────────────────────┘ │      │  └─────────────────────────────┘ │
            │                │                 │      │                │                 │
            │                ▼                 │      │                ▼                 │
            │  ┌─────────────────────────────┐ │      │  ┌─────────────────────────────┐ │
            │  │       Document Chunking     │ │      │  │     Keyword Detection       │ │
            │  │    (Text Splitter)          │ │      │  │   (Image Attachment?)       │ │
            │  └─────────────────────────────┘ │      │  └─────────────────────────────┘ │
            │                │                 │      │                │                 │
            │                ▼                 │      │                ▼                 │
            │  ┌─────────────────────────────┐ │      │  ┌─────────────────────────────┐ │
            │  │    Vector Embedding         │ │      │  │    Context Assembly         │ │
            │  │   (Google Gemini)           │ │      │  │  (Text + Image Context)     │ │
            │  └─────────────────────────────┘ │      │  └─────────────────────────────┘ │
            └─────────────────────────────────┘      └─────────────────────────────────┘
                                │                                      │
                                ▼                                      ▼
                    ┌─────────────────────────────────────────────────────────────┐
                    │                    STORAGE LAYER                            │
                    │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
                    │  │   ChromaDB      │  │   ChromaDB      │  │   File      │  │
                    │  │  Text Store     │  │  Image Store    │  │  Storage    │  │
                    │  └─────────────────┘  └─────────────────┘  └─────────────┘  │
                    └─────────────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
                    ┌─────────────────────────────────────────────────────────────┐
                    │                  LLM PROCESSING LAYER                       │
                    │  ┌─────────────────────────────────────────────────────────┐ │
                    │  │              Google Gemini LLM                          │ │
                    │  │  ┌─────────────────┐  ┌─────────────────────────────────┐│ │
                    │  │  │ Context-Aware   │  │     Response Generation         ││ │
                    │  │  │   Processing    │  │    (Streaming Support)          ││ │
                    │  │  └─────────────────┘  └─────────────────────────────────┘│ │
                    │  └─────────────────────────────────────────────────────────┘ │
                    └─────────────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
                    ┌─────────────────────────────────────────────────────────────┐
                    │                 RESPONSE PROCESSING                         │
                    │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
                    │  │ Output Validation│  │ Image Encoding  │  │  Response   │  │
                    │  │ (Guardrails AI) │  │   (Base64)      │  │  Assembly   │  │
                    │  └─────────────────┘  └─────────────────┘  └─────────────┘  │
                    └─────────────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
                    ┌─────────────────────────────────────────────────────────────┐
                    │                   CLIENT RESPONSE                           │
                    │  ┌─────────────────────────────────────────────────────────┐ │
                    │  │           Streaming FastAPI Response                   │ │
                    │  │  ┌─────────────────┐  ┌─────────────────────────────────┐│ │
                    │  │  │   Text Content  │  │     Image DataURIs              ││ │
                    │  │  │   (Markdown)    │  │   (Inline Attachments)          ││ │
                    │  │  └─────────────────┘  └─────────────────────────────────┘│ │
                    │  └─────────────────────────────────────────────────────────┘ │
                    └─────────────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
                    ┌─────────────────────────────────────────────────────────────┐
                    │                   MONITORING & LOGGING                      │
                    │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
                    │  │ Security Events │  │ Performance     │  │   Audit     │  │
                    │  │    Logging      │  │   Metrics       │  │   Trails    │  │
                    │  └─────────────────┘  └─────────────────┘  └─────────────┘  │
                    └─────────────────────────────────────────────────────────────┘
```

### 5.6 Query-to-Response Flow Diagram

The following detailed flow diagram illustrates the specific query processing workflow with image attachment logic:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  Input Validation│───▶│  Text Retrieval │
│  (e.g., "show   │    │  (Guardrails AI) │    │  (ChromaDB)     │
│   image of      │    └─────────────────┘    └─────────────────┘
│   diagram")     │         │                       │
└─────────────────┘         │                       │
                            ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Keyword Check  │───▶│  Image Retrieval │───▶│  LLM Generation │
│  (Image-related?│    │  (Separate Store)│    │  (Google Gemini)│
│   Yes: Attach)  │    └─────────────────┘    └─────────────────┘
└─────────────────┘         │                       │
                            ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Base64 Encode  │───▶│  Response Assembly│───▶│  Output Validation│
│  Matched Images │    │  (Text + Images) │    │  (Guardrails AI) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Streaming      │    │  Client Response │    │  End            │
│  Response       │    │  (Text + DataURI)│    └─────────────────┘
│  (FastAPI)      │    └─────────────────┘
└─────────────────┘

**Diagram Explanation**:
- **Start**: User submits a query (e.g., containing "image" keywords).
- **Validation**: Guardrails AI checks input for safety.
- **Retrieval**: Text from main vector store; images from separate store if keywords match.
- **Processing**: Base64 encode images for inline attachment.
- **LLM**: Generate response using combined context.
- **Output**: Validate and stream response with images.
- **End**: Deliver to client with DataURI for rendering.

---

## 6. Security Implementation Details

### 6.1 Guardrails Configuration

#### 6.1.1 Context-Aware Validation
```python
class RAGGuardrails:
    def _contains_toxic_language(self, text: str) -> bool:
        # Educational context detection
        educational_indicators = [
            "programming", "machine learning", "algorithm"
        ]
        
        has_educational_context = any(
            indicator in text.lower() 
            for indicator in educational_indicators
        )
        
        if has_educational_context:
            # Lenient validation for educational content
            return self._check_harmful_phrases_only(text)
        else:
            # Standard validation for other content
            return self._standard_toxic_check(text)
```

#### 6.1.2 PII Detection and Masking
```python
pii_patterns = {
    "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    "phone": r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
    "ssn": r'\b\d{3}-?\d{2}-?\d{4}\b',
    "credit_card": r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
}
```

### 6.2 Authentication and Authorization

#### 6.2.1 JWT Implementation
```python
# Token structure
{
    "sub": "user_id",
    "username": "user@example.com",
    "roles": ["user", "admin"],
    "exp": timestamp,
    "iat": timestamp
}
```

#### 6.2.2 Role-Based Access Control
```python
roles_permissions = {
    "user": ["query", "upload", "delete_own"],
    "admin": ["query", "upload", "delete_any", "stats", "monitor"],
    "system": ["auto_index", "background_tasks"]
}
```

### 6.3 Rate Limiting Strategy

| Endpoint | Rate Limit | Purpose |
|----------|------------|---------|
| /auth/login | 5/minute | Prevent brute force |
| /upload | 10/minute | Resource protection |
| /query | 30/minute | Balance usage/performance |
| /images/* | 30/minute | Bandwidth management |
| /security/stats | Admin only | Sensitive data protection |

---

## 7. Future Implementation Features

### 7.1 Advanced Security Features (From secure_app.py Analysis)

#### 7.1.1 Authentication & Authorization System
```python
# From secure_app.py - JWT-based authentication
class AuthenticationSystem:
    - JWT token-based authentication
    - User login/logout endpoints
    - Role-based access control (RBAC)
    - Token refresh mechanisms
    - Session management
```

#### 7.1.2 Rate Limiting & DDoS Protection
```python
# From secure_app.py - SlowAPI integration
class RateLimitingSystem:
    - Per-endpoint rate limiting
    - IP-based request throttling
    - Configurable rate limits
    - DDoS protection mechanisms
    - Rate limit exceeded handling
```

#### 7.1.3 Advanced Security Monitoring
```python
# From secure_app.py - Comprehensive logging
class SecurityMonitoring:
    - Security event logging
    - Threat detection and scoring
    - Anomaly detection algorithms
    - Real-time security alerts
    - Audit trail generation
    - Risk score calculation
```

#### 7.1.4 File Security & Quarantine System
```python
# From secure_app.py - Advanced file handling
class FileSecuritySystem:
    - Advanced file validation
    - Malicious file detection
    - File quarantine mechanisms
    - Content security scanning
    - Data poisoning detection
    - Secure file storage
```

#### 7.1.5 Enhanced CORS & Security Headers
```python
# From secure_app.py - Production security
class SecurityHeaders:
    - Restricted CORS policies
    - Security headers implementation
    - Content Security Policy (CSP)
    - X-Frame-Options protection
    - XSS protection headers
```

#### 7.1.6 Secure Vector Store & Document Management
```python
# From secure_app.py - Enterprise storage
class SecureStorage:
    - User-based access control
    - Document ownership tracking
    - Secure vector store operations
    - Encrypted document storage
    - Access logging and auditing
```

#### 7.1.7 Advanced Output Filtering
```python
# From secure_app.py - Response security
class OutputSecurity:
    - Multi-layer response validation
    - Content filtering pipelines
    - PII detection in responses
    - Safe error message handling
    - Response sanitization
```

### 7.2 AI/ML Enhancements

#### 7.2.1 Advanced Multimodal Processing
```python
# Planned capabilities
class AdvancedMultimodalProcessor:
    - Video content analysis
    - Audio transcription and analysis
    - 3D model processing
    - Real-time content streaming
    - Cross-modal semantic understanding
```

#### 7.2.2 Intelligent Content Curation
```python
# Future features
class ContentCurator:
    - Automatic content categorization
    - Quality scoring and ranking
    - Duplicate detection and merging
    - Content freshness tracking
    - Relevance optimization
```

#### 7.2.3 Advanced Query Understanding
```python
# Planned enhancements
class QueryIntelligence:
    - Intent classification
    - Query expansion and refinement
    - Context-aware query rewriting
    - Multi-turn conversation support
    - Personalized query suggestions
```

### 7.3 Performance and Scalability

#### 7.3.1 Distributed Architecture
```python
# Future implementation
class DistributedRAG:
    - Microservices architecture
    - Load balancing and auto-scaling
    - Distributed vector storage
    - Caching layer optimization
    - Edge computing support
```

#### 7.3.2 Advanced Caching
```python
# Planned features
class IntelligentCache:
    - Semantic caching
    - Predictive pre-loading
    - Cache invalidation strategies
    - Multi-level caching hierarchy
    - Performance analytics
```

### 7.4 Integration and Extensibility

#### 7.4.1 Enterprise Integrations
```python
# Future connectors
class EnterpriseConnectors:
    - Active Directory integration
    - SAML/OAuth2 providers
    - Enterprise document systems
    - Workflow automation tools
    - Business intelligence platforms
```

#### 7.4.2 Plugin Architecture
```python
# Extensibility framework
class PluginManager:
    - Custom validator plugins
    - Third-party LLM integrations
    - Custom document processors
    - External security tools
    - Analytics and monitoring plugins
```

---

## 8. Deployment Architecture

### 8.1 Production Deployment

#### 8.1.1 Container Strategy
```dockerfile
# Multi-stage Docker build
FROM python:3.12-slim as base
# Security hardening
# Dependency installation
# Application setup

FROM base as production
# Production optimizations
# Security configurations
# Monitoring setup
```

#### 8.1.2 Infrastructure Requirements

| Component | Standard App | Secure App | Scaling |
|-----------|-------------|------------|---------|
| CPU | 2 cores | 4 cores | Horizontal |
| Memory | 4GB | 8GB | Vertical |
| Storage | 50GB | 100GB | Network attached |
| Network | Basic | Load balanced | CDN enabled |

### 8.2 Security Hardening

#### 8.2.1 Network Security
- TLS 1.3 encryption
- Certificate pinning
- Network segmentation
- Firewall rules
- DDoS protection

#### 8.2.2 Application Security
- Security headers implementation
- Input sanitization
- Output encoding
- Session management
- Secure error handling

---

## 9. Monitoring and Observability

### 9.1 Security Monitoring

#### 9.1.1 Security Events
```python
security_events = {
    "authentication_failures": "Failed login attempts",
    "rate_limit_exceeded": "Rate limiting triggers",
    "validation_failures": "Guardrails validation failures",
    "unauthorized_access": "Access control violations",
    "suspicious_queries": "Potentially malicious queries"
}
```

#### 9.1.2 Performance Metrics
```python
performance_metrics = {
    "response_time": "Query processing latency",
    "throughput": "Requests per second",
    "error_rate": "Failed request percentage",
    "resource_usage": "CPU/Memory utilization",
    "cache_hit_rate": "Caching effectiveness"
}
```

### 9.2 Alerting Strategy

| Alert Type | Threshold | Action |
|------------|-----------|--------|
| High error rate | >5% | Immediate notification |
| Security breach | Any | Emergency response |
| Resource exhaustion | >90% | Auto-scaling trigger |
| Validation failures | >10% | Investigation required |

---

## 10. Testing Strategy

### 10.1 Security Testing

#### 10.1.1 Guardrails Testing
```python
# Test categories
test_categories = {
    "input_validation": "Malicious input detection",
    "output_filtering": "Response safety validation",
    "pii_detection": "Personal information masking",
    "context_awareness": "Educational content handling",
    "edge_cases": "Boundary condition testing"
}
```

#### 10.1.2 Penetration Testing
- Authentication bypass attempts
- Authorization escalation tests
- Input injection attacks
- Rate limiting bypass
- Data exfiltration attempts

### 10.2 Performance Testing

#### 10.2.1 Load Testing
- Concurrent user simulation
- Peak load handling
- Resource utilization monitoring
- Response time analysis
- Failure point identification

#### 10.2.2 Stress Testing
- System breaking point
- Recovery mechanisms
- Data integrity under stress
- Security maintenance under load
- Graceful degradation

---

## 12. High-Level Design for Core Components

### 12.1 Overview

This section provides a High-Level Design (HLD) for the core components of the RAG system: `app_google.py` (the main FastAPI application) and `rag_google.py` (the RAG utility module). These components handle the application logic, document processing, vector storage, and query handling, integrating multimodal features and security.

### 12.2 app_google.py - Main Application Layer

#### 12.2.1 Component Architecture

- **Role**: Serves as the entry point for the RAG system, providing RESTful APIs for uploads, queries, and document management.
- **Technologies**: FastAPI for web framework, APScheduler for background tasks, ChromaDB for vector storage, Google Gemini for embeddings and LLM.
- **Key Modules**:
  - FastAPI App: Handles HTTP requests and responses.
  - Multimodal Processor: Integrates with `src/core/multimodal_processor.py` for image and document analysis.
  - Vector Stores: Separate ChromaDB collections for text and images.
  - Security: Guardrails AI for input/output validation.

#### 12.2.2 Data Flow

```
1. Client Request (e.g., /upload or /query)
   ↓
2. Authentication & Rate Limiting (if applicable)
   ↓
3. Input Validation (Guardrails AI)
   ↓
4. Processing (Multimodal Extraction, Chunking)
   ↓
5. Vector Embedding & Storage
   ↓
6. Query Retrieval & LLM Generation
   ↓
7. Output Validation & Response
```

#### 12.2.3 Key Functions and Responsibilities

- **Endpoints**:
  - `/upload`: Processes documents, extracts multimodal content, and indexes into vector stores.
  - `/query`: Handles queries, retrieves context, generates responses, and attaches images if triggered.
  - `/delete`: Removes documents by source.
  - `/images/{filename}`: Serves extracted images securely.
- **Background Tasks**: APScheduler for auto-indexing uploads every 5 minutes.
- **Configuration**: Loads from `config_google.json` for API keys, thresholds, and tunables.
- **Image Handling**: Dynamic attachment based on keywords, using separate image vector store.

#### 12.2.4 Security Considerations

- Input/output validation via Guardrails AI.
- Path traversal protection for image serving.
- Rate limiting and error handling to prevent abuse.

### 12.3 rag_google.py - RAG Utility Layer

#### 12.3.1 Component Architecture

- **Role**: Provides utility functions for document loading, chunking, embedding, retrieval, and LLM interaction.
- **Technologies**: LangChain for document processing, ChromaDB for storage, Google Gemini for embeddings/LLM.
- **Key Modules**:
  - Document Loaders: PDF, text, and other formats.
  - Chunkers: Recursive text splitter for manageable pieces.
  - Embeddings: Google Generative AI for semantic vectors.
  - Retrievers: Similarity search with thresholds.
  - LLM: Google Gemini for context-aware responses.

#### 12.3.2 Data Flow

```
1. Load Documents from Sources
   ↓
2. Chunk into Smaller Pieces
   ↓
3. Generate Embeddings
   ↓
4. Store in ChromaDB
   ↓
5. Query with Similarity Search
   ↓
6. Assemble Context
   ↓
7. LLM Generation
```

#### 12.3.3 Key Functions and Responsibilities

- **load_docs(sources)**: Loads and processes documents from file paths or URLs.
- **chunk_docs(docs, chunk_size, chunk_overlap)**: Splits documents into chunks for better retrieval.
- **add_to_vectorstore(chunks, vectorstore)**: Embeds and stores chunks in ChromaDB.
- **retrieve(vectorstore, query, k, thresholds)**: Performs similarity search with fallback thresholds.
- **answer_query_with_context(query, context, llm)**: Generates LLM responses using retrieved context.
- **get_similar_queries_from_llm(query, llm)**: Suggests related queries for better context.

#### 12.3.4 Integration with app_google.py

- Functions are imported and used in endpoints (e.g., `retrieve` in `/query`).
- Supports both text and image retrieval via separate collections.
- Handles adaptive thresholds for relevance tuning.

### 12.4 Dependencies and Interfaces

- **External APIs**: Google Generative AI for embeddings and LLM.
- **Storage**: ChromaDB for persistent vector storage.
- **Security**: Guardrails AI for validation.
- **Multimodal**: Integrates with `src/core/multimodal_processor.py` for image/text extraction.

### 12.5 Scalability and Performance

- **Chunking Strategy**: Balances chunk size (1200) and overlap (150) for context preservation.
- **Retrieval Optimization**: Uses configurable `k` and thresholds to limit results.
- **Background Indexing**: Reduces latency for large document sets.
- **Error Handling**: Fallbacks for missing images or validation failures.

### 12.6 Future Enhancements

- Support for additional document types (e.g., video, audio).
- Advanced caching for embeddings.
- Multi-tenant support for shared vector stores.
- Integration with other LLMs or embeddings models.

---

## 11. Conclusion

### 11.1 System Capabilities

The Secure RAG System provides a comprehensive solution for intelligent document processing with enterprise-grade security. The dual-application architecture allows for flexible deployment options, from basic implementations to full enterprise security.

### 11.2 Key Differentiators

1. **Context-Aware Security**: Educational content-friendly validation
2. **Comprehensive Protection**: Multi-layered security approach
3. **Production Ready**: Enterprise-grade monitoring and logging
4. **Scalable Architecture**: Future-proof design with extensibility
5. **Compliance Ready**: Built-in audit trails and data protection

### 11.3 Next Steps

1. **Immediate**: Deploy current secure implementation
2. **Short-term**: Implement advanced threat detection
3. **Medium-term**: Add multi-tenant capabilities
4. **Long-term**: Develop distributed architecture

---

## Appendices

### Appendix A: Configuration Examples
### Appendix B: API Documentation
### Appendix C: Security Checklist
### Appendix D: Troubleshooting Guide
### Appendix E: Performance Tuning Guide

---

**Document End**

*This technical design document serves as the comprehensive guide for understanding, implementing, and maintaining the Secure RAG System with Guardrails AI integration.*
