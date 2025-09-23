# Secure RAG System - Technical Architecture Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Component Architecture](#component-architecture)
4. [Security Architecture](#security-architecture)
5. [Data Flow Architecture](#data-flow-architecture)
6. [Technology Stack](#technology-stack)

## System Overview

The Secure RAG (Retrieval-Augmented Generation) System is an enterprise-grade AI application that provides intelligent document search and question-answering capabilities with comprehensive security measures. The system implements all major security recommendations from industry best practices for AI systems.

### Key Features
- **Secure Document Processing**: Multi-layer validation and threat detection
- **Intelligent Query Processing**: Context-aware responses with anomaly detection
- **Enterprise Security**: JWT authentication, RBAC, encryption, and audit logging
- **Real-time Monitoring**: Comprehensive security event tracking and threat analysis
- **Scalable Architecture**: Modular design with separation of concerns

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                             │
├─────────────────────────────────────────────────────────────────┤
│  Web Clients  │  Mobile Apps  │  API Clients  │  Admin Console  │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API GATEWAY LAYER                          │
├─────────────────────────────────────────────────────────────────┤
│  Rate Limiting  │  CORS  │  Security Headers  │  Load Balancer  │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SECURITY LAYER                               │
├─────────────────────────────────────────────────────────────────┤
│  Authentication  │  Authorization  │  Input Validation  │  Audit │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                   APPLICATION LAYER                             │
├─────────────────────────────────────────────────────────────────┤
│  Document Service  │  Query Service  │  User Service  │  Monitor │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DATA LAYER                                  │
├─────────────────────────────────────────────────────────────────┤
│  Vector Store  │  Document Store  │  Metadata DB  │  Log Store  │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                  EXTERNAL SERVICES                              │
├─────────────────────────────────────────────────────────────────┤
│  Google Gemini API  │  Embedding Service  │  Monitoring Tools   │
└─────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### Core Components

#### 1. FastAPI Application (`secure_app.py`)
- **Purpose**: Main application entry point with security middleware
- **Responsibilities**:
  - HTTP request handling
  - Security middleware integration
  - Rate limiting enforcement
  - CORS configuration
  - Error handling and sanitization

#### 2. Configuration Management (`src/config/`)
- **Purpose**: Centralized configuration with validation
- **Components**:
  - `settings.py`: Pydantic-based configuration with validation
  - Environment variable management
  - Security parameter validation

#### 3. Security Layer (`src/security/`)
- **Purpose**: Authentication, authorization, and input validation
- **Components**:
  - `auth.py`: JWT authentication and RBAC
  - `validators.py`: Input sanitization and validation
  - Rate limiting and access control

#### 4. Core Services (`src/core/`)
- **Purpose**: Business logic and monitoring
- **Components**:
  - `monitoring.py`: Security logging and anomaly detection
  - `data_validation.py`: Data poisoning protection

#### 5. Repository Layer (`src/repositories/`)
- **Purpose**: Data access with security controls
- **Components**:
  - `vector_store.py`: Secure vector database operations
  - `document_store.py`: Secure file management

## Security Architecture

### Multi-Layer Security Model

```
┌─────────────────────────────────────────────────────────────────┐
│                    SECURITY LAYERS                              │
├─────────────────────────────────────────────────────────────────┤
│ Layer 7: Application Security (Business Logic Protection)       │
│ Layer 6: Data Security (Encryption & Access Control)           │
│ Layer 5: Output Security (PII Filtering & Sanitization)        │
│ Layer 4: Processing Security (Anomaly Detection)               │
│ Layer 3: Input Security (Validation & Sanitization)            │
│ Layer 2: Transport Security (HTTPS, Headers, CORS)             │
│ Layer 1: Network Security (Rate Limiting, DDoS Protection)     │
└─────────────────────────────────────────────────────────────────┘
```

### Security Components

#### Authentication & Authorization
- **JWT Tokens**: Stateless authentication with configurable expiration
- **Role-Based Access Control**: Granular permissions system
- **Session Management**: Secure token lifecycle management

#### Threat Detection
- **Prompt Injection Detection**: 15+ pattern-based detection rules
- **Anomaly Detection**: Behavioral analysis for suspicious activities
- **Data Poisoning Protection**: Content validation before indexing

#### Data Protection
- **Encryption at Rest**: Vector store content encryption
- **PII Masking**: Advanced personally identifiable information protection
- **Secure Storage**: Quarantine system for suspicious files

## Data Flow Architecture

### Document Processing Flow

```
Upload Request → Authentication → File Validation → Security Scan
     ↓
Content Extraction → Poisoning Detection → Chunking → Encryption
     ↓
Vector Generation → Secure Storage → Metadata Update → Audit Log
```

### Query Processing Flow

```
Query Request → Authentication → Input Validation → Anomaly Detection
     ↓
Vector Search → Context Retrieval → Response Generation → Output Filtering
     ↓
PII Masking → Security Logging → Response Delivery
```

## Technology Stack

### Backend Framework
- **FastAPI**: High-performance async web framework
- **Uvicorn**: ASGI server with production capabilities
- **Pydantic**: Data validation and settings management

### AI/ML Components
- **LangChain**: LLM orchestration and document processing
- **Google Gemini**: Large language model for generation
- **ChromaDB**: Vector database for embeddings storage
- **Text Embedding**: Google's text-embedding-004 model

### Security Stack
- **JWT**: JSON Web Tokens for authentication
- **Passlib**: Password hashing with bcrypt
- **Cryptography**: Encryption and secure key management
- **SlowAPI**: Rate limiting middleware

### Monitoring & Logging
- **Structlog**: Structured logging with security events
- **Python Logging**: Application and security audit trails
- **Custom Monitoring**: Real-time threat detection and alerting

### File Processing
- **PyPDF**: PDF document processing
- **python-docx**: Microsoft Word document handling
- **python-magic**: File type detection and validation

This architecture provides enterprise-grade security while maintaining high performance and scalability for RAG operations.
