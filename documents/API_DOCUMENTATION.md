# Secure RAG System - Complete API Documentation

## Table of Contents
1. [API Overview](#api-overview)
2. [Authentication](#authentication)
3. [API Endpoints](#api-endpoints)
4. [Request/Response Schemas](#requestresponse-schemas)
5. [Error Handling](#error-handling)
6. [Rate Limiting](#rate-limiting)
7. [Security Headers](#security-headers)

## API Overview

### Base URL
```
Production: https://your-domain.com/api/v1
Development: http://127.0.0.1:8002
```

### API Version
- **Current Version**: 2.0.0
- **Protocol**: REST API
- **Content-Type**: `application/json`
- **Authentication**: JWT Bearer Token

### Security Features
- JWT-based authentication
- Rate limiting per endpoint
- Input validation and sanitization
- Output filtering and PII masking
- Comprehensive audit logging

## Authentication

### Authentication Flow

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Auth
    
    Client->>API: POST /auth/login
    API->>Auth: Validate credentials
    Auth->>API: Generate JWT token
    API->>Client: Return access_token
    
    Client->>API: Request with Authorization header
    API->>Auth: Validate JWT token
    Auth->>API: Return user context
    API->>Client: Process request
```

### JWT Token Structure
```json
{
  "header": {
    "alg": "HS256",
    "typ": "JWT"
  },
  "payload": {
    "sub": "username",
    "active": true,
    "roles": ["admin", "user"],
    "permissions": ["read", "write", "delete"],
    "exp": 1640995200,
    "iat": 1640991600
  }
}
```

## API Endpoints

### 1. Authentication Endpoints

#### POST /auth/login
**Purpose**: Authenticate user and obtain JWT token

**Rate Limit**: 5 requests per minute per IP

**Request**:
```http
POST /auth/login HTTP/1.1
Content-Type: application/x-www-form-urlencoded

username=admin&password=secure_password_123
```

**Request Schema**:
```json
{
  "username": "string (required, 3-50 chars)",
  "password": "string (required, 8-128 chars)"
}
```

**Success Response (200)**:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

**Error Responses**:
```json
// 401 Unauthorized
{
  "detail": "Invalid credentials"
}

// 429 Too Many Requests
{
  "detail": "Rate limit exceeded. Please try again later."
}

// 422 Validation Error
{
  "detail": [
    {
      "loc": ["body", "username"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### 2. Document Management Endpoints

#### POST /upload
**Purpose**: Securely upload and process documents

**Authentication**: Required (Bearer Token)
**Rate Limit**: 10 requests per minute per user

**Request**:
```http
POST /upload HTTP/1.1
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
Content-Type: multipart/form-data

files: [file1.pdf, file2.docx]
```

**Request Schema**:
```json
{
  "files": "array of files (required)",
  "max_files": 10,
  "allowed_types": [".pdf", ".docx", ".txt"],
  "max_size_mb": 50
}
```

**Success Response (200)**:
```json
{
  "status": "success",
  "processed_files": [
    {
      "filename": "document1.pdf",
      "chunks": 15,
      "file_path": "uploads/document1.pdf"
    }
  ],
  "quarantined_files": [
    {
      "filename": "suspicious.pdf",
      "issues": ["suspicious_patterns", "encoding_attack"]
    }
  ],
  "message": "Processed 1 files, quarantined 1 files"
}
```

**Error Responses**:
```json
// 400 Bad Request - File validation error
{
  "detail": "File type not allowed. Allowed types: .pdf, .docx, .txt"
}

// 413 Request Entity Too Large
{
  "detail": "File too large. Maximum size: 50MB"
}

// 401 Unauthorized
{
  "detail": "Could not validate credentials"
}
```

#### DELETE /documents
**Purpose**: Securely delete documents

**Authentication**: Required (Bearer Token)
**Rate Limit**: 10 requests per minute per user

**Request**:
```http
DELETE /documents HTTP/1.1
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
Content-Type: application/json

{
  "paths": ["uploads/document1.pdf", "uploads/document2.docx"]
}
```

**Request Schema**:
```json
{
  "paths": {
    "type": "array",
    "items": "string",
    "minItems": 1,
    "maxItems": 100,
    "description": "Array of file paths to delete"
  }
}
```

**Success Response (200)**:
```json
{
  "status": "success",
  "deleted_files": ["uploads/document1.pdf"],
  "failed_deletions": ["uploads/nonexistent.pdf"],
  "message": "Deleted 1 files, 1 failures"
}
```

### 3. Query Processing Endpoints

#### POST /query
**Purpose**: Process RAG queries with security analysis

**Authentication**: Required (Bearer Token)
**Rate Limit**: 30 requests per minute per user

**Request**:
```http
POST /query HTTP/1.1
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
Content-Type: application/json

{
  "q": "What is the main conclusion of the research paper?"
}
```

**Request Schema**:
```json
{
  "q": {
    "type": "string",
    "minLength": 1,
    "maxLength": 1000,
    "description": "Query string for RAG processing"
  }
}
```

**Success Response (200)**:
```http
HTTP/1.1 200 OK
Content-Type: text/plain
Transfer-Encoding: chunked

Based on the provided context, the main conclusion of the research paper is...
```

**Error Responses**:
```json
// 400 Bad Request - Security block
{
  "detail": "Query blocked due to security concerns"
}

// 400 Bad Request - Validation error
{
  "detail": "Query cannot be empty"
}
```

### 4. System Monitoring Endpoints

#### GET /health
**Purpose**: System health check

**Authentication**: Not required
**Rate Limit**: 60 requests per minute per IP

**Request**:
```http
GET /health HTTP/1.1
```

**Success Response (200)**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "security_level": "enhanced",
  "version": "2.0.0"
}
```

#### GET /security/stats
**Purpose**: Security statistics (Admin only)

**Authentication**: Required (Admin role)
**Rate Limit**: 10 requests per minute per user

**Request**:
```http
GET /security/stats HTTP/1.1
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
```

**Success Response (200)**:
```json
{
  "vector_store": {
    "document_count": 150,
    "collection_name": "secure_rag_collection",
    "encrypted": true,
    "authorized_users": 5
  },
  "document_store": {
    "active_files": 45,
    "quarantined_files": 3,
    "total_size_bytes": 104857600,
    "quarantine_size_bytes": 2097152
  },
  "security_events": 1247,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Error Responses**:
```json
// 403 Forbidden
{
  "detail": "Admin access required"
}
```

## Request/Response Schemas

### Common Data Types

#### User Object
```json
{
  "sub": "string (username)",
  "active": "boolean",
  "roles": ["string"],
  "permissions": ["string"],
  "exp": "integer (unix timestamp)",
  "iat": "integer (unix timestamp)"
}
```

#### File Upload Object
```json
{
  "filename": "string",
  "content_type": "string",
  "size": "integer (bytes)",
  "chunks": "integer (optional)"
}
```

#### Security Event Object
```json
{
  "timestamp": "string (ISO 8601)",
  "event_type": "string",
  "threat_level": "low|medium|high|critical",
  "user_id": "string",
  "ip_address": "string",
  "risk_score": "float (0.0-1.0)",
  "metadata": "object"
}
```

### Validation Rules

#### Query Validation
- **Length**: 1-1000 characters
- **Blocked Patterns**: SQL injection, XSS, prompt injection
- **Sanitization**: HTML entities, special characters
- **Encoding**: UTF-8 only

#### File Validation
- **Types**: PDF, DOCX, TXT only
- **Size**: Maximum 50MB per file
- **Count**: Maximum 10 files per request
- **Content**: Scanned for malicious patterns

## Error Handling

### Standard Error Response Format
```json
{
  "detail": "string (error message)",
  "error_code": "string (optional)",
  "timestamp": "string (ISO 8601)",
  "request_id": "string (optional)"
}
```

### HTTP Status Codes

| Code | Description | Usage |
|------|-------------|-------|
| 200 | OK | Successful request |
| 201 | Created | Resource created successfully |
| 400 | Bad Request | Invalid request data |
| 401 | Unauthorized | Authentication required |
| 403 | Forbidden | Insufficient permissions |
| 413 | Payload Too Large | File size exceeded |
| 422 | Unprocessable Entity | Validation error |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |

### Error Categories

#### Authentication Errors
```json
{
  "detail": "Could not validate credentials",
  "error_code": "AUTH_INVALID_TOKEN"
}
```

#### Validation Errors
```json
{
  "detail": [
    {
      "loc": ["body", "q"],
      "msg": "ensure this value has at least 1 characters",
      "type": "value_error.any_str.min_length"
    }
  ]
}
```

#### Security Errors
```json
{
  "detail": "Query blocked due to security concerns",
  "error_code": "SECURITY_THREAT_DETECTED"
}
```

## Rate Limiting

### Rate Limit Headers
```http
X-RateLimit-Limit: 30
X-RateLimit-Remaining: 25
X-RateLimit-Reset: 1640995200
```

### Rate Limits by Endpoint

| Endpoint | Limit | Window |
|----------|-------|--------|
| POST /auth/login | 5 requests | 1 minute |
| POST /upload | 10 requests | 1 minute |
| POST /query | 30 requests | 1 minute |
| DELETE /documents | 10 requests | 1 minute |
| GET /health | 60 requests | 1 minute |
| GET /security/stats | 10 requests | 1 minute |

## Security Headers

### Standard Security Headers
```http
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000; includeSubDomains
Content-Security-Policy: default-src 'self'
Referrer-Policy: strict-origin-when-cross-origin
X-Process-Time: 0.123
```

### CORS Configuration
```http
Access-Control-Allow-Origin: http://localhost:3000
Access-Control-Allow-Methods: GET, POST, PUT, DELETE
Access-Control-Allow-Headers: Authorization, Content-Type
Access-Control-Allow-Credentials: true
```

This API documentation provides complete specifications for integrating with the Secure RAG System, including all security considerations and best practices.
