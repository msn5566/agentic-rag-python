# Secure RAG System - Troubleshooting Guide

## Table of Contents
1. [Common Issues](#common-issues)
2. [Authentication Problems](#authentication-problems)
3. [File Upload Issues](#file-upload-issues)
4. [Query Processing Errors](#query-processing-errors)
5. [Performance Issues](#performance-issues)
6. [Security Alerts](#security-alerts)
7. [Deployment Problems](#deployment-problems)
8. [Monitoring and Debugging](#monitoring-and-debugging)

## Common Issues

### 1. Application Won't Start

#### Symptom
```
Error: Could not validate credentials
ModuleNotFoundError: No module named 'xyz'
```

#### Diagnosis
```bash
# Check Python version
python --version  # Should be 3.9+

# Check virtual environment
which python
pip list

# Check environment variables
python -c "from src.config.settings import settings; print('Config loaded successfully')"
```

#### Solution
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Verify environment file
cp .env.example .env
# Edit .env with correct values

# Check file permissions
chmod 600 .env
chmod +x secure_app.py
```

### 2. Google API Connection Issues

#### Symptom
```
google.api_core.exceptions.Unauthenticated: 401 API key not valid
```

#### Diagnosis
```bash
# Test API key
curl -H "Authorization: Bearer $GOOGLE_API_KEY" \
  "https://generativelanguage.googleapis.com/v1/models"
```

#### Solution
```bash
# Verify API key in Google Cloud Console
# Ensure Gemini API is enabled
# Check API quotas and billing

# Update .env file
GOOGLE_API_KEY=your_valid_api_key_here
```

### 3. Vector Store Initialization Errors

#### Symptom
```
chromadb.errors.InvalidDimensionException: Embedding dimension mismatch
```

#### Diagnosis
```bash
# Check vector store directory
ls -la storage/chroma_google/

# Check embedding model configuration
grep EMBEDDING_MODEL .env
```

#### Solution
```bash
# Clear vector store (WARNING: This deletes all indexed data)
rm -rf storage/chroma_google/*

# Restart application to reinitialize
python secure_app.py
```

## Authentication Problems

### 1. JWT Token Issues

#### Symptom
```json
{
  "detail": "Could not validate credentials"
}
```

#### Diagnosis
```python
# Debug JWT token
import jwt
from src.config.settings import settings

token = "your_token_here"
try:
    payload = jwt.decode(token, settings.jwt_secret_key, algorithms=["HS256"])
    print("Token valid:", payload)
except jwt.ExpiredSignatureError:
    print("Token expired")
except jwt.InvalidTokenError as e:
    print("Token invalid:", e)
```

#### Solution
```bash
# Generate new JWT secret (minimum 32 characters)
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Update .env file
JWT_SECRET_KEY=your_new_secret_key_here

# Restart application
```

### 2. Login Failures

#### Symptom
```json
{
  "detail": "Invalid credentials"
}
```

#### Diagnosis
```python
# Check password hashing
from src.security.auth import AuthManager

auth = AuthManager()
hashed = auth.get_password_hash("your_password")
print("Hashed password:", hashed)
print("Verification:", auth.verify_password("your_password", hashed))
```

#### Solution
```python
# Reset admin password in secure_app.py
# Update the hardcoded credentials or implement user management
```

## File Upload Issues

### 1. File Type Rejection

#### Symptom
```json
{
  "detail": "File type not allowed. Allowed types: .pdf, .docx, .txt"
}
```

#### Diagnosis
```bash
# Check file MIME type
file --mime-type your_file.pdf

# Check file extension validation
python -c "
from src.security.validators import FileValidator
validator = FileValidator()
print(validator.validate_file_extension('test.pdf'))
"
```

#### Solution
```bash
# Update allowed extensions in .env
ALLOWED_FILE_EXTENSIONS=.pdf,.docx,.txt,.md

# Or convert file to supported format
```

### 2. File Size Limits

#### Symptom
```json
{
  "detail": "File too large. Maximum size: 50MB"
}
```

#### Diagnosis
```bash
# Check file size
ls -lh your_file.pdf

# Check current limit
grep MAX_FILE_SIZE_MB .env
```

#### Solution
```bash
# Increase file size limit in .env
MAX_FILE_SIZE_MB=100

# Or compress the file
```

### 3. File Quarantine Issues

#### Symptom
```json
{
  "quarantined_files": [
    {
      "filename": "document.pdf",
      "issues": ["suspicious_patterns"]
    }
  ]
}
```

#### Diagnosis
```python
# Check quarantine reason
from src.core.data_validation import DataPoisoningDetector

detector = DataPoisoningDetector()
with open("your_file.txt", "r") as f:
    content = f.read()
    is_safe, risk_score, threats = detector.validate_document_content("file.txt", content)
    print(f"Safe: {is_safe}, Risk: {risk_score}, Threats: {threats}")
```

#### Solution
```bash
# Review quarantined file manually
ls -la quarantine/

# If false positive, adjust detection thresholds in data_validation.py
# Or whitelist specific patterns
```

## Query Processing Errors

### 1. Query Blocked by Security

#### Symptom
```json
{
  "detail": "Query blocked due to security concerns"
}
```

#### Diagnosis
```python
# Test query security analysis
from src.core.monitoring import AnomalyDetector

detector = AnomalyDetector()
risk_score = detector.analyze_query("user123", "127.0.0.1", "your query here")
print(f"Risk score: {risk_score}")
```

#### Solution
```python
# Adjust risk thresholds in monitoring.py
# Or rephrase query to avoid suspicious patterns
```

### 2. No Results Found

#### Symptom
```
I cannot answer based on the provided information.
```

#### Diagnosis
```python
# Check vector store content
from src.repositories.vector_store import SecureVectorStore

store = SecureVectorStore()
results = store.similarity_search_with_score("test query", k=5, user_id="admin")
print(f"Found {len(results)} results")
```

#### Solution
```bash
# Re-index documents
python -c "
from rag_google import scan_and_index_uploads
scan_and_index_uploads('Manual')
"

# Lower relevance threshold in .env
MIN_RELEVANCE_THRESHOLD=0.2
```

### 3. Slow Query Response

#### Symptom
- Queries taking >30 seconds to respond
- Timeout errors

#### Diagnosis
```bash
# Check system resources
htop
df -h

# Check vector store size
du -sh storage/chroma_google/
```

#### Solution
```bash
# Optimize chunk size
CHUNK_SIZE=500
CHUNK_OVERLAP=100

# Add query caching
# Implement result pagination
```

## Performance Issues

### 1. High Memory Usage

#### Symptom
- Application consuming >4GB RAM
- Out of memory errors

#### Diagnosis
```bash
# Monitor memory usage
ps aux | grep python
free -h

# Profile memory usage
python -m memory_profiler secure_app.py
```

#### Solution
```python
# Implement document chunking for large files
# Add garbage collection
import gc
gc.collect()

# Reduce vector store cache size
# Use streaming for large responses
```

### 2. Slow File Processing

#### Symptom
- File uploads taking >5 minutes
- Processing timeouts

#### Diagnosis
```bash
# Check file processing pipeline
tail -f logs/app.log | grep "Processing"

# Profile document loading
python -c "
import time
from rag_google import load_docs
start = time.time()
docs = load_docs(['large_file.pdf'])
print(f'Processing time: {time.time() - start:.2f}s')
"
```

#### Solution
```python
# Implement async processing
# Add progress indicators
# Optimize PDF parsing
```

## Security Alerts

### 1. High Risk Score Alerts

#### Symptom
```
CRITICAL: High-risk query detected (score: 0.85)
```

#### Diagnosis
```bash
# Check security logs
tail -f logs/security.log | grep "CRITICAL"

# Analyze threat patterns
grep "threat_detected" logs/security.log | tail -10
```

#### Solution
```bash
# Review and adjust detection algorithms
# Implement IP blocking for repeated offenses
# Alert security team
```

### 2. Rate Limit Violations

#### Symptom
```json
{
  "detail": "Rate limit exceeded. Please try again later."
}
```

#### Diagnosis
```bash
# Check rate limit logs
grep "rate_limit_exceeded" logs/security.log

# Monitor request patterns
tail -f logs/app.log | grep "Rate limit"
```

#### Solution
```bash
# Adjust rate limits in .env
RATE_LIMIT_REQUESTS=50

# Implement user-based rate limiting
# Add request queuing
```

## Deployment Problems

### 1. Docker Container Issues

#### Symptom
```
Container exits with code 1
Port already in use
```

#### Diagnosis
```bash
# Check container logs
docker logs secure-rag-app

# Check port usage
netstat -tulpn | grep 8002

# Check container status
docker ps -a
```

#### Solution
```bash
# Stop conflicting services
sudo systemctl stop apache2
sudo systemctl stop nginx

# Use different port
APP_PORT=8003

# Rebuild container
docker-compose down
docker-compose up --build -d
```

### 2. SSL Certificate Issues

#### Symptom
```
SSL certificate verification failed
ERR_CERT_AUTHORITY_INVALID
```

#### Diagnosis
```bash
# Check certificate validity
openssl x509 -in cert.pem -text -noout

# Test SSL configuration
curl -I https://yourdomain.com
```

#### Solution
```bash
# Renew Let's Encrypt certificate
sudo certbot renew

# Update certificate paths in nginx.conf
# Restart nginx
sudo systemctl restart nginx
```

## Monitoring and Debugging

### 1. Enable Debug Logging

```python
# In secure_app.py, add debug configuration
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set environment variable
DEBUG=true
```

### 2. Health Check Endpoints

```bash
# Test application health
curl http://localhost:8002/health

# Check security statistics (admin only)
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8002/security/stats
```

### 3. Log Analysis Commands

```bash
# Monitor real-time logs
tail -f logs/app.log logs/security.log

# Search for specific errors
grep -i "error" logs/app.log | tail -20

# Analyze security events
jq '.threat_level' logs/security.log | sort | uniq -c

# Monitor query patterns
grep "query_processed" logs/security.log | \
  jq '.metadata.query' | head -10
```

### 4. Performance Monitoring

```bash
# Monitor API response times
curl -w "@curl-format.txt" -s -o /dev/null http://localhost:8002/health

# Create curl-format.txt:
echo "
     time_namelookup:  %{time_namelookup}s
        time_connect:  %{time_connect}s
     time_appconnect:  %{time_appconnect}s
    time_pretransfer:  %{time_pretransfer}s
       time_redirect:  %{time_redirect}s
  time_starttransfer:  %{time_starttransfer}s
                     ----------
          time_total:  %{time_total}s
" > curl-format.txt
```

### 5. Database Debugging

```python
# Check vector store statistics
from src.repositories.vector_store import SecureVectorStore

store = SecureVectorStore()
print(f"Collection count: {store.vectorstore._collection.count()}")
print(f"Authorized users: {len(store.authorized_users)}")

# Check document store
from src.repositories.document_store import SecureDocumentStore

doc_store = SecureDocumentStore()
files = doc_store.list_files("admin")
print(f"Stored files: {len(files)}")
```

This troubleshooting guide provides comprehensive solutions for common issues that may arise when deploying and operating the Secure RAG System.
