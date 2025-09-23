# Secure RAG System - Deployment Guide

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Production Configuration](#production-configuration)
6. [Monitoring Setup](#monitoring-setup)
7. [Backup and Recovery](#backup-and-recovery)
8. [Scaling Strategies](#scaling-strategies)

## Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+), Windows 10+, macOS 10.15+
- **Python**: 3.9 or higher
- **Memory**: Minimum 4GB RAM, Recommended 8GB+
- **Storage**: Minimum 10GB free space
- **Network**: HTTPS-capable domain (for production)

### Required Services
- **Google Cloud Platform**: Gemini API access
- **SSL Certificate**: For HTTPS in production
- **Reverse Proxy**: Nginx or Apache (recommended)
- **Process Manager**: PM2, systemd, or Docker

## Environment Setup

### 1. Clone and Setup Repository
```bash
# Clone repository
git clone <your-repo-url>
cd agentic-rag-python

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit environment variables
nano .env
```

### 3. Required Environment Variables
```bash
# Core Configuration
GOOGLE_API_KEY=your_google_api_key_here
JWT_SECRET_KEY=your_super_secure_jwt_secret_key_minimum_32_chars

# Server Configuration
APP_HOST=0.0.0.0
APP_PORT=8002
DEBUG=false

# Security Configuration
ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
ALLOWED_METHODS=GET,POST,PUT,DELETE
ALLOWED_HEADERS=Authorization,Content-Type

# Rate Limiting
RATE_LIMIT_REQUESTS=30
RATE_LIMIT_WINDOW=60

# File Upload Limits
MAX_FILE_SIZE_MB=50
MAX_FILES_PER_REQUEST=10
ALLOWED_FILE_EXTENSIONS=.pdf,.docx,.txt

# Model Configuration
EMBEDDING_MODEL=models/embedding-001
LLM_MODEL=gemini-pro
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MIN_RELEVANCE_THRESHOLD=0.3

# Storage Configuration
VECTOR_STORE_PERSIST_DIR=./storage/chroma_google
UPLOADS_DIR=./uploads
QUARANTINE_DIR=./quarantine
```

## Docker Deployment

### 1. Create Dockerfile

```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY ../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY .. .

# Create necessary directories
RUN mkdir -p uploads quarantine storage/chroma_google logs

# Set permissions
RUN chmod -R 755 uploads quarantine storage logs

# Expose port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Run application
CMD ["python", "secure_app.py"]
```

### 2. Create Docker Compose
```yaml
version: '3.8'

services:
  secure-rag:
    build: .
    container_name: secure-rag-app
    ports:
      - "8002:8002"
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
      - APP_HOST=0.0.0.0
      - APP_PORT=8002
    volumes:
      - ./uploads:/app/uploads
      - ./storage:/app/storage
      - ./quarantine:/app/quarantine
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8002/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  nginx:
    image: nginx:alpine
    container_name: secure-rag-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - secure-rag
    restart: unless-stopped

volumes:
  uploads:
  storage:
  quarantine:
  logs:
```

### 3. Nginx Configuration
```nginx
events {
    worker_connections 1024;
}

http {
    upstream secure_rag {
        server secure-rag:8002;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=auth:10m rate=5r/m;

    server {
        listen 80;
        server_name yourdomain.com www.yourdomain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name yourdomain.com www.yourdomain.com;

        # SSL Configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
        ssl_prefer_server_ciphers off;

        # Security Headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
        add_header Content-Security-Policy "default-src 'self'";

        # File upload size limit
        client_max_body_size 100M;

        # API endpoints
        location / {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://secure_rag;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Auth endpoints with stricter rate limiting
        location /auth/ {
            limit_req zone=auth burst=5 nodelay;
            proxy_pass http://secure_rag;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

### 4. Deploy with Docker
```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f secure-rag

# Scale application (if needed)
docker-compose up -d --scale secure-rag=3
```

## Cloud Deployment

### AWS Deployment

#### 1. EC2 Instance Setup
```bash
# Launch EC2 instance (t3.medium or larger)
# Security Group: Allow ports 22, 80, 443

# Connect to instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

#### 2. Application Deployment
```bash
# Clone repository
git clone <your-repo-url>
cd agentic-rag-python

# Setup environment
cp .env.example .env
nano .env  # Configure with production values

# Deploy with Docker Compose
docker-compose up -d
```

#### 3. SSL Certificate (Let's Encrypt)
```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### Google Cloud Platform Deployment

#### 1. Cloud Run Deployment
```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/secure-rag:$COMMIT_SHA', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/secure-rag:$COMMIT_SHA']
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: gcloud
    args:
      - 'run'
      - 'deploy'
      - 'secure-rag'
      - '--image'
      - 'gcr.io/$PROJECT_ID/secure-rag:$COMMIT_SHA'
      - '--region'
      - 'us-central1'
      - '--platform'
      - 'managed'
      - '--allow-unauthenticated'
```

#### 2. Deploy to Cloud Run
```bash
# Build and deploy
gcloud builds submit --config cloudbuild.yaml

# Set environment variables
gcloud run services update secure-rag \
  --set-env-vars GOOGLE_API_KEY=$GOOGLE_API_KEY \
  --set-env-vars JWT_SECRET_KEY=$JWT_SECRET_KEY \
  --region us-central1
```

## Production Configuration

### 1. Security Hardening
```bash
# Create dedicated user
sudo useradd -m -s /bin/bash ragapp
sudo usermod -aG docker ragapp

# Set file permissions
sudo chown -R ragapp:ragapp /app
sudo chmod 750 /app
sudo chmod 640 /app/.env

# Configure firewall
sudo ufw enable
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
```

### 2. Process Management (systemd)
```ini
# /etc/systemd/system/secure-rag.service
[Unit]
Description=Secure RAG Application
After=network.target

[Service]
Type=simple
User=ragapp
WorkingDirectory=/app
Environment=PATH=/app/.venv/bin
ExecStart=/app/.venv/bin/python secure_app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable secure-rag
sudo systemctl start secure-rag
sudo systemctl status secure-rag
```

### 3. Log Rotation
```bash
# /etc/logrotate.d/secure-rag
/app/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 ragapp ragapp
    postrotate
        systemctl reload secure-rag
    endscript
}
```

## Monitoring Setup

### 1. Health Monitoring Script
```bash
#!/bin/bash
# health_check.sh

ENDPOINT="https://yourdomain.com/health"
SLACK_WEBHOOK="your_slack_webhook_url"

response=$(curl -s -o /dev/null -w "%{http_code}" $ENDPOINT)

if [ $response -ne 200 ]; then
    curl -X POST -H 'Content-type: application/json' \
        --data '{"text":"🚨 Secure RAG System is DOWN! HTTP Status: '$response'"}' \
        $SLACK_WEBHOOK
fi
```

### 2. Monitoring with Prometheus
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'secure-rag'
    static_configs:
      - targets: ['localhost:8002']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

### 3. Log Monitoring with ELK Stack
```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.8.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    ports:
      - "9200:9200"

  logstash:
    image: docker.elastic.co/logstash/logstash:8.8.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
      - ./logs:/logs
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:8.8.0
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch
```

## Backup and Recovery

### 1. Automated Backup Script
```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups/secure-rag"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR/$DATE

# Backup vector store
tar -czf $BACKUP_DIR/$DATE/vector_store.tar.gz storage/

# Backup uploads
tar -czf $BACKUP_DIR/$DATE/uploads.tar.gz uploads/

# Backup configuration
cp .env $BACKUP_DIR/$DATE/

# Backup logs
tar -czf $BACKUP_DIR/$DATE/logs.tar.gz logs/

# Clean old backups (keep 30 days)
find $BACKUP_DIR -type d -mtime +30 -exec rm -rf {} \;

echo "Backup completed: $BACKUP_DIR/$DATE"
```

### 2. Recovery Procedure
```bash
#!/bin/bash
# restore.sh

BACKUP_DATE=$1
BACKUP_DIR="/backups/secure-rag/$BACKUP_DATE"

if [ -z "$BACKUP_DATE" ]; then
    echo "Usage: $0 <backup_date>"
    exit 1
fi

# Stop application
docker-compose down

# Restore vector store
tar -xzf $BACKUP_DIR/vector_store.tar.gz

# Restore uploads
tar -xzf $BACKUP_DIR/uploads.tar.gz

# Restore configuration
cp $BACKUP_DIR/.env .

# Start application
docker-compose up -d

echo "Recovery completed from backup: $BACKUP_DATE"
```

## Scaling Strategies

### 1. Horizontal Scaling with Load Balancer
```yaml
# docker-compose.scale.yml
version: '3.8'

services:
  secure-rag:
    build: .
    deploy:
      replicas: 3
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx-lb.conf:/etc/nginx/nginx.conf
    depends_on:
      - secure-rag
```

### 2. Database Scaling
```python
# For high-volume deployments, consider:
# - Separate vector database (Pinecone, Weaviate)
# - Redis for caching
# - PostgreSQL for metadata
# - Distributed file storage (S3, GCS)
```

### 3. Performance Optimization
```bash
# Optimize Python runtime
export PYTHONOPTIMIZE=1
export PYTHONDONTWRITEBYTECODE=1

# Increase worker processes
gunicorn secure_app:app -w 4 -k uvicorn.workers.UvicornWorker
```

This deployment guide provides comprehensive instructions for deploying the Secure RAG System in various environments with proper security, monitoring, and scaling considerations.
