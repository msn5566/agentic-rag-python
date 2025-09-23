"""
Secure configuration management for RAG application.
"""
import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import validator
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with validation and security."""
    
    # Google API Configuration
    google_api_key: str
    
    # Application Configuration
    app_host: str = "127.0.0.1"
    app_port: int = 8002
    app_debug: bool = False
    app_environment: str = "production"
    
    # Security Configuration
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    
    # CORS Configuration
    allowed_origins: str = "http://localhost:3000"
    allowed_methods: str = "GET,POST,PUT,DELETE"
    allowed_headers: str = "*"
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    
    # File Upload Configuration
    max_file_size_mb: int = 50
    allowed_file_extensions: str = ".pdf,.docx,.txt"
    max_files_per_request: int = 10
    
    # Database Configuration
    persist_dir: str = "storage/chroma_google"
    collection_name: str = "secure_rag_collection"
    
    # Model Configuration
    google_llm_model: str = "gemini-1.5-flash-latest"
    google_embed_model: str = "models/text-embedding-004"
    chunk_size: int = 1200
    chunk_overlap: int = 150
    retrieval_k: int = 4
    min_relevance: float = 0.3
    
    # Logging Configuration
    log_level: str = "INFO"
    log_file: str = "logs/app.log"
    
    @validator('google_api_key')
    def validate_api_key(cls, v):
        if not v or v == "your-google-api-key-here":
            raise ValueError("Google API key must be set and not be the placeholder value")
        return v
    
    @validator('jwt_secret_key')
    def validate_jwt_secret(cls, v):
        if not v or v == "your-super-secret-jwt-key-here-change-this":
            raise ValueError("JWT secret key must be set and not be the placeholder value")
        if len(v) < 32:
            raise ValueError("JWT secret key must be at least 32 characters long")
        return v
    
    # Helper methods to parse comma-separated strings
    def get_allowed_origins_list(self) -> List[str]:
        """Parse allowed_origins string into list."""
        return [origin.strip() for origin in self.allowed_origins.split(',') if origin.strip()]
    
    def get_allowed_methods_list(self) -> List[str]:
        """Parse allowed_methods string into list."""
        return [method.strip() for method in self.allowed_methods.split(',') if method.strip()]
    
    def get_allowed_headers_list(self) -> List[str]:
        """Parse allowed_headers string into list."""
        if self.allowed_headers == "*":
            return ["*"]
        return [header.strip() for header in self.allowed_headers.split(',') if header.strip()]
    
    def get_allowed_file_extensions_list(self) -> List[str]:
        """Parse allowed_file_extensions string into list."""
        return [ext.strip() for ext in self.allowed_file_extensions.split(',') if ext.strip()]
    
    @property
    def max_file_size_bytes(self) -> int:
        """Convert MB to bytes."""
        return self.max_file_size_mb * 1024 * 1024
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
