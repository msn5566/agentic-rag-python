"""
Authentication and authorization middleware for RAG application.
"""
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from src.config import settings
import logging

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT token handler
security = HTTPBearer()


class AuthManager:
    """Handles authentication and authorization."""
    
    def __init__(self):
        self.secret_key = settings.jwt_secret_key
        self.algorithm = settings.jwt_algorithm
        self.access_token_expire_minutes = settings.jwt_access_token_expire_minutes
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash a password."""
        return pwd_context.hash(password)
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire, "iat": datetime.utcnow()})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.JWTError as e:
            logger.warning(f"JWT validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )


# Global auth manager instance
auth_manager = AuthManager()


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Dependency to get current authenticated user."""
    token = credentials.credentials
    payload = auth_manager.verify_token(token)
    
    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return payload


async def get_current_active_user(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Dependency to get current active user."""
    if not current_user.get("active", True):
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self):
        self.requests = {}
        self.window_size = settings.rate_limit_window
        self.max_requests = settings.rate_limit_requests
    
    def is_allowed(self, client_ip: str) -> bool:
        """Check if request is allowed based on rate limiting."""
        now = datetime.utcnow()
        
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        
        # Remove old requests outside the window
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if (now - req_time).total_seconds() < self.window_size
        ]
        
        # Check if under limit
        if len(self.requests[client_ip]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[client_ip].append(now)
        return True


# Global rate limiter instance
rate_limiter = RateLimiter()


async def check_rate_limit(request: Request):
    """Dependency to check rate limiting."""
    client_ip = request.client.host
    
    if not rate_limiter.is_allowed(client_ip):
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )


def require_auth(require_active: bool = True):
    """Decorator to require authentication."""
    def decorator(func):
        if require_active:
            func.__annotations__["current_user"] = Depends(get_current_active_user)
        else:
            func.__annotations__["current_user"] = Depends(get_current_user)
        return func
    return decorator
