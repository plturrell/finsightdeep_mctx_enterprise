from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
import jwt
from jwt.exceptions import InvalidTokenError
from typing import Optional
from datetime import datetime, timedelta
import os

from .config import get_settings
from ..models.auth_models import User

settings = get_settings()

# Security schemes
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)

def create_access_token(data: dict, expires_delta: timedelta = None):
    """
    Create a JWT access token
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.api_secret_key, 
        algorithm="HS256"
    )
    return encoded_jwt

def get_current_user(
    api_key: str = Depends(api_key_header),
    bearer_auth: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)
) -> User:
    """
    Validate API key or JWT token and return the current user.
    This is a simplified version for the test environment.
    """
    # For testing - bypass auth in dev mode
    if settings.debug and not settings.api_key_required:
        return User(username="test_user", email="test@example.com", is_active=True, is_admin=True)
    
    # Check API key first
    if api_key:
        if api_key == settings.api_key:
            return User(username="api_user", email="api@example.com", is_active=True, is_admin=True)
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "ApiKey"},
            )
    
    # Then check JWT token
    if bearer_auth:
        try:
            payload = jwt.decode(
                bearer_auth.credentials, 
                settings.api_secret_key, 
                algorithms=["HS256"]
            )
            username = payload.get("sub")
            if username:
                return User(
                    username=username,
                    email=payload.get("email", f"{username}@example.com"),
                    is_active=True,
                    is_admin=payload.get("is_admin", False),
                )
        except InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    # If neither auth method succeeded
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer or ApiKey"},
    )