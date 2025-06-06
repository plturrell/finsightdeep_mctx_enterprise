from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, Depends, HTTPException, status, Body
from fastapi.security import OAuth2PasswordRequestForm
import logging
from jose import jwt, JWTError
import requests
import secrets
from passlib.context import CryptContext

from ..core.config import get_settings
from ..core.auth import (
    create_access_token, USERS_DB, OIDC_CLIENT_ID, OIDC_CLIENT_SECRET,
    OIDC_DISCOVERY_URL, JWT_SECRET_KEY, JWT_ALGORITHM, JWT_ACCESS_TOKEN_EXPIRE_MINUTES
)
from ..models.auth_models import Token, User, TokenRequest

router = APIRouter(prefix="/auth", tags=["auth"])
logger = logging.getLogger("mctx")
settings = get_settings()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Cache for OIDC metadata
oidc_metadata = None


def get_oidc_metadata():
    """
    Get OIDC provider metadata from discovery URL.
    
    Returns:
        dict: OIDC provider metadata
    """
    global oidc_metadata
    
    if not oidc_metadata and OIDC_DISCOVERY_URL:
        try:
            r = requests.get(OIDC_DISCOVERY_URL)
            r.raise_for_status()
            oidc_metadata = r.json()
        except Exception as e:
            logger.error(f"Failed to get OIDC metadata: {str(e)}")
            oidc_metadata = {}
    
    return oidc_metadata or {}


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify password against hashed password.
    
    Args:
        plain_password: Plain text password
        hashed_password: Hashed password from database
        
    Returns:
        bool: True if password matches hash
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Hash password for storage.
    
    Args:
        password: Plain text password
        
    Returns:
        str: Hashed password
    """
    return pwd_context.hash(password)


def authenticate_user(email: str, password: str) -> Optional[User]:
    """
    Authenticate user with email and password.
    
    Args:
        email: User email
        password: User password
        
    Returns:
        Optional[User]: Authenticated user or None
    """
    if email not in USERS_DB:
        return None
    
    user = USERS_DB[email]
    if not verify_password(password, user["hashed_password"]):
        return None
    
    return User(
        username=user["username"],
        email=user["email"],
        disabled=user["disabled"],
        roles=user["roles"]
    )


@router.post(
    "/token",
    response_model=Token,
    summary="Create access token",
    description="Creates a new access token using username/password credentials",
)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends()
) -> Token:
    """
    OAuth2 compatible token endpoint.
    
    Args:
        form_data: OAuth2 form data with username and password
        
    Returns:
        Token: Access token information
        
    Raises:
        HTTPException: If authentication fails
    """
    # Authenticate user
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        logger.warning(f"Failed login attempt for user: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if user is active
    if user.disabled:
        logger.warning(f"Login attempt for disabled user: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User is disabled",
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "scopes": form_data.scopes},
        expires_delta=access_token_expires,
    )
    
    logger.info(f"User {user.username} logged in successfully")
    
    # Return token
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.post(
    "/oauth/token",
    response_model=Token,
    summary="OAuth2 token endpoint",
    description="Handles various OAuth2 grant types including password, client_credentials, and authorization_code",
)
async def oauth_token(request: TokenRequest) -> Token:
    """
    OAuth 2.0 token endpoint supporting multiple grant types.
    
    Args:
        request: Token request with grant type and related parameters
        
    Returns:
        Token: Access token response
        
    Raises:
        HTTPException: If grant type is not supported or authentication fails
    """
    # Password grant type (resource owner password credentials)
    if request.grant_type == "password":
        if not request.username or not request.password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username and password required for password grant type",
            )
        
        # Authenticate user
        user = authenticate_user(request.username, request.password)
        if not user:
            logger.warning(f"Failed OAuth login attempt for user: {request.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Check if user is active
        if user.disabled:
            logger.warning(f"OAuth login attempt for disabled user: {request.username}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User is disabled",
            )
        
        # Parse scopes
        scopes = request.scope.split() if request.scope else []
        
        # Create access token
        access_token_expires = timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username, "scopes": scopes},
            expires_delta=access_token_expires,
        )
        
        # Create refresh token
        refresh_token = secrets.token_urlsafe(32)
        
        logger.info(f"User {user.username} logged in via OAuth password grant")
        
        # Return token
        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            refresh_token=refresh_token
        )
    
    # Client credentials grant type
    elif request.grant_type == "client_credentials":
        if not request.client_id or not request.client_secret:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Client ID and client secret required for client credentials grant type",
            )
        
        # In a real implementation, validate client credentials
        # For now, use a fixed valid client
        if request.client_id != "test_client" or request.client_secret != "test_secret":
            logger.warning(f"Invalid client credentials: {request.client_id}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid client credentials",
            )
        
        # Parse scopes
        scopes = request.scope.split() if request.scope else []
        
        # Create access token for client
        access_token_expires = timedelta(minutes=60)  # Longer expiry for machine-to-machine
        access_token = create_access_token(
            data={"sub": request.client_id, "scopes": scopes},
            expires_delta=access_token_expires,
        )
        
        logger.info(f"Client {request.client_id} authenticated via client credentials grant")
        
        # Return token without refresh token (as per OAuth spec)
        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=60 * 60,
        )
    
    # Authorization code grant type
    elif request.grant_type == "authorization_code":
        if not OIDC_DISCOVERY_URL or not OIDC_CLIENT_ID or not OIDC_CLIENT_SECRET:
            logger.error("OIDC not configured for authorization code grant")
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="OIDC provider not configured",
            )
            
        if not request.code or not request.redirect_uri:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Code and redirect_uri required for authorization code grant type",
            )
        
        # Get OIDC metadata
        metadata = get_oidc_metadata()
        token_endpoint = metadata.get("token_endpoint")
        
        if not token_endpoint:
            logger.error("OIDC token endpoint not found in metadata")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="OIDC configuration error",
            )
        
        # Exchange authorization code for tokens
        try:
            response = requests.post(
                token_endpoint,
                data={
                    "grant_type": "authorization_code",
                    "code": request.code,
                    "redirect_uri": request.redirect_uri,
                    "client_id": OIDC_CLIENT_ID,
                    "client_secret": OIDC_CLIENT_SECRET,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            response.raise_for_status()
            token_data = response.json()
            
            # Validate ID token
            id_token = token_data.get("id_token")
            if id_token:
                try:
                    # In production, use proper JWT validation with JWKS
                    # This is a simplified example
                    id_token_data = jwt.decode(
                        id_token,
                        options={"verify_signature": False}
                    )
                    logger.info(f"User authenticated via OIDC: {id_token_data.get('sub')}")
                except JWTError as e:
                    logger.error(f"Invalid ID token: {str(e)}")
            
            # Return token response from OIDC provider
            return Token(
                access_token=token_data["access_token"],
                token_type=token_data["token_type"],
                expires_in=token_data.get("expires_in", 3600),
                refresh_token=token_data.get("refresh_token")
            )
            
        except Exception as e:
            logger.error(f"OIDC token exchange failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to exchange authorization code for token",
            )
    
    # Refresh token grant type
    elif request.grant_type == "refresh_token":
        if not request.refresh_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Refresh token required for refresh_token grant type",
            )
        
        # In a real implementation, validate refresh token against a database
        # For now, just issue a new access token
        # This is a simplified example and not secure for production
        
        try:
            # Create a new access token
            access_token_expires = timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
            access_token = create_access_token(
                data={"sub": "refresh_user", "scopes": []},
                expires_delta=access_token_expires,
            )
            
            # Create a new refresh token
            new_refresh_token = secrets.token_urlsafe(32)
            
            logger.info(f"Token refreshed successfully")
            
            return Token(
                access_token=access_token,
                token_type="bearer",
                expires_in=JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
                refresh_token=new_refresh_token
            )
            
        except Exception as e:
            logger.error(f"Token refresh failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to refresh token",
            )
    
    # Unsupported grant type
    else:
        logger.warning(f"Unsupported grant type: {request.grant_type}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported grant type: {request.grant_type}",
        )