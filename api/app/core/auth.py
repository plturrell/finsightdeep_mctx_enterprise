from fastapi import Depends, HTTPException, status, Security, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2AuthorizationCodeBearer, APIKeyHeader
from jose import JWTError, jwt
from typing import Dict, Optional, List, Union, Any
import time
import logging
import os
import json
import requests
from datetime import datetime, timedelta
from urllib.parse import urlencode

from pydantic import BaseModel, EmailStr, Field

from ..core.config import get_settings
from ..models.auth_models import TokenData, User
from ..core.secrets import SecretManager, HanaSecrets

logger = logging.getLogger("mctx")
settings = get_settings()

# Security schemes
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/token")
api_key_header = APIKeyHeader(name=settings.API_KEY_NAME, auto_error=False)

# JWT settings
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# OIDC settings - initialized from secrets
OIDC_CLIENT_ID = None
OIDC_CLIENT_SECRET = None
OIDC_DISCOVERY_URL = None
OIDC_AUDIENCE = None

# Mock user database for development only
# In production, this would be replaced with a proper user database
USERS_DB = {
    "admin@example.com": {
        "username": "admin",
        "email": "admin@example.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "password"
        "disabled": False,
        "roles": ["admin"]
    }
}


class OIDCConfig:
    """Configuration for OIDC integration."""
    
    _instance = None
    _metadata = None
    _jwks = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize OIDC configuration."""
        global OIDC_CLIENT_ID, OIDC_CLIENT_SECRET, OIDC_DISCOVERY_URL, OIDC_AUDIENCE
        
        # Get configuration from environment or secrets
        self.client_id = os.getenv("OIDC_CLIENT_ID") or SecretManager.get_secret("OIDC_CLIENT_ID") or HanaSecrets.get_client_credentials()["client_id"]
        self.client_secret = os.getenv("OIDC_CLIENT_SECRET") or SecretManager.get_secret("OIDC_CLIENT_SECRET") or HanaSecrets.get_client_credentials()["client_secret"]
        self.discovery_url = os.getenv("OIDC_DISCOVERY_URL") or SecretManager.get_secret("OIDC_DISCOVERY_URL") or HanaSecrets.get_oauth_urls()["auth_url"].replace("/oauth/authorize", "/.well-known/openid-configuration")
        self.audience = os.getenv("OIDC_AUDIENCE") or SecretManager.get_secret("OIDC_AUDIENCE") or "api://default"
        
        # Update global variables
        OIDC_CLIENT_ID = self.client_id
        OIDC_CLIENT_SECRET = self.client_secret
        OIDC_DISCOVERY_URL = self.discovery_url
        OIDC_AUDIENCE = self.audience
        
        # Load metadata
        self._load_metadata()
    
    def _load_metadata(self):
        """Load OIDC metadata from discovery URL."""
        if not self.discovery_url:
            logger.warning("No OIDC discovery URL configured")
            return
        
        try:
            response = requests.get(self.discovery_url)
            response.raise_for_status()
            self._metadata = response.json()
            
            # Load JWKS
            if "jwks_uri" in self._metadata:
                jwks_response = requests.get(self._metadata["jwks_uri"])
                jwks_response.raise_for_status()
                self._jwks = jwks_response.json()
            
            logger.info("OIDC configuration loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load OIDC metadata: {str(e)}")
            self._metadata = {}
            self._jwks = {}
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get OIDC provider metadata."""
        if self._metadata is None:
            self._load_metadata()
        return self._metadata or {}
    
    def get_jwks(self) -> Dict[str, Any]:
        """Get OIDC provider JWKS."""
        if self._jwks is None and self._metadata and "jwks_uri" in self._metadata:
            try:
                response = requests.get(self._metadata["jwks_uri"])
                response.raise_for_status()
                self._jwks = response.json()
            except Exception as e:
                logger.error(f"Failed to load JWKS: {str(e)}")
                self._jwks = {}
        return self._jwks or {}
    
    def get_authorization_url(self, redirect_uri: str, state: str, scope: str = "openid profile email") -> str:
        """
        Get OIDC authorization URL.
        
        Args:
            redirect_uri: Redirect URI
            state: State parameter for CSRF protection
            scope: OAuth scopes
            
        Returns:
            Authorization URL
        """
        metadata = self.get_metadata()
        auth_endpoint = metadata.get("authorization_endpoint") or HanaSecrets.get_oauth_urls()["auth_url"]
        
        if not auth_endpoint:
            raise ValueError("No authorization endpoint found in OIDC metadata")
        
        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "scope": scope,
            "redirect_uri": redirect_uri,
            "state": state,
        }
        
        return f"{auth_endpoint}?{urlencode(params)}"
    
    def get_token_endpoint(self) -> str:
        """Get OIDC token endpoint."""
        metadata = self.get_metadata()
        return metadata.get("token_endpoint") or HanaSecrets.get_oauth_urls()["token_url"]
    
    def get_userinfo_endpoint(self) -> Optional[str]:
        """Get OIDC userinfo endpoint."""
        metadata = self.get_metadata()
        return metadata.get("userinfo_endpoint")
    
    def exchange_code_for_token(self, code: str, redirect_uri: str) -> Dict[str, Any]:
        """
        Exchange authorization code for token.
        
        Args:
            code: Authorization code
            redirect_uri: Redirect URI
            
        Returns:
            Token response
            
        Raises:
            HTTPException: If token exchange fails
        """
        token_endpoint = self.get_token_endpoint()
        if not token_endpoint:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="OIDC token endpoint not configured"
            )
        
        try:
            response = requests.post(
                token_endpoint,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": redirect_uri,
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to exchange code for token: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to exchange code for token: {str(e)}"
            )
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """
        Verify and decode JWT token.
        
        Args:
            token: JWT token
            
        Returns:
            Decoded token claims
            
        Raises:
            JWTError: If token verification fails
        """
        try:
            from jwcrypto import jwk, jwt as jwt_lib
            import json
            
            # Get JWKS
            jwks_data = self.get_jwks()
            if not jwks_data:
                # Fall back to local verification
                return jwt.decode(
                    token,
                    JWT_SECRET_KEY,
                    algorithms=[JWT_ALGORITHM],
                    audience=OIDC_AUDIENCE
                )
            
            # Create JWK keyset
            keyset = jwk.JWKSet.from_json(json.dumps(jwks_data))
            
            # Create JWT object
            token_obj = jwt_lib.JWT(jwt=token, algs=["RS256", "ES256"])
            
            # Verify token
            token_obj.validate(keyset)
            
            # Get claims
            claims = json.loads(token_obj.claims)
            
            # Verify audience if configured
            if OIDC_AUDIENCE and "aud" in claims:
                if OIDC_AUDIENCE not in claims["aud"]:
                    raise JWTError("Invalid audience")
            
            # Verify expiry
            if "exp" in claims:
                exp_timestamp = claims["exp"]
                if datetime.utcfromtimestamp(exp_timestamp) < datetime.utcnow():
                    raise JWTError("Token expired")
            
            return claims
        
        except Exception as e:
            logger.error(f"Token verification failed: {str(e)}")
            raise JWTError(f"Token verification failed: {str(e)}")


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Data to encode in the token
        expires_delta: Optional token expiration time
        
    Returns:
        JWT token string
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    
    return encoded_jwt


async def get_current_user_from_token(token: str = Depends(oauth2_scheme)) -> User:
    """
    Validate and extract user information from JWT token.
    
    Args:
        token: JWT token
        
    Returns:
        User object
        
    Raises:
        HTTPException: If token validation fails
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Try to verify with OIDC first
        try:
            oidc_config = OIDCConfig()
            payload = oidc_config.verify_token(token)
        except Exception:
            # Fall back to local JWT verification
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    # Check if user is in local database
    for email, user_data in USERS_DB.items():
        if user_data["username"] == token_data.username:
            return User(**user_data)
    
    # User not in database, but token is valid
    # Create a new user with basic access
    return User(
        username=token_data.username,
        email=f"{token_data.username}@example.com",  # Placeholder
        disabled=False,
        roles=["user"]
    )


async def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> Optional[str]:
    """
    Verify API key from request header.
    
    Args:
        api_key: API key from request header
        
    Returns:
        API key if valid, None otherwise
        
    Raises:
        HTTPException: If API key is required but invalid
    """
    if settings.API_KEY_REQUIRED:
        if api_key != settings.API_KEY:
            logger.warning("Invalid API key attempt")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid or missing API key",
            )
        return api_key
    return None


async def get_current_user(
    request: Request,
    token: Optional[str] = Depends(oauth2_scheme),
    api_key: Optional[str] = Depends(verify_api_key)
) -> Optional[User]:
    """
    Get current user from either JWT token or API key.
    
    This function provides a unified authentication mechanism supporting
    both OAuth tokens and API keys.
    
    Args:
        request: HTTP request
        token: Optional JWT token
        api_key: Optional API key
        
    Returns:
        User object if authenticated, None otherwise
        
    Raises:
        HTTPException: If authentication fails
    """
    # If API key authentication is used and valid, return a system user
    if api_key and not token:
        user = User(
            username="api_client",
            email="api@system",
            disabled=False,
            roles=["api"]
        )
        # Store user in request state
        request.state.user = user
        return user
    
    # If token is provided, validate it
    if token:
        user = await get_current_user_from_token(token)
        # Store user in request state
        request.state.user = user
        return user
    
    # If neither token nor API key is provided, and API key is not required
    if not settings.API_KEY_REQUIRED:
        request.state.user = None
        return None
    
    # Otherwise, authentication is required but not provided
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_user_id(request: Request, current_user: Optional[User] = Depends(get_current_user)) -> Optional[str]:
    """
    Extract user ID from authenticated user.
    
    Args:
        request: HTTP request
        current_user: Authenticated user or None
        
    Returns:
        User ID string or None
    """
    if current_user:
        return current_user.username
    return None


class RoleChecker:
    """
    Role-based access control checker.
    """
    def __init__(self, required_roles: List[str]):
        self.required_roles = required_roles
    
    def __call__(self, request: Request, user: User = Depends(get_current_user)) -> User:
        """
        Check if user has required roles.
        
        Args:
            request: HTTP request
            user: Authenticated user
            
        Returns:
            User if roles are sufficient
            
        Raises:
            HTTPException: If user lacks required roles
        """
        if not user:
            logger.warning("Unauthenticated access attempt to protected resource")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        for role in self.required_roles:
            if role not in user.roles:
                logger.warning(f"User {user.username} attempted to access resource requiring role {role}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Role '{role}' required",
                )
        return user


def get_oauth_credentials() -> Dict[str, str]:
    """
    Get OAuth client credentials.
    
    Returns:
        Dict with client_id and client_secret
    """
    # Initialize OIDC config if needed
    oidc_config = OIDCConfig()
    
    return {
        "client_id": oidc_config.client_id,
        "client_secret": oidc_config.client_secret,
    }