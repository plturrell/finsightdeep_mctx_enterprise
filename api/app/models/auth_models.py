from typing import List, Optional
from pydantic import BaseModel, EmailStr, Field


class TokenData(BaseModel):
    """
    JWT token data model.
    """
    username: Optional[str] = None
    scopes: List[str] = []


class Token(BaseModel):
    """
    OAuth token response model.
    """
    access_token: str
    token_type: str
    expires_in: int
    refresh_token: Optional[str] = None


class User(BaseModel):
    """
    User model.
    """
    username: str
    email: str
    disabled: bool = False
    roles: List[str] = []


class UserInDB(User):
    """
    User model with password hash for database storage.
    """
    hashed_password: str


class TokenRequest(BaseModel):
    """
    OAuth token request model.
    """
    grant_type: str = Field(..., description="OAuth2 grant type")
    username: Optional[str] = Field(None, description="Username for password grant")
    password: Optional[str] = Field(None, description="Password for password grant")
    refresh_token: Optional[str] = Field(None, description="Refresh token for refresh_token grant")
    scope: Optional[str] = Field(None, description="Requested scopes")
    client_id: Optional[str] = Field(None, description="OAuth client ID")
    client_secret: Optional[str] = Field(None, description="OAuth client secret")
    code: Optional[str] = Field(None, description="Authorization code for code grant")
    redirect_uri: Optional[str] = Field(None, description="Redirect URI for code grant")