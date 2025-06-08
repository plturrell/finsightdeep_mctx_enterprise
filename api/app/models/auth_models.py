from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class User(BaseModel):
    """User model for authentication."""
    username: str
    email: str
    is_active: bool = True
    is_admin: bool = False
    
    class Config:
        schema_extra = {
            "example": {
                "username": "johndoe",
                "email": "john@example.com",
                "is_active": True,
                "is_admin": False
            }
        }


class Token(BaseModel):
    """Token model for authentication."""
    access_token: str
    token_type: str = "bearer"
    
    class Config:
        schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer"
            }
        }


class TokenData(BaseModel):
    """Data extracted from token."""
    username: Optional[str] = None
    email: Optional[str] = None
    is_admin: bool = False
    exp: Optional[datetime] = None