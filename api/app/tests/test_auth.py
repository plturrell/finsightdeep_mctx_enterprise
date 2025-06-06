import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from ..core.auth import USERS_DB
from ..models.auth_models import Token


def test_token_endpoint_valid_credentials(test_client):
    """Test token endpoint with valid credentials."""
    # Valid credentials from the mock USERS_DB
    response = test_client.post(
        "/api/v1/auth/token",
        data={
            "username": "admin@example.com",
            "password": "password",
            "scope": "read:mcts write:mcts",
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert "expires_in" in data


def test_token_endpoint_invalid_credentials(test_client):
    """Test token endpoint with invalid credentials."""
    response = test_client.post(
        "/api/v1/auth/token",
        data={
            "username": "admin@example.com",
            "password": "wrong_password",
            "scope": "",
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    
    assert response.status_code == 401
    data = response.json()
    assert "Incorrect username or password" in data.get("detail", "")


def test_token_endpoint_disabled_user(test_client):
    """Test token endpoint with disabled user."""
    # Temporarily modify the USERS_DB for this test
    original_status = USERS_DB["admin@example.com"]["disabled"]
    USERS_DB["admin@example.com"]["disabled"] = True
    
    try:
        response = test_client.post(
            "/api/v1/auth/token",
            data={
                "username": "admin@example.com",
                "password": "password",
                "scope": "",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "User is disabled" in data.get("detail", "")
    finally:
        # Restore original status
        USERS_DB["admin@example.com"]["disabled"] = original_status


def test_oauth_token_password_grant(test_client):
    """Test OAuth token endpoint with password grant type."""
    response = test_client.post(
        "/api/v1/auth/oauth/token",
        json={
            "grant_type": "password",
            "username": "admin@example.com",
            "password": "password",
            "scope": "read:mcts write:mcts"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert "refresh_token" in data


def test_oauth_token_client_credentials_grant(test_client):
    """Test OAuth token endpoint with client credentials grant type."""
    # Mock client validation
    with patch("app.routers.auth.request.client_id", "test_client"), \
         patch("app.routers.auth.request.client_secret", "test_secret"):
        
        response = test_client.post(
            "/api/v1/auth/oauth/token",
            json={
                "grant_type": "client_credentials",
                "client_id": "test_client",
                "client_secret": "test_secret",
                "scope": "api:read api:write"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert "refresh_token" not in data  # No refresh token for client credentials


def test_oauth_token_invalid_grant_type(test_client):
    """Test OAuth token endpoint with invalid grant type."""
    response = test_client.post(
        "/api/v1/auth/oauth/token",
        json={
            "grant_type": "invalid_grant",
            "username": "admin@example.com",
            "password": "password"
        }
    )
    
    assert response.status_code == 400
    data = response.json()
    assert "Unsupported grant type" in data.get("detail", "")


def test_protected_endpoint_with_valid_token(test_client, auth_headers):
    """Test accessing a protected endpoint with a valid token."""
    # Create a test endpoint that requires authentication
    from ..main import app
    from fastapi import Depends
    from ..core.auth import get_current_user
    
    @app.get("/api/v1/test/protected")
    def protected_endpoint(current_user = Depends(get_current_user)):
        return {"message": "Access granted", "user": current_user.username}
    
    # Test with valid token
    response = test_client.get("/api/v1/test/protected", headers=auth_headers)
    
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Access granted"
    assert data["user"] == "test_user"


def test_protected_endpoint_without_token(test_client):
    """Test accessing a protected endpoint without a token."""
    # Use the test endpoint from previous test
    response = test_client.get("/api/v1/test/protected")
    
    # With API_KEY_REQUIRED=False in test settings, this should return 401
    assert response.status_code == 401
    data = response.json()
    assert "Authentication required" in data.get("detail", "")


def test_protected_endpoint_with_api_key(test_client):
    """Test accessing a protected endpoint with API key."""
    # Use the test endpoint from previous test
    response = test_client.get(
        "/api/v1/test/protected",
        headers={"X-API-Key": "test_api_key"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Access granted"
    assert data["user"] == "api_client"