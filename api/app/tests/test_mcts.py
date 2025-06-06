import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from ..models.mcts_models import MCTSRequest


def test_search_endpoint_success(
    test_client, sample_mcts_request, mock_mctx, mock_hana_manager
):
    """Test successful search request."""
    # Make request to search endpoint
    response = test_client.post(
        "/api/v1/mcts/search",
        json=sample_mcts_request.dict()
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    
    # Check expected data structure
    assert "action" in data
    assert "action_weights" in data
    assert "search_statistics" in data
    assert "duration_ms" in data["search_statistics"]
    
    # Verify mocks were called correctly
    mock_mctx["gumbel_muzero_policy"].assert_called_once()
    mock_hana_manager.save_search_history.assert_called_once()


def test_search_with_invalid_search_type(test_client, sample_mcts_request):
    """Test search with invalid search type."""
    # Modify request to have invalid search type
    invalid_request = sample_mcts_request.dict()
    invalid_request["search_type"] = "invalid_type"
    
    # Make request
    response = test_client.post(
        "/api/v1/mcts/search",
        json=invalid_request
    )
    
    # Check response
    assert response.status_code == 400
    data = response.json()
    assert "Invalid search type" in data["message"]


def test_search_with_batch_size_too_large(
    test_client, sample_mcts_request, mock_settings
):
    """Test search with batch size exceeding limit."""
    # Modify request to have large batch size
    invalid_request = sample_mcts_request.dict()
    invalid_request["root_input"]["batch_size"] = mock_settings.MAX_BATCH_SIZE + 1
    
    # Make request
    response = test_client.post(
        "/api/v1/mcts/search",
        json=invalid_request
    )
    
    # Check response
    assert response.status_code == 400
    data = response.json()
    assert "Batch size exceeds maximum allowed" in data["message"]


def test_search_with_auth(test_client, sample_mcts_request, auth_headers):
    """Test search with authentication."""
    # Make request with auth headers
    response = test_client.post(
        "/api/v1/mcts/search",
        json=sample_mcts_request.dict(),
        headers=auth_headers
    )
    
    # Check response
    assert response.status_code == 200


def test_health_endpoint(test_client):
    """Test health check endpoint."""
    response = test_client.get("/api/v1/mcts/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


@pytest.mark.parametrize("error_type", [
    "ValueError", 
    "KeyError",
    "Exception"
])
def test_search_with_internal_error(
    test_client, sample_mcts_request, mock_mctx, error_type
):
    """Test search with internal server error."""
    # Make mock raise an exception
    error_class = globals()[error_type]
    mock_mctx["gumbel_muzero_policy"].side_effect = error_class("Test error")
    
    # Make request
    response = test_client.post(
        "/api/v1/mcts/search",
        json=sample_mcts_request.dict()
    )
    
    # Check response
    assert response.status_code == 500
    data = response.json()
    assert "Failed to execute MCTS search" in data["message"]


def test_rate_limiting(test_client, sample_mcts_request):
    """Test rate limiting middleware."""
    # Patch rate limiter to simulate rate limit exceeded
    with patch("app.core.rate_limit.RateLimiter.check_rate_limit") as mock_rate_limit:
        # First call allowed, second call denied
        mock_rate_limit.side_effect = [
            (True, {"limit": 10, "remaining": 9, "reset": 0}),
            (False, {"limit": 10, "remaining": 0, "reset": 0})
        ]
        
        # First request should succeed
        response1 = test_client.post(
            "/api/v1/mcts/search",
            json=sample_mcts_request.dict()
        )
        assert response1.status_code == 200
        
        # Second request should be rate limited
        response2 = test_client.post(
            "/api/v1/mcts/search",
            json=sample_mcts_request.dict()
        )
        assert response2.status_code == 429
        
        # Check rate limit headers
        assert "Retry-After" in response2.headers