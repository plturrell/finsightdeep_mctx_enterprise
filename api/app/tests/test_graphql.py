import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from ..models.mcts_models import MCTSRequest


def test_graphql_health_query(test_client):
    """Test GraphQL health query."""
    query = """
    query {
        health
    }
    """
    
    response = test_client.post(
        "/api/v1/graphql",
        json={"query": query}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert "health" in data["data"]
    assert "status" in json.loads(data["data"]["health"])
    assert json.loads(data["data"]["health"])["status"] == "ok"


def test_graphql_run_search_mutation(test_client, mock_mctx, mock_hana_manager):
    """Test GraphQL run_search mutation."""
    mutation = """
    mutation ($rootInput: RootInputType!, $searchParams: SearchParamsInput!, $searchType: String!) {
        runSearch(rootInput: $rootInput, searchParams: $searchParams, searchType: $searchType) {
            action
            actionWeights
            searchStatistics {
                durationMs
                numExpandedNodes
                maxDepthReached
            }
        }
    }
    """
    
    variables = {
        "rootInput": {
            "priorLogits": [[0.1, 0.2], [0.3, 0.4]],
            "value": [0.5, 0.6],
            "embedding": [0, 0],
            "batchSize": 2,
            "numActions": 2
        },
        "searchParams": {
            "numSimulations": 10,
            "maxDepth": 5,
            "maxNumConsideredActions": 2
        },
        "searchType": "gumbel_muzero"
    }
    
    response = test_client.post(
        "/api/v1/graphql",
        json={"query": mutation, "variables": variables}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check for errors
    assert "errors" not in data, f"GraphQL errors: {data.get('errors')}"
    
    # Check data structure
    assert "data" in data
    assert "runSearch" in data["data"]
    assert "action" in data["data"]["runSearch"]
    assert "actionWeights" in data["data"]["runSearch"]
    assert "searchStatistics" in data["data"]["runSearch"]
    
    # Verify mocks were called
    mock_mctx["gumbel_muzero_policy"].assert_called_once()
    mock_hana_manager.save_search_history.assert_called_once()


def test_graphql_queue_search_mutation(test_client, mock_mctx):
    """Test GraphQL queue_search mutation."""
    mutation = """
    mutation ($rootInput: RootInputType!, $searchParams: SearchParamsInput!, $searchType: String!) {
        queueSearch(rootInput: $rootInput, searchParams: $searchParams, searchType: $searchType) {
            taskId
            status
            message
        }
    }
    """
    
    variables = {
        "rootInput": {
            "priorLogits": [[0.1, 0.2], [0.3, 0.4]],
            "value": [0.5, 0.6],
            "embedding": [0, 0],
            "batchSize": 2,
            "numActions": 2
        },
        "searchParams": {
            "numSimulations": 10,
            "maxDepth": 5,
            "maxNumConsideredActions": 2
        },
        "searchType": "gumbel_muzero"
    }
    
    # Mock Celery task
    with patch("app.worker.tasks.run_large_search.apply_async") as mock_apply_async:
        mock_task = MagicMock()
        mock_task.id = "test-task-id"
        mock_apply_async.return_value = mock_task
        
        response = test_client.post(
            "/api/v1/graphql",
            json={"query": mutation, "variables": variables}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check for errors
        assert "errors" not in data, f"GraphQL errors: {data.get('errors')}"
        
        # Check data structure
        assert "data" in data
        assert "queueSearch" in data["data"]
        assert data["data"]["queueSearch"]["taskId"] == "test-task-id"
        assert data["data"]["queueSearch"]["status"] == "queued"
        
        # Verify mocks were called
        mock_apply_async.assert_called_once()


def test_graphql_search_history_query(test_client, mock_hana_manager):
    """Test GraphQL search_history query."""
    query = """
    query {
        searchHistory(limit: 5) {
            id
            timestamp
            userId
            searchType
            batchSize
            numSimulations
        }
    }
    """
    
    # Mock database response
    mock_history_entries = [
        {
            "ID": 1,
            "TIMESTAMP": datetime.datetime.now(),
            "USER_ID": "test_user",
            "SEARCH_TYPE": "gumbel_muzero",
            "BATCH_SIZE": 2,
            "NUM_SIMULATIONS": 32,
            "MAX_DEPTH": 50,
            "DURATION_MS": 123.45,
            "NUM_EXPANDED_NODES": 64,
            "MAX_DEPTH_REACHED": 10,
        }
    ]
    mock_hana_manager.get_search_history.return_value = mock_history_entries
    
    response = test_client.post(
        "/api/v1/graphql",
        json={"query": query}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check for errors
    assert "errors" not in data, f"GraphQL errors: {data.get('errors')}"
    
    # Check data structure
    assert "data" in data
    assert "searchHistory" in data["data"]
    assert len(data["data"]["searchHistory"]) == 1
    assert data["data"]["searchHistory"][0]["id"] == 1