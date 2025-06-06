import os
import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import jax
import jax.numpy as jnp

# Set environment variables for testing
os.environ["TESTING"] = "True"
os.environ["DEBUG"] = "True"
os.environ["JWT_SECRET_KEY"] = "test_secret_key"
os.environ["API_KEY"] = "test_api_key"

# Import app after setting environment variables
from ..main import app
from ..core.config import get_settings, Settings
from ..models.mcts_models import MCTSRequest, RootInput, SearchParams
from ..models.auth_models import User


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for pytest-asyncio."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
def test_client():
    """
    Create a TestClient instance for testing FastAPI endpoints.
    
    Returns:
        TestClient: FastAPI test client
    """
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="function")
def mock_settings():
    """
    Create a mock settings object for testing.
    
    Returns:
        MagicMock: Mock settings
    """
    settings = MagicMock(spec=Settings)
    settings.DEBUG = True
    settings.API_V1_STR = "/api/v1"
    settings.PROJECT_NAME = "MCTX API Test"
    settings.VERSION = "test"
    settings.MAX_BATCH_SIZE = 64
    settings.MAX_NUM_SIMULATIONS = 100
    settings.API_KEY_REQUIRED = False
    settings.API_KEY = "test_api_key"
    settings.API_KEY_NAME = "X-API-Key"
    return settings


@pytest.fixture(scope="function")
def mock_jax():
    """
    Mock JAX for testing without actual computation.
    
    Yields:
        None
    """
    # Store original functions
    original_muzero_policy = jax.random.PRNGKey
    
    # Create mock return values
    def mock_prng_key(seed):
        return jnp.array([0, seed], dtype=jnp.uint32)
    
    # Apply mocks
    jax.random.PRNGKey = mock_prng_key
    
    yield
    
    # Restore original functions
    jax.random.PRNGKey = original_muzero_policy


@pytest.fixture(scope="function")
def mock_hana_manager():
    """
    Mock SAP HANA database manager.
    
    Returns:
        MagicMock: Mock HANA manager
    """
    with patch("app.db.hana_connector.HANAConnectionManager") as mock:
        instance = mock.return_value
        instance.save_search_history.return_value = 123
        instance.update_daily_statistics.return_value = None
        instance.get_search_history.return_value = []
        yield instance


@pytest.fixture(scope="function")
def mock_mctx():
    """
    Mock MCTX library for testing.
    
    Returns:
        MagicMock: Mock MCTX module
    """
    with patch("mctx.muzero_policy") as mock_muzero, \
         patch("mctx.gumbel_muzero_policy") as mock_gumbel, \
         patch("mctx.stochastic_muzero_policy") as mock_stochastic, \
         patch("mctx.RootFnOutput") as mock_root_output:
        
        # Create mock policy output
        mock_policy_output = MagicMock()
        mock_policy_output.action = jnp.array([0, 1])
        mock_policy_output.action_weights = jnp.array([[0.5, 0.5], [0.3, 0.7]])
        
        # Create mock search tree
        mock_policy_output.search_tree = MagicMock()
        mock_policy_output.search_tree.node_visits = jnp.array([[10, 5], [8, 7]])
        
        # Set return values for mock policy functions
        mock_muzero.return_value = mock_policy_output
        mock_gumbel.return_value = mock_policy_output
        mock_stochastic.return_value = mock_policy_output
        
        yield {
            "muzero_policy": mock_muzero,
            "gumbel_muzero_policy": mock_gumbel,
            "stochastic_muzero_policy": mock_stochastic,
            "RootFnOutput": mock_root_output
        }


@pytest.fixture(scope="function")
def sample_mcts_request():
    """
    Create a sample MCTS request for testing.
    
    Returns:
        MCTSRequest: Sample request
    """
    return MCTSRequest(
        root_input=RootInput(
            prior_logits=[[0.1, 0.2], [0.3, 0.4]],
            value=[0.5, 0.6],
            embedding=[0, 0],
            batch_size=2,
            num_actions=2
        ),
        search_params=SearchParams(
            num_simulations=10,
            max_depth=5,
            max_num_considered_actions=2
        ),
        search_type="gumbel_muzero"
    )


@pytest.fixture(scope="function")
def auth_headers():
    """
    Create authentication headers for testing.
    
    Returns:
        dict: Headers with authentication
    """
    # Create a JWT token
    import jwt
    import time
    from datetime import datetime, timedelta
    
    secret_key = "test_secret_key"
    expiry = datetime.utcnow() + timedelta(minutes=30)
    
    payload = {
        "sub": "test_user",
        "exp": expiry,
        "scopes": ["read:mcts", "write:mcts"]
    }
    
    token = jwt.encode(payload, secret_key, algorithm="HS256")
    
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="function")
def mock_user():
    """
    Create a mock user for testing.
    
    Returns:
        User: Mock user
    """
    return User(
        username="test_user",
        email="test@example.com",
        disabled=False,
        roles=["admin"]
    )