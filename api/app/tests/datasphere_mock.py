"""Mock implementation of DataSphere connector for testing."""

import logging
import json
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger("mctx")

class DataSphereAuth:
    """
    Mock authentication manager for SAP DataSphere.
    """
    
    def __init__(self):
        """Initialize the auth manager."""
        self.access_token = None
        self.token_expiry = None
        self.client_credentials = None
        self.oauth_urls = None
        self.api_url = None
    
    def initialize(self):
        """Initialize with client credentials."""
        logger.info("Initializing mock DataSphere auth")
        self.client_credentials = {
            "client_id": "mock-client-id",
            "client_secret": "mock-client-secret"
        }
        self.oauth_urls = {
            "auth_url": "https://mock-auth-url.com/oauth/authorize",
            "token_url": "https://mock-auth-url.com/oauth/token"
        }
        self.api_url = "https://mock-api-url.com"
    
    def get_access_token(self) -> str:
        """
        Get a mock access token.
        
        Returns:
            str: Access token
        """
        logger.info("MOCK: Getting DataSphere access token")
        self.access_token = "mock-ds-token-123456"
        return self.access_token


class DataSphereConnector:
    """
    Mock connector for SAP DataSphere services.
    """
    
    def __init__(self):
        """Initialize the connector."""
        self.auth = DataSphereAuth()
        self.api_url = None
        self.space_id = None
        self.is_initialized = False
    
    def initialize(self):
        """Initialize the connector with configuration."""
        logger.info("Initializing mock DataSphere connector")
        self.auth.initialize()
        self.api_url = "https://mock-api-url.com"
        self.space_id = "DP_001"
        self.is_initialized = True
    
    def get_spaces(self) -> List[Dict[str, Any]]:
        """
        Get mock list of available spaces.
        
        Returns:
            List of spaces
        """
        logger.info("MOCK: Getting DataSphere spaces")
        return [
            {"id": "DP_001", "name": "Development Space"},
            {"id": "DP_002", "name": "Test Space"}
        ]
    
    def get_space_assets(self, space_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get mock list of assets in a space.
        
        Args:
            space_id: Optional space ID, uses default if not provided
            
        Returns:
            List of assets
        """
        logger.info(f"MOCK: Getting assets for space {space_id or self.space_id}")
        return [
            {"id": "asset1", "name": "Test Dataset", "type": "VIEW"},
            {"id": "asset2", "name": "Metrics Table", "type": "TABLE"}
        ]
    
    def create_dataset(
        self,
        name: str,
        description: str,
        columns: List[Dict[str, Any]],
        data: Optional[List[Dict[str, Any]]] = None,
        space_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a mock dataset.
        
        Args:
            name: Dataset name
            description: Dataset description
            columns: Column definitions
            data: Optional initial data
            space_id: Optional space ID
            
        Returns:
            Created dataset info
        """
        logger.info(f"MOCK: Creating dataset {name}")
        return {
            "id": "new-dataset-123",
            "name": name,
            "description": description,
            "technicalName": name.lower().replace(" ", "_"),
            "type": "VIEW"
        }
    
    def save_search_results(
        self,
        search_type: str,
        batch_size: int,
        num_simulations: int,
        duration_ms: float,
        num_expanded_nodes: int,
        results: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Mock save MCTS search results to DataSphere.
        
        Args:
            search_type: Type of search algorithm
            batch_size: Number of batch items
            num_simulations: Number of simulations
            duration_ms: Search duration in milliseconds
            num_expanded_nodes: Number of expanded nodes
            results: Search results
            user_id: Optional user ID
            
        Returns:
            Created record
        """
        logger.info(f"MOCK: Saving search results to DataSphere")
        return {"id": "ds-12345", "status": "success"}
    
    def run_analytics_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Run a mock analytics query.
        
        Args:
            query: SQL query
            
        Returns:
            Query results
        """
        logger.info(f"MOCK: Running analytics query: {query}")
        return [
            {"count": 42, "avg_duration": 125.7},
            {"count": 18, "avg_duration": 87.3}
        ]


# Create a global instance
datasphere = DataSphereConnector()