import os
import json
import logging
import requests
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

from ..core.config import get_settings
from ..core.exceptions import ConfigurationError, ModelError
from ..core.secrets import DataSphereSecrets, SecretManager

logger = logging.getLogger("mctx")
settings = get_settings()


class DataSphereAuth:
    """
    Authentication manager for SAP DataSphere.
    
    Handles OAuth token acquisition and renewal.
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
        # Initialize DataSphere secrets
        DataSphereSecrets.initialize()
        
        # Get client credentials and URLs
        self.client_credentials = DataSphereSecrets.get_client_credentials()
        self.oauth_urls = DataSphereSecrets.get_oauth_urls()
        self.api_url = DataSphereSecrets.get_api_url()
        
        # Validate configuration
        if not self.client_credentials.get("client_id") or not self.client_credentials.get("client_secret"):
            raise ConfigurationError(
                message="Missing SAP DataSphere client credentials",
                details={"error": "client_id and client_secret must be provided"}
            )
        
        if not self.oauth_urls.get("token_url"):
            raise ConfigurationError(
                message="Missing SAP DataSphere token URL",
                details={"error": "token_url must be provided"}
            )
    
    def get_access_token(self) -> str:
        """
        Get a valid access token, refreshing if necessary.
        
        Returns:
            str: Access token
            
        Raises:
            ConfigurationError: If authentication fails
        """
        # Initialize if needed
        if not self.client_credentials:
            self.initialize()
        
        # Check if token is valid
        current_time = datetime.now()
        if self.access_token and self.token_expiry and current_time < self.token_expiry:
            # Token is still valid
            return self.access_token
        
        # Get new token
        try:
            response = requests.post(
                self.oauth_urls["token_url"],
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.client_credentials["client_id"],
                    "client_secret": self.client_credentials["client_secret"],
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            
            response.raise_for_status()
            token_data = response.json()
            
            self.access_token = token_data["access_token"]
            # Set expiry with a safety margin
            expires_in = token_data.get("expires_in", 3600)
            self.token_expiry = current_time + timedelta(seconds=expires_in - 300)  # 5 min safety margin
            
            logger.info("Obtained new DataSphere access token")
            return self.access_token
            
        except Exception as e:
            logger.error(f"Failed to get DataSphere access token: {str(e)}")
            raise ConfigurationError(
                message="Failed to authenticate with SAP DataSphere",
                details={"error": str(e)}
            )


class DataSphereConnector:
    """
    Connector for SAP DataSphere services.
    
    This class provides methods for interacting with SAP DataSphere
    APIs including data persistence, analytics, and ML operations.
    """
    
    def __init__(self):
        """Initialize the connector."""
        self.auth = DataSphereAuth()
        self.api_url = None
        self.space_id = None
        self.is_initialized = False
    
    def initialize(self):
        """Initialize the connector with configuration."""
        if self.is_initialized:
            return
        
        try:
            # Initialize auth
            self.auth.initialize()
            
            # Get API URL and space ID
            self.api_url = DataSphereSecrets.get_api_url()
            self.space_id = DataSphereSecrets.get_space_id()
            
            if not self.api_url:
                raise ConfigurationError(
                    message="Missing SAP DataSphere API URL",
                    details={"error": "api_url must be provided"}
                )
                
            if not self.space_id:
                raise ConfigurationError(
                    message="Missing SAP DataSphere space ID",
                    details={"error": "space_id must be provided"}
                )
            
            self.is_initialized = True
            logger.info("DataSphere connector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize DataSphere connector: {str(e)}")
            raise
    
    def _get_headers(self) -> Dict[str, str]:
        """
        Get headers for API requests with valid access token.
        
        Returns:
            Dict: Request headers
        """
        if not self.is_initialized:
            self.initialize()
        
        return {
            "Authorization": f"Bearer {self.auth.get_access_token()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
    
    def get_spaces(self) -> List[Dict[str, Any]]:
        """
        Get list of available spaces.
        
        Returns:
            List of spaces
            
        Raises:
            ModelError: If API request fails
        """
        if not self.is_initialized:
            self.initialize()
        
        try:
            response = requests.get(
                f"{self.api_url}/api/v1/dwc/catalog/spaces",
                headers=self._get_headers()
            )
            
            response.raise_for_status()
            result = response.json()
            
            return result.get("value", [])
        except Exception as e:
            logger.error(f"Failed to get DataSphere spaces: {str(e)}")
            raise ModelError(
                message="Failed to get DataSphere spaces",
                details={"error": str(e)}
            )
    
    def get_space_assets(self, space_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of assets in a space.
        
        Args:
            space_id: Optional space ID, uses default if not provided
            
        Returns:
            List of assets
            
        Raises:
            ModelError: If API request fails
        """
        if not self.is_initialized:
            self.initialize()
        
        space_id = space_id or self.space_id
        
        try:
            response = requests.get(
                f"{self.api_url}/api/v1/dwc/catalog/spaces('{space_id}')/assets",
                headers=self._get_headers()
            )
            
            response.raise_for_status()
            result = response.json()
            
            return result.get("value", [])
        except Exception as e:
            logger.error(f"Failed to get DataSphere assets: {str(e)}")
            raise ModelError(
                message=f"Failed to get assets for space {space_id}",
                details={"error": str(e)}
            )
    
    def create_dataset(
        self,
        name: str,
        description: str,
        columns: List[Dict[str, Any]],
        data: Optional[List[Dict[str, Any]]] = None,
        space_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new dataset in DataSphere.
        
        Args:
            name: Dataset name
            description: Dataset description
            columns: Column definitions
            data: Optional initial data
            space_id: Optional space ID
            
        Returns:
            Created dataset info
            
        Raises:
            ModelError: If API request fails
        """
        if not self.is_initialized:
            self.initialize()
        
        space_id = space_id or self.space_id
        
        try:
            # Create dataset definition
            definition = {
                "name": name,
                "description": description,
                "technicalName": name.lower().replace(" ", "_"),
                "type": "VIEW",
                "columns": columns,
            }
            
            # Create dataset
            response = requests.post(
                f"{self.api_url}/api/v1/dwc/catalog/spaces('{space_id}')/datasets",
                headers=self._get_headers(),
                json=definition
            )
            
            response.raise_for_status()
            dataset = response.json()
            
            # If data is provided, insert it
            if data and dataset.get("id"):
                dataset_id = dataset["id"]
                
                # Insert data
                data_response = requests.post(
                    f"{self.api_url}/api/v1/dwc/catalog/spaces('{space_id}')/datasets('{dataset_id}')/data",
                    headers=self._get_headers(),
                    json={"data": data}
                )
                
                data_response.raise_for_status()
            
            return dataset
        except Exception as e:
            logger.error(f"Failed to create DataSphere dataset: {str(e)}")
            raise ModelError(
                message=f"Failed to create dataset '{name}'",
                details={"error": str(e)}
            )
    
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
        Save MCTS search results to DataSphere.
        
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
            
        Raises:
            ModelError: If API request fails
        """
        if not self.is_initialized:
            self.initialize()
        
        try:
            # Prepare data
            record = {
                "searchType": search_type,
                "batchSize": batch_size,
                "numSimulations": num_simulations,
                "durationMs": duration_ms,
                "numExpandedNodes": num_expanded_nodes,
                "timestamp": datetime.now().isoformat(),
                "userId": user_id or "anonymous",
                "results": json.dumps(results),
            }
            
            # Make API request to store data
            response = requests.post(
                f"{self.api_url}/api/v1/dwc/catalog/spaces('{self.space_id}')/tables('MCTS_SEARCH_HISTORY')/data",
                headers=self._get_headers(),
                json={"data": [record]}
            )
            
            response.raise_for_status()
            result = response.json()
            
            logger.info(f"Saved search results to DataSphere, record ID: {result.get('id')}")
            return result
        except Exception as e:
            logger.error(f"Failed to save search results to DataSphere: {str(e)}")
            raise ModelError(
                message="Failed to save search results",
                details={"error": str(e)}
            )
    
    def run_analytics_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Run an analytics query on DataSphere.
        
        Args:
            query: SQL query
            
        Returns:
            Query results
            
        Raises:
            ModelError: If query fails
        """
        if not self.is_initialized:
            self.initialize()
        
        try:
            # Make API request to run query
            response = requests.post(
                f"{self.api_url}/api/v1/dwc/analytics/query",
                headers=self._get_headers(),
                json={"query": query, "spaceId": self.space_id}
            )
            
            response.raise_for_status()
            result = response.json()
            
            return result.get("results", [])
        except Exception as e:
            logger.error(f"Failed to run analytics query on DataSphere: {str(e)}")
            raise ModelError(
                message="Failed to run analytics query",
                details={"error": str(e), "query": query}
            )


# Create a global instance
datasphere = DataSphereConnector()