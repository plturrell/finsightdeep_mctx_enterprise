import os
import json
import logging
import requests
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from contextlib import contextmanager
from datetime import datetime, timedelta

from ..core.config import get_settings
from ..core.exceptions import ConfigurationError, ModelError
from ..core.secrets import HanaSecrets, SecretManager

logger = logging.getLogger("mctx")
settings = get_settings()


class HanaCloudAuth:
    """
    Authentication manager for SAP HANA Cloud.
    
    Handles OAuth token acquisition and renewal.
    """
    
    def __init__(self):
        """Initialize the auth manager."""
        self.access_token = None
        self.token_expiry = None
        self.client_credentials = None
        self.oauth_urls = None
    
    def initialize(self):
        """Initialize with client credentials."""
        # Initialize secrets if needed
        HanaSecrets.initialize()
        
        # Get client credentials and URLs
        self.client_credentials = HanaSecrets.get_client_credentials()
        self.oauth_urls = HanaSecrets.get_oauth_urls()
        
        # Validate configuration
        if not self.client_credentials.get("client_id") or not self.client_credentials.get("client_secret"):
            raise ConfigurationError(
                message="Missing SAP HANA Cloud client credentials",
                details={"error": "client_id and client_secret must be provided"}
            )
        
        if not self.oauth_urls.get("token_url"):
            raise ConfigurationError(
                message="Missing SAP HANA Cloud token URL",
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
            
            return self.access_token
            
        except Exception as e:
            logger.error(f"Failed to get HANA Cloud access token: {str(e)}")
            raise ConfigurationError(
                message="Failed to authenticate with SAP HANA Cloud",
                details={"error": str(e)}
            )


class HanaCloudConnector:
    """
    Connector for SAP HANA Cloud services.
    
    This class provides methods for interacting with SAP HANA Cloud
    APIs including data persistence and retrieval.
    """
    
    def __init__(self):
        """Initialize the connector."""
        self.auth = HanaCloudAuth()
        self.api_url = None
        self.is_initialized = False
    
    def initialize(self):
        """Initialize the connector with configuration."""
        if self.is_initialized:
            return
        
        try:
            # Initialize auth
            self.auth.initialize()
            
            # Get API URL
            self.api_url = HanaSecrets.get_api_url()
            if not self.api_url:
                raise ConfigurationError(
                    message="Missing SAP HANA Cloud API URL",
                    details={"error": "api_url must be provided"}
                )
            
            self.is_initialized = True
            logger.info("HANA Cloud connector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize HANA Cloud connector: {str(e)}")
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
    
    def save_search_history(
        self,
        search_type: str,
        batch_size: int,
        num_simulations: int,
        max_depth: Optional[int],
        config: Dict[str, Any],
        duration_ms: float,
        num_expanded_nodes: int,
        max_depth_reached: int,
        result: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> str:
        """
        Save search history to SAP HANA Cloud.
        
        Args:
            search_type: Type of search algorithm used
            batch_size: Number of batch items
            num_simulations: Number of simulations performed
            max_depth: Maximum allowed depth
            config: Search configuration
            duration_ms: Search duration in milliseconds
            num_expanded_nodes: Number of nodes expanded
            max_depth_reached: Maximum depth reached
            result: Search results
            user_id: Optional user identifier
            
        Returns:
            str: ID of the created record
            
        Raises:
            ModelError: If operation fails
        """
        if not self.is_initialized:
            self.initialize()
        
        try:
            # Prepare data
            data = {
                "timestamp": datetime.now().isoformat(),
                "userId": user_id,
                "searchType": search_type,
                "batchSize": batch_size,
                "numSimulations": num_simulations,
                "maxDepth": max_depth,
                "config": config,
                "durationMs": duration_ms,
                "numExpandedNodes": num_expanded_nodes,
                "maxDepthReached": max_depth_reached,
                "result": result,
            }
            
            # Make API request
            response = requests.post(
                f"{self.api_url}/api/v1/dwc/catalog/spaces('DP_001')/searchHistory",
                headers=self._get_headers(),
                json=data
            )
            
            response.raise_for_status()
            result_data = response.json()
            
            record_id = result_data["id"]
            logger.info(f"Saved search history to HANA Cloud with ID: {record_id}")
            
            return record_id
            
        except Exception as e:
            logger.error(f"Failed to save search history to HANA Cloud: {str(e)}")
            raise ModelError(
                message="Failed to save search history",
                details={"error": str(e)}
            )
    
    def get_search_history(
        self,
        limit: int = 100,
        offset: int = 0,
        user_id: Optional[str] = None,
        search_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get search history records from SAP HANA Cloud.
        
        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            user_id: Filter by user ID
            search_type: Filter by search type
            start_date: Filter by start date (YYYY-MM-DD)
            end_date: Filter by end date (YYYY-MM-DD)
            
        Returns:
            List of search history records
            
        Raises:
            ModelError: If operation fails
        """
        if not self.is_initialized:
            self.initialize()
        
        try:
            # Build query parameters
            params = {
                "$top": limit,
                "$skip": offset,
                "$orderby": "timestamp desc",
            }
            
            # Add filters
            filters = []
            if user_id:
                filters.append(f"userId eq '{user_id}'")
            if search_type:
                filters.append(f"searchType eq '{search_type}'")
            if start_date:
                filters.append(f"date(timestamp) ge {start_date}")
            if end_date:
                filters.append(f"date(timestamp) le {end_date}")
            
            if filters:
                params["$filter"] = " and ".join(filters)
            
            # Make API request
            response = requests.get(
                f"{self.api_url}/api/v1/dwc/catalog/spaces('DP_001')/searchHistory",
                headers=self._get_headers(),
                params=params
            )
            
            response.raise_for_status()
            result_data = response.json()
            
            return result_data.get("value", [])
            
        except Exception as e:
            logger.error(f"Failed to get search history from HANA Cloud: {str(e)}")
            raise ModelError(
                message="Failed to retrieve search history",
                details={"error": str(e)}
            )
    
    def update_statistics(self, date_key: str, search_type: str) -> None:
        """
        Update aggregated statistics in SAP HANA Cloud.
        
        Args:
            date_key: Date in 'YYYY-MM-DD' format
            search_type: Type of search algorithm
            
        Raises:
            ModelError: If operation fails
        """
        if not self.is_initialized:
            self.initialize()
        
        try:
            # Get current statistics
            response = requests.get(
                f"{self.api_url}/api/v1/dwc/catalog/spaces('DP_001')/searchStatistics",
                headers=self._get_headers(),
                params={
                    "$filter": f"dateKey eq '{date_key}' and searchType eq '{search_type}'"
                }
            )
            
            response.raise_for_status()
            existing_stats = response.json().get("value", [])
            
            # Calculate new statistics
            agg_response = requests.get(
                f"{self.api_url}/api/v1/dwc/catalog/spaces('DP_001')/searchHistory/$aggregate",
                headers=self._get_headers(),
                params={
                    "$apply": (
                        f"filter(searchType eq '{search_type}' and date(timestamp) eq {date_key})/"
                        "aggregate("
                        "totalSearches with count as count,"
                        "avgDurationMs with average as average,"
                        "avgExpandedNodes with average as average,"
                        "maxBatchSize with max as maximum,"
                        "maxNumSimulations with max as maximum"
                        ")"
                    )
                }
            )
            
            agg_response.raise_for_status()
            agg_data = agg_response.json().get("value", [{}])[0]
            
            # Prepare statistics data
            stats_data = {
                "dateKey": date_key,
                "searchType": search_type,
                "totalSearches": agg_data.get("count", 0),
                "avgDurationMs": agg_data.get("avgDurationMs", 0),
                "avgExpandedNodes": agg_data.get("avgExpandedNodes", 0),
                "maxBatchSize": agg_data.get("maxBatchSize", 0),
                "maxNumSimulations": agg_data.get("maxNumSimulations", 0),
            }
            
            if existing_stats:
                # Update existing record
                stats_id = existing_stats[0]["id"]
                response = requests.patch(
                    f"{self.api_url}/api/v1/dwc/catalog/spaces('DP_001')/searchStatistics('{stats_id}')",
                    headers=self._get_headers(),
                    json=stats_data
                )
            else:
                # Create new record
                response = requests.post(
                    f"{self.api_url}/api/v1/dwc/catalog/spaces('DP_001')/searchStatistics",
                    headers=self._get_headers(),
                    json=stats_data
                )
            
            response.raise_for_status()
            logger.info(f"Updated statistics in HANA Cloud for {date_key}, search type: {search_type}")
            
        except Exception as e:
            logger.error(f"Failed to update statistics in HANA Cloud: {str(e)}")
            raise ModelError(
                message="Failed to update statistics",
                details={"error": str(e)}
            )


# Create a global instance
hana_cloud = HanaCloudConnector()