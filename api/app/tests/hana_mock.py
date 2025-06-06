"""Mock implementation of HANA connector for testing."""

import logging
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger("mctx")

class HANAConnectionManager:
    """
    Mock HANA connection manager for testing.
    """
    def __init__(self):
        """Initialize the mock connection manager."""
        self.connection_pool = []
        self.is_initialized = False
        self.tables_created = False
        
    def initialize(self):
        """Initialize the connection manager."""
        logger.info("Initializing mock HANA connection manager")
        self.is_initialized = True
        self._setup_tables()
        
    def _setup_tables(self):
        """Mock table setup method."""
        logger.info("Mock: Setting up HANA database tables")
        self.tables_created = True
        
    def get_connection(self):
        """Get a mock connection."""
        class MockCursor:
            def __init__(self):
                self.description = [
                    ("ID", None, None, None, None, None, None),
                    ("TIMESTAMP", None, None, None, None, None, None),
                    ("USER_ID", None, None, None, None, None, None),
                    ("SEARCH_TYPE", None, None, None, None, None, None),
                    ("BATCH_SIZE", None, None, None, None, None, None),
                    ("CONFIG", None, None, None, None, None, None)
                ]
                
            def __enter__(self):
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
                
            def execute(self, query, params=None):
                logger.info(f"MOCK: Executing query: {query}")
                if "SELECT CURRENT_IDENTITY_VALUE()" in query:
                    self.last_result = [12345]
                elif "SELECT 'HANA connection successful'" in query:
                    self.last_result = ["HANA connection successful"]
                elif "SELECT 1 FROM DUMMY" in query:
                    self.last_result = [1]
                else:
                    self.last_result = []
                return None
                
            def fetchone(self):
                return self.last_result
                
            def fetchall(self):
                if hasattr(self, 'last_result') and self.last_result:
                    return [self.last_result]
                return [["Result 1"], ["Result 2"]]

        class MockConnection:
            def __init__(self):
                self.closed = False
                self._cursor = MockCursor()
                
            def cursor(self):
                return self._cursor
                
            def commit(self):
                logger.info("MOCK: Committing transaction")
                
            def rollback(self):
                logger.info("MOCK: Rolling back transaction")
                
            def close(self):
                self.closed = True
                
            def __enter__(self):
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                if not self.closed:
                    self.close()
        
        return MockConnection()
        
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
    ) -> int:
        """Mock save search history."""
        logger.info("MOCK: Saving search history to HANA")
        return 12345

    def update_daily_statistics(self, date_key: str, search_type: str) -> None:
        """Mock update daily statistics."""
        logger.info(f"MOCK: Updating statistics for {date_key}, {search_type}")
        return True
        
    def get_search_history(
        self,
        limit: int = 100,
        offset: int = 0,
        user_id: Optional[str] = None,
        search_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Mock get search history."""
        logger.info("MOCK: Getting search history from HANA")
        return [{
            "ID": 12345,
            "TIMESTAMP": "2025-06-05T13:45:00",
            "USER_ID": user_id or "test_user",
            "SEARCH_TYPE": search_type or "test_search",
            "BATCH_SIZE": 2,
            "NUM_SIMULATIONS": 10,
            "CONFIG": json.dumps({"test": True}),
            "RESULT": json.dumps({"action": [0, 1]})
        }]

# Create a global instance
hana_manager = HANAConnectionManager()