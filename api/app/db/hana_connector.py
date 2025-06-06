import os
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from functools import wraps
from contextlib import contextmanager

from hdbcli import dbapi
from pydantic import BaseModel

from ..core.config import get_settings
from ..core.exceptions import ConfigurationError, ModelError
from ..core.secrets import HanaSecrets, SecretManager

logger = logging.getLogger("mctx")
settings = get_settings()


class HANAConnectionManager:
    """
    Manager for SAP HANA database connections.
    
    Handles connection pooling and provides utility methods for database operations.
    """
    def __init__(self):
        """Initialize the connection manager."""
        self.connection_pool = []
        self.is_initialized = False
        self.retry_attempts = 3
        self.retry_delay = 2  # seconds
        
    def initialize(self):
        """Initialize the connection manager with configuration."""
        if self.is_initialized:
            return
        
        try:
            # Initialize HANA secrets
            HanaSecrets.initialize()
            
            # Set up tables
            self._setup_tables()
            self.is_initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize HANA connector: {str(e)}")
            raise
    
    def _get_connection_params(self) -> Dict[str, Any]:
        """Get connection parameters from configuration."""
        # Get connection parameters from secrets
        params = HanaSecrets.get_db_connection_params()
        
        # Add additional parameters
        return {
            "address": params["host"],
            "port": params["port"],
            "user": params["user"],
            "password": params["password"],
            "encrypt": True,
            "sslValidateCertificate": True,
            "timeout": 30,
        }
    
    @contextmanager
    def get_connection(self):
        """
        Get a connection from the pool or create a new one.
        
        Implements automatic retry mechanism for cloud network instability.
        
        Yields:
            Connection: Database connection
            
        Raises:
            ModelError: If connection fails after all retry attempts
        """
        if not self.is_initialized:
            self.initialize()
        
        conn = None
        attempt = 0
        last_error = None
        
        while attempt < self.retry_attempts:
            try:
                # Try to get connection from pool
                if self.connection_pool:
                    conn = self.connection_pool.pop()
                    # Check if connection is still alive
                    try:
                        with conn.cursor() as cursor:
                            cursor.execute("SELECT 1 FROM DUMMY")
                        # Connection is good
                        break
                    except Exception:
                        # Connection is dead, create new one
                        try:
                            conn.close()
                        except Exception:
                            pass
                        conn = None
                
                # Create new connection if needed
                if conn is None:
                    conn_params = self._get_connection_params()
                    # Set workload class for resource management
                    conn_params["workload_class"] = "MCTS_SERVICE"
                    conn = dbapi.connect(**conn_params)
                    break
                    
            except dbapi.Error as e:
                attempt += 1
                last_error = e
                logger.warning(f"HANA connection attempt {attempt} failed: {str(e)}")
                if attempt < self.retry_attempts:
                    import time
                    time.sleep(self.retry_delay)
                    # Exponential backoff
                    self.retry_delay *= 1.5
            except Exception as e:
                # Don't retry on non-connection errors
                if conn and not conn.closed:
                    try:
                        conn.close()
                    except Exception:
                        pass
                raise ModelError(
                    message="Unexpected database error",
                    details={"error": str(e)}
                )
        
        if conn is None:
            logger.error(f"SAP HANA connection failed after {self.retry_attempts} attempts")
            raise ModelError(
                message="Failed to connect to SAP HANA database after multiple attempts",
                details={
                    "error": str(last_error) if last_error else "Unknown error", 
                    "error_code": getattr(last_error, "errorcode", None) if last_error else None,
                    "attempts": attempt
                }
            )
        
        try:
            yield conn
            
            # Return connection to pool
            if conn and not conn.closed:
                self.connection_pool.append(conn)
                
        except Exception as e:
            logger.exception(f"Unexpected error with SAP HANA connection: {str(e)}")
            if conn and not conn.closed:
                try:
                    conn.close()
                except Exception:
                    pass
            raise ModelError(
                message="Unexpected database error",
                details={"error": str(e)}
            )
    
    def _setup_tables(self):
        """
        Initialize the database schema.
        
        Creates tables if they don't exist.
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Create search_history table
                    cursor.execute("""
                    CREATE TABLE IF NOT EXISTS MCTX_SEARCH_HISTORY (
                        ID BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                        TIMESTAMP TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        USER_ID VARCHAR(100),
                        SEARCH_TYPE VARCHAR(50) NOT NULL,
                        BATCH_SIZE INTEGER NOT NULL,
                        NUM_SIMULATIONS INTEGER NOT NULL,
                        MAX_DEPTH INTEGER,
                        CONFIG NCLOB,
                        DURATION_MS DECIMAL(10,2),
                        NUM_EXPANDED_NODES INTEGER,
                        MAX_DEPTH_REACHED INTEGER,
                        RESULT NCLOB
                    )
                    """)
                    
                    # Create search_statistics table for aggregated metrics
                    cursor.execute("""
                    CREATE TABLE IF NOT EXISTS MCTX_SEARCH_STATISTICS (
                        ID BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                        DATE_KEY DATE NOT NULL,
                        SEARCH_TYPE VARCHAR(50) NOT NULL,
                        TOTAL_SEARCHES INTEGER NOT NULL,
                        AVG_DURATION_MS DECIMAL(10,2),
                        AVG_EXPANDED_NODES DECIMAL(10,2),
                        MAX_BATCH_SIZE INTEGER,
                        MAX_NUM_SIMULATIONS INTEGER,
                        T4_OPTIMIZED_COUNT INTEGER,
                        DISTRIBUTED_COUNT INTEGER,
                        AVG_DEVICES INTEGER,
                        UNIQUE CONSTRAINT UX_MCTX_STATS_DATE_TYPE (DATE_KEY, SEARCH_TYPE)
                    )
                    """)
                
                conn.commit()
                logger.info("SAP HANA database schema initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SAP HANA schema: {str(e)}")
            raise ConfigurationError(
                message="Failed to initialize database schema",
                details={"error": str(e)}
            )
    
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
        """
        Save search history to the database.
        
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
            int: ID of the inserted record
            
        Raises:
            ModelError: If database operation fails
        """
        if not self.is_initialized:
            self.initialize()
            
        try:
            import json
            
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                    INSERT INTO MCTX_SEARCH_HISTORY (
                        USER_ID, SEARCH_TYPE, BATCH_SIZE, NUM_SIMULATIONS, MAX_DEPTH,
                        CONFIG, DURATION_MS, NUM_EXPANDED_NODES, MAX_DEPTH_REACHED, RESULT
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        user_id, search_type, batch_size, num_simulations, max_depth,
                        json.dumps(config), duration_ms, num_expanded_nodes, max_depth_reached,
                        json.dumps(result)
                    ))
                    
                    # Get the inserted ID
                    cursor.execute("SELECT CURRENT_IDENTITY_VALUE() FROM DUMMY")
                    record_id = cursor.fetchone()[0]
                
                conn.commit()
                logger.info(f"Saved search history record with ID: {record_id}")
                return record_id
                
        except Exception as e:
            logger.error(f"Failed to save search history: {str(e)}")
            raise ModelError(
                message="Failed to save search history",
                details={"error": str(e)}
            )
    
    def update_daily_statistics(self, date_key: str, search_type: str) -> None:
        """
        Update aggregated daily statistics.
        
        Args:
            date_key: Date in 'YYYY-MM-DD' format
            search_type: Type of search algorithm
            
        Raises:
            ModelError: If database operation fails
        """
        if not self.is_initialized:
            self.initialize()
            
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Check if statistics entry exists for this date and search type
                    cursor.execute("""
                    SELECT COUNT(*) FROM MCTX_SEARCH_STATISTICS 
                    WHERE DATE_KEY = ? AND SEARCH_TYPE = ?
                    """, (date_key, search_type))
                    
                    count = cursor.fetchone()[0]
                    
                    if count > 0:
                        # Update existing record
                        cursor.execute("""
                        UPDATE MCTX_SEARCH_STATISTICS SET
                            TOTAL_SEARCHES = (
                                SELECT COUNT(*) FROM MCTX_SEARCH_HISTORY 
                                WHERE SEARCH_TYPE = ? 
                                AND DATE(TIMESTAMP) = ?
                            ),
                            AVG_DURATION_MS = (
                                SELECT AVG(DURATION_MS) FROM MCTX_SEARCH_HISTORY 
                                WHERE SEARCH_TYPE = ? 
                                AND DATE(TIMESTAMP) = ?
                            ),
                            AVG_EXPANDED_NODES = (
                                SELECT AVG(NUM_EXPANDED_NODES) FROM MCTX_SEARCH_HISTORY 
                                WHERE SEARCH_TYPE = ? 
                                AND DATE(TIMESTAMP) = ?
                            ),
                            MAX_BATCH_SIZE = (
                                SELECT MAX(BATCH_SIZE) FROM MCTX_SEARCH_HISTORY 
                                WHERE SEARCH_TYPE = ? 
                                AND DATE(TIMESTAMP) = ?
                            ),
                            MAX_NUM_SIMULATIONS = (
                                SELECT MAX(NUM_SIMULATIONS) FROM MCTX_SEARCH_HISTORY 
                                WHERE SEARCH_TYPE = ? 
                                AND DATE(TIMESTAMP) = ?
                            ),
                            T4_OPTIMIZED_COUNT = (
                                SELECT COUNT(*) FROM MCTX_SEARCH_HISTORY 
                                WHERE SEARCH_TYPE = ? 
                                AND DATE(TIMESTAMP) = ?
                                AND JSON_VALUE(CONFIG, '$.optimizations.use_t4') = 'true'
                            ),
                            DISTRIBUTED_COUNT = (
                                SELECT COUNT(*) FROM MCTX_SEARCH_HISTORY 
                                WHERE SEARCH_TYPE = ? 
                                AND DATE(TIMESTAMP) = ?
                                AND JSON_VALUE(CONFIG, '$.optimizations.distributed') = 'true'
                            ),
                            AVG_DEVICES = (
                                SELECT AVG(CAST(JSON_VALUE(CONFIG, '$.optimizations.num_devices') AS INTEGER)) 
                                FROM MCTX_SEARCH_HISTORY 
                                WHERE SEARCH_TYPE = ? 
                                AND DATE(TIMESTAMP) = ?
                                AND JSON_VALUE(CONFIG, '$.optimizations.distributed') = 'true'
                            )
                        WHERE DATE_KEY = ? AND SEARCH_TYPE = ?
                        """, (
                            search_type, date_key,
                            search_type, date_key,
                            search_type, date_key,
                            search_type, date_key,
                            search_type, date_key,
                            search_type, date_key,
                            search_type, date_key,
                            search_type, date_key,
                            date_key, search_type
                        ))
                    else:
                        # Insert new record
                        cursor.execute("""
                        INSERT INTO MCTX_SEARCH_STATISTICS (
                            DATE_KEY, SEARCH_TYPE, TOTAL_SEARCHES,
                            AVG_DURATION_MS, AVG_EXPANDED_NODES,
                            MAX_BATCH_SIZE, MAX_NUM_SIMULATIONS,
                            T4_OPTIMIZED_COUNT, DISTRIBUTED_COUNT, AVG_DEVICES
                        )
                        SELECT
                            ?, ?,
                            COUNT(*),
                            AVG(DURATION_MS),
                            AVG(NUM_EXPANDED_NODES),
                            MAX(BATCH_SIZE),
                            MAX(NUM_SIMULATIONS),
                            SUM(CASE WHEN JSON_VALUE(CONFIG, '$.optimizations.use_t4') = 'true' THEN 1 ELSE 0 END),
                            SUM(CASE WHEN JSON_VALUE(CONFIG, '$.optimizations.distributed') = 'true' THEN 1 ELSE 0 END),
                            AVG(CASE WHEN JSON_VALUE(CONFIG, '$.optimizations.distributed') = 'true' 
                                THEN CAST(JSON_VALUE(CONFIG, '$.optimizations.num_devices') AS INTEGER) 
                                ELSE 1 END)
                        FROM MCTX_SEARCH_HISTORY
                        WHERE SEARCH_TYPE = ?
                        AND DATE(TIMESTAMP) = ?
                        """, (
                            date_key, search_type,
                            search_type, date_key
                        ))
                
                conn.commit()
                logger.info(f"Updated statistics for {date_key}, search type: {search_type}")
                
        except Exception as e:
            logger.error(f"Failed to update daily statistics: {str(e)}")
            raise ModelError(
                message="Failed to update daily statistics",
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
        config_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get search history records.
        
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
            ModelError: If database operation fails
        """
        if not self.is_initialized:
            self.initialize()
            
        try:
            import json
            
            # Use HANA JSON Table Function for optimized JSON query performance
            query = """
            SELECT
                ID, TIMESTAMP, USER_ID, SEARCH_TYPE,
                BATCH_SIZE, NUM_SIMULATIONS, MAX_DEPTH,
                DURATION_MS, NUM_EXPANDED_NODES, MAX_DEPTH_REACHED,
                CONFIG, RESULT
            FROM MCTX_SEARCH_HISTORY
            WHERE 1=1
            """
            
            params = []
            
            if user_id:
                query += " AND USER_ID = ?"
                params.append(user_id)
            
            if search_type:
                query += " AND SEARCH_TYPE = ?"
                params.append(search_type)
            
            if start_date:
                query += " AND DATE(TIMESTAMP) >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND DATE(TIMESTAMP) <= ?"
                params.append(end_date)
                
            # Add JSON config filter using HANA's JSON Table function
            if config_filter:
                query += f" AND {config_filter}"
            
            query += " ORDER BY TIMESTAMP DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, params)
                    columns = [col[0] for col in cursor.description]
                    results = []
                    
                    for row in cursor.fetchall():
                        record = dict(zip(columns, row))
                        # Parse JSON fields
                        if record["CONFIG"]:
                            record["CONFIG"] = json.loads(record["CONFIG"])
                        if record["RESULT"]:
                            record["RESULT"] = json.loads(record["RESULT"])
                        results.append(record)
                    
                    return results
                    
        except Exception as e:
            logger.error(f"Failed to get search history: {str(e)}")
            raise ModelError(
                message="Failed to retrieve search history",
                details={"error": str(e)}
            )


# Create a global instance
hana_manager = HANAConnectionManager()