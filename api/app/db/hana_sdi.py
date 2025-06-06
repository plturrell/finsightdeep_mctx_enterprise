#!/usr/bin/env python3
"""
SAP HANA Smart Data Integration (SDI) module for MCTS service.

This module implements Smart Data Integration capabilities to connect
MCTS search results with other data sources for enhanced analytics.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from contextlib import contextmanager

from hdbcli import dbapi
from ..core.config import get_settings
from ..core.exceptions import ConfigurationError, ModelError
from .hana_connector import hana_manager

logger = logging.getLogger("mctx")
settings = get_settings()


class HANASmartDataIntegration:
    """
    Implements Smart Data Integration for MCTS service.
    
    Enables cross-system analytics, remote data processing, and
    performance optimization for MCTS service data.
    """
    
    def __init__(self):
        """Initialize the SDI manager."""
        self.is_initialized = False
        
    def initialize(self):
        """Initialize the SDI manager."""
        if self.is_initialized:
            return
            
        try:
            # Initialize HANA connection
            hana_manager.initialize()
            
            # Set up remote sources and virtual tables
            self._setup_remote_sources()
            self._setup_virtual_tables()
            
            self.is_initialized = True
            logger.info("HANA Smart Data Integration initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize HANA SDI: {str(e)}")
            raise ConfigurationError(
                message="Failed to initialize HANA Smart Data Integration",
                details={"error": str(e)}
            )
    
    def _setup_remote_sources(self):
        """Set up remote data sources for Smart Data Integration."""
        with hana_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                # Check if HADOOP_SOURCE exists
                cursor.execute("""
                SELECT COUNT(*) FROM SYS.REMOTE_SOURCES 
                WHERE SOURCE_NAME = 'HADOOP_SOURCE'
                """)
                
                count = cursor.fetchone()[0]
                
                if count == 0:
                    # Create HADOOP remote source for storing large MCTS trees
                    cursor.execute("""
                    CREATE REMOTE SOURCE "HADOOP_SOURCE"
                    ADAPTER "HADOOP"
                    CONFIGURATION 'HDFS_HOST=hadoop-namenode;HDFS_PORT=8020'
                    WITH CREDENTIAL TYPE 'PASSWORD' USING 'user=hdfs;password=hdfs_password'
                    """)
                
                # Check if MODEL_REPO_SOURCE exists
                cursor.execute("""
                SELECT COUNT(*) FROM SYS.REMOTE_SOURCES 
                WHERE SOURCE_NAME = 'MODEL_REPO_SOURCE'
                """)
                
                count = cursor.fetchone()[0]
                
                if count == 0:
                    # Create MODEL_REPO remote source for ML model metadata
                    cursor.execute("""
                    CREATE REMOTE SOURCE "MODEL_REPO_SOURCE"
                    ADAPTER "ODBC"
                    CONFIGURATION 'DRIVER=HANAODBC;SERVERNODE=model-repo:30015'
                    WITH CREDENTIAL TYPE 'PASSWORD' USING 'user=MODEL_REPO_USER;password=model_repo_password'
                    """)
            
            conn.commit()
    
    def _setup_virtual_tables(self):
        """Set up virtual tables for cross-system analytics."""
        with hana_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                # Create virtual table for MCTS tree storage
                cursor.execute("""
                CREATE VIRTUAL TABLE "MCTS_LARGE_TREES" AT "HADOOP_SOURCE"."default"."mcts_trees"
                """)
                
                # Create virtual table for ML model metadata
                cursor.execute("""
                CREATE VIRTUAL TABLE "ML_MODEL_METADATA" AT "MODEL_REPO_SOURCE"."MODELS"."MODEL_METADATA"
                """)
                
                # Create virtual schema for easier access
                cursor.execute("""
                CREATE VIRTUAL SCHEMA "MCTS_ANALYTICS"
                """)
                
                # Add virtual tables to schema
                cursor.execute("""
                ADD VIRTUAL TABLE "MCTS_ANALYTICS"."MCTS_LARGE_TREES" AT "HADOOP_SOURCE"."default"."mcts_trees"
                """)
                
                cursor.execute("""
                ADD VIRTUAL TABLE "MCTS_ANALYTICS"."ML_MODEL_METADATA" AT "MODEL_REPO_SOURCE"."MODELS"."MODEL_METADATA"
                """)
            
            conn.commit()
    
    def save_large_tree(self, tree_id: str, tree_data: Dict[str, Any], user_id: Optional[str] = None) -> str:
        """
        Save a large MCTS tree to the Hadoop storage.
        
        Args:
            tree_id: Unique identifier for the tree
            tree_data: Tree data to store
            user_id: Optional user identifier
            
        Returns:
            Storage ID for the saved tree
            
        Raises:
            ModelError: If storage operation fails
        """
        if not self.is_initialized:
            self.initialize()
            
        try:
            import json
            
            with hana_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Convert tree data to JSON
                    tree_json = json.dumps(tree_data)
                    
                    # Store in Hadoop through virtual table
                    cursor.execute("""
                    INSERT INTO "MCTS_ANALYTICS"."MCTS_LARGE_TREES" 
                    (TREE_ID, USER_ID, TIMESTAMP, TREE_DATA) 
                    VALUES (?, ?, CURRENT_TIMESTAMP, ?)
                    """, (tree_id, user_id, tree_json))
                
                conn.commit()
                
                logger.info(f"Saved large tree with ID: {tree_id}")
                return tree_id
                
        except Exception as e:
            logger.error(f"Failed to save large tree: {str(e)}")
            raise ModelError(
                message="Failed to save large MCTS tree",
                details={"error": str(e), "tree_id": tree_id}
            )
    
    def get_cross_system_stats(self) -> Dict[str, Any]:
        """
        Get cross-system statistics using Smart Data Integration.
        
        Returns:
            Dictionary containing cross-system statistics
        """
        if not self.is_initialized:
            self.initialize()
            
        try:
            with hana_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Query for cross-system MCTS performance
                    cursor.execute("""
                    SELECT 
                        s.SEARCH_TYPE,
                        s.AVG_DURATION_MS,
                        m.MODEL_VERSION,
                        m.TRAINING_DATASET,
                        m.PARAMETERS_COUNT
                    FROM 
                        MCTX_SEARCH_STATISTICS s
                    JOIN 
                        "MCTS_ANALYTICS"."ML_MODEL_METADATA" m
                    ON 
                        s.SEARCH_TYPE = m.MODEL_TYPE
                    WHERE 
                        s.DATE_KEY > ADD_DAYS(CURRENT_DATE, -30)
                    ORDER BY 
                        s.DATE_KEY DESC
                    """)
                    
                    cross_system_stats = []
                    for row in cursor.fetchall():
                        cross_system_stats.append({
                            "search_type": row[0],
                            "avg_duration_ms": row[1],
                            "model_version": row[2],
                            "training_dataset": row[3],
                            "parameters_count": row[4]
                        })
                    
                    # Query for large tree statistics
                    cursor.execute("""
                    SELECT 
                        COUNT(*) as TREE_COUNT,
                        AVG(JSON_LENGTH(TREE_DATA)) as AVG_TREE_SIZE
                    FROM 
                        "MCTS_ANALYTICS"."MCTS_LARGE_TREES"
                    WHERE 
                        TIMESTAMP > ADD_DAYS(CURRENT_TIMESTAMP, -30)
                    """)
                    
                    tree_stats = cursor.fetchone()
                    large_tree_stats = {
                        "tree_count": tree_stats[0],
                        "avg_tree_size": tree_stats[1]
                    }
                    
                    return {
                        "cross_system_stats": cross_system_stats,
                        "large_tree_stats": large_tree_stats
                    }
                    
        except Exception as e:
            logger.error(f"Failed to get cross-system stats: {str(e)}")
            return {
                "cross_system_stats": [],
                "large_tree_stats": {"tree_count": 0, "avg_tree_size": 0}
            }


# Create a global instance
hana_sdi = HANASmartDataIntegration()