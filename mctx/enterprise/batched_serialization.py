# Copyright 2023 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Batched serialization for large MCTS trees with SAP HANA.

This module provides memory-efficient serialization and deserialization for
extremely large MCTS tree structures when working with SAP HANA.
"""

import gc
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union, Generator, Set

import numpy as np

from mctx.enterprise.hana_integration import HanaConnection

# Configure default batch sizes
DEFAULT_NODE_BATCH_SIZE = int(os.environ.get('MCTX_NODE_BATCH_SIZE', '10000'))
DEFAULT_SERIALIZATION_CHUNK_SIZE = int(os.environ.get('MCTX_SERIALIZATION_CHUNK_SIZE', '50000'))
DEFAULT_MEMORY_LIMIT_MB = int(os.environ.get('MCTX_SERIALIZATION_MEMORY_LIMIT_MB', '1024'))

# Configure logging
logger = logging.getLogger("mctx.enterprise.batched_serialization")


class BatchedTreeSerializer:
    """Memory-efficient serializer for large MCTS trees.
    
    This class provides methods to serialize and deserialize large MCTS trees
    in batches to avoid out-of-memory errors when working with very large
    tree structures.
    """
    
    def __init__(self, 
                 connection: HanaConnection,
                 node_batch_size: int = DEFAULT_NODE_BATCH_SIZE,
                 serialization_chunk_size: int = DEFAULT_SERIALIZATION_CHUNK_SIZE,
                 memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB):
        """Initialize a new batched tree serializer.
        
        Args:
            connection: SAP HANA connection.
            node_batch_size: Maximum number of nodes to process in a batch.
            serialization_chunk_size: Maximum number of nodes to serialize at once.
            memory_limit_mb: Memory limit in MB before forcing garbage collection.
        """
        self.connection = connection
        self.node_batch_size = node_batch_size
        self.serialization_chunk_size = serialization_chunk_size
        self.memory_limit_mb = memory_limit_mb
        
        # Table names
        self.schema = connection.config.schema
        self.trees_table = f"{self.schema}.MCTS_TREES"
        self.nodes_table = f"{self.schema}.MCTS_NODES"
        self.batches_table = f"{self.schema}.MCTS_SERIALIZATION_BATCHES"
        
        # Ensure serialization batches table exists
        self._ensure_batches_table_exists()
    
    def _ensure_batches_table_exists(self) -> None:
        """Ensure the serialization batches table exists."""
        conn = self.connection.get_connection()
        cursor = conn.cursor()
        
        try:
            # Check if table exists
            query = f"""
            SELECT COUNT(*) 
            FROM SYS.TABLES 
            WHERE SCHEMA_NAME = '{self.schema}' 
            AND TABLE_NAME = 'MCTS_SERIALIZATION_BATCHES'
            """
            cursor.execute(query)
            result = cursor.fetchone()
            
            if result and result[0] == 0:
                # Create table if it doesn't exist
                create_table_query = f"""
                CREATE TABLE {self.batches_table} (
                    batch_id VARCHAR(36) NOT NULL,
                    tree_id VARCHAR(36) NOT NULL,
                    batch_number INTEGER NOT NULL,
                    node_count INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (batch_id),
                    FOREIGN KEY (tree_id) REFERENCES {self.trees_table}(tree_id) ON DELETE CASCADE
                )
                """
                cursor.execute(create_table_query)
                
                # Create index on tree_id
                cursor.execute(f"CREATE INDEX IDX_BATCHES_TREE_ID ON {self.batches_table}(tree_id)")
                
                logger.info(f"Created serialization batches table: {self.batches_table}")
        finally:
            self.connection.release_connection(conn)
    
    def _check_memory_usage(self) -> None:
        """Check memory usage and force garbage collection if needed."""
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        
        if memory_mb > self.memory_limit_mb:
            logger.info(f"Memory usage ({memory_mb:.1f} MB) exceeded limit "
                       f"({self.memory_limit_mb} MB), forcing garbage collection")
            gc.collect()
    
    def batch_serialize_tree(self, tree: Dict[str, Any]) -> str:
        """Serialize a large tree to the database in batches.
        
        Args:
            tree: The tree structure to serialize.
            
        Returns:
            The tree_id of the serialized tree.
        """
        # Start timer
        start_time = time.time()
        
        # Extract tree metadata
        tree_id = tree.get('tree_id', str(uuid.uuid4()))
        nodes = tree.get('nodes', [])
        total_nodes = len(nodes)
        
        # Create a copy of the tree without nodes for metadata storage
        tree_metadata = {k: v for k, v in tree.items() if k != 'nodes'}
        tree_metadata['tree_id'] = tree_id
        
        # Add serialization metadata
        if 'metadata' not in tree_metadata:
            tree_metadata['metadata'] = {}
        
        if isinstance(tree_metadata['metadata'], str):
            metadata = json.loads(tree_metadata['metadata'])
        else:
            metadata = tree_metadata['metadata']
        
        metadata['batched_serialization'] = True
        metadata['total_nodes'] = total_nodes
        metadata['node_batch_size'] = self.node_batch_size
        metadata['serialization_time'] = time.time()
        
        tree_metadata['metadata'] = json.dumps(metadata)
        
        # Save tree metadata
        self._save_tree_metadata(tree_metadata)
        
        # Batch the nodes
        if total_nodes > 0:
            # Process nodes in chunks to avoid memory issues
            batch_number = 0
            for node_batch in self._batch_nodes(nodes):
                batch_id = str(uuid.uuid4())
                batch_count = len(node_batch)
                
                # Save batch metadata
                self._save_batch_metadata(batch_id, tree_id, batch_number, batch_count)
                
                # Save batch nodes
                self._save_nodes_batch(node_batch, tree_id)
                
                batch_number += 1
                
                # Force garbage collection if needed
                self._check_memory_usage()
        
        # Log serialization stats
        serialization_time = time.time() - start_time
        logger.info(f"Serialized tree {tree_id} with {total_nodes} nodes "
                   f"in {batch_number} batches ({serialization_time:.2f}s)")
        
        return tree_id
    
    def _batch_nodes(self, nodes: List[Dict[str, Any]]) -> Generator[List[Dict[str, Any]], None, None]:
        """Split a list of nodes into batches.
        
        Args:
            nodes: The list of nodes to batch.
            
        Yields:
            Batches of nodes.
        """
        total_nodes = len(nodes)
        
        for i in range(0, total_nodes, self.node_batch_size):
            end_idx = min(i + self.node_batch_size, total_nodes)
            yield nodes[i:end_idx]
    
    def _save_tree_metadata(self, tree_metadata: Dict[str, Any]) -> None:
        """Save tree metadata to the database.
        
        Args:
            tree_metadata: The tree metadata to save.
        """
        conn = self.connection.get_connection()
        cursor = conn.cursor()
        
        try:
            # Check if tree exists
            query = f"SELECT COUNT(*) FROM {self.trees_table} WHERE tree_id = :tree_id"
            cursor.execute(query, {'tree_id': tree_metadata['tree_id']})
            result = cursor.fetchone()
            
            if result and result[0] > 0:
                # Update existing tree
                update_query = f"""
                UPDATE {self.trees_table}
                SET name = :name,
                    batch_size = :batch_size,
                    num_actions = :num_actions,
                    num_simulations = :num_simulations,
                    metadata = :metadata,
                    updated_at = CURRENT_TIMESTAMP
                WHERE tree_id = :tree_id
                """
                cursor.execute(update_query, tree_metadata)
            else:
                # Insert new tree
                insert_query = f"""
                INSERT INTO {self.trees_table} (
                    tree_id, name, batch_size, num_actions, num_simulations, metadata
                ) VALUES (
                    :tree_id, :name, :batch_size, :num_actions, :num_simulations, :metadata
                )
                """
                cursor.execute(insert_query, tree_metadata)
        finally:
            self.connection.release_connection(conn)
    
    def _save_batch_metadata(self, batch_id: str, tree_id: str, 
                            batch_number: int, node_count: int) -> None:
        """Save batch metadata to the database.
        
        Args:
            batch_id: The batch ID.
            tree_id: The tree ID.
            batch_number: The batch number.
            node_count: The number of nodes in the batch.
        """
        conn = self.connection.get_connection()
        cursor = conn.cursor()
        
        try:
            query = f"""
            INSERT INTO {self.batches_table} (
                batch_id, tree_id, batch_number, node_count
            ) VALUES (
                :batch_id, :tree_id, :batch_number, :node_count
            )
            """
            cursor.execute(query, {
                'batch_id': batch_id,
                'tree_id': tree_id,
                'batch_number': batch_number,
                'node_count': node_count
            })
        finally:
            self.connection.release_connection(conn)
    
    def _save_nodes_batch(self, nodes: List[Dict[str, Any]], tree_id: str) -> None:
        """Save a batch of nodes to the database.
        
        Args:
            nodes: The list of nodes to save.
            tree_id: The tree ID.
        """
        if not nodes:
            return
        
        conn = self.connection.get_connection()
        cursor = conn.cursor()
        
        try:
            # Save nodes in smaller chunks if needed
            for i in range(0, len(nodes), self.serialization_chunk_size):
                chunk = nodes[i:i + self.serialization_chunk_size]
                
                # Prepare parameters for batch insert
                params = []
                for node in chunk:
                    # Ensure node has an ID
                    node_id = node.get('id', str(uuid.uuid4()))
                    
                    # Prepare node state and action for serialization
                    state = json.dumps(node.get('state', {})) if isinstance(node.get('state'), (dict, list)) else node.get('state', '')
                    action = json.dumps(node.get('action', {})) if isinstance(node.get('action'), (dict, list)) else node.get('action', '')
                    
                    params.append({
                        'id': node_id,
                        'tree_id': tree_id,
                        'parent_id': node.get('parent_id', ''),
                        'visit_count': node.get('visit_count', 0),
                        'value': node.get('value', 0.0),
                        'state': state,
                        'action': action
                    })
                
                # Batch insert
                for param in params:
                    query = f"""
                    INSERT INTO {self.nodes_table} (
                        id, tree_id, parent_id, visit_count, value, state, action
                    ) VALUES (
                        :id, :tree_id, :parent_id, :visit_count, :value, :state, :action
                    )
                    """
                    cursor.execute(query, param)
        finally:
            self.connection.release_connection(conn)
    
    def batch_deserialize_tree(self, tree_id: str, 
                              include_nodes: bool = True,
                              max_nodes: Optional[int] = None) -> Dict[str, Any]:
        """Deserialize a tree from the database in batches.
        
        Args:
            tree_id: The ID of the tree to deserialize.
            include_nodes: Whether to include nodes in the result.
            max_nodes: Maximum number of nodes to deserialize (for partial loading).
            
        Returns:
            The deserialized tree structure.
        """
        # Start timer
        start_time = time.time()
        
        # Get tree metadata
        tree = self._get_tree_metadata(tree_id)
        
        if not tree:
            logger.warning(f"Tree {tree_id} not found")
            return {}
        
        if include_nodes:
            # Check if tree uses batched serialization
            metadata = tree.get('metadata', {})
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            
            is_batched = metadata.get('batched_serialization', False)
            total_nodes = metadata.get('total_nodes', 0)
            
            # Apply max_nodes limit if specified
            if max_nodes is not None:
                node_limit = min(max_nodes, total_nodes)
            else:
                node_limit = total_nodes
            
            # Get nodes
            if is_batched:
                # Get nodes in batches
                nodes = self._get_batched_nodes(tree_id, node_limit)
            else:
                # Get nodes directly (for backward compatibility)
                nodes = self._get_nodes(tree_id, node_limit)
            
            tree['nodes'] = nodes
        
        # Log deserialization stats
        deserialization_time = time.time() - start_time
        node_count = len(tree.get('nodes', [])) if include_nodes else 'N/A'
        logger.info(f"Deserialized tree {tree_id} with {node_count} nodes "
                   f"in {deserialization_time:.2f}s")
        
        return tree
    
    def _get_tree_metadata(self, tree_id: str) -> Dict[str, Any]:
        """Get tree metadata from the database.
        
        Args:
            tree_id: The ID of the tree.
            
        Returns:
            The tree metadata.
        """
        conn = self.connection.get_connection()
        cursor = conn.cursor()
        
        try:
            query = f"SELECT * FROM {self.trees_table} WHERE tree_id = :tree_id"
            cursor.execute(query, {'tree_id': tree_id})
            row = cursor.fetchone()
            
            if not row:
                return {}
            
            # Convert row to dict
            columns = [desc[0] for desc in cursor.description]
            tree = dict(zip(columns, row))
            
            # Parse metadata if it's a string
            if 'metadata' in tree and isinstance(tree['metadata'], str):
                try:
                    tree['metadata'] = json.loads(tree['metadata'])
                except json.JSONDecodeError:
                    pass
            
            return tree
        finally:
            self.connection.release_connection(conn)
    
    def _get_batched_nodes(self, tree_id: str, max_nodes: int) -> List[Dict[str, Any]]:
        """Get nodes for a tree in batches.
        
        Args:
            tree_id: The ID of the tree.
            max_nodes: Maximum number of nodes to retrieve.
            
        Returns:
            The list of nodes.
        """
        # Get batch information
        batches = self._get_batch_info(tree_id)
        
        if not batches:
            logger.warning(f"No batches found for tree {tree_id}")
            return []
        
        # Sort batches by batch number
        batches.sort(key=lambda b: b['batch_number'])
        
        # Load nodes batch by batch until max_nodes is reached
        nodes = []
        nodes_loaded = 0
        
        for batch in batches:
            batch_nodes = self._get_batch_nodes(tree_id, batch['batch_number'])
            
            # Calculate how many nodes to take from this batch
            nodes_remaining = max_nodes - nodes_loaded
            if nodes_remaining <= 0:
                break
            
            if len(batch_nodes) > nodes_remaining:
                batch_nodes = batch_nodes[:nodes_remaining]
            
            nodes.extend(batch_nodes)
            nodes_loaded += len(batch_nodes)
            
            # Force garbage collection if needed
            self._check_memory_usage()
            
            if nodes_loaded >= max_nodes:
                break
        
        return nodes
    
    def _get_batch_info(self, tree_id: str) -> List[Dict[str, Any]]:
        """Get batch information for a tree.
        
        Args:
            tree_id: The ID of the tree.
            
        Returns:
            The list of batch information.
        """
        conn = self.connection.get_connection()
        cursor = conn.cursor()
        
        try:
            query = f"""
            SELECT batch_id, tree_id, batch_number, node_count, created_at
            FROM {self.batches_table}
            WHERE tree_id = :tree_id
            ORDER BY batch_number
            """
            cursor.execute(query, {'tree_id': tree_id})
            rows = cursor.fetchall()
            
            if not rows:
                return []
            
            # Convert rows to dicts
            columns = [desc[0] for desc in cursor.description]
            batches = [dict(zip(columns, row)) for row in rows]
            
            return batches
        finally:
            self.connection.release_connection(conn)
    
    def _get_batch_nodes(self, tree_id: str, batch_number: int) -> List[Dict[str, Any]]:
        """Get nodes for a specific batch.
        
        Args:
            tree_id: The ID of the tree.
            batch_number: The batch number.
            
        Returns:
            The list of nodes for the batch.
        """
        conn = self.connection.get_connection()
        cursor = conn.cursor()
        
        try:
            # Get batch size for pagination
            query = f"""
            SELECT node_count
            FROM {self.batches_table}
            WHERE tree_id = :tree_id AND batch_number = :batch_number
            """
            cursor.execute(query, {
                'tree_id': tree_id,
                'batch_number': batch_number
            })
            row = cursor.fetchone()
            
            if not row:
                return []
            
            batch_size = row[0]
            
            # Calculate OFFSET based on batch number and size
            offset = batch_number * self.node_batch_size
            
            # Get nodes for this batch
            query = f"""
            SELECT id, tree_id, parent_id, visit_count, value, state, action
            FROM {self.nodes_table}
            WHERE tree_id = :tree_id
            ORDER BY id
            LIMIT {batch_size} OFFSET {offset}
            """
            cursor.execute(query, {'tree_id': tree_id})
            rows = cursor.fetchall()
            
            if not rows:
                return []
            
            # Convert rows to dicts
            columns = [desc[0] for desc in cursor.description]
            nodes = []
            
            for row in rows:
                node = dict(zip(columns, row))
                
                # Parse state and action if they're strings
                for field in ['state', 'action']:
                    if field in node and isinstance(node[field], str):
                        try:
                            node[field] = json.loads(node[field])
                        except json.JSONDecodeError:
                            pass
                
                nodes.append(node)
            
            return nodes
        finally:
            self.connection.release_connection(conn)
    
    def _get_nodes(self, tree_id: str, max_nodes: int) -> List[Dict[str, Any]]:
        """Get nodes for a tree directly (without batching).
        
        Args:
            tree_id: The ID of the tree.
            max_nodes: Maximum number of nodes to retrieve.
            
        Returns:
            The list of nodes.
        """
        conn = self.connection.get_connection()
        cursor = conn.cursor()
        
        try:
            query = f"""
            SELECT id, tree_id, parent_id, visit_count, value, state, action
            FROM {self.nodes_table}
            WHERE tree_id = :tree_id
            ORDER BY id
            LIMIT {max_nodes}
            """
            cursor.execute(query, {'tree_id': tree_id})
            rows = cursor.fetchall()
            
            if not rows:
                return []
            
            # Convert rows to dicts
            columns = [desc[0] for desc in cursor.description]
            nodes = []
            
            for row in rows:
                node = dict(zip(columns, row))
                
                # Parse state and action if they're strings
                for field in ['state', 'action']:
                    if field in node and isinstance(node[field], str):
                        try:
                            node[field] = json.loads(node[field])
                        except json.JSONDecodeError:
                            pass
                
                nodes.append(node)
            
            return nodes
        finally:
            self.connection.release_connection(conn)
    
    def delete_tree(self, tree_id: str) -> bool:
        """Delete a tree and all its batches and nodes.
        
        Args:
            tree_id: The ID of the tree to delete.
            
        Returns:
            True if the tree was deleted, False otherwise.
        """
        conn = self.connection.get_connection()
        cursor = conn.cursor()
        
        try:
            # Delete nodes first (due to foreign key constraints)
            query = f"DELETE FROM {self.nodes_table} WHERE tree_id = :tree_id"
            cursor.execute(query, {'tree_id': tree_id})
            
            # Delete batches
            query = f"DELETE FROM {self.batches_table} WHERE tree_id = :tree_id"
            cursor.execute(query, {'tree_id': tree_id})
            
            # Delete tree
            query = f"DELETE FROM {self.trees_table} WHERE tree_id = :tree_id"
            cursor.execute(query, {'tree_id': tree_id})
            
            return True
        except Exception as e:
            logger.error(f"Error deleting tree {tree_id}: {e}")
            return False
        finally:
            self.connection.release_connection(conn)


# Convenience functions

def batch_serialize_tree(connection: HanaConnection, tree: Dict[str, Any]) -> str:
    """Serialize a tree to the database in batches.
    
    Args:
        connection: SAP HANA connection.
        tree: The tree structure to serialize.
        
    Returns:
        The tree_id of the serialized tree.
    """
    serializer = BatchedTreeSerializer(connection)
    return serializer.batch_serialize_tree(tree)


def batch_deserialize_tree(connection: HanaConnection, 
                          tree_id: str, 
                          include_nodes: bool = True,
                          max_nodes: Optional[int] = None) -> Dict[str, Any]:
    """Deserialize a tree from the database in batches.
    
    Args:
        connection: SAP HANA connection.
        tree_id: The ID of the tree to deserialize.
        include_nodes: Whether to include nodes in the result.
        max_nodes: Maximum number of nodes to deserialize (for partial loading).
        
    Returns:
        The deserialized tree structure.
    """
    serializer = BatchedTreeSerializer(connection)
    return serializer.batch_deserialize_tree(tree_id, include_nodes, max_nodes)


def delete_tree(connection: HanaConnection, tree_id: str) -> bool:
    """Delete a tree and all its batches and nodes.
    
    Args:
        connection: SAP HANA connection.
        tree_id: The ID of the tree to delete.
        
    Returns:
        True if the tree was deleted, False otherwise.
    """
    serializer = BatchedTreeSerializer(connection)
    return serializer.delete_tree(tree_id)