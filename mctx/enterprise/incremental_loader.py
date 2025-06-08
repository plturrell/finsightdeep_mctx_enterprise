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
"""Incremental loading system for large MCTS trees with SAP HANA.

This module provides memory-efficient incremental loading for extremely large
MCTS tree structures when working with SAP HANA.
"""

import gc
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union, Generator, Set, Callable

import numpy as np

from mctx.enterprise.hana_integration import HanaConnection
from mctx.enterprise.batched_serialization import BatchedTreeSerializer

# Configure default settings
DEFAULT_PAGE_SIZE = int(os.environ.get('MCTX_INCREMENTAL_PAGE_SIZE', '1000'))
DEFAULT_MAX_DEPTH = int(os.environ.get('MCTX_INCREMENTAL_MAX_DEPTH', '100'))
DEFAULT_PREFETCH_SIZE = int(os.environ.get('MCTX_INCREMENTAL_PREFETCH_SIZE', '5000'))
DEFAULT_MEMORY_LIMIT_MB = int(os.environ.get('MCTX_INCREMENTAL_MEMORY_LIMIT_MB', '1024'))

# Configure logging
logger = logging.getLogger("mctx.enterprise.incremental_loader")


class TreeNodePage:
    """A page of tree nodes for incremental loading."""
    
    def __init__(self, nodes: List[Dict[str, Any]], 
                has_more: bool = False, 
                next_page_token: Optional[str] = None):
        """Initialize a new tree node page.
        
        Args:
            nodes: The list of nodes in this page.
            has_more: Whether there are more pages available.
            next_page_token: Token for retrieving the next page.
        """
        self.nodes = nodes
        self.has_more = has_more
        self.next_page_token = next_page_token
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the page to a dictionary.
        
        Returns:
            Dictionary representation of the page.
        """
        return {
            'nodes': self.nodes,
            'has_more': self.has_more,
            'next_page_token': self.next_page_token
        }


class IncrementalTreeLoader:
    """Memory-efficient incremental loader for large MCTS trees.
    
    This class provides methods to load large MCTS trees incrementally, which
    helps avoid out-of-memory errors when working with very large tree structures.
    """
    
    def __init__(self, 
                connection: HanaConnection,
                page_size: int = DEFAULT_PAGE_SIZE,
                max_depth: int = DEFAULT_MAX_DEPTH,
                prefetch_size: int = DEFAULT_PREFETCH_SIZE,
                memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB):
        """Initialize a new incremental tree loader.
        
        Args:
            connection: SAP HANA connection.
            page_size: Number of nodes to load per page.
            max_depth: Maximum depth to traverse when loading by depth.
            prefetch_size: Number of nodes to prefetch for faster access.
            memory_limit_mb: Memory limit in MB before forcing garbage collection.
        """
        self.connection = connection
        self.page_size = page_size
        self.max_depth = max_depth
        self.prefetch_size = prefetch_size
        self.memory_limit_mb = memory_limit_mb
        
        # Table names
        self.schema = connection.config.schema
        self.trees_table = f"{self.schema}.MCTS_TREES"
        self.nodes_table = f"{self.schema}.MCTS_NODES"
        
        # Serializer for batched operations
        self.serializer = BatchedTreeSerializer(connection)
        
        # Node cache for prefetching
        self._node_cache: Dict[str, Dict[str, Any]] = {}
        self._prefetch_counts: Dict[str, int] = {}
    
    def _check_memory_usage(self) -> None:
        """Check memory usage and force garbage collection if needed."""
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        
        if memory_mb > self.memory_limit_mb:
            logger.info(f"Memory usage ({memory_mb:.1f} MB) exceeded limit "
                       f"({self.memory_limit_mb} MB), forcing garbage collection")
            
            # Clear node cache to free memory
            self._clear_node_cache()
            
            # Force garbage collection
            gc.collect()
    
    def _clear_node_cache(self) -> None:
        """Clear the node cache to free memory."""
        self._node_cache.clear()
        self._prefetch_counts.clear()
    
    def get_tree_metadata(self, tree_id: str) -> Dict[str, Any]:
        """Get tree metadata without loading nodes.
        
        Args:
            tree_id: The ID of the tree.
            
        Returns:
            The tree metadata.
        """
        return self.serializer._get_tree_metadata(tree_id)
    
    def load_tree_by_pages(self, tree_id: str, 
                          page_token: Optional[str] = None) -> TreeNodePage:
        """Load a tree incrementally by pages.
        
        Args:
            tree_id: The ID of the tree to load.
            page_token: Token for retrieving a specific page.
            
        Returns:
            A page of tree nodes.
        """
        # Start timer
        start_time = time.time()
        
        # Parse page token
        offset = 0
        if page_token:
            try:
                offset = int(page_token)
            except ValueError:
                logger.warning(f"Invalid page token: {page_token}, using default offset")
        
        # Get nodes for this page
        nodes = self._get_nodes_page(tree_id, offset, self.page_size)
        
        # Check if there are more pages
        has_more = len(nodes) == self.page_size
        next_page_token = str(offset + self.page_size) if has_more else None
        
        # Log timing
        loading_time = time.time() - start_time
        logger.info(f"Loaded page of {len(nodes)} nodes for tree {tree_id} "
                   f"(offset={offset}) in {loading_time:.2f}s")
        
        return TreeNodePage(nodes, has_more, next_page_token)
    
    def _get_nodes_page(self, tree_id: str, offset: int, limit: int) -> List[Dict[str, Any]]:
        """Get a page of nodes for a tree.
        
        Args:
            tree_id: The ID of the tree.
            offset: Offset for pagination.
            limit: Maximum number of nodes to retrieve.
            
        Returns:
            The list of nodes for the page.
        """
        conn = self.connection.get_connection()
        cursor = conn.cursor()
        
        try:
            query = f"""
            SELECT id, tree_id, parent_id, visit_count, value, state, action
            FROM {self.nodes_table}
            WHERE tree_id = :tree_id
            ORDER BY id
            LIMIT {limit} OFFSET {offset}
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
    
    def load_tree_by_depth(self, tree_id: str, 
                         max_depth: Optional[int] = None, 
                         start_node_id: Optional[str] = None) -> Dict[str, Any]:
        """Load a tree incrementally by depth.
        
        Args:
            tree_id: The ID of the tree to load.
            max_depth: Maximum depth to traverse (default is class max_depth).
            start_node_id: ID of the node to start from (default is root).
            
        Returns:
            The tree structure with nodes up to the specified depth.
        """
        # Start timer
        start_time = time.time()
        
        # Use default max_depth if not specified
        if max_depth is None:
            max_depth = self.max_depth
        
        # Get tree metadata
        tree = self.get_tree_metadata(tree_id)
        
        if not tree:
            logger.warning(f"Tree {tree_id} not found")
            return {}
        
        # Get root node if start_node_id is not specified
        if not start_node_id:
            root_nodes = self._get_root_nodes(tree_id)
            if not root_nodes:
                logger.warning(f"No root nodes found for tree {tree_id}")
                tree['nodes'] = []
                return tree
            
            start_node_id = root_nodes[0]['id']
        
        # Load nodes by BFS up to max_depth
        nodes = self._load_nodes_by_bfs(tree_id, start_node_id, max_depth)
        
        # Add nodes to tree
        tree['nodes'] = nodes
        
        # Log timing
        loading_time = time.time() - start_time
        logger.info(f"Loaded tree {tree_id} with {len(nodes)} nodes "
                   f"up to depth {max_depth} in {loading_time:.2f}s")
        
        return tree
    
    def _get_root_nodes(self, tree_id: str) -> List[Dict[str, Any]]:
        """Get root nodes for a tree.
        
        Args:
            tree_id: The ID of the tree.
            
        Returns:
            The list of root nodes.
        """
        conn = self.connection.get_connection()
        cursor = conn.cursor()
        
        try:
            # Root nodes have empty or NULL parent_id
            query = f"""
            SELECT id, tree_id, parent_id, visit_count, value, state, action
            FROM {self.nodes_table}
            WHERE tree_id = :tree_id AND (parent_id IS NULL OR parent_id = '')
            ORDER BY id
            LIMIT 10
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
    
    def _load_nodes_by_bfs(self, tree_id: str, 
                          start_node_id: str, 
                          max_depth: int) -> List[Dict[str, Any]]:
        """Load nodes by breadth-first search up to a maximum depth.
        
        Args:
            tree_id: The ID of the tree.
            start_node_id: ID of the node to start from.
            max_depth: Maximum depth to traverse.
            
        Returns:
            The list of nodes up to the specified depth.
        """
        # Clear node cache
        self._clear_node_cache()
        
        # Get the start node
        start_node = self._get_node_by_id(tree_id, start_node_id)
        if not start_node:
            logger.warning(f"Start node {start_node_id} not found for tree {tree_id}")
            return []
        
        # Initialize BFS
        all_nodes: Dict[str, Dict[str, Any]] = {start_node_id: start_node}
        queue: List[Tuple[str, int]] = [(start_node_id, 0)]  # (node_id, depth)
        queue_index = 0
        
        # Prefetch children of the start node
        self._prefetch_children(tree_id, start_node_id)
        
        while queue_index < len(queue):
            node_id, depth = queue[queue_index]
            queue_index += 1
            
            # Stop if we've reached the maximum depth
            if depth >= max_depth:
                continue
            
            # Get children of this node
            children = self._get_node_children(tree_id, node_id)
            
            # Add children to the queue and nodes list
            for child in children:
                child_id = child['id']
                
                # Skip if we've already processed this node
                if child_id in all_nodes:
                    continue
                
                all_nodes[child_id] = child
                queue.append((child_id, depth + 1))
                
                # Prefetch children for the next level if we're not near the max depth
                if depth < max_depth - 1:
                    self._prefetch_children(tree_id, child_id)
            
            # Check memory usage periodically
            if queue_index % 100 == 0:
                self._check_memory_usage()
        
        # Convert the nodes dict to a list
        return list(all_nodes.values())
    
    def _get_node_by_id(self, tree_id: str, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node by its ID.
        
        Args:
            tree_id: The ID of the tree.
            node_id: The ID of the node.
            
        Returns:
            The node or None if not found.
        """
        # Check cache first
        if node_id in self._node_cache:
            # Update prefetch count
            self._prefetch_counts[node_id] = self._prefetch_counts.get(node_id, 0) + 1
            return self._node_cache[node_id]
        
        # Query the database
        conn = self.connection.get_connection()
        cursor = conn.cursor()
        
        try:
            query = f"""
            SELECT id, tree_id, parent_id, visit_count, value, state, action
            FROM {self.nodes_table}
            WHERE tree_id = :tree_id AND id = :node_id
            """
            cursor.execute(query, {'tree_id': tree_id, 'node_id': node_id})
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Convert row to dict
            columns = [desc[0] for desc in cursor.description]
            node = dict(zip(columns, row))
            
            # Parse state and action if they're strings
            for field in ['state', 'action']:
                if field in node and isinstance(node[field], str):
                    try:
                        node[field] = json.loads(node[field])
                    except json.JSONDecodeError:
                        pass
            
            return node
        finally:
            self.connection.release_connection(conn)
    
    def _get_node_children(self, tree_id: str, node_id: str) -> List[Dict[str, Any]]:
        """Get the children of a node.
        
        Args:
            tree_id: The ID of the tree.
            node_id: The ID of the parent node.
            
        Returns:
            The list of child nodes.
        """
        # Check cache for children first
        children = []
        for cached_id, node in self._node_cache.items():
            if node.get('parent_id') == node_id and node.get('tree_id') == tree_id:
                children.append(node)
                # Update prefetch count
                self._prefetch_counts[cached_id] = self._prefetch_counts.get(cached_id, 0) + 1
        
        if children:
            return children
        
        # Query the database
        conn = self.connection.get_connection()
        cursor = conn.cursor()
        
        try:
            query = f"""
            SELECT id, tree_id, parent_id, visit_count, value, state, action
            FROM {self.nodes_table}
            WHERE tree_id = :tree_id AND parent_id = :parent_id
            ORDER BY visit_count DESC
            """
            cursor.execute(query, {'tree_id': tree_id, 'parent_id': node_id})
            rows = cursor.fetchall()
            
            if not rows:
                return []
            
            # Convert rows to dicts
            columns = [desc[0] for desc in cursor.description]
            children = []
            
            for row in rows:
                node = dict(zip(columns, row))
                
                # Parse state and action if they're strings
                for field in ['state', 'action']:
                    if field in node and isinstance(node[field], str):
                        try:
                            node[field] = json.loads(node[field])
                        except json.JSONDecodeError:
                            pass
                
                children.append(node)
                
                # Add to cache
                self._node_cache[node['id']] = node
                self._prefetch_counts[node['id']] = 1
            
            return children
        finally:
            self.connection.release_connection(conn)
    
    def _prefetch_children(self, tree_id: str, node_id: str) -> None:
        """Prefetch children of a node for faster access.
        
        Args:
            tree_id: The ID of the tree.
            node_id: The ID of the parent node.
        """
        # Check if we need to clean the cache first
        if len(self._node_cache) >= self.prefetch_size:
            # Remove least accessed nodes
            items = sorted(self._prefetch_counts.items(), key=lambda x: x[1])
            to_remove = len(items) // 4  # Remove 25% of the least accessed nodes
            
            for i in range(to_remove):
                node_id_to_remove = items[i][0]
                if node_id_to_remove in self._node_cache:
                    del self._node_cache[node_id_to_remove]
                del self._prefetch_counts[node_id_to_remove]
        
        conn = self.connection.get_connection()
        cursor = conn.cursor()
        
        try:
            # Get all descendants within 2 levels
            query = f"""
            SELECT c.id, c.tree_id, c.parent_id, c.visit_count, c.value, c.state, c.action
            FROM {self.nodes_table} c
            WHERE c.tree_id = :tree_id AND c.parent_id = :parent_id
            ORDER BY c.visit_count DESC
            LIMIT 100
            """
            cursor.execute(query, {'tree_id': tree_id, 'parent_id': node_id})
            rows = cursor.fetchall()
            
            if not rows:
                return
            
            # Convert rows to dicts and add to cache
            columns = [desc[0] for desc in cursor.description]
            
            for row in rows:
                node = dict(zip(columns, row))
                
                # Parse state and action if they're strings
                for field in ['state', 'action']:
                    if field in node and isinstance(node[field], str):
                        try:
                            node[field] = json.loads(node[field])
                        except json.JSONDecodeError:
                            pass
                
                # Add to cache
                self._node_cache[node['id']] = node
                self._prefetch_counts[node['id']] = 0
            
            # Prefetch next level for nodes with high visit counts
            for i, row in enumerate(rows):
                if i >= 5:  # Only prefetch for top 5 nodes
                    break
                
                node = dict(zip(columns, row))
                child_id = node['id']
                
                # Prefetch grandchildren for nodes with high visit counts
                query = f"""
                SELECT gc.id, gc.tree_id, gc.parent_id, gc.visit_count, gc.value, gc.state, gc.action
                FROM {self.nodes_table} gc
                WHERE gc.tree_id = :tree_id AND gc.parent_id = :parent_id
                ORDER BY gc.visit_count DESC
                LIMIT 20
                """
                cursor.execute(query, {'tree_id': tree_id, 'parent_id': child_id})
                gc_rows = cursor.fetchall()
                
                for gc_row in gc_rows:
                    gc_node = dict(zip(columns, gc_row))
                    
                    # Parse state and action if they're strings
                    for field in ['state', 'action']:
                        if field in gc_node and isinstance(gc_node[field], str):
                            try:
                                gc_node[field] = json.loads(gc_node[field])
                            except json.JSONDecodeError:
                                pass
                    
                    # Add to cache
                    self._node_cache[gc_node['id']] = gc_node
                    self._prefetch_counts[gc_node['id']] = 0
        finally:
            self.connection.release_connection(conn)
    
    def load_subtree(self, tree_id: str, 
                    node_id: str, 
                    max_depth: int = 3, 
                    max_nodes: int = 1000) -> Dict[str, Any]:
        """Load a subtree starting from a specific node.
        
        Args:
            tree_id: The ID of the tree.
            node_id: The ID of the node to start from.
            max_depth: Maximum depth to traverse from the start node.
            max_nodes: Maximum number of nodes to load.
            
        Returns:
            The subtree structure.
        """
        # Start timer
        start_time = time.time()
        
        # Get tree metadata
        tree = self.get_tree_metadata(tree_id)
        
        if not tree:
            logger.warning(f"Tree {tree_id} not found")
            return {}
        
        # Get the start node
        start_node = self._get_node_by_id(tree_id, node_id)
        if not start_node:
            logger.warning(f"Start node {node_id} not found for tree {tree_id}")
            tree['nodes'] = []
            return tree
        
        # Load subtree
        nodes = self._load_subtree(tree_id, node_id, max_depth, max_nodes)
        
        # Add nodes to tree
        tree['nodes'] = nodes
        
        # Log timing
        loading_time = time.time() - start_time
        logger.info(f"Loaded subtree for node {node_id} in tree {tree_id} "
                   f"with {len(nodes)} nodes in {loading_time:.2f}s")
        
        return tree
    
    def _load_subtree(self, tree_id: str, 
                     node_id: str, 
                     max_depth: int, 
                     max_nodes: int) -> List[Dict[str, Any]]:
        """Load a subtree starting from a specific node.
        
        Args:
            tree_id: The ID of the tree.
            node_id: The ID of the node to start from.
            max_depth: Maximum depth to traverse from the start node.
            max_nodes: Maximum number of nodes to load.
            
        Returns:
            The list of nodes in the subtree.
        """
        # Clear node cache
        self._clear_node_cache()
        
        # Get the start node
        start_node = self._get_node_by_id(tree_id, node_id)
        if not start_node:
            return []
        
        # Initialize BFS
        all_nodes: Dict[str, Dict[str, Any]] = {node_id: start_node}
        queue: List[Tuple[str, int]] = [(node_id, 0)]  # (node_id, depth)
        queue_index = 0
        
        # Prefetch children of the start node
        self._prefetch_children(tree_id, node_id)
        
        while queue_index < len(queue) and len(all_nodes) < max_nodes:
            current_id, depth = queue[queue_index]
            queue_index += 1
            
            # Stop if we've reached the maximum depth
            if depth >= max_depth:
                continue
            
            # Get children of this node
            children = self._get_node_children(tree_id, current_id)
            
            # Add children to the queue and nodes list
            for child in children:
                child_id = child['id']
                
                # Skip if we've already processed this node
                if child_id in all_nodes:
                    continue
                
                all_nodes[child_id] = child
                
                # Stop if we've reached the maximum number of nodes
                if len(all_nodes) >= max_nodes:
                    break
                
                queue.append((child_id, depth + 1))
                
                # Prefetch children for the next level if we're not near the max depth
                if depth < max_depth - 1:
                    self._prefetch_children(tree_id, child_id)
            
            # Check memory usage periodically
            if queue_index % 100 == 0:
                self._check_memory_usage()
        
        # Convert the nodes dict to a list
        return list(all_nodes.values())
    
    def load_path_to_node(self, tree_id: str, node_id: str) -> Dict[str, Any]:
        """Load the path from the root to a specific node.
        
        Args:
            tree_id: The ID of the tree.
            node_id: The ID of the target node.
            
        Returns:
            The tree structure with the path to the node.
        """
        # Start timer
        start_time = time.time()
        
        # Get tree metadata
        tree = self.get_tree_metadata(tree_id)
        
        if not tree:
            logger.warning(f"Tree {tree_id} not found")
            return {}
        
        # Get the path from the node to the root
        path_nodes = self._get_path_to_root(tree_id, node_id)
        
        if not path_nodes:
            logger.warning(f"Node {node_id} not found or no path to root for tree {tree_id}")
            tree['nodes'] = []
            return tree
        
        # Add nodes to tree
        tree['nodes'] = path_nodes
        
        # Log timing
        loading_time = time.time() - start_time
        logger.info(f"Loaded path to node {node_id} in tree {tree_id} "
                   f"with {len(path_nodes)} nodes in {loading_time:.2f}s")
        
        return tree
    
    def _get_path_to_root(self, tree_id: str, node_id: str) -> List[Dict[str, Any]]:
        """Get the path from a node to the root.
        
        Args:
            tree_id: The ID of the tree.
            node_id: The ID of the node.
            
        Returns:
            The list of nodes in the path from the node to the root.
        """
        # Clear node cache
        self._clear_node_cache()
        
        # Get the target node
        node = self._get_node_by_id(tree_id, node_id)
        if not node:
            return []
        
        # Initialize path
        path_nodes: Dict[str, Dict[str, Any]] = {node_id: node}
        current_id = node_id
        
        # Traverse up to the root
        while True:
            # Get the current node
            current_node = path_nodes[current_id]
            parent_id = current_node.get('parent_id')
            
            # Stop if we've reached the root or a node without a parent
            if not parent_id or parent_id == '' or parent_id in path_nodes:
                break
            
            # Get the parent node
            parent_node = self._get_node_by_id(tree_id, parent_id)
            if not parent_node:
                break
            
            # Add parent to path
            path_nodes[parent_id] = parent_node
            current_id = parent_id
        
        # Get children of nodes in the path
        for path_id in list(path_nodes.keys()):
            # Get immediate children of this path node
            children = self._get_node_children(tree_id, path_id)
            
            # Add children to the path
            for child in children[:10]:  # Limit to 10 children per node
                child_id = child['id']
                if child_id not in path_nodes:
                    path_nodes[child_id] = child
        
        # Convert the path dict to a list
        return list(path_nodes.values())
    
    def load_high_value_nodes(self, tree_id: str, 
                             max_nodes: int = 100,
                             value_threshold: Optional[float] = None) -> Dict[str, Any]:
        """Load the highest value nodes from a tree.
        
        Args:
            tree_id: The ID of the tree.
            max_nodes: Maximum number of nodes to load.
            value_threshold: Minimum value threshold for nodes.
            
        Returns:
            The tree structure with high value nodes.
        """
        # Start timer
        start_time = time.time()
        
        # Get tree metadata
        tree = self.get_tree_metadata(tree_id)
        
        if not tree:
            logger.warning(f"Tree {tree_id} not found")
            return {}
        
        # Get high value nodes
        high_value_nodes = self._get_high_value_nodes(tree_id, max_nodes, value_threshold)
        
        if not high_value_nodes:
            logger.warning(f"No high value nodes found for tree {tree_id}")
            tree['nodes'] = []
            return tree
        
        # Add nodes to tree
        tree['nodes'] = high_value_nodes
        
        # Log timing
        loading_time = time.time() - start_time
        logger.info(f"Loaded {len(high_value_nodes)} high value nodes for tree {tree_id} "
                   f"in {loading_time:.2f}s")
        
        return tree
    
    def _get_high_value_nodes(self, tree_id: str, 
                             max_nodes: int,
                             value_threshold: Optional[float]) -> List[Dict[str, Any]]:
        """Get the highest value nodes from a tree.
        
        Args:
            tree_id: The ID of the tree.
            max_nodes: Maximum number of nodes to load.
            value_threshold: Minimum value threshold for nodes.
            
        Returns:
            The list of high value nodes.
        """
        conn = self.connection.get_connection()
        cursor = conn.cursor()
        
        try:
            # Build query based on parameters
            if value_threshold is not None:
                query = f"""
                SELECT id, tree_id, parent_id, visit_count, value, state, action
                FROM {self.nodes_table}
                WHERE tree_id = :tree_id AND value >= :value_threshold
                ORDER BY value DESC, visit_count DESC
                LIMIT :max_nodes
                """
                params = {
                    'tree_id': tree_id,
                    'value_threshold': value_threshold,
                    'max_nodes': max_nodes
                }
            else:
                query = f"""
                SELECT id, tree_id, parent_id, visit_count, value, state, action
                FROM {self.nodes_table}
                WHERE tree_id = :tree_id
                ORDER BY value DESC, visit_count DESC
                LIMIT :max_nodes
                """
                params = {
                    'tree_id': tree_id,
                    'max_nodes': max_nodes
                }
            
            cursor.execute(query, params)
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
            
            # Get path to root for each high value node
            high_value_nodes: Dict[str, Dict[str, Any]] = {}
            for node in nodes:
                node_id = node['id']
                high_value_nodes[node_id] = node
                
                # Get path to root
                path_nodes = self._get_path_to_root(tree_id, node_id)
                
                # Add path nodes
                for path_node in path_nodes:
                    path_id = path_node['id']
                    if path_id not in high_value_nodes:
                        high_value_nodes[path_id] = path_node
            
            return list(high_value_nodes.values())
        finally:
            self.connection.release_connection(conn)
    
    def iterate_tree_nodes(self, tree_id: str, 
                          batch_size: int = 1000,
                          node_filter: Optional[Callable[[Dict[str, Any]], bool]] = None) -> Generator[List[Dict[str, Any]], None, None]:
        """Iterate through all nodes in a tree in batches.
        
        Args:
            tree_id: The ID of the tree.
            batch_size: Number of nodes to load per batch.
            node_filter: Optional filter function for nodes.
            
        Yields:
            Batches of nodes.
        """
        offset = 0
        while True:
            # Get a batch of nodes
            nodes = self._get_nodes_page(tree_id, offset, batch_size)
            
            if not nodes:
                break
            
            # Apply filter if provided
            if node_filter:
                nodes = [node for node in nodes if node_filter(node)]
            
            if nodes:
                yield nodes
            
            # Check if we've reached the end
            if len(nodes) < batch_size:
                break
            
            offset += batch_size
            
            # Check memory usage
            self._check_memory_usage()


# Convenience functions

def load_tree_by_pages(connection: HanaConnection, 
                     tree_id: str, 
                     page_token: Optional[str] = None,
                     page_size: int = DEFAULT_PAGE_SIZE) -> Dict[str, Any]:
    """Load a tree incrementally by pages.
    
    Args:
        connection: SAP HANA connection.
        tree_id: The ID of the tree to load.
        page_token: Token for retrieving a specific page.
        page_size: Number of nodes to load per page.
        
    Returns:
        A dictionary with the page of tree nodes.
    """
    loader = IncrementalTreeLoader(connection, page_size=page_size)
    page = loader.load_tree_by_pages(tree_id, page_token)
    return page.to_dict()


def load_tree_by_depth(connection: HanaConnection, 
                     tree_id: str, 
                     max_depth: int = DEFAULT_MAX_DEPTH,
                     start_node_id: Optional[str] = None) -> Dict[str, Any]:
    """Load a tree incrementally by depth.
    
    Args:
        connection: SAP HANA connection.
        tree_id: The ID of the tree to load.
        max_depth: Maximum depth to traverse.
        start_node_id: ID of the node to start from (default is root).
        
    Returns:
        The tree structure with nodes up to the specified depth.
    """
    loader = IncrementalTreeLoader(connection, max_depth=max_depth)
    return loader.load_tree_by_depth(tree_id, max_depth, start_node_id)


def load_subtree(connection: HanaConnection, 
                tree_id: str, 
                node_id: str, 
                max_depth: int = 3, 
                max_nodes: int = 1000) -> Dict[str, Any]:
    """Load a subtree starting from a specific node.
    
    Args:
        connection: SAP HANA connection.
        tree_id: The ID of the tree.
        node_id: The ID of the node to start from.
        max_depth: Maximum depth to traverse from the start node.
        max_nodes: Maximum number of nodes to load.
        
    Returns:
        The subtree structure.
    """
    loader = IncrementalTreeLoader(connection)
    return loader.load_subtree(tree_id, node_id, max_depth, max_nodes)


def load_path_to_node(connection: HanaConnection, 
                     tree_id: str, 
                     node_id: str) -> Dict[str, Any]:
    """Load the path from the root to a specific node.
    
    Args:
        connection: SAP HANA connection.
        tree_id: The ID of the tree.
        node_id: The ID of the target node.
        
    Returns:
        The tree structure with the path to the node.
    """
    loader = IncrementalTreeLoader(connection)
    return loader.load_path_to_node(tree_id, node_id)


def load_high_value_nodes(connection: HanaConnection, 
                        tree_id: str, 
                        max_nodes: int = 100,
                        value_threshold: Optional[float] = None) -> Dict[str, Any]:
    """Load the highest value nodes from a tree.
    
    Args:
        connection: SAP HANA connection.
        tree_id: The ID of the tree.
        max_nodes: Maximum number of nodes to load.
        value_threshold: Minimum value threshold for nodes.
        
    Returns:
        The tree structure with high value nodes.
    """
    loader = IncrementalTreeLoader(connection)
    return loader.load_high_value_nodes(tree_id, max_nodes, value_threshold)