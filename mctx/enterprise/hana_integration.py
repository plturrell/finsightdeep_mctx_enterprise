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
"""SAP HANA integration for MCTX."""

import base64
import datetime
import functools
import io
import json
import pickle
import time
import uuid
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, TypeVar, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_map, tree_flatten, tree_unflatten

try:
    import hdbcli
    import hdbcli.dbapi
    from hdbcli.dbapi import Connection as HDBConnection
    HANA_AVAILABLE = True
except ImportError:
    HANA_AVAILABLE = False

from mctx._src import base
from mctx._src import tree as tree_lib

# Type definitions
T = TypeVar('T')
Tree = tree_lib.Tree
ArrayTree = base.ArrayTree


class HanaConfig(NamedTuple):
    """Configuration for SAP HANA connection.
    
    Attributes:
        host: Hostname or IP address of the SAP HANA server.
        port: Port number for the SAP HANA server.
        user: Username for authentication.
        password: Password for authentication.
        schema: Schema name for MCTX tables.
        encryption: Whether to use encryption for the connection.
        autocommit: Whether to autocommit transactions.
        timeout: Connection timeout in seconds.
        pool_size: Connection pool size.
        use_compression: Whether to compress data before storing.
        compression_level: Compression level (1-9, 9 being highest).
        enable_caching: Whether to enable result caching.
        cache_ttl: Cache time-to-live in seconds.
    """
    host: str
    port: int
    user: str
    password: str
    schema: str = "MCTX"
    encryption: bool = True
    autocommit: bool = True
    timeout: int = 30
    pool_size: int = 10
    use_compression: bool = True
    compression_level: int = 6
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour


class HanaConnection:
    """SAP HANA connection manager for MCTX.
    
    This class handles connection pooling, reconnection, and transaction
    management for SAP HANA database operations.
    """
    
    def __init__(self, config: HanaConfig):
        """Initialize a new SAP HANA connection manager.
        
        Args:
            config: SAP HANA connection configuration.
            
        Raises:
            ImportError: If the hdbcli package is not available.
        """
        if not HANA_AVAILABLE:
            raise ImportError(
                "SAP HANA integration requires the hdbcli package. "
                "Please install it with: pip install hdbcli"
            )
        
        self.config = config
        self._pool = []
        self._in_use = set()
        self._schema_initialized = False
        self._last_ping = 0
        
        # Initialize connection pool
        self._initialize_pool()
        
        # Initialize schema
        self._initialize_schema()
    
    def _initialize_pool(self):
        """Initialize the connection pool."""
        for _ in range(self.config.pool_size):
            conn = self._create_connection()
            if conn is not None:
                self._pool.append(conn)
    
    def _create_connection(self) -> Optional[HDBConnection]:
        """Create a new SAP HANA connection.
        
        Returns:
            A new SAP HANA connection, or None if connection failed.
        """
        try:
            conn = hdbcli.dbapi.connect(
                address=self.config.host,
                port=self.config.port,
                user=self.config.user,
                password=self.config.password,
                encrypt=self.config.encryption,
                autocommit=self.config.autocommit,
                timeout=self.config.timeout,
            )
            return conn
        except Exception as e:
            print(f"Error connecting to SAP HANA: {e}")
            return None
    
    def _initialize_schema(self):
        """Initialize the schema and tables for MCTX."""
        if self._schema_initialized:
            return
        
        conn = self.get_connection()
        if conn is None:
            return
        
        try:
            cursor = conn.cursor()
            
            # Create schema if it doesn't exist
            cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {self.config.schema}")
            
            # Create MCTS trees table
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.config.schema}.MCTS_TREES (
                    tree_id VARCHAR(36) PRIMARY KEY,
                    name VARCHAR(255),
                    batch_size INTEGER,
                    num_actions INTEGER,
                    num_simulations INTEGER,
                    tree_data BLOB,
                    metadata NCLOB,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            """)
            
            # Create model cache table
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.config.schema}.MODEL_CACHE (
                    model_id VARCHAR(36) PRIMARY KEY,
                    name VARCHAR(255),
                    model_type VARCHAR(50),
                    model_data BLOB,
                    metadata NCLOB,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            """)
            
            # Create simulation results table
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.config.schema}.SIMULATION_RESULTS (
                    result_id VARCHAR(36) PRIMARY KEY,
                    tree_id VARCHAR(36),
                    model_id VARCHAR(36),
                    batch_idx INTEGER,
                    visit_counts BLOB,
                    visit_probs BLOB,
                    value BLOB,
                    qvalues BLOB,
                    metadata NCLOB,
                    created_at TIMESTAMP,
                    FOREIGN KEY (tree_id) REFERENCES {self.config.schema}.MCTS_TREES(tree_id),
                    FOREIGN KEY (model_id) REFERENCES {self.config.schema}.MODEL_CACHE(model_id)
                )
            """)
            
            # Create indexes
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_tree_name 
                ON {self.config.schema}.MCTS_TREES(name)
            """)
            
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_model_name 
                ON {self.config.schema}.MODEL_CACHE(name)
            """)
            
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_sim_tree_id 
                ON {self.config.schema}.SIMULATION_RESULTS(tree_id)
            """)
            
            self._schema_initialized = True
        except Exception as e:
            print(f"Error initializing schema: {e}")
        finally:
            self.release_connection(conn)
    
    def get_connection(self) -> Optional[HDBConnection]:
        """Get a connection from the pool.
        
        Returns:
            A SAP HANA connection from the pool, or a new connection if
            the pool is empty.
        """
        # First, ping the database if we haven't in a while
        current_time = time.time()
        if current_time - self._last_ping > 60:  # Ping every minute
            self._ping_connections()
            self._last_ping = current_time
        
        # Try to get a connection from the pool
        if self._pool:
            conn = self._pool.pop()
            self._in_use.add(conn)
            return conn
        
        # If pool is empty, create a new connection
        conn = self._create_connection()
        if conn is not None:
            self._in_use.add(conn)
        return conn
    
    def release_connection(self, conn: HDBConnection):
        """Release a connection back to the pool.
        
        Args:
            conn: The connection to release.
        """
        if conn in self._in_use:
            self._in_use.remove(conn)
            if len(self._pool) < self.config.pool_size:
                self._pool.append(conn)
            else:
                conn.close()
    
    def _ping_connections(self):
        """Ping all idle connections to keep them alive."""
        for i in range(len(self._pool)):
            try:
                conn = self._pool[i]
                cursor = conn.cursor()
                cursor.execute("SELECT 1 FROM DUMMY")
                cursor.close()
            except Exception:
                # Replace bad connection
                try:
                    conn.close()
                except Exception:
                    pass
                self._pool[i] = self._create_connection()
    
    def close_all(self):
        """Close all connections in the pool."""
        # Close connections in the pool
        for conn in self._pool:
            try:
                conn.close()
            except Exception:
                pass
        self._pool = []
        
        # Close connections in use
        for conn in self._in_use:
            try:
                conn.close()
            except Exception:
                pass
        self._in_use = set()


class HanaTreeSerializer:
    """Serializes and deserializes MCTS trees for SAP HANA storage."""
    
    def __init__(self, config: HanaConfig):
        """Initialize the tree serializer.
        
        Args:
            config: SAP HANA configuration.
        """
        self.config = config
    
    def serialize_tree(self, tree: Tree) -> Tuple[bytes, Dict[str, Any]]:
        """Serialize a tree for storage in SAP HANA.
        
        Args:
            tree: The MCTS tree to serialize.
            
        Returns:
            A tuple of (serialized_data, metadata).
        """
        # Get tree shape information for metadata
        batch_size = tree_lib.infer_batch_size(tree)
        num_nodes = tree.node_visits.shape[1]
        num_actions = tree.num_actions
        
        # Create metadata
        metadata = {
            "batch_size": batch_size,
            "num_nodes": num_nodes,
            "num_actions": num_actions,
            "data_type": str(tree.node_values.dtype),
            "tree_structure": "flat_arrays",
            "compression": "pickle" if self.config.use_compression else "none",
        }
        
        # Extract all arrays from the tree
        tree_dict = {
            "node_visits": tree.node_visits,
            "raw_values": tree.raw_values,
            "node_values": tree.node_values,
            "parents": tree.parents,
            "action_from_parent": tree.action_from_parent,
            "children_index": tree.children_index,
            "children_prior_logits": tree.children_prior_logits,
            "children_visits": tree.children_visits,
            "children_rewards": tree.children_rewards,
            "children_discounts": tree.children_discounts,
            "children_values": tree.children_values,
        }
        
        # Handle embeddings specially
        embeddings_flat, embeddings_treedef = tree_flatten(tree.embeddings)
        tree_dict["embeddings_flat"] = embeddings_flat
        metadata["embeddings_treedef"] = str(embeddings_treedef)
        
        # Handle extra_data
        if tree.extra_data is not None:
            try:
                extra_flat, extra_treedef = tree_flatten(tree.extra_data)
                tree_dict["extra_data_flat"] = extra_flat
                metadata["extra_data_treedef"] = str(extra_treedef)
            except Exception:
                # If extra_data can't be flattened, store as is
                tree_dict["extra_data"] = tree.extra_data
                metadata["extra_data_format"] = "raw"
        
        # Include root_invalid_actions
        tree_dict["root_invalid_actions"] = tree.root_invalid_actions
        
        # Convert JAX arrays to NumPy arrays
        tree_dict = tree_map(lambda x: np.asarray(x) if hasattr(x, "shape") else x, tree_dict)
        
        # Serialize with compression
        if self.config.use_compression:
            serialized_data = pickle.dumps(tree_dict, protocol=4)
        else:
            # Without compression, we use a simple dictionary format
            buffer = io.BytesIO()
            np.savez(buffer, **tree_dict)
            serialized_data = buffer.getvalue()
        
        return serialized_data, metadata
    
    def deserialize_tree(self, serialized_data: bytes, metadata: Dict[str, Any]) -> Tree:
        """Deserialize a tree from SAP HANA storage.
        
        Args:
            serialized_data: The serialized tree data.
            metadata: Metadata about the tree.
            
        Returns:
            The deserialized MCTS tree.
        """
        compression = metadata.get("compression", "none")
        
        if compression == "pickle":
            tree_dict = pickle.loads(serialized_data)
        else:
            buffer = io.BytesIO(serialized_data)
            tree_dict = dict(np.load(buffer, allow_pickle=True))
        
        # Handle embeddings
        if "embeddings_treedef" in metadata:
            embeddings_flat = tree_dict["embeddings_flat"]
            embeddings_treedef = eval(metadata["embeddings_treedef"])
            embeddings = tree_unflatten(embeddings_treedef, embeddings_flat)
        else:
            # Default empty embeddings
            embeddings = {}
        
        # Handle extra_data
        if "extra_data_treedef" in metadata:
            extra_flat = tree_dict["extra_data_flat"]
            extra_treedef = eval(metadata["extra_data_treedef"])
            extra_data = tree_unflatten(extra_treedef, extra_flat)
        elif "extra_data_format" in metadata and metadata["extra_data_format"] == "raw":
            extra_data = tree_dict["extra_data"]
        else:
            extra_data = None
        
        # Convert NumPy arrays to JAX arrays
        tree_dict = tree_map(
            lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x, 
            tree_dict
        )
        
        # Create the tree
        tree = Tree(
            node_visits=tree_dict["node_visits"],
            raw_values=tree_dict["raw_values"],
            node_values=tree_dict["node_values"],
            parents=tree_dict["parents"],
            action_from_parent=tree_dict["action_from_parent"],
            children_index=tree_dict["children_index"],
            children_prior_logits=tree_dict["children_prior_logits"],
            children_visits=tree_dict["children_visits"],
            children_rewards=tree_dict["children_rewards"],
            children_discounts=tree_dict["children_discounts"],
            children_values=tree_dict["children_values"],
            embeddings=embeddings,
            root_invalid_actions=tree_dict["root_invalid_actions"],
            extra_data=extra_data
        )
        
        return tree


class HanaModelCache:
    """Caches models in SAP HANA for fast retrieval and sharing."""
    
    def __init__(self, connection: HanaConnection):
        """Initialize the model cache.
        
        Args:
            connection: The SAP HANA connection manager.
        """
        self.connection = connection
        self._local_cache = {}
    
    def save_model(self, model_id: str, model, name: str = None, 
                   model_type: str = "default", metadata: Dict[str, Any] = None) -> str:
        """Save a model to the cache.
        
        Args:
            model_id: Unique identifier for the model. If None, a new ID will be generated.
            model: The model to save.
            name: A human-readable name for the model.
            model_type: The type of model.
            metadata: Additional metadata to store with the model.
            
        Returns:
            The model ID.
        """
        if model_id is None:
            model_id = str(uuid.uuid4())
        
        # Serialize the model
        try:
            model_data = pickle.dumps(model, protocol=4)
        except Exception as e:
            print(f"Error serializing model: {e}")
            return None
        
        # Store in local cache
        self._local_cache[model_id] = model
        
        # Store in database
        conn = self.connection.get_connection()
        if conn is None:
            return model_id  # Return ID even if DB storage fails
        
        try:
            cursor = conn.cursor()
            now = datetime.datetime.now()
            metadata_json = json.dumps(metadata or {})
            
            # Check if model exists
            cursor.execute(
                f"SELECT COUNT(*) FROM {self.connection.config.schema}.MODEL_CACHE "
                f"WHERE model_id = ?",
                (model_id,)
            )
            count = cursor.fetchone()[0]
            
            if count > 0:
                # Update existing model
                cursor.execute(
                    f"UPDATE {self.connection.config.schema}.MODEL_CACHE "
                    f"SET model_data = ?, name = ?, model_type = ?, "
                    f"metadata = ?, updated_at = ? WHERE model_id = ?",
                    (model_data, name, model_type, metadata_json, now, model_id)
                )
            else:
                # Insert new model
                cursor.execute(
                    f"INSERT INTO {self.connection.config.schema}.MODEL_CACHE "
                    f"(model_id, name, model_type, model_data, metadata, created_at, updated_at) "
                    f"VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (model_id, name, model_type, model_data, metadata_json, now, now)
                )
        except Exception as e:
            print(f"Error saving model to database: {e}")
        finally:
            self.connection.release_connection(conn)
        
        return model_id
    
    def load_model(self, model_id: str) -> Any:
        """Load a model from the cache.
        
        Args:
            model_id: The ID of the model to load.
            
        Returns:
            The loaded model, or None if not found.
        """
        # Check local cache first
        if model_id in self._local_cache:
            return self._local_cache[model_id]
        
        # Load from database
        conn = self.connection.get_connection()
        if conn is None:
            return None
        
        model = None
        try:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT model_data FROM {self.connection.config.schema}.MODEL_CACHE "
                f"WHERE model_id = ?",
                (model_id,)
            )
            row = cursor.fetchone()
            
            if row is not None:
                model_data = row[0]
                model = pickle.loads(model_data)
                
                # Add to local cache
                self._local_cache[model_id] = model
        except Exception as e:
            print(f"Error loading model from database: {e}")
        finally:
            self.connection.release_connection(conn)
        
        return model
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a model from the cache.
        
        Args:
            model_id: The ID of the model to delete.
            
        Returns:
            True if the model was deleted, False otherwise.
        """
        # Remove from local cache
        if model_id in self._local_cache:
            del self._local_cache[model_id]
        
        # Remove from database
        conn = self.connection.get_connection()
        if conn is None:
            return False
        
        success = False
        try:
            cursor = conn.cursor()
            cursor.execute(
                f"DELETE FROM {self.connection.config.schema}.MODEL_CACHE "
                f"WHERE model_id = ?",
                (model_id,)
            )
            success = cursor.rowcount > 0
        except Exception as e:
            print(f"Error deleting model from database: {e}")
        finally:
            self.connection.release_connection(conn)
        
        return success
    
    def list_models(self, model_type: str = None) -> List[Dict[str, Any]]:
        """List all models in the cache.
        
        Args:
            model_type: Filter by model type.
            
        Returns:
            A list of model metadata.
        """
        conn = self.connection.get_connection()
        if conn is None:
            return []
        
        models = []
        try:
            cursor = conn.cursor()
            query = (
                f"SELECT model_id, name, model_type, metadata, created_at, updated_at "
                f"FROM {self.connection.config.schema}.MODEL_CACHE"
            )
            params = []
            
            if model_type is not None:
                query += " WHERE model_type = ?"
                params.append(model_type)
            
            cursor.execute(query, params)
            
            for row in cursor.fetchall():
                model_id, name, model_type, metadata_json, created_at, updated_at = row
                metadata = json.loads(metadata_json) if metadata_json else {}
                
                models.append({
                    "model_id": model_id,
                    "name": name,
                    "model_type": model_type,
                    "metadata": metadata,
                    "created_at": created_at,
                    "updated_at": updated_at,
                })
        except Exception as e:
            print(f"Error listing models from database: {e}")
        finally:
            self.connection.release_connection(conn)
        
        return models


def connect_to_hana(config: HanaConfig) -> HanaConnection:
    """Connect to SAP HANA database.
    
    Args:
        config: SAP HANA connection configuration.
        
    Returns:
        A HanaConnection object.
    """
    return HanaConnection(config)


def save_tree_to_hana(
    connection: HanaConnection,
    tree: Tree,
    tree_id: str = None,
    name: str = None,
    metadata: Dict[str, Any] = None
) -> str:
    """Save a tree to SAP HANA.
    
    Args:
        connection: The SAP HANA connection.
        tree: The tree to save.
        tree_id: Unique identifier for the tree. If None, a new ID will be generated.
        name: A human-readable name for the tree.
        metadata: Additional metadata to store with the tree.
        
    Returns:
        The tree ID.
    """
    if tree_id is None:
        tree_id = str(uuid.uuid4())
    
    # Create serializer
    serializer = HanaTreeSerializer(connection.config)
    
    # Serialize tree
    tree_data, tree_metadata = serializer.serialize_tree(tree)
    
    # Combine metadata
    combined_metadata = tree_metadata.copy()
    if metadata:
        combined_metadata.update(metadata)
    metadata_json = json.dumps(combined_metadata)
    
    # Get tree dimensions
    batch_size = tree_lib.infer_batch_size(tree)
    num_actions = tree.num_actions
    num_simulations = tree.num_simulations
    
    # Save to database
    conn = connection.get_connection()
    if conn is None:
        return None
    
    try:
        cursor = conn.cursor()
        now = datetime.datetime.now()
        
        # Check if tree exists
        cursor.execute(
            f"SELECT COUNT(*) FROM {connection.config.schema}.MCTS_TREES "
            f"WHERE tree_id = ?",
            (tree_id,)
        )
        count = cursor.fetchone()[0]
        
        if count > 0:
            # Update existing tree
            cursor.execute(
                f"UPDATE {connection.config.schema}.MCTS_TREES "
                f"SET tree_data = ?, name = ?, batch_size = ?, "
                f"num_actions = ?, num_simulations = ?, "
                f"metadata = ?, updated_at = ? WHERE tree_id = ?",
                (tree_data, name, batch_size, num_actions, 
                 num_simulations, metadata_json, now, tree_id)
            )
        else:
            # Insert new tree
            cursor.execute(
                f"INSERT INTO {connection.config.schema}.MCTS_TREES "
                f"(tree_id, name, batch_size, num_actions, num_simulations, "
                f"tree_data, metadata, created_at, updated_at) "
                f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (tree_id, name, batch_size, num_actions, 
                 num_simulations, tree_data, metadata_json, now, now)
            )
    except Exception as e:
        print(f"Error saving tree to database: {e}")
        return None
    finally:
        connection.release_connection(conn)
    
    return tree_id


def load_tree_from_hana(
    connection: HanaConnection,
    tree_id: str
) -> Optional[Tuple[Tree, Dict[str, Any]]]:
    """Load a tree from SAP HANA.
    
    Args:
        connection: The SAP HANA connection.
        tree_id: The ID of the tree to load.
        
    Returns:
        A tuple of (tree, metadata), or None if not found.
    """
    conn = connection.get_connection()
    if conn is None:
        return None
    
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT tree_data, metadata FROM {connection.config.schema}.MCTS_TREES "
            f"WHERE tree_id = ?",
            (tree_id,)
        )
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        tree_data, metadata_json = row
        metadata = json.loads(metadata_json)
        
        # Create serializer
        serializer = HanaTreeSerializer(connection.config)
        
        # Deserialize tree
        tree = serializer.deserialize_tree(tree_data, metadata)
        
        return tree, metadata
    except Exception as e:
        print(f"Error loading tree from database: {e}")
        return None
    finally:
        connection.release_connection(conn)


def save_model_to_hana(
    connection: HanaConnection,
    model,
    model_id: str = None,
    name: str = None,
    model_type: str = "default",
    metadata: Dict[str, Any] = None
) -> str:
    """Save a model to SAP HANA.
    
    Args:
        connection: The SAP HANA connection.
        model: The model to save.
        model_id: Unique identifier for the model. If None, a new ID will be generated.
        name: A human-readable name for the model.
        model_type: The type of model.
        metadata: Additional metadata to store with the model.
        
    Returns:
        The model ID.
    """
    cache = HanaModelCache(connection)
    return cache.save_model(model_id, model, name, model_type, metadata)


def load_model_from_hana(
    connection: HanaConnection,
    model_id: str
) -> Any:
    """Load a model from SAP HANA.
    
    Args:
        connection: The SAP HANA connection.
        model_id: The ID of the model to load.
        
    Returns:
        The loaded model, or None if not found.
    """
    cache = HanaModelCache(connection)
    return cache.load_model(model_id)


def save_simulation_results(
    connection: HanaConnection,
    tree_id: str,
    model_id: str,
    batch_idx: int,
    summary: tree_lib.SearchSummary,
    metadata: Dict[str, Any] = None
) -> str:
    """Save simulation results to SAP HANA.
    
    Args:
        connection: The SAP HANA connection.
        tree_id: The ID of the tree used for simulation.
        model_id: The ID of the model used for simulation.
        batch_idx: The batch index of the result.
        summary: The search summary.
        metadata: Additional metadata to store with the result.
        
    Returns:
        The result ID.
    """
    result_id = str(uuid.uuid4())
    
    # Serialize the summary components
    visit_counts = pickle.dumps(np.asarray(summary.visit_counts))
    visit_probs = pickle.dumps(np.asarray(summary.visit_probs))
    value = pickle.dumps(np.asarray(summary.value))
    qvalues = pickle.dumps(np.asarray(summary.qvalues))
    
    # Convert metadata to JSON
    metadata_json = json.dumps(metadata or {})
    
    # Save to database
    conn = connection.get_connection()
    if conn is None:
        return None
    
    try:
        cursor = conn.cursor()
        now = datetime.datetime.now()
        
        cursor.execute(
            f"INSERT INTO {connection.config.schema}.SIMULATION_RESULTS "
            f"(result_id, tree_id, model_id, batch_idx, "
            f"visit_counts, visit_probs, value, qvalues, "
            f"metadata, created_at) "
            f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (result_id, tree_id, model_id, batch_idx,
             visit_counts, visit_probs, value, qvalues,
             metadata_json, now)
        )
    except Exception as e:
        print(f"Error saving simulation results to database: {e}")
        return None
    finally:
        connection.release_connection(conn)
    
    return result_id


def load_simulation_results(
    connection: HanaConnection,
    result_id: str = None,
    tree_id: str = None,
    batch_idx: int = None
) -> List[Tuple[str, tree_lib.SearchSummary, Dict[str, Any]]]:
    """Load simulation results from SAP HANA.
    
    Args:
        connection: The SAP HANA connection.
        result_id: The ID of the specific result to load.
        tree_id: Filter by tree ID.
        batch_idx: Filter by batch index.
        
    Returns:
        A list of tuples (result_id, summary, metadata).
    """
    conn = connection.get_connection()
    if conn is None:
        return []
    
    results = []
    try:
        cursor = conn.cursor()
        query = (
            f"SELECT result_id, visit_counts, visit_probs, value, qvalues, metadata "
            f"FROM {connection.config.schema}.SIMULATION_RESULTS"
        )
        params = []
        where_clauses = []
        
        if result_id is not None:
            where_clauses.append("result_id = ?")
            params.append(result_id)
        
        if tree_id is not None:
            where_clauses.append("tree_id = ?")
            params.append(tree_id)
        
        if batch_idx is not None:
            where_clauses.append("batch_idx = ?")
            params.append(batch_idx)
        
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
        
        cursor.execute(query, params)
        
        for row in cursor.fetchall():
            result_id, visit_counts_data, visit_probs_data, value_data, qvalues_data, metadata_json = row
            
            # Deserialize the summary components
            visit_counts = jnp.asarray(pickle.loads(visit_counts_data))
            visit_probs = jnp.asarray(pickle.loads(visit_probs_data))
            value = jnp.asarray(pickle.loads(value_data))
            qvalues = jnp.asarray(pickle.loads(qvalues_data))
            
            # Create search summary
            summary = tree_lib.SearchSummary(
                visit_counts=visit_counts,
                visit_probs=visit_probs,
                value=value,
                qvalues=qvalues
            )
            
            # Parse metadata
            metadata = json.loads(metadata_json) if metadata_json else {}
            
            results.append((result_id, summary, metadata))
    except Exception as e:
        print(f"Error loading simulation results from database: {e}")
    finally:
        connection.release_connection(conn)
    
    return results


def batch_tree_operations(
    connection: HanaConnection,
    operations: List[Dict[str, Any]]
) -> List[Any]:
    """Perform batch operations on trees in SAP HANA.
    
    Args:
        connection: The SAP HANA connection.
        operations: A list of operation dictionaries, each with:
            - "operation": "save_tree", "load_tree", "save_model",
              "load_model", "save_results", or "load_results"
            - Other parameters specific to the operation
            
    Returns:
        A list of results, one for each operation.
    """
    results = []
    
    for op in operations:
        operation = op.get("operation")
        
        if operation == "save_tree":
            result = save_tree_to_hana(
                connection,
                op.get("tree"),
                op.get("tree_id"),
                op.get("name"),
                op.get("metadata")
            )
        elif operation == "load_tree":
            result = load_tree_from_hana(
                connection,
                op.get("tree_id")
            )
        elif operation == "save_model":
            result = save_model_to_hana(
                connection,
                op.get("model"),
                op.get("model_id"),
                op.get("name"),
                op.get("model_type"),
                op.get("metadata")
            )
        elif operation == "load_model":
            result = load_model_from_hana(
                connection,
                op.get("model_id")
            )
        elif operation == "save_results":
            result = save_simulation_results(
                connection,
                op.get("tree_id"),
                op.get("model_id"),
                op.get("batch_idx"),
                op.get("summary"),
                op.get("metadata")
            )
        elif operation == "load_results":
            result = load_simulation_results(
                connection,
                op.get("result_id"),
                op.get("tree_id"),
                op.get("batch_idx")
            )
        else:
            result = None
        
        results.append(result)
    
    return results