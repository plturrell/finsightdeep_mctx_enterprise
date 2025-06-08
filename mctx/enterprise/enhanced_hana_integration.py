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
"""Enhanced SAP HANA integration for MCTX with complete CRUD operations."""

import datetime
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

# Import base integration
from mctx.enterprise.hana_integration import (
    HANA_AVAILABLE,
    HanaConfig,
    HanaConnection,
    HanaModelCache,
    HanaTreeSerializer,
    Tree,
    connect_to_hana,
    load_model_from_hana,
    load_simulation_results,
    load_tree_from_hana,
    save_model_to_hana,
    save_simulation_results,
    save_tree_to_hana,
)

logger = logging.getLogger("mctx.enterprise.hana")


def delete_tree_from_hana(
    connection: HanaConnection,
    tree_id: str,
    cascade: bool = False
) -> bool:
    """Delete a tree from SAP HANA.
    
    Args:
        connection: The SAP HANA connection.
        tree_id: The ID of the tree to delete.
        cascade: If True, also delete associated simulation results.
        
    Returns:
        True if the tree was deleted, False otherwise.
    """
    conn = connection.get_connection()
    if conn is None:
        return False
    
    success = False
    try:
        cursor = conn.cursor()
        
        # If cascade is True, delete associated simulation results
        if cascade:
            cursor.execute(
                f"DELETE FROM {connection.config.schema}.SIMULATION_RESULTS "
                f"WHERE tree_id = ?",
                (tree_id,)
            )
            logger.info(f"Deleted associated simulation results for tree {tree_id}")
        
        # Delete the tree
        cursor.execute(
            f"DELETE FROM {connection.config.schema}.MCTS_TREES "
            f"WHERE tree_id = ?",
            (tree_id,)
        )
        
        success = cursor.rowcount > 0
        if success:
            logger.info(f"Successfully deleted tree {tree_id}")
        else:
            logger.warning(f"Tree {tree_id} not found for deletion")
    except Exception as e:
        logger.error(f"Error deleting tree from database: {e}")
    finally:
        connection.release_connection(conn)
    
    return success


def delete_simulation_results(
    connection: HanaConnection,
    result_id: str = None,
    tree_id: str = None,
    model_id: str = None,
    older_than: Optional[datetime.datetime] = None
) -> int:
    """Delete simulation results from SAP HANA.
    
    Args:
        connection: The SAP HANA connection.
        result_id: The ID of the specific result to delete.
        tree_id: Filter by tree ID.
        model_id: Filter by model ID.
        older_than: Delete results older than this timestamp.
        
    Returns:
        The number of results deleted.
    """
    if not any([result_id, tree_id, model_id, older_than]):
        logger.error("At least one filter parameter must be provided")
        return 0
    
    conn = connection.get_connection()
    if conn is None:
        return 0
    
    deleted_count = 0
    try:
        cursor = conn.cursor()
        query = f"DELETE FROM {connection.config.schema}.SIMULATION_RESULTS WHERE "
        
        where_clauses = []
        params = []
        
        if result_id is not None:
            where_clauses.append("result_id = ?")
            params.append(result_id)
        
        if tree_id is not None:
            where_clauses.append("tree_id = ?")
            params.append(tree_id)
        
        if model_id is not None:
            where_clauses.append("model_id = ?")
            params.append(model_id)
        
        if older_than is not None:
            where_clauses.append("created_at < ?")
            params.append(older_than)
        
        query += " AND ".join(where_clauses)
        
        cursor.execute(query, params)
        deleted_count = cursor.rowcount
        logger.info(f"Deleted {deleted_count} simulation results")
    except Exception as e:
        logger.error(f"Error deleting simulation results: {e}")
    finally:
        connection.release_connection(conn)
    
    return deleted_count


def list_trees(
    connection: HanaConnection,
    name_filter: str = None,
    min_batch_size: int = None,
    max_batch_size: int = None,
    min_num_simulations: int = None,
    max_num_simulations: int = None,
    created_after: datetime.datetime = None,
    created_before: datetime.datetime = None,
    limit: int = 100,
    offset: int = 0,
    order_by: str = "created_at DESC"
) -> List[Dict[str, Any]]:
    """List trees in SAP HANA with pagination and filtering.
    
    Args:
        connection: The SAP HANA connection.
        name_filter: Filter trees by name (supports LIKE syntax).
        min_batch_size: Filter by minimum batch size.
        max_batch_size: Filter by maximum batch size.
        min_num_simulations: Filter by minimum number of simulations.
        max_num_simulations: Filter by maximum number of simulations.
        created_after: Filter trees created after this timestamp.
        created_before: Filter trees created before this timestamp.
        limit: Maximum number of trees to return.
        offset: Number of trees to skip.
        order_by: Field to order by (e.g., "created_at DESC").
        
    Returns:
        A list of tree metadata dictionaries.
    """
    conn = connection.get_connection()
    if conn is None:
        return []
    
    trees = []
    try:
        cursor = conn.cursor()
        
        query = (
            f"SELECT tree_id, name, batch_size, num_actions, num_simulations, "
            f"metadata, created_at, updated_at "
            f"FROM {connection.config.schema}.MCTS_TREES WHERE 1=1"
        )
        
        params = []
        
        # Add filters
        if name_filter is not None:
            query += " AND name LIKE ?"
            params.append(f"%{name_filter}%")
        
        if min_batch_size is not None:
            query += " AND batch_size >= ?"
            params.append(min_batch_size)
        
        if max_batch_size is not None:
            query += " AND batch_size <= ?"
            params.append(max_batch_size)
        
        if min_num_simulations is not None:
            query += " AND num_simulations >= ?"
            params.append(min_num_simulations)
        
        if max_num_simulations is not None:
            query += " AND num_simulations <= ?"
            params.append(max_num_simulations)
        
        if created_after is not None:
            query += " AND created_at >= ?"
            params.append(created_after)
        
        if created_before is not None:
            query += " AND created_at <= ?"
            params.append(created_before)
        
        # Add ordering, limit, and offset
        query += f" ORDER BY {order_by} LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        
        for row in cursor.fetchall():
            tree_id, name, batch_size, num_actions, num_simulations, metadata_json, created_at, updated_at = row
            metadata = json.loads(metadata_json) if metadata_json else {}
            
            trees.append({
                "tree_id": tree_id,
                "name": name,
                "batch_size": batch_size,
                "num_actions": num_actions,
                "num_simulations": num_simulations,
                "metadata": metadata,
                "created_at": created_at,
                "updated_at": updated_at
            })
    except Exception as e:
        logger.error(f"Error listing trees from database: {e}")
    finally:
        connection.release_connection(conn)
    
    return trees


def bulk_delete_trees(
    connection: HanaConnection,
    tree_ids: List[str],
    cascade: bool = False
) -> Tuple[int, List[str]]:
    """Delete multiple trees in a single operation.
    
    Args:
        connection: The SAP HANA connection.
        tree_ids: List of tree IDs to delete.
        cascade: If True, also delete associated simulation results.
        
    Returns:
        A tuple of (number of trees deleted, list of failed tree IDs).
    """
    if not tree_ids:
        return 0, []
    
    conn = connection.get_connection()
    if conn is None:
        return 0, tree_ids
    
    successful_deletions = 0
    failed_deletions = []
    
    try:
        cursor = conn.cursor()
        
        # If cascade is True, delete associated simulation results first
        if cascade:
            # Use a parameterized query with a variable number of placeholders
            placeholders = ", ".join(["?"] * len(tree_ids))
            cursor.execute(
                f"DELETE FROM {connection.config.schema}.SIMULATION_RESULTS "
                f"WHERE tree_id IN ({placeholders})",
                tree_ids
            )
            logger.info(f"Deleted simulation results for {len(tree_ids)} trees")
        
        # Delete the trees
        placeholders = ", ".join(["?"] * len(tree_ids))
        cursor.execute(
            f"DELETE FROM {connection.config.schema}.MCTS_TREES "
            f"WHERE tree_id IN ({placeholders})",
            tree_ids
        )
        
        successful_deletions = cursor.rowcount
        
        # Determine which deletions failed
        if successful_deletions < len(tree_ids):
            # Get IDs of successfully deleted trees
            cursor.execute(
                f"SELECT tree_id FROM {connection.config.schema}.MCTS_TREES "
                f"WHERE tree_id IN ({placeholders})",
                tree_ids
            )
            remaining_ids = [row[0] for row in cursor.fetchall()]
            failed_deletions = [tree_id for tree_id in tree_ids if tree_id in remaining_ids]
        
        logger.info(f"Successfully deleted {successful_deletions} trees")
        if failed_deletions:
            logger.warning(f"Failed to delete {len(failed_deletions)} trees")
    except Exception as e:
        logger.error(f"Error during bulk tree deletion: {e}")
        return 0, tree_ids
    finally:
        connection.release_connection(conn)
    
    return successful_deletions, failed_deletions


def update_tree_metadata(
    connection: HanaConnection,
    tree_id: str,
    metadata_updates: Dict[str, Any]
) -> bool:
    """Update only the metadata of a tree without modifying the tree data.
    
    Args:
        connection: The SAP HANA connection.
        tree_id: The ID of the tree to update.
        metadata_updates: Dictionary of metadata fields to update.
        
    Returns:
        True if the metadata was updated, False otherwise.
    """
    conn = connection.get_connection()
    if conn is None:
        return False
    
    success = False
    try:
        cursor = conn.cursor()
        
        # Get current metadata
        cursor.execute(
            f"SELECT metadata FROM {connection.config.schema}.MCTS_TREES "
            f"WHERE tree_id = ?",
            (tree_id,)
        )
        row = cursor.fetchone()
        
        if row is None:
            logger.warning(f"Tree {tree_id} not found for metadata update")
            return False
        
        # Update metadata
        metadata_json = row[0]
        current_metadata = json.loads(metadata_json) if metadata_json else {}
        current_metadata.update(metadata_updates)
        
        # Save updated metadata
        now = datetime.datetime.now()
        cursor.execute(
            f"UPDATE {connection.config.schema}.MCTS_TREES "
            f"SET metadata = ?, updated_at = ? WHERE tree_id = ?",
            (json.dumps(current_metadata), now, tree_id)
        )
        
        success = cursor.rowcount > 0
        if success:
            logger.info(f"Successfully updated metadata for tree {tree_id}")
    except Exception as e:
        logger.error(f"Error updating tree metadata: {e}")
    finally:
        connection.release_connection(conn)
    
    return success


def bulk_save_trees(
    connection: HanaConnection,
    trees: List[Tuple[Tree, str, Dict[str, Any]]]
) -> List[str]:
    """Save multiple trees in a single transaction.
    
    Args:
        connection: The SAP HANA connection.
        trees: List of tuples (tree, name, metadata).
        
    Returns:
        List of tree IDs that were successfully saved.
    """
    if not trees:
        return []
    
    conn = connection.get_connection()
    if conn is None:
        return []
    
    tree_ids = []
    serializer = HanaTreeSerializer(connection.config)
    
    try:
        # Disable autocommit to handle as a transaction
        autocommit_state = conn.getautocommit()
        conn.setautocommit(False)
        
        cursor = conn.cursor()
        now = datetime.datetime.now()
        
        for tree, name, metadata in trees:
            tree_id = str(uuid.uuid4())
            
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
            
            # Insert new tree
            cursor.execute(
                f"INSERT INTO {connection.config.schema}.MCTS_TREES "
                f"(tree_id, name, batch_size, num_actions, num_simulations, "
                f"tree_data, metadata, created_at, updated_at) "
                f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (tree_id, name, batch_size, num_actions, 
                 num_simulations, tree_data, metadata_json, now, now)
            )
            
            tree_ids.append(tree_id)
        
        # Commit the transaction
        conn.commit()
        logger.info(f"Successfully saved {len(tree_ids)} trees in bulk operation")
    except Exception as e:
        logger.error(f"Error in bulk tree save operation: {e}")
        # Roll back the transaction
        try:
            conn.rollback()
        except Exception:
            pass
        tree_ids = []
    finally:
        # Restore autocommit state
        if conn:
            try:
                conn.setautocommit(autocommit_state)
            except Exception:
                pass
            connection.release_connection(conn)
    
    return tree_ids


def get_database_statistics(
    connection: HanaConnection
) -> Dict[str, Any]:
    """Get statistics about the HANA database usage.
    
    Args:
        connection: The SAP HANA connection.
        
    Returns:
        Dictionary with database statistics.
    """
    conn = connection.get_connection()
    if conn is None:
        return {}
    
    stats = {}
    try:
        cursor = conn.cursor()
        
        # Get tree count and size
        cursor.execute(
            f"SELECT COUNT(*), SUM(LENGTH(tree_data)) "
            f"FROM {connection.config.schema}.MCTS_TREES"
        )
        tree_count, tree_size = cursor.fetchone()
        
        # Get model count and size
        cursor.execute(
            f"SELECT COUNT(*), SUM(LENGTH(model_data)) "
            f"FROM {connection.config.schema}.MODEL_CACHE"
        )
        model_count, model_size = cursor.fetchone()
        
        # Get results count
        cursor.execute(
            f"SELECT COUNT(*) FROM {connection.config.schema}.SIMULATION_RESULTS"
        )
        results_count = cursor.fetchone()[0]
        
        # Get most recent activity
        cursor.execute(
            f"SELECT MAX(updated_at) FROM {connection.config.schema}.MCTS_TREES"
        )
        last_tree_activity = cursor.fetchone()[0]
        
        cursor.execute(
            f"SELECT MAX(updated_at) FROM {connection.config.schema}.MODEL_CACHE"
        )
        last_model_activity = cursor.fetchone()[0]
        
        cursor.execute(
            f"SELECT MAX(created_at) FROM {connection.config.schema}.SIMULATION_RESULTS"
        )
        last_result_activity = cursor.fetchone()[0]
        
        # Compile statistics
        stats = {
            "trees": {
                "count": tree_count,
                "total_size_bytes": tree_size,
                "avg_size_bytes": tree_size / tree_count if tree_count else 0,
                "last_activity": last_tree_activity
            },
            "models": {
                "count": model_count,
                "total_size_bytes": model_size,
                "avg_size_bytes": model_size / model_count if model_count else 0,
                "last_activity": last_model_activity
            },
            "simulation_results": {
                "count": results_count,
                "last_activity": last_result_activity
            },
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting database statistics: {e}")
    finally:
        connection.release_connection(conn)
    
    return stats


def clean_old_data(
    connection: HanaConnection,
    older_than_days: int = 30,
    simulation_results_only: bool = True
) -> Dict[str, int]:
    """Clean up old data from the database.
    
    Args:
        connection: The SAP HANA connection.
        older_than_days: Delete data older than this many days.
        simulation_results_only: If True, only delete simulation results.
        
    Returns:
        Dictionary with count of deleted items by type.
    """
    conn = connection.get_connection()
    if conn is None:
        return {"error": "No connection"}
    
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=older_than_days)
    deleted_counts = {
        "simulation_results": 0,
        "trees": 0,
        "models": 0
    }
    
    try:
        # Disable autocommit to handle as a transaction
        autocommit_state = conn.getautocommit()
        conn.setautocommit(False)
        
        cursor = conn.cursor()
        
        # Delete old simulation results
        cursor.execute(
            f"DELETE FROM {connection.config.schema}.SIMULATION_RESULTS "
            f"WHERE created_at < ?",
            (cutoff_date,)
        )
        deleted_counts["simulation_results"] = cursor.rowcount
        
        if not simulation_results_only:
            # Delete old trees
            cursor.execute(
                f"DELETE FROM {connection.config.schema}.MCTS_TREES "
                f"WHERE updated_at < ? AND tree_id NOT IN "
                f"(SELECT DISTINCT tree_id FROM {connection.config.schema}.SIMULATION_RESULTS)",
                (cutoff_date,)
            )
            deleted_counts["trees"] = cursor.rowcount
            
            # Delete old models
            cursor.execute(
                f"DELETE FROM {connection.config.schema}.MODEL_CACHE "
                f"WHERE updated_at < ? AND model_id NOT IN "
                f"(SELECT DISTINCT model_id FROM {connection.config.schema}.SIMULATION_RESULTS)",
                (cutoff_date,)
            )
            deleted_counts["models"] = cursor.rowcount
        
        # Commit the transaction
        conn.commit()
        logger.info(f"Cleaned up old data: {deleted_counts}")
    except Exception as e:
        logger.error(f"Error cleaning up old data: {e}")
        # Roll back the transaction
        try:
            conn.rollback()
        except Exception:
            pass
    finally:
        # Restore autocommit state
        if conn:
            try:
                conn.setautocommit(autocommit_state)
            except Exception:
                pass
            connection.release_connection(conn)
    
    return deleted_counts


# For backwards compatibility
def extended_batch_operations(
    connection: HanaConnection,
    operations: List[Dict[str, Any]]
) -> List[Any]:
    """Extended batch operations on HANA with additional operation types.
    
    Args:
        connection: The SAP HANA connection.
        operations: A list of operation dictionaries, each with:
            - "operation": One of the supported operation types:
              - "save_tree", "load_tree", "delete_tree", "list_trees",
              - "save_model", "load_model", "delete_model", "list_models",
              - "save_results", "load_results", "delete_results",
              - "update_tree_metadata", "bulk_save_trees", "bulk_delete_trees",
              - "get_database_statistics", "clean_old_data"
            - Other parameters specific to the operation
            
    Returns:
        A list of results, one for each operation.
    """
    from mctx.enterprise.hana_integration import batch_tree_operations
    
    # First handle operations from the base module
    base_operations = [
        "save_tree", "load_tree", "save_model", "load_model", 
        "save_results", "load_results"
    ]
    
    # Split operations into base and extended
    base_ops = []
    ext_ops = []
    
    for op in operations:
        operation_type = op.get("operation")
        if operation_type in base_operations:
            base_ops.append(op)
        else:
            ext_ops.append(op)
    
    # Get base operation results
    results = []
    if base_ops:
        results = batch_tree_operations(connection, base_ops)
    
    # Handle extended operations
    for op in ext_ops:
        operation_type = op.get("operation")
        
        if operation_type == "delete_tree":
            result = delete_tree_from_hana(
                connection,
                op.get("tree_id"),
                op.get("cascade", False)
            )
        elif operation_type == "list_trees":
            result = list_trees(
                connection,
                name_filter=op.get("name_filter"),
                min_batch_size=op.get("min_batch_size"),
                max_batch_size=op.get("max_batch_size"),
                min_num_simulations=op.get("min_num_simulations"),
                max_num_simulations=op.get("max_num_simulations"),
                created_after=op.get("created_after"),
                created_before=op.get("created_before"),
                limit=op.get("limit", 100),
                offset=op.get("offset", 0),
                order_by=op.get("order_by", "created_at DESC")
            )
        elif operation_type == "delete_results":
            result = delete_simulation_results(
                connection,
                result_id=op.get("result_id"),
                tree_id=op.get("tree_id"),
                model_id=op.get("model_id"),
                older_than=op.get("older_than")
            )
        elif operation_type == "update_tree_metadata":
            result = update_tree_metadata(
                connection,
                op.get("tree_id"),
                op.get("metadata_updates", {})
            )
        elif operation_type == "bulk_save_trees":
            result = bulk_save_trees(
                connection,
                op.get("trees", [])
            )
        elif operation_type == "bulk_delete_trees":
            result = bulk_delete_trees(
                connection,
                op.get("tree_ids", []),
                op.get("cascade", False)
            )
        elif operation_type == "get_database_statistics":
            result = get_database_statistics(
                connection
            )
        elif operation_type == "clean_old_data":
            result = clean_old_data(
                connection,
                op.get("older_than_days", 30),
                op.get("simulation_results_only", True)
            )
        else:
            logger.warning(f"Unknown operation type: {operation_type}")
            result = None
        
        results.append(result)
    
    return results