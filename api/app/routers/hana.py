from fastapi import APIRouter, Depends, HTTPException, Query, Body, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import datetime
import logging
import os
import json
from uuid import UUID

from ..core.auth import get_current_user
from ..models.auth_models import User

# Initialize logger
logger = logging.getLogger("mctx.api.hana")

# Check if HANA is available
try:
    import hdbcli.dbapi
    from mctx.enterprise.hana_integration import (
        HANA_AVAILABLE, HanaConfig, connect_to_hana,
        save_tree_to_hana, load_tree_from_hana, save_model_to_hana,
        load_model_from_hana, save_simulation_results, load_simulation_results
    )
    from mctx.enterprise.enhanced_hana_integration import (
        delete_tree_from_hana, delete_simulation_results, list_trees,
        update_tree_metadata, bulk_delete_trees, get_database_statistics,
        clean_old_data
    )
    HANA_ENABLED = HANA_AVAILABLE and os.environ.get("HANA_ENABLED", "True").lower() in ("true", "1", "yes")
except ImportError:
    HANA_AVAILABLE = False
    HANA_ENABLED = False
    logger.warning("SAP HANA libraries not available. HANA endpoints will be disabled.")

# Pydantic models
class TreeMetadata(BaseModel):
    name: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
class TreeFilter(BaseModel):
    name_filter: Optional[str] = None
    min_batch_size: Optional[int] = None
    max_batch_size: Optional[int] = None
    min_num_simulations: Optional[int] = None
    max_num_simulations: Optional[int] = None
    created_after: Optional[datetime.datetime] = None
    created_before: Optional[datetime.datetime] = None
    limit: int = 100
    offset: int = 0
    order_by: str = "created_at DESC"
    
class DeleteParams(BaseModel):
    cascade: bool = False
    
class CleanupParams(BaseModel):
    older_than_days: int = 30
    simulation_results_only: bool = True
    
class BulkDeleteRequest(BaseModel):
    tree_ids: List[str]
    cascade: bool = False

class TreeResponse(BaseModel):
    tree_id: str
    name: Optional[str]
    batch_size: int
    num_actions: int
    num_simulations: int
    metadata: Dict[str, Any]
    created_at: datetime.datetime
    updated_at: datetime.datetime

class DBStatisticsResponse(BaseModel):
    trees: Dict[str, Any]
    models: Dict[str, Any]
    simulation_results: Dict[str, Any]
    timestamp: str

# Create router
router = APIRouter(
    prefix="/api/hana",
    tags=["hana"],
    dependencies=[Depends(get_current_user)],
    responses={404: {"description": "Not found"}},
)

# Helper function to get HANA connection
def get_hana_connection():
    """Get a HANA connection from environment variables."""
    if not HANA_ENABLED:
        raise HTTPException(
            status_code=501,
            detail="SAP HANA integration is not enabled or available"
        )
    
    try:
        hana_host = os.environ.get("HANA_HOST")
        hana_port = int(os.environ.get("HANA_PORT", "443"))
        hana_user = os.environ.get("HANA_USER")
        hana_password = os.environ.get("HANA_PASSWORD")
        hana_schema = os.environ.get("HANA_SCHEMA", "MCTX")
        
        if not all([hana_host, hana_user, hana_password]):
            raise HTTPException(
                status_code=500,
                detail="Missing required HANA configuration"
            )
        
        config = HanaConfig(
            host=hana_host,
            port=hana_port,
            user=hana_user,
            password=hana_password,
            schema=hana_schema,
        )
        
        return connect_to_hana(config)
    except Exception as e:
        logger.error(f"Failed to create HANA connection: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"HANA connection error: {str(e)}"
        )

# API endpoints
@router.get("/status/", response_model=Dict[str, Any])
async def check_hana_status(current_user: User = Depends(get_current_user)):
    """Check the status of the SAP HANA connection."""
    if not HANA_ENABLED:
        return {
            "available": False,
            "enabled": False,
            "message": "SAP HANA integration is not enabled or available"
        }
    
    try:
        conn = get_hana_connection()
        hdb_conn = conn.get_connection()
        
        if hdb_conn:
            cursor = hdb_conn.cursor()
            cursor.execute("SELECT * FROM DUMMY")
            result = cursor.fetchall()
            conn.release_connection(hdb_conn)
            
            return {
                "available": True,
                "enabled": True,
                "connected": True,
                "message": "SAP HANA connection successful",
                "version": hdb_conn.client_info.version,
                "schema": conn.config.schema
            }
    except Exception as e:
        logger.error(f"HANA connection test failed: {str(e)}")
        return {
            "available": True,
            "enabled": True,
            "connected": False,
            "message": f"SAP HANA connection failed: {str(e)}"
        }

@router.get("/trees/", response_model=List[TreeResponse])
async def get_trees(
    name: Optional[str] = None,
    min_batch: Optional[int] = None,
    max_batch: Optional[int] = None,
    min_sims: Optional[int] = None,
    max_sims: Optional[int] = None,
    limit: int = 100,
    offset: int = 0,
    current_user: User = Depends(get_current_user)
):
    """List trees with filtering and pagination."""
    conn = get_hana_connection()
    try:
        trees = list_trees(
            conn,
            name_filter=name,
            min_batch_size=min_batch,
            max_batch_size=max_batch,
            min_num_simulations=min_sims,
            max_num_simulations=max_sims,
            limit=limit,
            offset=offset
        )
        return trees
    finally:
        conn.close_all()

@router.get("/trees/{tree_id}", response_model=Dict[str, Any])
async def get_tree_metadata(
    tree_id: str = Path(..., description="The tree ID"),
    current_user: User = Depends(get_current_user)
):
    """Get tree metadata by ID."""
    conn = get_hana_connection()
    try:
        # Try to load the tree
        result = load_tree_from_hana(conn, tree_id)
        
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Tree with ID {tree_id} not found"
            )
        
        tree, metadata = result
        
        # Get tree dimensions
        batch_size = tree.node_visits.shape[0] if hasattr(tree, 'node_visits') else 0
        num_actions = tree.num_actions if hasattr(tree, 'num_actions') else 0
        
        # Return metadata and basic info
        return {
            "tree_id": tree_id,
            "metadata": metadata,
            "dimensions": {
                "batch_size": batch_size,
                "num_actions": num_actions,
                "num_nodes": tree.node_visits.shape[1] if hasattr(tree, 'node_visits') else 0,
            }
        }
    finally:
        conn.close_all()

@router.patch("/trees/{tree_id}", response_model=Dict[str, Any])
async def update_tree(
    tree_id: str = Path(..., description="The tree ID"),
    tree_data: TreeMetadata = Body(...),
    current_user: User = Depends(get_current_user)
):
    """Update tree metadata."""
    conn = get_hana_connection()
    try:
        # Update the tree metadata
        success = update_tree_metadata(
            conn,
            tree_id,
            tree_data.metadata
        )
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Tree with ID {tree_id} not found or update failed"
            )
        
        return {
            "tree_id": tree_id,
            "success": True,
            "message": "Tree metadata updated successfully"
        }
    finally:
        conn.close_all()

@router.delete("/trees/{tree_id}", response_model=Dict[str, Any])
async def delete_tree(
    tree_id: str = Path(..., description="The tree ID"),
    params: DeleteParams = Depends(),
    current_user: User = Depends(get_current_user)
):
    """Delete a tree by ID."""
    conn = get_hana_connection()
    try:
        # Delete the tree
        success = delete_tree_from_hana(
            conn,
            tree_id,
            cascade=params.cascade
        )
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Tree with ID {tree_id} not found or deletion failed"
            )
        
        return {
            "tree_id": tree_id,
            "success": True,
            "message": "Tree deleted successfully",
            "cascade": params.cascade
        }
    finally:
        conn.close_all()

@router.post("/trees/bulk-delete", response_model=Dict[str, Any])
async def bulk_delete_trees_endpoint(
    request: BulkDeleteRequest,
    current_user: User = Depends(get_current_user)
):
    """Delete multiple trees in a single operation."""
    conn = get_hana_connection()
    try:
        # Delete the trees
        deleted_count, failed_ids = bulk_delete_trees(
            conn,
            request.tree_ids,
            cascade=request.cascade
        )
        
        return {
            "success": True,
            "deleted_count": deleted_count,
            "failed_ids": failed_ids,
            "message": f"Deleted {deleted_count} trees"
        }
    finally:
        conn.close_all()

@router.delete("/results/{result_id}", response_model=Dict[str, Any])
async def delete_result(
    result_id: str = Path(..., description="The result ID"),
    current_user: User = Depends(get_current_user)
):
    """Delete a simulation result by ID."""
    conn = get_hana_connection()
    try:
        # Delete the result
        count = delete_simulation_results(conn, result_id=result_id)
        
        if count == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Result with ID {result_id} not found or deletion failed"
            )
        
        return {
            "result_id": result_id,
            "success": True,
            "message": "Result deleted successfully"
        }
    finally:
        conn.close_all()

@router.delete("/results", response_model=Dict[str, Any])
async def delete_results_by_filter(
    tree_id: Optional[str] = None,
    model_id: Optional[str] = None,
    days: Optional[int] = None,
    current_user: User = Depends(get_current_user)
):
    """Delete simulation results by filter criteria."""
    if not any([tree_id, model_id, days]):
        raise HTTPException(
            status_code=400,
            detail="At least one filter parameter must be provided"
        )
    
    conn = get_hana_connection()
    try:
        older_than = None
        if days:
            older_than = datetime.datetime.now() - datetime.timedelta(days=days)
        
        # Delete the results
        count = delete_simulation_results(
            conn,
            tree_id=tree_id,
            model_id=model_id,
            older_than=older_than
        )
        
        return {
            "success": True,
            "deleted_count": count,
            "message": f"Deleted {count} results"
        }
    finally:
        conn.close_all()

@router.get("/statistics", response_model=DBStatisticsResponse)
async def get_statistics(
    current_user: User = Depends(get_current_user)
):
    """Get database statistics."""
    conn = get_hana_connection()
    try:
        stats = get_database_statistics(conn)
        return stats
    finally:
        conn.close_all()

@router.post("/cleanup", response_model=Dict[str, Any])
async def cleanup_database(
    params: CleanupParams = Body(...),
    current_user: User = Depends(get_current_user)
):
    """Clean up old data from the database."""
    conn = get_hana_connection()
    try:
        result = clean_old_data(
            conn,
            older_than_days=params.older_than_days,
            simulation_results_only=params.simulation_results_only
        )
        
        return {
            "success": True,
            "deleted_counts": result,
            "message": "Cleanup completed successfully"
        }
    finally:
        conn.close_all()