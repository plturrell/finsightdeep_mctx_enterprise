"""API endpoints for memory-optimized SAP HANA integration with MCTX.

This module provides REST API endpoints for interacting with large MCTS trees in
SAP HANA, using memory-optimized techniques like batched serialization and incremental loading.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Body, Path
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional, Any, Union
import time
from pydantic import BaseModel, Field, validator
import json
from datetime import datetime
import logging
from enum import Enum

from ..core.auth import get_current_user
from ..models.auth_models import User
from ..models.mcts_models import TreeModel, NodeModel
from ..db.hana_connector import HanaConnection, get_hana_connection

# Import enterprise modules
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from mctx.enterprise.batched_serialization import batch_serialize_tree, batch_deserialize_tree
from mctx.enterprise.incremental_loader import (
    load_tree_by_pages,
    load_tree_by_depth,
    load_subtree,
    load_path_to_node,
    load_high_value_nodes
)
from mctx.enterprise.hana_query_cache import (
    global_query_cache,
    invalidate_tree_cache
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/optimized",
    tags=["optimized"],
    responses={404: {"description": "Not found"}},
)

# Enums and models

class LoadingStrategy(str, Enum):
    BY_PAGES = "by_pages"
    BY_DEPTH = "by_depth"
    SUBTREE = "subtree"
    PATH = "path"
    HIGH_VALUE = "high_value"

class TreePageResponse(BaseModel):
    nodes: List[Dict[str, Any]]
    has_more: bool
    next_page_token: Optional[str] = None
    execution_time: float

class TreeResponse(BaseModel):
    tree: Dict[str, Any]
    execution_time: float

class SubtreeRequest(BaseModel):
    node_id: str = Field(..., description="ID of the node to start from")
    max_depth: int = Field(3, description="Maximum depth to traverse")
    max_nodes: int = Field(1000, description="Maximum number of nodes to load")

class PathRequest(BaseModel):
    node_id: str = Field(..., description="ID of the target node")

class HighValueRequest(BaseModel):
    max_nodes: int = Field(100, description="Maximum number of nodes to load")
    value_threshold: Optional[float] = Field(None, description="Minimum value threshold")

# Endpoints

@router.post("/trees/", response_model=Dict[str, Any])
async def save_tree_batched(
    tree: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_user)
):
    """Save a large tree to the database using batched serialization."""
    conn = get_hana_connection()
    
    start_time = time.time()
    
    try:
        # Serialize the tree in batches
        tree_id = batch_serialize_tree(conn, tree)
        
        # Log performance metrics
        execution_time = time.time() - start_time
        logger.info(f"Batched tree serialization completed in {execution_time:.3f}s for tree {tree_id}")
        
        return {
            'tree_id': tree_id,
            'message': 'Tree saved successfully using batched serialization',
            'execution_time': execution_time,
            'node_count': len(tree.get('nodes', []))
        }
    except Exception as e:
        logger.error(f"Error in batched tree serialization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trees/{tree_id}/pages", response_model=TreePageResponse)
async def get_tree_by_pages(
    tree_id: str = Path(..., description="ID of the tree to retrieve"),
    page_token: Optional[str] = Query(None, description="Token for retrieving a specific page"),
    page_size: int = Query(1000, description="Number of nodes per page"),
    current_user: User = Depends(get_current_user)
):
    """Get a tree incrementally by pages."""
    conn = get_hana_connection()
    
    start_time = time.time()
    
    try:
        # Load the tree by pages
        result = load_tree_by_pages(conn, tree_id, page_token, page_size)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Create response
        response = {
            'nodes': result['nodes'],
            'has_more': result['has_more'],
            'next_page_token': result['next_page_token'],
            'execution_time': execution_time
        }
        
        # Log performance metrics
        logger.info(f"Tree page loading completed in {execution_time:.3f}s for tree {tree_id}")
        
        return response
    except Exception as e:
        logger.error(f"Error loading tree by pages: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trees/{tree_id}/depth/{max_depth}", response_model=TreeResponse)
async def get_tree_by_depth(
    tree_id: str = Path(..., description="ID of the tree to retrieve"),
    max_depth: int = Path(..., description="Maximum depth to traverse"),
    start_node_id: Optional[str] = Query(None, description="ID of the node to start from"),
    current_user: User = Depends(get_current_user)
):
    """Get a tree incrementally by depth using breadth-first search."""
    conn = get_hana_connection()
    
    start_time = time.time()
    
    try:
        # Load the tree by depth
        tree = load_tree_by_depth(conn, tree_id, max_depth, start_node_id)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Create response
        response = {
            'tree': tree,
            'execution_time': execution_time
        }
        
        # Log performance metrics
        logger.info(f"Tree depth loading completed in {execution_time:.3f}s for tree {tree_id}")
        
        return response
    except Exception as e:
        logger.error(f"Error loading tree by depth: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/trees/{tree_id}/subtree", response_model=TreeResponse)
async def get_subtree(
    tree_id: str = Path(..., description="ID of the tree"),
    request: SubtreeRequest = Body(...),
    current_user: User = Depends(get_current_user)
):
    """Get a subtree starting from a specific node."""
    conn = get_hana_connection()
    
    start_time = time.time()
    
    try:
        # Load the subtree
        tree = load_subtree(
            conn, 
            tree_id, 
            request.node_id, 
            request.max_depth, 
            request.max_nodes
        )
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Create response
        response = {
            'tree': tree,
            'execution_time': execution_time
        }
        
        # Log performance metrics
        logger.info(f"Subtree loading completed in {execution_time:.3f}s for tree {tree_id}")
        
        return response
    except Exception as e:
        logger.error(f"Error loading subtree: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/trees/{tree_id}/path", response_model=TreeResponse)
async def get_path_to_node(
    tree_id: str = Path(..., description="ID of the tree"),
    request: PathRequest = Body(...),
    current_user: User = Depends(get_current_user)
):
    """Get the path from the root to a specific node."""
    conn = get_hana_connection()
    
    start_time = time.time()
    
    try:
        # Load the path
        tree = load_path_to_node(conn, tree_id, request.node_id)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Create response
        response = {
            'tree': tree,
            'execution_time': execution_time
        }
        
        # Log performance metrics
        logger.info(f"Path loading completed in {execution_time:.3f}s for tree {tree_id}")
        
        return response
    except Exception as e:
        logger.error(f"Error loading path: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/trees/{tree_id}/high-value", response_model=TreeResponse)
async def get_high_value_nodes(
    tree_id: str = Path(..., description="ID of the tree"),
    request: HighValueRequest = Body(...),
    current_user: User = Depends(get_current_user)
):
    """Get the highest value nodes from a tree."""
    conn = get_hana_connection()
    
    start_time = time.time()
    
    try:
        # Load high value nodes
        tree = load_high_value_nodes(
            conn, 
            tree_id, 
            request.max_nodes, 
            request.value_threshold
        )
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Create response
        response = {
            'tree': tree,
            'execution_time': execution_time
        }
        
        # Log performance metrics
        logger.info(f"High-value nodes loading completed in {execution_time:.3f}s for tree {tree_id}")
        
        return response
    except Exception as e:
        logger.error(f"Error loading high-value nodes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/trees/{tree_id}")
async def delete_tree_optimized(
    tree_id: str = Path(..., description="ID of the tree to delete"),
    current_user: User = Depends(get_current_user)
):
    """Delete a tree and all its batched data."""
    conn = get_hana_connection()
    
    try:
        # Use the batched serializer to delete the tree
        from mctx.enterprise.batched_serialization import BatchedTreeSerializer
        serializer = BatchedTreeSerializer(conn)
        success = serializer.delete_tree(tree_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Tree with ID {tree_id} not found")
        
        # Invalidate cache
        invalidate_tree_cache(tree_id)
        
        return {
            'tree_id': tree_id,
            'message': 'Tree deleted successfully'
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting tree {tree_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))