from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import time
from datetime import datetime
import os
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mctx.test")

# Check for HANA availability
try:
    import hdbcli.dbapi
    HANA_AVAILABLE = True
except ImportError:
    HANA_AVAILABLE = False
    logger.warning("SAP HANA client not available - HANA endpoints will not work")

# Initialize FastAPI application
app = FastAPI(
    title="MCTX Test API",
    version="0.1.0",
    description="Simple test API for MCTX Docker testing",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """
    Root endpoint providing basic API information.
    """
    return {
        "name": "MCTX Test API",
        "version": "0.1.0",
        "description": "Docker test for FastAPI"
    }

# Health check endpoint
@app.get("/health/", tags=["health"])
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": {
            "python_version": os.environ.get("PYTHONPATH", "Not set"),
            "debug": os.environ.get("DEBUG", "Not set"),
            "host": os.environ.get("HOST", "Not set")
        }
    }

# Test endpoint
@app.get("/test/", tags=["test"])
async def test_endpoint():
    """Test endpoint with extended information."""
    logger.info("Test endpoint called")
    return {
        "message": "Test endpoint working correctly",
        "timestamp": datetime.utcnow().isoformat(),
        "server_info": {
            "hostname": os.environ.get("HOSTNAME", "unknown"),
            "platform": os.name,
            "python_version": os.environ.get("PYTHONVERSION", "3.9")
        }
    }

# Metrics endpoint
@app.get("/metrics/", tags=["metrics"])
async def metrics():
    """Sample metrics endpoint."""
    logger.info("Metrics endpoint called")
    return {
        "metrics": {
            "requests_processed": 100,
            "average_response_time": 45.2,
            "cpu_usage": 12.5,
            "memory_usage": "128MB"
        },
        "timestamp": datetime.utcnow().isoformat()
    }

# Enhanced HANA API router support
@app.get("/hana/test/", tags=["hana"])
async def hana_test():
    """Test SAP HANA connection."""
    logger.info("HANA test endpoint called")
    
    if not HANA_AVAILABLE:
        raise HTTPException(
            status_code=500, 
            detail="SAP HANA client library not available. Please install hdbcli package."
        )
    
    try:
        # Get HANA configuration from environment variables
        hana_host = os.environ.get("HANA_HOST", "")
        hana_port = int(os.environ.get("HANA_PORT", "443"))
        hana_user = os.environ.get("HANA_USER", "")
        hana_password = os.environ.get("HANA_PASSWORD", "")
        
        # Test connection
        conn = hdbcli.dbapi.connect(
            address=hana_host,
            port=hana_port,
            user=hana_user,
            password=hana_password,
            encrypt=True,
            timeout=30
        )
        
        # Execute a simple query
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM DUMMY")
        result = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return {
            "status": "connected",
            "message": "Successfully connected to SAP HANA",
            "result": str(result),
            "connection_info": {
                "host": hana_host,
                "port": hana_port,
                "user": hana_user,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"HANA connection failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to connect to SAP HANA: {str(e)}"
        )

# Memory-optimized API endpoints
@app.get("/optimized/test/", tags=["optimized"])
async def optimized_test():
    """Test memory-optimized functionality."""
    logger.info("Memory-optimized test endpoint called")
    return {
        "status": "available",
        "message": "Memory-optimized API endpoints are available",
        "features": [
            "Batched serialization for large trees",
            "Incremental loading of tree nodes",
            "Memory-efficient processing",
            "Subtree loading",
            "High-value node prioritization"
        ],
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/optimized/trees/", tags=["optimized"])
async def create_batched_tree():
    """Create a tree using batched serialization (simulation)."""
    logger.info("Batched tree creation endpoint called")
    return {
        "tree_id": "test-tree-123",
        "message": "Tree saved successfully using batched serialization (simulation)",
        "execution_time": 0.25,
        "node_count": 1000
    }

@app.get("/optimized/trees/{tree_id}/depth/{max_depth}", tags=["optimized"])
async def get_tree_by_depth(tree_id: str, max_depth: int):
    """Get a tree by depth (simulation)."""
    logger.info(f"Get tree by depth endpoint called: tree_id={tree_id}, max_depth={max_depth}")
    return {
        "tree": {
            "tree_id": tree_id,
            "name": "Test Tree",
            "nodes": [
                {"id": "root", "parent_id": "", "visit_count": 100, "value": 0.5},
                {"id": "child1", "parent_id": "root", "visit_count": 50, "value": 0.7},
                {"id": "child2", "parent_id": "root", "visit_count": 30, "value": 0.4}
            ],
            "metadata": {
                "depth": max_depth,
                "loaded_nodes": 3,
                "total_nodes": 1000
            }
        },
        "execution_time": 0.05
    }

# For full HANA API functionality, use the dedicated routers/hana.py module
# This would be included in the main.py application as follows:
#
# from .routers import hana as hana_router
# app.include_router(hana_router.router)
#
# The test_main.py file provides basic connectivity testing only.