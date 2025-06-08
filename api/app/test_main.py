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
    description="Simple test API for MCTX Docker testing"
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

# SAP HANA connection test endpoint
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