from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from starlette.middleware.base import BaseHTTPMiddleware
import logging
import time
from datetime import datetime
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mctx.api")

# Import routers and middleware
from .core.config import get_settings
from .core.logging import LoggingMiddleware
from .core.auth import get_current_user
from .routers import auth, mcts, tasks
try:
    from .routers import hana
    HANA_ROUTER_AVAILABLE = True
except ImportError:
    HANA_ROUTER_AVAILABLE = False
    logger.warning("HANA router not available, endpoints will be disabled")

# Get settings
settings = get_settings()

# Initialize FastAPI application
app = FastAPI(
    title="MCTX Enterprise API",
    version="1.0.0",
    description="Enterprise-grade Monte Carlo Tree Search API with SAP HANA integration",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add logging middleware
app.add_middleware(LoggingMiddleware)

# Include routers
app.include_router(auth.router, prefix="/api")
app.include_router(mcts.router)
app.include_router(tasks.router)

# Include HANA router if available
if HANA_ROUTER_AVAILABLE:
    app.include_router(hana.router)

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Add custom security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyHeader": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key"
        },
        "BearerToken": {
            "type": "http",
            "scheme": "bearer"
        }
    }
    
    # Apply security to all routes
    openapi_schema["security"] = [
        {"ApiKeyHeader": []},
        {"BearerToken": []}
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """
    Root endpoint providing basic API information.
    """
    return {
        "name": "MCTX Enterprise API",
        "version": "1.0.0",
        "description": "Enterprise-grade Monte Carlo Tree Search API with SAP HANA integration",
        "documentation": "/api/docs"
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
            "debug": settings.debug,
            "host": settings.host
        }
    }

# Example secured endpoint
@app.get("/secure-endpoint/", tags=["example"])
async def secure_endpoint(current_user = Depends(get_current_user)):
    """
    Example of a secured endpoint requiring authentication.
    """
    return {
        "message": "You have access to this secure endpoint",
        "user": current_user.username,
        "timestamp": datetime.utcnow().isoformat()
    }