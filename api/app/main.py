from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging
import time
from contextlib import asynccontextmanager

from .core.config import get_settings, setup_logging
from .core.exceptions import MCTXException
from .core.rate_limit import RateLimitMiddleware
from .core.auth import get_user_id
from .routers import mcts, auth, tasks, graphql, health
from .models.mcts_models import ErrorResponse

# Set up logging
logger = setup_logging()
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for the FastAPI application.
    
    Handles startup and shutdown events.
    
    Args:
        app: FastAPI application instance
    """
    # Startup
    logger.info("Starting MCTX API server", 
                extra={"event": "server_startup"})
    
    # Initialize JAX
    try:
        import jax
        # Log JAX configuration
        logger.info(
            f"JAX initialized successfully",
            extra={
                "event": "jax_init",
                "devices": str(jax.devices()),
                "backend": jax.default_backend(),
            }
        )
    except Exception as e:
        logger.error(
            f"Failed to initialize JAX: {str(e)}",
            extra={"event": "jax_init_error", "error": str(e)}
        )
    
    yield
    
    # Shutdown
    logger.info("Shutting down MCTX API server", 
                extra={"event": "server_shutdown"})


# Initialize FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="API for Monte Carlo Tree Search using the MCTX library",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting middleware
app.add_middleware(
    RateLimitMiddleware,
    get_user_role=None  # This would extract user role from request in production
)


# Exception handlers
@app.exception_handler(MCTXException)
async def mctx_exception_handler(request: Request, exc: MCTXException):
    """
    Handle MCTX-specific exceptions.
    """
    from .core.exceptions import http_exception_handler
    http_exc = http_exception_handler(exc)
    
    return JSONResponse(
        status_code=http_exc.status_code,
        content={"status_code": http_exc.status_code, "message": http_exc.detail["message"], "details": http_exc.detail.get("details")},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle request validation errors.
    """
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"status_code": status.HTTP_422_UNPROCESSABLE_ENTITY, "message": "Validation error", "details": exc.errors()},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Handle all other exceptions.
    """
    # Log the exception
    logger.exception(f"Unhandled exception: {str(exc)}")
    
    # In debug mode, return detailed error information
    if settings.DEBUG:
        detail = {"error": str(exc), "type": type(exc).__name__}
    else:
        detail = None
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"status_code": status.HTTP_500_INTERNAL_SERVER_ERROR, "message": "Internal server error", "details": detail},
    )


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware to log HTTP requests and responses.
    """
    start_time = time.time()
    
    # Extract request details
    method = request.method
    path = request.url.path
    client_ip = request.client.host if request.client else "unknown"
    
    # Process the request
    try:
        response = await call_next(request)
        status_code = response.status_code
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Log request details
        logger.info(
            f"{method} {path} {status_code}",
            extra={
                "event": "http_request",
                "method": method,
                "path": path,
                "status_code": status_code,
                "client_ip": client_ip,
                "duration_ms": duration_ms,
            }
        )
        
        return response
    except Exception as e:
        # Log exceptions
        duration_ms = (time.time() - start_time) * 1000
        logger.exception(
            f"Exception during request processing: {str(e)}",
            extra={
                "event": "http_request_error",
                "method": method,
                "path": path,
                "client_ip": client_ip,
                "duration_ms": duration_ms,
                "error": str(e),
            }
        )
        raise


# Include routers
app.include_router(mcts.router, prefix=settings.API_V1_STR)
app.include_router(auth.router, prefix=settings.API_V1_STR)
app.include_router(tasks.router, prefix=settings.API_V1_STR)
app.include_router(graphql.router, prefix=settings.API_V1_STR)
app.include_router(health.router)


# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """
    Root endpoint providing basic API information.
    """
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs": "/api/docs",
    }