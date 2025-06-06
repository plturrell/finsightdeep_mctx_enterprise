from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
import time
import logging
from typing import Any, Dict

from ..models.mcts_models import MCTSRequest, SearchResult, ErrorResponse
from ..services.mcts_service import MCTSService
from ..core.exceptions import MCTXException, http_exception_handler
from ..core.config import get_settings

router = APIRouter(prefix="/mcts", tags=["mcts"])
logger = logging.getLogger("mctx")
settings = get_settings()


@router.post(
    "/search",
    response_model=SearchResult,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Run Monte Carlo Tree Search",
    description="Runs MCTS with specified parameters and returns selected actions and search statistics",
)
async def run_search(
    request: MCTSRequest,
    http_request: Request
) -> SearchResult:
    """
    Run Monte Carlo Tree Search with the provided configuration.
    
    This endpoint performs MCTS search using the specified algorithm type
    and parameters. It supports various MCTS algorithms including MuZero,
    Gumbel MuZero, and Stochastic MuZero.
    
    Args:
        request: Search configuration including root state and parameters
        http_request: FastAPI request object
        
    Returns:
        SearchResult containing selected actions and search statistics
        
    Raises:
        HTTPException: On validation or execution errors
    """
    start_time = time.time()
    client_ip = http_request.client.host if http_request.client else "unknown"
    
    try:
        mcts_service = MCTSService()
        result = mcts_service.run_search(request)
        
        # Log successful request
        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            f"MCTS search successful",
            extra={
                "event": "api_request",
                "endpoint": "/mcts/search",
                "method": "POST",
                "client_ip": client_ip,
                "search_type": request.search_type,
                "batch_size": request.root_input.batch_size,
                "num_simulations": request.search_params.num_simulations,
                "duration_ms": duration_ms,
                "status_code": 200,
            }
        )
        
        return result
        
    except MCTXException as e:
        # Log exception
        duration_ms = (time.time() - start_time) * 1000
        logger.error(
            f"MCTS search failed: {str(e)}",
            extra={
                "event": "api_request_error",
                "endpoint": "/mcts/search",
                "method": "POST",
                "client_ip": client_ip,
                "search_type": request.search_type,
                "batch_size": request.root_input.batch_size,
                "num_simulations": request.search_params.num_simulations,
                "duration_ms": duration_ms,
                "error": str(e),
                "error_type": type(e).__name__,
            }
        )
        
        # Convert to HTTPException
        http_exc = http_exception_handler(e)
        raise http_exc
        
    except Exception as e:
        # Log unexpected exceptions
        duration_ms = (time.time() - start_time) * 1000
        logger.exception(
            f"Unexpected error in MCTS search: {str(e)}",
            extra={
                "event": "api_request_error",
                "endpoint": "/mcts/search",
                "method": "POST",
                "client_ip": client_ip,
                "duration_ms": duration_ms,
                "error": str(e),
                "error_type": type(e).__name__,
            }
        )
        
        # Return 500 error
        raise HTTPException(
            status_code=500,
            detail={
                "message": "An unexpected error occurred",
                "details": {"error": str(e)} if settings.DEBUG else None,
            },
        )


@router.get(
    "/health",
    response_model=Dict[str, Any],
    summary="Health check endpoint",
    description="Returns health status of the MCTS API service",
)
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint to verify API service status.
    
    Returns:
        Dictionary with status information
    """
    return {
        "status": "ok",
        "version": settings.VERSION,
        "timestamp": time.time(),
    }