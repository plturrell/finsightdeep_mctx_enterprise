from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import logging
import time
from typing import Callable

logger = logging.getLogger("mctx.api")


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging API requests and responses."""
    
    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        start_time = time.time()
        
        # Log request details
        logger.info(
            f"Request: {request.method} {request.url.path}"
            f" - Client: {request.client.host if request.client else 'unknown'}"
        )
        
        response = None
        try:
            response = await call_next(request)
            
            # Log response details
            process_time = time.time() - start_time
            logger.info(
                f"Response: {response.status_code} - "
                f"Took: {process_time:.4f}s"
            )
            return response
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            raise
        finally:
            # If response was not created, log error
            if response is None:
                logger.error(f"No response generated - took {time.time() - start_time:.4f}s")