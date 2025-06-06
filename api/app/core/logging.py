import logging
import json
import time
import uuid
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union

# Create a global logger
logger = logging.getLogger("mctx")


class CustomJSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.default_fields = kwargs

    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        
        # Add any default fields
        log_data.update(self.default_fields)
        
        # Add any extra attributes
        if hasattr(record, "extra"):
            log_data.update(record.extra)
        
        # Add exception info if available
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


def setup_json_logging():
    """
    Configure JSON structured logging.
    """
    # Remove existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create and configure handler with JSON formatter
    handler = logging.StreamHandler()
    formatter = CustomJSONFormatter(app="mctx-api")
    handler.setFormatter(formatter)
    
    # Add handler to root logger
    root_logger.addHandler(handler)


def with_logging(func: Callable) -> Callable:
    """
    Decorator for logging function entry and exit with timing.
    
    Args:
        func: The function to wrap with logging
        
    Returns:
        Wrapped function with logging added
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        function_name = func.__name__
        request_id = str(uuid.uuid4())
        
        # Log function entry
        logger.info(
            f"Entering {function_name}",
            extra={
                "function": function_name,
                "request_id": request_id,
                "event": "function_enter",
            }
        )
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            
            # Log successful function exit
            duration = (time.time() - start_time) * 1000  # Convert to ms
            logger.info(
                f"Exiting {function_name}",
                extra={
                    "function": function_name,
                    "request_id": request_id,
                    "event": "function_exit",
                    "duration_ms": duration,
                    "status": "success",
                }
            )
            
            return result
        
        except Exception as e:
            # Log exception
            duration = (time.time() - start_time) * 1000  # Convert to ms
            logger.exception(
                f"Exception in {function_name}: {str(e)}",
                extra={
                    "function": function_name,
                    "request_id": request_id,
                    "event": "function_exception",
                    "duration_ms": duration,
                    "status": "error",
                    "error_type": type(e).__name__,
                }
            )
            raise
    
    return wrapper


def log_api_request(
    endpoint: str,
    method: str,
    status_code: int,
    duration: float,
    params: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None
):
    """
    Log API request details.
    
    Args:
        endpoint: API endpoint path
        method: HTTP method
        status_code: HTTP status code
        duration: Request duration in milliseconds
        params: Optional request parameters
        error: Optional error message
    """
    log_data = {
        "endpoint": endpoint,
        "method": method,
        "status_code": status_code,
        "duration_ms": duration,
        "event": "api_request",
    }
    
    if params:
        log_data["params"] = params
    
    if error:
        log_data["error"] = error
        logger.error(f"API request error: {endpoint}", extra=log_data)
    else:
        logger.info(f"API request: {endpoint}", extra=log_data)


def log_search_metrics(
    search_type: str,
    batch_size: int,
    num_simulations: int,
    duration: float,
    num_expanded_nodes: int,
    max_depth_reached: int
):
    """
    Log metrics from MCTS search operations.
    
    Args:
        search_type: Type of search algorithm used
        batch_size: Number of parallel searches
        num_simulations: Number of simulations performed
        duration: Search duration in milliseconds
        num_expanded_nodes: Number of nodes expanded
        max_depth_reached: Maximum depth reached in search
    """
    logger.info(
        f"MCTS search metrics: {search_type}",
        extra={
            "event": "search_metrics",
            "search_type": search_type,
            "batch_size": batch_size,
            "num_simulations": num_simulations,
            "duration_ms": duration,
            "num_expanded_nodes": num_expanded_nodes,
            "max_depth_reached": max_depth_reached,
            "nodes_per_second": num_expanded_nodes / (duration / 1000) if duration > 0 else 0,
        }
    )