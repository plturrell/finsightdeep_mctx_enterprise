from typing import Any, Dict, Optional

from fastapi import HTTPException, status


class MCTXException(Exception):
    """
    Base exception class for all MCTX exceptions.
    
    This allows catching all MCTX-specific exceptions with a single except clause.
    """
    pass


class InvalidInputError(MCTXException):
    """
    Exception raised when input data is invalid.
    """
    def __init__(
        self, 
        message: str = "Invalid input data", 
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.details = details
        super().__init__(self.message)


class ModelError(MCTXException):
    """
    Exception raised when there's an error in model operations.
    """
    def __init__(
        self, 
        message: str = "Error in model operation", 
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.details = details
        super().__init__(self.message)


class ConfigurationError(MCTXException):
    """
    Exception raised when there's an error in the configuration.
    """
    def __init__(
        self, 
        message: str = "Configuration error", 
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.details = details
        super().__init__(self.message)


class ResourceExceededError(MCTXException):
    """
    Exception raised when resource limits are exceeded.
    """
    def __init__(
        self, 
        message: str = "Resource limits exceeded", 
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.details = details
        super().__init__(self.message)


class ValidationError(MCTXException):
    """
    Exception raised when validation fails.
    """
    def __init__(
        self, 
        message: str = "Validation failed", 
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.details = details
        super().__init__(self.message)


def http_exception_handler(exc: MCTXException) -> HTTPException:
    """
    Convert MCTX exceptions to HTTPExceptions.
    
    Args:
        exc: MCTX exception to convert
        
    Returns:
        HTTPException with appropriate status code and details
    """
    if isinstance(exc, InvalidInputError):
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": exc.message, "details": exc.details},
        )
    elif isinstance(exc, ModelError):
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": exc.message, "details": exc.details},
        )
    elif isinstance(exc, ConfigurationError):
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": exc.message, "details": exc.details},
        )
    elif isinstance(exc, ResourceExceededError):
        return HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"message": exc.message, "details": exc.details},
        )
    elif isinstance(exc, ValidationError):
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": exc.message, "details": exc.details},
        )
    else:
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": str(exc)},
        )