"""
Rate Limiting Module

This module provides rate limiting functionality to protect API endpoints from abuse,
implementing a token bucket algorithm with support for role-based limits.
"""

import time
from datetime import datetime
import logging
from typing import Dict, Tuple, Optional, Callable, Any
from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from pydantic import BaseModel

from .config import get_settings

logger = logging.getLogger("mctx")
settings = get_settings()


class RateLimitSettings(BaseModel):
    """Settings for rate limiting."""
    enabled: bool = True
    # Rate limits by endpoint (requests per minute)
    rate_limits: Dict[str, int] = {
        "default": 60,  # Default rate limit for all endpoints
        "/api/v1/mcts/search": 20,  # Limit for compute-intensive endpoints
        "/api/v1/auth/token": 10,  # Limit for authentication endpoints
    }
    # Additional limits by user role (multiplier of base rate)
    role_multipliers: Dict[str, float] = {
        "default": 1.0,  # Default multiplier
        "admin": 5.0,    # Admins get 5x the base rate
        "premium": 3.0,  # Premium users get 3x the base rate
    }
    # Window size in seconds
    window_size: int = 60
    # Cleanup interval for stale entries (seconds)
    cleanup_interval: int = 300


class RateLimiter:
    """
    Rate limiter implementation with token bucket algorithm.
    
    This class tracks request rates by IP and user ID, applying
    different limits based on endpoint and user role.
    """
    def __init__(self, settings: Optional[RateLimitSettings] = None):
        """
        Initialize rate limiter.
        
        Args:
            settings: Rate limit settings
        """
        self.settings = settings or RateLimitSettings()
        self._buckets: Dict[str, Dict[str, Any]] = {}
        self._last_cleanup = time.time()
    
    def _get_bucket_key(self, request: Request) -> str:
        """
        Get unique key for rate limit bucket.
        
        Args:
            request: HTTP request
            
        Returns:
            Unique bucket key based on client IP and path
        """
        client_host = request.client.host if request.client else "unknown"
        return f"{client_host}:{request.url.path}"
    
    def _get_rate_limit(self, endpoint: str, user_role: Optional[str] = None) -> int:
        """
        Get rate limit for endpoint and user role.
        
        Args:
            endpoint: API endpoint path
            user_role: Optional user role
            
        Returns:
            Rate limit in requests per minute
        """
        # Get base rate for endpoint, or default
        base_rate = self.settings.rate_limits.get(
            endpoint, self.settings.rate_limits["default"]
        )
        
        # Apply role multiplier if available
        multiplier = 1.0
        if user_role:
            multiplier = self.settings.role_multipliers.get(
                user_role, self.settings.role_multipliers["default"]
            )
        
        return int(base_rate * multiplier)
    
    def _cleanup_stale_buckets(self):
        """Remove stale buckets that haven't been used recently."""
        current_time = time.time()
        
        # Only run cleanup periodically
        if current_time - self._last_cleanup < self.settings.cleanup_interval:
            return
        
        self._last_cleanup = current_time
        stale_threshold = current_time - (self.settings.window_size * 2)
        
        # Remove stale buckets
        stale_keys = [
            key for key, bucket in self._buckets.items()
            if bucket["last_update"] < stale_threshold
        ]
        
        for key in stale_keys:
            del self._buckets[key]
        
        if stale_keys:
            logger.debug(f"Cleaned up {len(stale_keys)} stale rate limit buckets")
    
    def check_rate_limit(
        self, request: Request, user_role: Optional[str] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request exceeds rate limit.
        
        Uses token bucket algorithm to track request rates.
        
        Args:
            request: HTTP request
            user_role: Optional user role for role-based limits
            
        Returns:
            Tuple of (is_allowed, limit_info)
        """
        if not self.settings.enabled:
            return True, {"limit": "unlimited", "remaining": "unlimited"}
        
        # Clean up stale buckets periodically
        self._cleanup_stale_buckets()
        
        # Get bucket key and rate limit
        key = self._get_bucket_key(request)
        rate_limit = self._get_rate_limit(request.url.path, user_role)
        
        # Get or create bucket
        current_time = time.time()
        if key not in self._buckets:
            # Initialize new bucket with full tokens
            self._buckets[key] = {
                "tokens": rate_limit,
                "last_update": current_time,
                "request_count": 0,
            }
        
        bucket = self._buckets[key]
        
        # Calculate token refill based on time elapsed
        time_passed = current_time - bucket["last_update"]
        token_refill = time_passed * (rate_limit / self.settings.window_size)
        
        # Refill bucket (up to maximum)
        bucket["tokens"] = min(rate_limit, bucket["tokens"] + token_refill)
        bucket["last_update"] = current_time
        
        # Check if request can be processed
        if bucket["tokens"] >= 1.0:
            # Consume one token
            bucket["tokens"] -= 1.0
            bucket["request_count"] += 1
            
            # Request is allowed
            remaining = int(bucket["tokens"])
            return True, {
                "limit": rate_limit,
                "remaining": remaining,
                "reset": int(current_time + (self.settings.window_size * (1 - (remaining / rate_limit)))),
            }
        else:
            # Request exceeds rate limit
            time_to_reset = (1.0 - bucket["tokens"]) * (self.settings.window_size / rate_limit)
            return False, {
                "limit": rate_limit,
                "remaining": 0,
                "reset": int(current_time + time_to_reset),
            }


# Global rate limiter instance
rate_limiter = RateLimiter()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting.
    
    This middleware applies rate limits to all API endpoints
    based on client IP, endpoint, and user role.
    """
    
    def __init__(
        self,
        app: ASGIApp,
        get_user_role: Optional[Callable[[Request], Optional[str]]] = None,
    ):
        """
        Initialize rate limit middleware.
        
        Args:
            app: ASGI application
            get_user_role: Optional function to extract user role from request
        """
        super().__init__(app)
        self.get_user_role = get_user_role
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process request and apply rate limits.
        
        Args:
            request: HTTP request
            call_next: Next middleware in chain
            
        Returns:
            HTTP response
            
        Raises:
            HTTPException: If rate limit is exceeded
        """
        # Skip rate limiting for non-API paths
        if not request.url.path.startswith(settings.API_V1_STR):
            return await call_next(request)
        
        # Get user role if available
        user_role = None
        if self.get_user_role:
            user_role = await self.get_user_role(request)
        
        # Check rate limit
        allowed, limit_info = rate_limiter.check_rate_limit(request, user_role)
        
        if allowed:
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers to response
            response.headers["X-RateLimit-Limit"] = str(limit_info["limit"])
            response.headers["X-RateLimit-Remaining"] = str(limit_info["remaining"])
            response.headers["X-RateLimit-Reset"] = str(limit_info["reset"])
            
            return response
        else:
            # Log rate limit exceeded
            client_host = request.client.host if request.client else "unknown"
            logger.warning(
                f"Rate limit exceeded for {client_host} on {request.url.path}",
                extra={
                    "event": "rate_limit_exceeded",
                    "client_ip": client_host,
                    "path": request.url.path,
                    "user_role": user_role,
                }
            )
            
            # Return 429 Too Many Requests
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "message": "Rate limit exceeded",
                    "details": {
                        "limit": limit_info["limit"],
                        "reset": limit_info["reset"],
                    }
                },
                headers={
                    "Retry-After": str(limit_info["reset"] - int(time.time())),
                    "X-RateLimit-Limit": str(limit_info["limit"]),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(limit_info["reset"]),
                }
            )