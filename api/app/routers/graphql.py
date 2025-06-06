from fastapi import APIRouter, Depends, Request, HTTPException, status
from fastapi.responses import JSONResponse
import logging
from typing import Dict, Any, Optional
from starlette.graphql import GraphQLApp
from graphql import format_error

from ..core.auth import get_current_user, get_user_id
from ..models.auth_models import User
from ..graphql.schema import schema

router = APIRouter(prefix="/graphql", tags=["graphql"])
logger = logging.getLogger("mctx")


class AuthenticatedGraphQLApp(GraphQLApp):
    """
    GraphQL app with authentication support.
    
    This extends the standard GraphQLApp to inject user information
    into the GraphQL context.
    """
    async def handle_graphql(self, request: Request) -> JSONResponse:
        """
        Handle GraphQL request with authentication.
        
        Args:
            request: HTTP request
            
        Returns:
            JSON response
        """
        # Extract user information
        try:
            # Get user from request state (set by authentication middleware)
            user = getattr(request.state, "user", None)
            user_id = user.username if user else None
        except Exception:
            user = None
            user_id = None
        
        # Add user info to request context
        request.state.user_id = user_id
        
        # Process request with parent class
        return await super().handle_graphql(request)


# Create GraphQL app with authentication
graphql_app = AuthenticatedGraphQLApp(schema=schema)


@router.post(
    "",
    summary="GraphQL endpoint",
    description="Process GraphQL queries and mutations"
)
async def graphql_route(
    request: Request,
    user: Optional[User] = Depends(get_current_user)
):
    """
    GraphQL endpoint that supports authentication.
    
    This route delegates to the GraphQLApp but adds user information
    to the request context.
    
    Args:
        request: HTTP request
        user: Authenticated user
        
    Returns:
        GraphQL response
    """
    # Add user to request state
    request.state.user = user
    if user:
        request.state.user_id = user.username
    
    # Execute GraphQL query
    return await graphql_app.handle_graphql(request)


@router.get(
    "",
    summary="GraphQL endpoint (GET)",
    description="Process GraphQL queries (GET method)"
)
async def graphql_get_route(
    request: Request,
    user: Optional[User] = Depends(get_current_user)
):
    """
    GraphQL GET endpoint that supports authentication.
    
    Args:
        request: HTTP request
        user: Authenticated user
        
    Returns:
        GraphQL response
    """
    # Add user to request state
    request.state.user = user
    if user:
        request.state.user_id = user.username
    
    # Execute GraphQL query
    return await graphql_app.handle_graphql(request)