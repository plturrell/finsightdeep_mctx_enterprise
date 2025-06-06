import graphene
from typing import Dict, List, Optional, Any, Union
from graphql import GraphQLResolveInfo
import time
import logging
import datetime
import jax
import jax.numpy as jnp
import mctx

from ..core.exceptions import InvalidInputError, ModelError
from ..core.logging import log_search_metrics
from ..core.config import get_settings
from ..services.mcts_service import MCTSService
from ..db.hana_connector import hana_manager

logger = logging.getLogger("mctx")
settings = get_settings()


# Input types
class SearchParamsInput(graphene.InputObjectType):
    """GraphQL input type for search parameters."""
    num_simulations = graphene.Int(default_value=32)
    max_depth = graphene.Int(required=False)
    max_num_considered_actions = graphene.Int(default_value=16)
    dirichlet_fraction = graphene.Float(required=False)
    dirichlet_alpha = graphene.Float(required=False)


class RootInputType(graphene.InputObjectType):
    """GraphQL input type for root node input."""
    prior_logits = graphene.List(graphene.List(graphene.Float))
    value = graphene.List(graphene.Float)
    embedding = graphene.List(graphene.Float)
    batch_size = graphene.Int()
    num_actions = graphene.Int()


# Object types
class SearchStatistics(graphene.ObjectType):
    """GraphQL type for search statistics."""
    duration_ms = graphene.Float()
    num_expanded_nodes = graphene.Int()
    max_depth_reached = graphene.Int()


class SearchResult(graphene.ObjectType):
    """GraphQL type for search results."""
    action = graphene.List(graphene.Int)
    action_weights = graphene.List(graphene.List(graphene.Float))
    search_statistics = graphene.Field(SearchStatistics)


class SearchHistoryEntry(graphene.ObjectType):
    """GraphQL type for search history entry."""
    id = graphene.Int()
    timestamp = graphene.String()
    user_id = graphene.String()
    search_type = graphene.String()
    batch_size = graphene.Int()
    num_simulations = graphene.Int()
    max_depth = graphene.Int()
    duration_ms = graphene.Float()
    num_expanded_nodes = graphene.Int()
    max_depth_reached = graphene.Int()


class SearchStatisticsAggregate(graphene.ObjectType):
    """GraphQL type for aggregated search statistics."""
    date_key = graphene.String()
    search_type = graphene.String()
    total_searches = graphene.Int()
    avg_duration_ms = graphene.Float()
    avg_expanded_nodes = graphene.Float()
    max_batch_size = graphene.Int()
    max_num_simulations = graphene.Int()


class TaskInfo(graphene.ObjectType):
    """GraphQL type for task information."""
    task_id = graphene.String()
    status = graphene.String()
    message = graphene.String(required=False)
    progress = graphene.Int(required=False)
    error = graphene.String(required=False)


# Query type
class Query(graphene.ObjectType):
    """Root GraphQL query type."""
    health = graphene.Field(
        graphene.JSONString,
        description="Health check endpoint"
    )
    
    search_history = graphene.List(
        SearchHistoryEntry,
        limit=graphene.Int(default_value=10),
        offset=graphene.Int(default_value=0),
        user_id=graphene.String(required=False),
        search_type=graphene.String(required=False),
        start_date=graphene.String(required=False),
        end_date=graphene.String(required=False),
        description="Get search history records"
    )
    
    search_statistics = graphene.List(
        SearchStatisticsAggregate,
        start_date=graphene.String(required=False),
        end_date=graphene.String(required=False),
        search_type=graphene.String(required=False),
        description="Get aggregated search statistics by date"
    )
    
    task_status = graphene.Field(
        TaskInfo,
        task_id=graphene.String(required=True),
        description="Get status of an asynchronous task"
    )

    def resolve_health(self, info):
        """Resolve health check query."""
        return {
            "status": "ok",
            "version": settings.VERSION,
            "timestamp": time.time(),
        }
    
    def resolve_search_history(
        self, 
        info, 
        limit=10, 
        offset=0,
        user_id=None,
        search_type=None,
        start_date=None,
        end_date=None
    ):
        """Resolve search history query."""
        try:
            # Get search history from database
            history = hana_manager.get_search_history(
                limit=limit,
                offset=offset,
                user_id=user_id,
                search_type=search_type,
                start_date=start_date,
                end_date=end_date,
            )
            
            return [
                SearchHistoryEntry(
                    id=entry["ID"],
                    timestamp=str(entry["TIMESTAMP"]),
                    user_id=entry["USER_ID"],
                    search_type=entry["SEARCH_TYPE"],
                    batch_size=entry["BATCH_SIZE"],
                    num_simulations=entry["NUM_SIMULATIONS"],
                    max_depth=entry["MAX_DEPTH"],
                    duration_ms=entry["DURATION_MS"],
                    num_expanded_nodes=entry["NUM_EXPANDED_NODES"],
                    max_depth_reached=entry["MAX_DEPTH_REACHED"],
                )
                for entry in history
            ]
        except Exception as e:
            logger.error(f"Error getting search history: {str(e)}")
            raise Exception(f"Failed to get search history: {str(e)}")
    
    def resolve_search_statistics(
        self,
        info,
        start_date=None,
        end_date=None,
        search_type=None
    ):
        """Resolve search statistics query."""
        try:
            # In a real implementation, this would query the database
            # For now, return mock data
            return [
                SearchStatisticsAggregate(
                    date_key="2023-01-01",
                    search_type="gumbel_muzero",
                    total_searches=120,
                    avg_duration_ms=155.3,
                    avg_expanded_nodes=64.5,
                    max_batch_size=32,
                    max_num_simulations=128
                ),
                SearchStatisticsAggregate(
                    date_key="2023-01-02",
                    search_type="gumbel_muzero",
                    total_searches=145,
                    avg_duration_ms=162.1,
                    avg_expanded_nodes=72.3,
                    max_batch_size=32,
                    max_num_simulations=256
                ),
            ]
        except Exception as e:
            logger.error(f"Error getting search statistics: {str(e)}")
            raise Exception(f"Failed to get search statistics: {str(e)}")
    
    def resolve_task_status(self, info, task_id):
        """Resolve task status query."""
        try:
            # This would check the task status in a real implementation
            # For demonstration, return mock data
            return TaskInfo(
                task_id=task_id,
                status="completed",
                message="Task completed successfully",
                progress=100
            )
        except Exception as e:
            logger.error(f"Error getting task status: {str(e)}")
            raise Exception(f"Failed to get task status: {str(e)}")


# Mutation type
class RunSearch(graphene.Mutation):
    """GraphQL mutation to run MCTS search."""
    class Arguments:
        root_input = RootInputType(required=True)
        search_params = SearchParamsInput(required=True)
        search_type = graphene.String(default_value="gumbel_muzero")
    
    action = graphene.List(graphene.Int)
    action_weights = graphene.List(graphene.List(graphene.Float))
    search_statistics = graphene.Field(SearchStatistics)
    
    def mutate(
        self, 
        info, 
        root_input, 
        search_params, 
        search_type="gumbel_muzero"
    ):
        """Execute MCTS search mutation."""
        try:
            # Convert GraphQL input to MCTS request
            from ..models.mcts_models import MCTSRequest, RootInput, SearchParams
            
            # Create request object
            request = MCTSRequest(
                root_input=RootInput(
                    prior_logits=root_input.prior_logits,
                    value=root_input.value,
                    embedding=root_input.embedding,
                    batch_size=root_input.batch_size,
                    num_actions=root_input.num_actions
                ),
                search_params=SearchParams(
                    num_simulations=search_params.num_simulations,
                    max_depth=search_params.max_depth,
                    max_num_considered_actions=search_params.max_num_considered_actions,
                    dirichlet_fraction=search_params.dirichlet_fraction,
                    dirichlet_alpha=search_params.dirichlet_alpha
                ),
                search_type=search_type
            )
            
            # Get user ID from context if available
            context = info.context
            user_id = getattr(context, "user_id", None)
            
            # Run search
            mcts_service = MCTSService()
            result = mcts_service.run_search(request, user_id=user_id)
            
            # Convert to GraphQL type
            return RunSearch(
                action=result.action,
                action_weights=result.action_weights,
                search_statistics=SearchStatistics(
                    duration_ms=result.search_statistics["duration_ms"],
                    num_expanded_nodes=result.search_statistics["num_expanded_nodes"],
                    max_depth_reached=result.search_statistics["max_depth_reached"]
                )
            )
            
        except InvalidInputError as e:
            logger.error(f"Invalid input for GraphQL search: {str(e)}")
            raise graphene.GraphQLError(
                message=str(e),
                extensions={"code": "BAD_USER_INPUT", "details": e.details}
            )
        except Exception as e:
            logger.exception(f"Error in GraphQL search mutation: {str(e)}")
            raise graphene.GraphQLError(
                message=f"Failed to execute MCTS search: {str(e)}",
                extensions={"code": "INTERNAL_SERVER_ERROR"}
            )


class QueueSearch(graphene.Mutation):
    """GraphQL mutation to queue an asynchronous MCTS search."""
    class Arguments:
        root_input = RootInputType(required=True)
        search_params = SearchParamsInput(required=True)
        search_type = graphene.String(default_value="gumbel_muzero")
        callback_url = graphene.String(required=False)
    
    task_id = graphene.String()
    status = graphene.String()
    message = graphene.String()
    
    def mutate(
        self, 
        info, 
        root_input, 
        search_params, 
        search_type="gumbel_muzero",
        callback_url=None
    ):
        """Queue an asynchronous MCTS search."""
        try:
            # Convert GraphQL input to MCTS request
            from ..models.mcts_models import MCTSRequest, RootInput, SearchParams
            
            # Create request object
            request = MCTSRequest(
                root_input=RootInput(
                    prior_logits=root_input.prior_logits,
                    value=root_input.value,
                    embedding=root_input.embedding,
                    batch_size=root_input.batch_size,
                    num_actions=root_input.num_actions
                ),
                search_params=SearchParams(
                    num_simulations=search_params.num_simulations,
                    max_depth=search_params.max_depth,
                    max_num_considered_actions=search_params.max_num_considered_actions,
                    dirichlet_fraction=search_params.dirichlet_fraction,
                    dirichlet_alpha=search_params.dirichlet_alpha
                ),
                search_type=search_type
            )
            
            # Get user ID from context if available
            context = info.context
            user_id = getattr(context, "user_id", None)
            
            # Queue task
            from ..worker.tasks import run_large_search
            
            search_config = request.dict()
            task = run_large_search.apply_async(
                args=[search_config, user_id, callback_url],
                countdown=0,
            )
            
            logger.info(
                f"Queued MCTS search via GraphQL (task_id={task.id})",
                extra={
                    "event": "graphql_task_queued",
                    "task_id": task.id,
                }
            )
            
            return QueueSearch(
                task_id=task.id,
                status="queued",
                message="Search queued for processing"
            )
            
        except Exception as e:
            logger.exception(f"Error in GraphQL queue search mutation: {str(e)}")
            raise graphene.GraphQLError(
                message=f"Failed to queue MCTS search: {str(e)}",
                extensions={"code": "INTERNAL_SERVER_ERROR"}
            )


class Mutation(graphene.ObjectType):
    """Root GraphQL mutation type."""
    run_search = RunSearch.Field(description="Run MCTS search synchronously")
    queue_search = QueueSearch.Field(description="Queue MCTS search for asynchronous processing")


# Create schema
schema = graphene.Schema(query=Query, mutation=Mutation)