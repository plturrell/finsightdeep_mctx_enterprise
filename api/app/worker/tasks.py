import logging
import time
import datetime
import os
import json
from typing import Any, Dict, List, Optional, Tuple, Union
import asyncio
from functools import wraps

import jax
import jax.numpy as jnp
from fastapi import BackgroundTasks
from celery import Celery
from celery.signals import worker_ready

from ..core.config import get_settings
from ..core.logging import log_search_metrics
from ..db.hana_connector import hana_manager
from ..models.mcts_models import MCTSRequest

# Configure logger
logger = logging.getLogger("mctx")
settings = get_settings()

# Configure Celery
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery("mctx_worker", broker=REDIS_URL, backend=REDIS_URL)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    broker_connection_retry_on_startup=True,
    result_expires=86400,  # 1 day
)


@worker_ready.connect
def on_worker_ready(**kwargs):
    """Log when the worker is ready to process tasks."""
    logger.info(
        "Celery worker is ready to process tasks",
        extra={
            "event": "worker_ready",
            "hostname": celery_app.conf.get("worker_hostname", "unknown"),
        }
    )


@celery_app.task(name="mcts.run_large_search", bind=True)
def run_large_search(
    self,
    search_config: Dict[str, Any],
    user_id: Optional[str] = None,
    callback_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    Celery task for running large MCTS searches.
    
    This is designed for searches that are too large to run
    synchronously in the API.
    
    Args:
        self: Celery task instance
        search_config: MCTS search configuration
        user_id: Optional user ID for tracking
        callback_url: Optional URL to call with results
        
    Returns:
        Dict containing task results
    """
    try:
        import mctx
        import requests
        
        logger.info(
            f"Starting large MCTS search (task_id={self.request.id})",
            extra={
                "event": "task_start",
                "task_id": self.request.id,
                "search_type": search_config.get("search_type"),
                "batch_size": search_config.get("root_input", {}).get("batch_size"),
                "num_simulations": search_config.get("search_params", {}).get("num_simulations"),
            }
        )
        
        # Update task state
        self.update_state(state="RUNNING", meta={"progress": 0})
        
        # Start timing
        start_time = time.time()
        
        # Unpack configuration
        request = MCTSRequest(**search_config)
        root_input = request.root_input
        search_params = request.search_params
        search_type = request.search_type
        
        # Convert input arrays to JAX arrays
        prior_logits = jnp.array(root_input.prior_logits)
        value = jnp.array(root_input.value)
        embedding = jnp.array(root_input.embedding)
        
        # Create RootFnOutput
        root = mctx.RootFnOutput(
            prior_logits=prior_logits,
            value=value,
            embedding=embedding,
        )
        
        # Get a random key
        rng_key = jax.random.PRNGKey(int(time.time() * 1000) % (2**32))
        
        # Update progress
        self.update_state(state="RUNNING", meta={"progress": 10})
        
        # Create mock qvalues and recurrent function
        # In production, this would come from the actual model
        qvalues = jnp.ones((root_input.batch_size, root_input.num_actions))
        
        def recurrent_fn(params, rng_key, action, embedding):
            batch_size = action.shape[0]
            
            # Simple mock reward
            reward = jnp.ones_like(action, dtype=jnp.float32)
            discount = jnp.ones_like(reward)
            
            # Create dummy prior logits
            prior_logits = jnp.zeros((batch_size, qvalues.shape[1]), dtype=jnp.float32)
            
            # Create dummy value
            value = jnp.zeros_like(reward)
            
            # Increment embedding
            next_embedding = embedding + 1
            
            recurrent_fn_output = mctx.RecurrentFnOutput(
                reward=reward,
                discount=discount,
                prior_logits=prior_logits,
                value=value
            )
            
            return recurrent_fn_output, next_embedding
        
        # Update progress
        self.update_state(state="RUNNING", meta={"progress": 30})
        
        # Run the appropriate search algorithm
        if search_type == "muzero":
            policy_output = mctx.muzero_policy(
                params=(),
                rng_key=rng_key,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=search_params.num_simulations,
                max_depth=search_params.max_depth,
                dirichlet_fraction=search_params.dirichlet_fraction,
                dirichlet_alpha=search_params.dirichlet_alpha,
            )
        elif search_type == "gumbel_muzero":
            policy_output = mctx.gumbel_muzero_policy(
                params=(),
                rng_key=rng_key,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=search_params.num_simulations,
                max_depth=search_params.max_depth,
                max_num_considered_actions=search_params.max_num_considered_actions,
            )
        elif search_type == "stochastic_muzero":
            policy_output = mctx.stochastic_muzero_policy(
                params=(),
                rng_key=rng_key,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=search_params.num_simulations,
                max_depth=search_params.max_depth,
            )
        else:
            raise ValueError(f"Invalid search type: {search_type}")
        
        # Update progress
        self.update_state(state="RUNNING", meta={"progress": 80})
        
        # Calculate duration and metrics
        duration_ms = (time.time() - start_time) * 1000
        search_tree = policy_output.search_tree
        action = policy_output.action.tolist()
        action_weights = policy_output.action_weights.tolist()
        
        # Calculate metrics
        num_expanded_nodes = jnp.sum(search_tree.node_visits > 0).item()
        max_depth_reached = 0  # In a real implementation, track this during search
        
        # Log metrics
        log_search_metrics(
            search_type=search_type,
            batch_size=root_input.batch_size,
            num_simulations=search_params.num_simulations,
            duration=duration_ms,
            num_expanded_nodes=num_expanded_nodes,
            max_depth_reached=max_depth_reached
        )
        
        # Update progress
        self.update_state(state="RUNNING", meta={"progress": 90})
        
        # Prepare results
        result = {
            "action": action,
            "action_weights": action_weights,
            "search_statistics": {
                "duration_ms": duration_ms,
                "num_expanded_nodes": num_expanded_nodes,
                "max_depth_reached": max_depth_reached,
                "task_id": self.request.id,
                "completion_time": datetime.datetime.now().isoformat(),
            }
        }
        
        # Save to database
        try:
            today = datetime.datetime.now().strftime("%Y-%m-%d")
            
            # Save search history
            hana_manager.save_search_history(
                search_type=search_type,
                batch_size=root_input.batch_size,
                num_simulations=search_params.num_simulations,
                max_depth=search_params.max_depth,
                config={
                    "task_id": self.request.id,
                    "search_params": dict(search_params),
                    "root_input_shape": {
                        "batch_size": root_input.batch_size,
                        "num_actions": root_input.num_actions
                    }
                },
                duration_ms=duration_ms,
                num_expanded_nodes=num_expanded_nodes,
                max_depth_reached=max_depth_reached,
                result=result,
                user_id=user_id
            )
            
            # Update daily statistics
            hana_manager.update_daily_statistics(
                date_key=today,
                search_type=search_type
            )
        except Exception as e:
            logger.error(f"Failed to save search history: {str(e)}")
        
        # Send results to callback URL if provided
        if callback_url:
            try:
                requests.post(
                    callback_url,
                    json={
                        "task_id": self.request.id,
                        "status": "completed",
                        "result": result
                    },
                    timeout=10
                )
            except Exception as e:
                logger.error(f"Failed to send results to callback URL: {str(e)}")
        
        logger.info(
            f"Completed large MCTS search (task_id={self.request.id})",
            extra={
                "event": "task_complete",
                "task_id": self.request.id,
                "duration_ms": duration_ms,
                "num_expanded_nodes": num_expanded_nodes,
            }
        )
        
        return result
        
    except Exception as e:
        logger.exception(f"Error in large MCTS search task: {str(e)}")
        
        # Send error to callback URL if provided
        if callback_url:
            try:
                requests.post(
                    callback_url,
                    json={
                        "task_id": self.request.id,
                        "status": "failed",
                        "error": str(e)
                    },
                    timeout=10
                )
            except Exception as callback_error:
                logger.error(f"Failed to send error to callback URL: {str(callback_error)}")
        
        # Re-raise the exception to mark the task as failed
        raise


class BackgroundTaskManager:
    """
    Manager for background tasks using FastAPI's BackgroundTasks.
    
    This class provides methods for running lighter-weight tasks
    that don't need the full Celery infrastructure.
    """
    
    @staticmethod
    async def run_search_async(
        search_config: Dict[str, Any],
        background_tasks: BackgroundTasks,
    ) -> Dict[str, Any]:
        """
        Run search asynchronously using FastAPI's BackgroundTasks.
        
        Args:
            search_config: MCTS search configuration
            background_tasks: FastAPI BackgroundTasks
            
        Returns:
            Dict with task information
        """
        task_id = f"bg-{int(time.time())}-{os.urandom(4).hex()}"
        
        logger.info(
            f"Scheduling background search (task_id={task_id})",
            extra={
                "event": "background_task_created",
                "task_id": task_id,
                "search_type": search_config.get("search_type"),
            }
        )
        
        # Add search task to background tasks
        background_tasks.add_task(
            BackgroundTaskManager._execute_search,
            task_id,
            search_config,
        )
        
        return {
            "task_id": task_id,
            "status": "scheduled",
            "message": "Search scheduled for background execution",
        }
    
    @staticmethod
    async def _execute_search(
        task_id: str,
        search_config: Dict[str, Any],
    ) -> None:
        """
        Execute search in background.
        
        Args:
            task_id: Task ID
            search_config: Search configuration
        """
        import mctx
        
        logger.info(
            f"Starting background search (task_id={task_id})",
            extra={
                "event": "background_task_start",
                "task_id": task_id,
            }
        )
        
        # This would normally be implemented similar to the Celery task,
        # but simplified here for brevity
        try:
            # Simulate search execution
            await asyncio.sleep(2)
            
            logger.info(
                f"Completed background search (task_id={task_id})",
                extra={
                    "event": "background_task_complete",
                    "task_id": task_id,
                }
            )
            
        except Exception as e:
            logger.exception(
                f"Error in background search (task_id={task_id}): {str(e)}",
                extra={
                    "event": "background_task_error",
                    "task_id": task_id,
                    "error": str(e),
                }
            )