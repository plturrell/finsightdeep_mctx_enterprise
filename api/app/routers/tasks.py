from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status, Body
import logging
from typing import Dict, Any, Optional, List
import time

from ..models.mcts_models import MCTSRequest, ErrorResponse
from ..worker.tasks import celery_app, run_large_search, BackgroundTaskManager
from ..core.auth import get_current_user, get_user_id, RoleChecker
from ..models.auth_models import User
from ..core.config import get_settings

router = APIRouter(prefix="/tasks", tags=["tasks"])
logger = logging.getLogger("mctx")
settings = get_settings()

# Role-based access control
require_admin = RoleChecker(["admin"])


@router.post(
    "/search/async",
    summary="Run MCTS search asynchronously",
    description="Schedules a search to run in the background and returns immediately with a task ID",
    response_model=Dict[str, Any],
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def run_search_async(
    request: MCTSRequest,
    background_tasks: BackgroundTasks,
    user_id: Optional[str] = Depends(get_user_id),
):
    """
    Run MCTS search asynchronously using FastAPI background tasks.
    
    This endpoint is for smaller searches that can be handled by
    the API server without needing a dedicated worker.
    
    Args:
        request: MCTS search configuration
        background_tasks: FastAPI background tasks
        user_id: User ID from authentication
        
    Returns:
        Dict with task ID and status
    """
    # Run search in background
    search_config = request.dict()
    result = await BackgroundTaskManager.run_search_async(
        search_config=search_config,
        background_tasks=background_tasks,
    )
    
    return result


@router.post(
    "/search/queue",
    summary="Queue large MCTS search",
    description="Queues a large search for processing by workers and returns a task ID for tracking",
    response_model=Dict[str, Any],
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def queue_search(
    request: MCTSRequest,
    callback_url: Optional[str] = None,
    user_id: Optional[str] = Depends(get_user_id),
):
    """
    Queue a large MCTS search for asynchronous processing.
    
    This endpoint is for larger searches that need to be
    processed by dedicated Celery workers.
    
    Args:
        request: MCTS search configuration
        callback_url: Optional URL to call with results
        user_id: User ID from authentication
        
    Returns:
        Dict with task ID and status
    """
    # Validate inputs
    if request.root_input.batch_size > settings.MAX_BATCH_SIZE * 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": f"Batch size exceeds maximum allowed for queued tasks ({settings.MAX_BATCH_SIZE * 2})",
                "details": {"batch_size": request.root_input.batch_size}
            }
        )
    
    # Create task
    search_config = request.dict()
    task = run_large_search.apply_async(
        args=[search_config, user_id, callback_url],
        countdown=0,  # Start immediately
    )
    
    logger.info(
        f"Queued large MCTS search task (task_id={task.id})",
        extra={
            "event": "task_queued",
            "task_id": task.id,
            "search_type": request.search_type,
            "batch_size": request.root_input.batch_size,
            "num_simulations": request.search_params.num_simulations,
            "user_id": user_id,
        }
    )
    
    return {
        "task_id": task.id,
        "status": "queued",
        "message": "Search queued for processing",
    }


@router.get(
    "/status/{task_id}",
    summary="Get task status",
    description="Retrieve the status and result of a task by its ID",
    response_model=Dict[str, Any],
    responses={
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def get_task_status(
    task_id: str,
    user_id: Optional[str] = Depends(get_user_id),
):
    """
    Get status of a task.
    
    Args:
        task_id: Celery task ID
        user_id: User ID from authentication
        
    Returns:
        Dict with task status and result if available
    """
    try:
        # Check if this is a background task or Celery task
        if task_id.startswith("bg-"):
            # For background tasks, we don't have persistent storage
            # In a production system, you would store task states in a database
            return {
                "task_id": task_id,
                "status": "unknown",
                "message": "Background task status not available",
            }
        
        # For Celery tasks
        task = run_large_search.AsyncResult(task_id)
        
        if task.state == "PENDING":
            response = {
                "task_id": task_id,
                "status": "pending",
                "message": "Task is pending execution",
            }
        elif task.state == "STARTED" or task.state == "RUNNING":
            response = {
                "task_id": task_id,
                "status": "running",
                "message": "Task is currently running",
            }
            # Add progress if available
            if task.info and isinstance(task.info, dict) and "progress" in task.info:
                response["progress"] = task.info["progress"]
        elif task.state == "SUCCESS":
            response = {
                "task_id": task_id,
                "status": "completed",
                "result": task.result,
            }
        elif task.state == "FAILURE":
            response = {
                "task_id": task_id,
                "status": "failed",
                "error": str(task.result) if task.result else "Unknown error",
            }
        else:
            response = {
                "task_id": task_id,
                "status": task.state.lower(),
                "message": "Task is in an unknown state",
            }
        
        return response
        
    except Exception as e:
        logger.error(f"Error retrieving task status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": "Failed to retrieve task status",
                "details": {"error": str(e)} if settings.DEBUG else None,
            }
        )


@router.delete(
    "/{task_id}",
    summary="Cancel task",
    description="Cancel a running or pending task",
    response_model=Dict[str, Any],
    responses={
        404: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    dependencies=[Depends(get_current_user)]  # Require authentication
)
async def cancel_task(
    task_id: str,
    user: User = Depends(get_current_user),
):
    """
    Cancel a running or pending task.
    
    Args:
        task_id: Celery task ID
        user: Authenticated user
        
    Returns:
        Dict with cancellation status
    """
    try:
        # Background tasks can't be canceled
        if task_id.startswith("bg-"):
            return {
                "task_id": task_id,
                "status": "error",
                "message": "Background tasks cannot be canceled",
            }
        
        # Revoke Celery task
        task = run_large_search.AsyncResult(task_id)
        
        # Check if task exists
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"message": f"Task {task_id} not found"}
            )
        
        # Revoke the task
        task.revoke(terminate=True)
        
        logger.info(
            f"Task {task_id} canceled by user {user.username}",
            extra={
                "event": "task_canceled",
                "task_id": task_id,
                "user": user.username,
            }
        )
        
        return {
            "task_id": task_id,
            "status": "canceled",
            "message": "Task has been canceled",
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error canceling task: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": "Failed to cancel task",
                "details": {"error": str(e)} if settings.DEBUG else None,
            }
        )


@router.get(
    "/",
    summary="List tasks",
    description="List all active tasks (admin only)",
    response_model=List[Dict[str, Any]],
    dependencies=[Depends(require_admin)],  # Require admin role
)
async def list_tasks():
    """
    List all active tasks (admin only).
    
    Returns:
        List of active tasks
    """
    try:
        # Get active tasks from Celery
        i = celery_app.control.inspect()
        active_tasks = i.active() or {}
        scheduled_tasks = i.scheduled() or {}
        reserved_tasks = i.reserved() or {}
        
        # Combine all tasks
        all_tasks = []
        
        # Process active tasks
        for worker, tasks in active_tasks.items():
            for task in tasks:
                all_tasks.append({
                    "task_id": task["id"],
                    "status": "active",
                    "worker": worker,
                    "name": task["name"],
                    "time_start": task["time_start"],
                    "args": task["args"],
                    "kwargs": task["kwargs"],
                })
        
        # Process scheduled tasks
        for worker, tasks in scheduled_tasks.items():
            for task in tasks:
                all_tasks.append({
                    "task_id": task["request"]["id"],
                    "status": "scheduled",
                    "worker": worker,
                    "name": task["request"]["name"],
                    "eta": task["eta"],
                    "args": task["request"]["args"],
                    "kwargs": task["request"]["kwargs"],
                })
        
        # Process reserved tasks
        for worker, tasks in reserved_tasks.items():
            for task in tasks:
                all_tasks.append({
                    "task_id": task["id"],
                    "status": "reserved",
                    "worker": worker,
                    "name": task["name"],
                    "args": task["args"],
                    "kwargs": task["kwargs"],
                })
        
        return all_tasks
        
    except Exception as e:
        logger.error(f"Error listing tasks: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": "Failed to list tasks",
                "details": {"error": str(e)} if settings.DEBUG else None,
            }
        )