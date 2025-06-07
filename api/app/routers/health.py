from fastapi import APIRouter, Depends, HTTPException, Request
import psutil
import time
import os
import logging
import socket
import platform
import json
from typing import Dict, Any, Optional
from datetime import datetime
import jax

router = APIRouter(
    prefix="/health",
    tags=["health"],
    responses={404: {"description": "Not found"}},
)

# Initialize logger
logger = logging.getLogger("mctx.health")

# Cache for GPU info to avoid repeated expensive checks
gpu_info_cache = {"timestamp": 0, "data": None, "ttl": 60}  # TTL in seconds


def get_system_info() -> Dict[str, Any]:
    """Get basic system information."""
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "total_memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "uptime_seconds": int(time.time() - psutil.boot_time()),
    }


def get_process_info() -> Dict[str, Any]:
    """Get information about the current process."""
    process = psutil.Process()
    return {
        "pid": process.pid,
        "cpu_percent": process.cpu_percent(),
        "memory_percent": process.memory_percent(),
        "memory_mb": round(process.memory_info().rss / (1024**2), 2),
        "threads": process.num_threads(),
        "open_files": len(process.open_files()),
        "connections": len(process.connections()),
        "running_time": int(time.time() - process.create_time()),
    }


def get_gpu_info() -> Optional[Dict[str, Any]]:
    """Get GPU information if available."""
    # Check if we should use cached data
    current_time = time.time()
    if current_time - gpu_info_cache["timestamp"] < gpu_info_cache["ttl"] and gpu_info_cache["data"]:
        return gpu_info_cache["data"]

    # Try to get GPU info
    try:
        # Check if we have GPU available via JAX
        devices = jax.devices()
        gpu_devices = [d for d in devices if d.platform == 'gpu']
        
        if not gpu_devices:
            gpu_info_cache["data"] = {"available": False, "message": "No GPU devices found via JAX"}
            gpu_info_cache["timestamp"] = current_time
            return gpu_info_cache["data"]
        
        # Try to get more detailed info via NVIDIA tools if available
        try:
            import pynvml
            pynvml.nvmlInit()
            
            gpu_info = []
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                
                gpu_info.append({
                    "index": i,
                    "name": name,
                    "memory_total_mb": memory.total // (1024**2),
                    "memory_used_mb": memory.used // (1024**2),
                    "memory_free_mb": memory.free // (1024**2),
                    "utilization_gpu": utilization.gpu,
                    "utilization_memory": utilization.memory,
                    "temperature_c": temperature,
                })
            
            pynvml.nvmlShutdown()
            
            result = {
                "available": True,
                "count": device_count,
                "devices": gpu_info,
            }
            
            # Update cache
            gpu_info_cache["data"] = result
            gpu_info_cache["timestamp"] = current_time
            
            return result
            
        except (ImportError, Exception) as e:
            # Fall back to basic JAX info
            result = {
                "available": True,
                "count": len(gpu_devices),
                "devices": [{"name": str(d), "index": i} for i, d in enumerate(gpu_devices)],
                "jax_platform": jax.default_backend(),
                "error_msg": f"Could not get detailed GPU info: {str(e)}"
            }
            
            # Update cache
            gpu_info_cache["data"] = result
            gpu_info_cache["timestamp"] = current_time
            
            return result
            
    except Exception as e:
        result = {"available": False, "error": str(e)}
        
        # Update cache
        gpu_info_cache["data"] = result
        gpu_info_cache["timestamp"] = current_time
        
        return result


def get_service_info() -> Dict[str, Any]:
    """Get MCTX service information."""
    return {
        "service_name": "mctx-nvidia",
        "version": os.environ.get("MCTX_VERSION", "unknown"),
        "jax_version": getattr(jax, "__version__", "unknown"),
        "t4_optimizations": os.environ.get("MCTX_ENABLE_T4_OPTIMIZATIONS", "0") == "1",
        "precision": os.environ.get("MCTX_PRECISION", "fp32"),
        "tensor_cores": os.environ.get("MCTX_TENSOR_CORES", "0") == "1",
        "model_path": os.environ.get("MCTX_MODEL_PATH", "unknown"),
        "model_type": os.environ.get("MCTX_MODEL_TYPE", "default"),
    }


def check_service_health() -> Dict[str, bool]:
    """Perform basic health checks for the service."""
    checks = {
        "disk_space": True,
        "memory": True,
        "cpu": True,
        "gpu": True,
    }
    
    # Check if disk space is critically low
    try:
        disk_usage = psutil.disk_usage("/")
        if disk_usage.percent > 90:
            checks["disk_space"] = False
    except Exception:
        checks["disk_space"] = False
    
    # Check if memory is critically low
    try:
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            checks["memory"] = False
    except Exception:
        checks["memory"] = False
    
    # Check if CPU is overloaded
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        if cpu_percent > 95:
            checks["cpu"] = False
    except Exception:
        checks["cpu"] = False
    
    # Check GPU if expected to be available
    if os.environ.get("JAX_PLATFORM_NAME", "") == "gpu":
        try:
            gpu_info = get_gpu_info()
            if not gpu_info or not gpu_info.get("available", False):
                checks["gpu"] = False
        except Exception:
            checks["gpu"] = False
    
    return checks


@router.get("/")
async def health_check(request: Request) -> Dict[str, Any]:
    """Basic health check endpoint."""
    start_time = time.time()
    
    # Get system information
    system_info = get_system_info()
    process_info = get_process_info()
    service_info = get_service_info()
    gpu_info = get_gpu_info()
    health_checks = check_service_health()
    
    # Determine overall health status
    is_healthy = all(health_checks.values())
    
    # Create response
    response = {
        "status": "healthy" if is_healthy else "unhealthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": service_info,
        "checks": health_checks,
        "system": system_info,
        "process": process_info,
    }
    
    # Only include GPU info if available
    if gpu_info:
        response["gpu"] = gpu_info
    
    # Add response time
    response["response_time_ms"] = round((time.time() - start_time) * 1000, 2)
    
    # Log if unhealthy
    if not is_healthy:
        logger.warning(f"Unhealthy status detected: {json.dumps(response)}")
    
    return response


@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """Readiness check endpoint (for Kubernetes)."""
    health_checks = check_service_health()
    is_ready = all(health_checks.values())
    
    if not is_ready:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}


@router.get("/live")
async def liveness_check() -> Dict[str, Any]:
    """Liveness check endpoint (for Kubernetes)."""
    # Simple check that the service is responding
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}