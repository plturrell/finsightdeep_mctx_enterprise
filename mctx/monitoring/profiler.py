"""
MCTX Performance Profiling

Tools for profiling and optimizing Monte Carlo Tree Search performance.
Provides deep insights into computational efficiency, memory usage,
and optimization opportunities.
"""

import time
import timeit
import functools
import logging
import threading
import atexit
from typing import Dict, List, Any, Optional, Callable, Union, TypeVar, Tuple
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager

import numpy as np
import jax
import jax.numpy as jnp

# Configure logging
logger = logging.getLogger("mctx.monitoring.profiler")

# Type variables for generic functions
T = TypeVar('T')
R = TypeVar('R')


class ResourceType(str, Enum):
    """Types of resources that can be monitored."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    GPU_MEMORY = "gpu_memory"
    NETWORK = "network"
    DISK = "disk"


@dataclass
class ResourceSnapshot:
    """Snapshot of resource utilization."""
    timestamp: float = 0.0
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    gpu_percent: Optional[List[float]] = None
    gpu_memory_percent: Optional[List[float]] = None
    gpu_memory_used_mb: Optional[List[float]] = None
    gpu_memory_available_mb: Optional[List[float]] = None
    disk_read_mb: float = 0.0
    disk_write_mb: float = 0.0
    network_sent_mb: float = 0.0
    network_received_mb: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class FunctionProfile:
    """Profile of a function's performance."""
    name: str
    calls: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    last_call_time: float = 0.0
    memory_before: Optional[float] = None
    memory_after: Optional[float] = None
    memory_peak: Optional[float] = None
    gpu_memory_before: Optional[float] = None
    gpu_memory_after: Optional[float] = None
    gpu_memory_peak: Optional[float] = None
    
    def update(self, execution_time: float) -> None:
        """
        Update profile with a new execution time.
        
        Args:
            execution_time: Time in seconds for function execution
        """
        self.calls += 1
        self.total_time += execution_time
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.avg_time = self.total_time / self.calls
        self.last_call_time = execution_time


class ResourceMonitor:
    """
    Monitors system resources during MCTS execution.
    
    Tracks CPU, memory, GPU, disk, and network utilization
    to identify potential bottlenecks.
    """
    
    def __init__(self, 
                 collect_interval_ms: int = 1000,
                 track_cpu: bool = True,
                 track_memory: bool = True, 
                 track_gpu: bool = True,
                 track_disk: bool = False,
                 track_network: bool = False):
        """
        Initialize the resource monitor.
        
        Args:
            collect_interval_ms: Interval for collecting metrics (ms)
            track_cpu: Whether to track CPU usage
            track_memory: Whether to track memory usage
            track_gpu: Whether to track GPU usage
            track_disk: Whether to track disk I/O
            track_network: Whether to track network I/O
        """
        self.collect_interval_ms = collect_interval_ms
        self.track_cpu = track_cpu
        self.track_memory = track_memory
        self.track_gpu = track_gpu
        self.track_disk = track_disk
        self.track_network = track_network
        
        self.snapshots = []
        self.is_collecting = False
        self.collection_thread = None
        self.stop_collection = threading.Event()
        
        # Initialize baseline values
        self.baseline_snapshot = None
        
        # Import optional dependencies
        self._init_dependencies()
        
        # Register cleanup
        atexit.register(self.stop_collection)
        
        logger.info("Resource monitor initialized")
    
    def _init_dependencies(self) -> None:
        """Initialize optional dependencies for resource monitoring."""
        self.psutil = None
        self.pynvml = None
        
        # Try to import psutil for system monitoring
        try:
            import psutil
            self.psutil = psutil
            logger.info("psutil available for system monitoring")
        except ImportError:
            logger.warning("psutil not available, some metrics will be disabled")
        
        # Try to import pynvml for GPU monitoring
        if self.track_gpu:
            try:
                import pynvml
                self.pynvml = pynvml
                # Initialize NVML
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                logger.info(f"pynvml available for GPU monitoring, detected {self.gpu_count} GPUs")
            except (ImportError, Exception) as e:
                logger.warning(f"pynvml not available or error initializing: {e}")
                self.pynvml = None
                self.gpu_count = 0
    
    def start_collection(self) -> None:
        """Start collecting resource metrics."""
        if self.is_collecting:
            logger.warning("Resource collection already active")
            return
        
        self.snapshots = []
        self.stop_collection.clear()
        self.is_collecting = True
        
        # Take baseline snapshot
        self.baseline_snapshot = self.take_snapshot()
        self.snapshots.append(self.baseline_snapshot)
        
        # Start collection thread
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True
        )
        self.collection_thread.start()
        
        logger.info(f"Started resource monitoring with interval of {self.collect_interval_ms}ms")
    
    def stop_collection(self) -> List[ResourceSnapshot]:
        """
        Stop collecting resource metrics.
        
        Returns:
            List of resource snapshots
        """
        if not self.is_collecting:
            return self.snapshots
        
        # Signal thread to stop
        self.stop_collection.set()
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=1.0)
        
        self.is_collecting = False
        
        # Take final snapshot
        final_snapshot = self.take_snapshot()
        self.snapshots.append(final_snapshot)
        
        logger.info(f"Stopped resource monitoring, collected {len(self.snapshots)} snapshots")
        return self.snapshots
    
    def take_snapshot(self) -> ResourceSnapshot:
        """
        Take a snapshot of current resource utilization.
        
        Returns:
            Resource snapshot
        """
        snapshot = ResourceSnapshot(timestamp=time.time())
        
        # CPU and memory metrics (via psutil)
        if self.psutil:
            if self.track_cpu:
                try:
                    snapshot.cpu_percent = self.psutil.cpu_percent(interval=0.1)
                except Exception as e:
                    logger.error(f"Error collecting CPU metrics: {e}")
            
            if self.track_memory:
                try:
                    memory = self.psutil.virtual_memory()
                    snapshot.memory_percent = memory.percent
                    snapshot.memory_used_mb = memory.used / (1024 * 1024)
                    snapshot.memory_available_mb = memory.available / (1024 * 1024)
                except Exception as e:
                    logger.error(f"Error collecting memory metrics: {e}")
            
            if self.track_disk:
                try:
                    disk_io = self.psutil.disk_io_counters()
                    snapshot.disk_read_mb = disk_io.read_bytes / (1024 * 1024)
                    snapshot.disk_write_mb = disk_io.write_bytes / (1024 * 1024)
                except Exception as e:
                    logger.error(f"Error collecting disk metrics: {e}")
            
            if self.track_network:
                try:
                    net_io = self.psutil.net_io_counters()
                    snapshot.network_sent_mb = net_io.bytes_sent / (1024 * 1024)
                    snapshot.network_received_mb = net_io.bytes_recv / (1024 * 1024)
                except Exception as e:
                    logger.error(f"Error collecting network metrics: {e}")
        
        # GPU metrics (via pynvml)
        if self.pynvml and self.track_gpu and hasattr(self, 'gpu_count'):
            try:
                gpu_utilization = []
                gpu_memory_utilization = []
                gpu_memory_used = []
                gpu_memory_available = []
                
                for i in range(self.gpu_count):
                    handle = self.pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # GPU utilization
                    utilization = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_utilization.append(utilization.gpu)
                    
                    # GPU memory
                    memory_info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_memory_utilization.append(memory_info.used / memory_info.total * 100)
                    gpu_memory_used.append(memory_info.used / (1024 * 1024))
                    gpu_memory_available.append((memory_info.total - memory_info.used) / (1024 * 1024))
                
                snapshot.gpu_percent = gpu_utilization
                snapshot.gpu_memory_percent = gpu_memory_utilization
                snapshot.gpu_memory_used_mb = gpu_memory_used
                snapshot.gpu_memory_available_mb = gpu_memory_available
            except Exception as e:
                logger.error(f"Error collecting GPU metrics: {e}")
        
        return snapshot
    
    def _collection_loop(self) -> None:
        """Continuously collect resource metrics at specified intervals."""
        while not self.stop_collection.is_set():
            # Take snapshot
            snapshot = self.take_snapshot()
            self.snapshots.append(snapshot)
            
            # Sleep for the specified interval
            self.stop_collection.wait(self.collect_interval_ms / 1000)
    
    def get_resource_usage(self, resource_type: ResourceType) -> Tuple[List[float], List[float]]:
        """
        Get resource usage over time.
        
        Args:
            resource_type: Type of resource to get usage for
            
        Returns:
            Tuple of (timestamps, values)
        """
        if not self.snapshots:
            return [], []
        
        timestamps = [s.timestamp - self.snapshots[0].timestamp for s in self.snapshots]
        
        if resource_type == ResourceType.CPU:
            values = [s.cpu_percent for s in self.snapshots]
        elif resource_type == ResourceType.MEMORY:
            values = [s.memory_percent for s in self.snapshots]
        elif resource_type == ResourceType.GPU:
            if self.snapshots[0].gpu_percent is None:
                return timestamps, []
            # Average across GPUs
            values = [np.mean(s.gpu_percent) if s.gpu_percent else 0 for s in self.snapshots]
        elif resource_type == ResourceType.GPU_MEMORY:
            if self.snapshots[0].gpu_memory_percent is None:
                return timestamps, []
            # Average across GPUs
            values = [np.mean(s.gpu_memory_percent) if s.gpu_memory_percent else 0 for s in self.snapshots]
        elif resource_type == ResourceType.DISK:
            # Get change in disk I/O
            values = []
            prev_read = self.snapshots[0].disk_read_mb
            prev_write = self.snapshots[0].disk_write_mb
            for s in self.snapshots:
                read_delta = s.disk_read_mb - prev_read
                write_delta = s.disk_write_mb - prev_write
                values.append(read_delta + write_delta)
                prev_read = s.disk_read_mb
                prev_write = s.disk_write_mb
        elif resource_type == ResourceType.NETWORK:
            # Get change in network I/O
            values = []
            prev_sent = self.snapshots[0].network_sent_mb
            prev_recv = self.snapshots[0].network_received_mb
            for s in self.snapshots:
                sent_delta = s.network_sent_mb - prev_sent
                recv_delta = s.network_received_mb - prev_recv
                values.append(sent_delta + recv_delta)
                prev_sent = s.network_sent_mb
                prev_recv = s.network_received_mb
        else:
            values = []
        
        return timestamps, values
    
    def get_peak_usage(self, resource_type: ResourceType) -> float:
        """
        Get peak usage for a resource type.
        
        Args:
            resource_type: Type of resource
            
        Returns:
            Peak usage value
        """
        if not self.snapshots:
            return 0.0
        
        if resource_type == ResourceType.CPU:
            return max(s.cpu_percent for s in self.snapshots)
        elif resource_type == ResourceType.MEMORY:
            return max(s.memory_percent for s in self.snapshots)
        elif resource_type == ResourceType.GPU:
            if all(s.gpu_percent is None for s in self.snapshots):
                return 0.0
            return max(np.max(s.gpu_percent) if s.gpu_percent else 0 for s in self.snapshots)
        elif resource_type == ResourceType.GPU_MEMORY:
            if all(s.gpu_memory_percent is None for s in self.snapshots):
                return 0.0
            return max(np.max(s.gpu_memory_percent) if s.gpu_memory_percent else 0 for s in self.snapshots)
        else:
            return 0.0
    
    def get_memory_increase(self) -> float:
        """
        Calculate the increase in memory usage during monitoring.
        
        Returns:
            Memory increase in MB
        """
        if not self.snapshots or len(self.snapshots) < 2:
            return 0.0
        
        start_memory = self.snapshots[0].memory_used_mb
        end_memory = self.snapshots[-1].memory_used_mb
        
        return end_memory - start_memory
    
    def get_gpu_memory_increase(self) -> List[float]:
        """
        Calculate the increase in GPU memory usage during monitoring.
        
        Returns:
            List of GPU memory increases in MB
        """
        if not self.snapshots or len(self.snapshots) < 2:
            return []
        
        if (self.snapshots[0].gpu_memory_used_mb is None or 
            self.snapshots[-1].gpu_memory_used_mb is None):
            return []
        
        start_memory = self.snapshots[0].gpu_memory_used_mb
        end_memory = self.snapshots[-1].gpu_memory_used_mb
        
        return [end - start for start, end in zip(start_memory, end_memory)]
    
    def plot_resource_usage(self, resource_types: Optional[List[ResourceType]] = None) -> Any:
        """
        Plot resource usage over time.
        
        Args:
            resource_types: List of resource types to plot
            
        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            
            if not resource_types:
                resource_types = [
                    ResourceType.CPU,
                    ResourceType.MEMORY
                ]
                
                # Add GPU if available
                if any(s.gpu_percent is not None for s in self.snapshots):
                    resource_types.append(ResourceType.GPU)
                    resource_types.append(ResourceType.GPU_MEMORY)
            
            fig, axes = plt.subplots(len(resource_types), 1, figsize=(10, 3*len(resource_types)))
            if len(resource_types) == 1:
                axes = [axes]
            
            for i, resource_type in enumerate(resource_types):
                timestamps, values = self.get_resource_usage(resource_type)
                
                if not timestamps or not values:
                    continue
                
                axes[i].plot(timestamps, values)
                axes[i].set_title(f"{resource_type.value.replace('_', ' ').title()} Usage")
                axes[i].set_xlabel("Time (s)")
                
                if resource_type in [ResourceType.CPU, ResourceType.MEMORY, 
                                    ResourceType.GPU, ResourceType.GPU_MEMORY]:
                    axes[i].set_ylabel("Percent")
                    axes[i].set_ylim(0, 100)
                else:
                    axes[i].set_ylabel("MB")
            
            plt.tight_layout()
            return fig
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return None


class PerformanceProfiler:
    """
    Comprehensive performance profiling for MCTS operations.
    
    Tracks function execution times, memory usage, and computational
    efficiency to identify optimization opportunities.
    """
    
    def __init__(self, 
                 track_memory: bool = True,
                 track_gpu_memory: bool = True,
                 save_profiles: bool = False,
                 profile_dir: str = 'mcts_profiles'):
        """
        Initialize the performance profiler.
        
        Args:
            track_memory: Whether to track memory usage
            track_gpu_memory: Whether to track GPU memory usage
            save_profiles: Whether to save profiles to disk
            profile_dir: Directory to save profiles in
        """
        self.track_memory = track_memory
        self.track_gpu_memory = track_gpu_memory
        self.save_profiles = save_profiles
        self.profile_dir = profile_dir
        
        # Function profiles
        self.profiles = {}
        
        # Call stack for nested profiling
        self.call_stack = []
        
        # Import optional dependencies
        self._init_dependencies()
        
        logger.info("Performance profiler initialized")
    
    def _init_dependencies(self) -> None:
        """Initialize optional dependencies for profiling."""
        self.psutil = None
        self.pynvml = None
        
        # Try to import psutil for memory monitoring
        if self.track_memory:
            try:
                import psutil
                self.psutil = psutil
                self.process = psutil.Process()
                logger.info("psutil available for memory monitoring")
            except ImportError:
                logger.warning("psutil not available, memory profiling will be disabled")
                self.track_memory = False
        
        # Try to import pynvml for GPU monitoring
        if self.track_gpu_memory:
            try:
                import pynvml
                self.pynvml = pynvml
                # Initialize NVML
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                logger.info(f"pynvml available for GPU monitoring, detected {self.gpu_count} GPUs")
            except (ImportError, Exception) as e:
                logger.warning(f"pynvml not available or error initializing: {e}")
                self.pynvml = None
                self.gpu_count = 0
                self.track_gpu_memory = False
    
    def profile(self, func_or_name: Union[Callable[..., R], str]) -> Callable[[Callable[..., R]], Callable[..., R]]:
        """
        Decorator to profile a function.
        
        Args:
            func_or_name: Function to profile or name for the profile
            
        Returns:
            Decorated function
        """
        def decorator(func: Callable[..., R]) -> Callable[..., R]:
            profile_name = func.__name__ if callable(func_or_name) else func_or_name
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> R:
                return self._profile_function(profile_name, func, *args, **kwargs)
            
            return wrapper
        
        if callable(func_or_name):
            return decorator(func_or_name)
        
        return decorator
    
    def _profile_function(self, name: str, func: Callable[..., R], *args, **kwargs) -> R:
        """
        Profile a function execution.
        
        Args:
            name: Profile name
            func: Function to profile
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        # Create profile if it doesn't exist
        if name not in self.profiles:
            self.profiles[name] = FunctionProfile(name=name)
        
        # Get memory usage before execution
        if self.track_memory and self.psutil:
            try:
                memory_before = self.process.memory_info().rss / (1024 * 1024)
                self.profiles[name].memory_before = memory_before
            except Exception as e:
                logger.error(f"Error getting memory usage: {e}")
        
        # Get GPU memory usage before execution
        if self.track_gpu_memory and self.pynvml:
            try:
                gpu_memory_before = []
                for i in range(self.gpu_count):
                    handle = self.pynvml.nvmlDeviceGetHandleByIndex(i)
                    memory_info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_memory_before.append(memory_info.used / (1024 * 1024))
                
                self.profiles[name].gpu_memory_before = gpu_memory_before
            except Exception as e:
                logger.error(f"Error getting GPU memory usage: {e}")
        
        # Time execution
        start_time = time.time()
        
        # Add to call stack
        self.call_stack.append(name)
        
        # Execute function
        try:
            result = func(*args, **kwargs)
        finally:
            # Remove from call stack
            if self.call_stack and self.call_stack[-1] == name:
                self.call_stack.pop()
            
            # Calculate execution time
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Update profile
            self.profiles[name].update(execution_time)
            
            # Get memory usage after execution
            if self.track_memory and self.psutil:
                try:
                    memory_after = self.process.memory_info().rss / (1024 * 1024)
                    self.profiles[name].memory_after = memory_after
                    
                    # Update peak memory if needed
                    if (self.profiles[name].memory_peak is None or 
                        memory_after > self.profiles[name].memory_peak):
                        self.profiles[name].memory_peak = memory_after
                except Exception as e:
                    logger.error(f"Error getting memory usage: {e}")
            
            # Get GPU memory usage after execution
            if self.track_gpu_memory and self.pynvml:
                try:
                    gpu_memory_after = []
                    for i in range(self.gpu_count):
                        handle = self.pynvml.nvmlDeviceGetHandleByIndex(i)
                        memory_info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
                        gpu_memory_after.append(memory_info.used / (1024 * 1024))
                    
                    self.profiles[name].gpu_memory_after = gpu_memory_after
                    
                    # Update peak GPU memory if needed
                    if (self.profiles[name].gpu_memory_peak is None or 
                        max(gpu_memory_after) > max(self.profiles[name].gpu_memory_peak or [0])):
                        self.profiles[name].gpu_memory_peak = gpu_memory_after
                except Exception as e:
                    logger.error(f"Error getting GPU memory usage: {e}")
        
        return result
    
    @contextmanager
    def profile_section(self, name: str) -> None:
        """
        Context manager to profile a section of code.
        
        Args:
            name: Name for the profile
        """
        # Start profiling
        self._profile_start(name)
        
        try:
            yield
        finally:
            # End profiling
            self._profile_end(name)
    
    def _profile_start(self, name: str) -> None:
        """
        Start profiling a section.
        
        Args:
            name: Section name
        """
        # Create profile if it doesn't exist
        if name not in self.profiles:
            self.profiles[name] = FunctionProfile(name=name)
        
        # Get memory usage before execution
        if self.track_memory and self.psutil:
            try:
                memory_before = self.process.memory_info().rss / (1024 * 1024)
                self.profiles[name].memory_before = memory_before
            except Exception as e:
                logger.error(f"Error getting memory usage: {e}")
        
        # Get GPU memory usage before execution
        if self.track_gpu_memory and self.pynvml:
            try:
                gpu_memory_before = []
                for i in range(self.gpu_count):
                    handle = self.pynvml.nvmlDeviceGetHandleByIndex(i)
                    memory_info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_memory_before.append(memory_info.used / (1024 * 1024))
                
                self.profiles[name].gpu_memory_before = gpu_memory_before
            except Exception as e:
                logger.error(f"Error getting GPU memory usage: {e}")
        
        # Add to call stack
        self.call_stack.append(name)
        
        # Store start time
        setattr(self, f"_start_time_{name}", time.time())
    
    def _profile_end(self, name: str) -> None:
        """
        End profiling a section.
        
        Args:
            name: Section name
        """
        # Calculate execution time
        start_time = getattr(self, f"_start_time_{name}", time.time())
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Remove from call stack
        if self.call_stack and self.call_stack[-1] == name:
            self.call_stack.pop()
        
        # Update profile
        self.profiles[name].update(execution_time)
        
        # Get memory usage after execution
        if self.track_memory and self.psutil:
            try:
                memory_after = self.process.memory_info().rss / (1024 * 1024)
                self.profiles[name].memory_after = memory_after
                
                # Update peak memory if needed
                if (self.profiles[name].memory_peak is None or 
                    memory_after > self.profiles[name].memory_peak):
                    self.profiles[name].memory_peak = memory_after
            except Exception as e:
                logger.error(f"Error getting memory usage: {e}")
        
        # Get GPU memory usage after execution
        if self.track_gpu_memory and self.pynvml:
            try:
                gpu_memory_after = []
                for i in range(self.gpu_count):
                    handle = self.pynvml.nvmlDeviceGetHandleByIndex(i)
                    memory_info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_memory_after.append(memory_info.used / (1024 * 1024))
                
                self.profiles[name].gpu_memory_after = gpu_memory_after
                
                # Update peak GPU memory if needed
                if (self.profiles[name].gpu_memory_peak is None or 
                    max(gpu_memory_after) > max(self.profiles[name].gpu_memory_peak or [0])):
                    self.profiles[name].gpu_memory_peak = gpu_memory_after
            except Exception as e:
                logger.error(f"Error getting GPU memory usage: {e}")
        
        # Clean up start time attribute
        if hasattr(self, f"_start_time_{name}"):
            delattr(self, f"_start_time_{name}")
    
    def reset_profiles(self) -> None:
        """Reset all profiles."""
        self.profiles = {}
        self.call_stack = []
        logger.info("Reset all performance profiles")
    
    def get_profile(self, name: str) -> Optional[FunctionProfile]:
        """
        Get a profile by name.
        
        Args:
            name: Profile name
            
        Returns:
            Function profile or None if not found
        """
        return self.profiles.get(name)
    
    def get_profiles_sorted_by_time(self) -> List[FunctionProfile]:
        """
        Get profiles sorted by total execution time.
        
        Returns:
            List of profiles sorted by total time
        """
        return sorted(self.profiles.values(), key=lambda p: p.total_time, reverse=True)
    
    def get_profiles_sorted_by_calls(self) -> List[FunctionProfile]:
        """
        Get profiles sorted by number of calls.
        
        Returns:
            List of profiles sorted by calls
        """
        return sorted(self.profiles.values(), key=lambda p: p.calls, reverse=True)
    
    def get_profiles_sorted_by_avg_time(self) -> List[FunctionProfile]:
        """
        Get profiles sorted by average execution time.
        
        Returns:
            List of profiles sorted by average time
        """
        return sorted(self.profiles.values(), key=lambda p: p.avg_time, reverse=True)
    
    def profile_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all profiles.
        
        Returns:
            Profile summary
        """
        total_time = sum(p.total_time for p in self.profiles.values())
        total_calls = sum(p.calls for p in self.profiles.values())
        
        top_by_time = self.get_profiles_sorted_by_time()[:5]
        top_by_calls = self.get_profiles_sorted_by_calls()[:5]
        top_by_avg = self.get_profiles_sorted_by_avg_time()[:5]
        
        # Memory increase
        memory_increase = {}
        for name, profile in self.profiles.items():
            if profile.memory_before is not None and profile.memory_after is not None:
                memory_increase[name] = profile.memory_after - profile.memory_before
        
        return {
            "total_time": total_time,
            "total_calls": total_calls,
            "profile_count": len(self.profiles),
            "top_by_time": [{"name": p.name, "time": p.total_time, "calls": p.calls} for p in top_by_time],
            "top_by_calls": [{"name": p.name, "calls": p.calls, "time": p.total_time} for p in top_by_calls],
            "top_by_avg": [{"name": p.name, "avg_time": p.avg_time, "calls": p.calls} for p in top_by_avg],
            "memory_increase": memory_increase
        }
    
    def print_profile_summary(self) -> None:
        """Print a summary of all profiles."""
        summary = self.profile_summary()
        
        print("\n===== Performance Profile Summary =====")
        print(f"Total time: {summary['total_time']:.3f}s")
        print(f"Total calls: {summary['total_calls']}")
        print(f"Profile count: {summary['profile_count']}")
        
        print("\nTop functions by total time:")
        for i, profile in enumerate(summary['top_by_time']):
            print(f"{i+1}. {profile['name']}: {profile['time']:.3f}s ({profile['calls']} calls)")
        
        print("\nTop functions by calls:")
        for i, profile in enumerate(summary['top_by_calls']):
            print(f"{i+1}. {profile['name']}: {profile['calls']} calls ({profile['time']:.3f}s)")
        
        print("\nTop functions by average time:")
        for i, profile in enumerate(summary['top_by_avg']):
            print(f"{i+1}. {profile['name']}: {profile['avg_time']:.6f}s/call ({profile['calls']} calls)")
        
        if summary['memory_increase']:
            print("\nMemory increases:")
            for name, increase in sorted(summary['memory_increase'].items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"{name}: {increase:.2f} MB")
    
    def plot_profile_summary(self) -> Any:
        """
        Plot a summary of profiles.
        
        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            
            # Get profiles sorted by total time
            profiles = self.get_profiles_sorted_by_time()[:10]
            
            if not profiles:
                logger.warning("No profiles to plot")
                return None
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot total time
            names = [p.name for p in profiles]
            times = [p.total_time for p in profiles]
            calls = [p.calls for p in profiles]
            
            ax1.barh(names, times, alpha=0.7)
            ax1.set_title("Total Execution Time")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Function")
            
            # Plot calls
            ax2.barh(names, calls, alpha=0.7, color='orange')
            ax2.set_title("Number of Calls")
            ax2.set_xlabel("Calls")
            ax2.set_ylabel("Function")
            
            plt.tight_layout()
            return fig
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return None


def time_function(func: Callable[..., R], *args, **kwargs) -> Tuple[R, float]:
    """
    Time a function execution.
    
    Args:
        func: Function to time
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Tuple of (result, execution_time)
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    return result, end_time - start_time


def benchmark_function(func: Callable[..., R], *args, 
                     repeat: int = 5, number: int = 100, 
                     **kwargs) -> Dict[str, float]:
    """
    Benchmark a function execution.
    
    Args:
        func: Function to benchmark
        *args: Function arguments
        repeat: Number of times to repeat the timing
        number: Number of times to call the function in each timing
        **kwargs: Function keyword arguments
        
    Returns:
        Dictionary with benchmark results
    """
    # Create a wrapper function that calls func with args and kwargs
    def wrapper():
        func(*args, **kwargs)
    
    # Run timeit
    timer = timeit.Timer(wrapper)
    times = timer.repeat(repeat=repeat, number=number)
    
    # Calculate statistics
    min_time = min(times) / number
    avg_time = sum(times) / (repeat * number)
    max_time = max(times) / number
    
    return {
        "min_time": min_time,
        "avg_time": avg_time,
        "max_time": max_time,
        "total_runs": repeat * number
    }