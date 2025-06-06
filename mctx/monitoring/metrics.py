"""
MCTX Metrics Collection

Comprehensive metrics collection for Monte Carlo Tree Search processes.
Provides detailed insights into search performance, tree characteristics,
and resource utilization.
"""

import time
import functools
import threading
import logging
from typing import Dict, List, Any, Optional, Callable, Union, TypeVar, Generic
from dataclasses import dataclass, field

import numpy as np
import jax
import jax.numpy as jnp

from mctx._src.tree import Tree

# Type variables for generic functions
T = TypeVar('T')
R = TypeVar('R')

# Configure logging
logger = logging.getLogger("mctx.monitoring")


@dataclass
class SearchMetrics:
    """
    Comprehensive metrics collected during a Monte Carlo Tree Search.
    
    Provides detailed information about the search process, tree structure,
    resource utilization, and decision quality.
    """
    # Basic search statistics
    search_time_ms: float = 0.0
    iterations: int = 0
    simulation_calls: int = 0
    
    # Tree metrics
    max_depth: int = 0
    tree_size: int = 0
    leaf_nodes: int = 0
    branching_factor: float = 0.0
    
    # Visit and value statistics
    max_visits: int = 0
    avg_visits: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0
    avg_value: float = 0.0
    
    # Decision quality metrics
    exploration_rate: float = 0.0
    value_uncertainty: float = 0.0
    
    # Performance metrics
    iterations_per_second: float = 0.0
    simulations_per_second: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_memory_usage_mb: float = 0.0
    
    # Device utilization
    device_type: str = "cpu"
    device_count: int = 1
    device_utilization: Dict[str, float] = field(default_factory=dict)
    
    # T4-specific metrics
    tensor_core_utilization: Optional[float] = None
    t4_optimizations_enabled: bool = False
    t4_memory_efficiency: Optional[float] = None
    
    # Distributed metrics
    is_distributed: bool = False
    node_count: int = 1
    communication_overhead_ms: float = 0.0
    load_balance_score: Optional[float] = None
    
    # Time breakdown
    recurrent_fn_time_ms: float = 0.0
    tree_traversal_time_ms: float = 0.0
    node_selection_time_ms: float = 0.0
    backpropagation_time_ms: float = 0.0
    
    # Custom metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert metrics to a dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, dict):
                result[key] = value.copy()
            else:
                result[key] = value
        return result
    
    def update(self, other: 'SearchMetrics') -> None:
        """Update metrics with values from another SearchMetrics object."""
        for key, value in other.__dict__.items():
            if key != 'custom_metrics':
                setattr(self, key, value)
        
        # Merge custom metrics
        for key, value in other.custom_metrics.items():
            self.custom_metrics[key] = value


class MCTSMetricsCollector:
    """
    Collects comprehensive metrics during Monte Carlo Tree Search.
    
    Provides tools for monitoring, profiling, and analyzing search
    performance with minimal overhead.
    """
    
    def __init__(self, 
                 collect_interval_ms: int = 500,
                 resource_monitoring: bool = True,
                 device_monitoring: bool = True):
        """
        Initialize the metrics collector.
        
        Args:
            collect_interval_ms: Interval for continuous metric collection (ms)
            resource_monitoring: Whether to monitor system resources
            device_monitoring: Whether to monitor device utilization
        """
        self.collect_interval_ms = collect_interval_ms
        self.resource_monitoring = resource_monitoring
        self.device_monitoring = device_monitoring
        
        self.metrics = SearchMetrics()
        self.start_time = 0.0
        self.continuous_collection = False
        self.collection_thread = None
        self.stop_collection = threading.Event()
        
        # Performance tracking
        self.iteration_timestamps = []
        self.simulation_timestamps = []
        self.custom_timestamps = {}
        
        logger.info("Metrics collector initialized")
    
    def start_collection(self) -> None:
        """Start metrics collection for a search process."""
        self.metrics = SearchMetrics()
        self.start_time = time.time()
        self.iteration_timestamps = []
        self.simulation_timestamps = []
        self.custom_timestamps = {}
        self.stop_collection.clear()
        
        # Check for JAX devices
        devices = jax.devices()
        self.metrics.device_count = len(devices)
        self.metrics.device_type = str(devices[0].platform).lower()
        
        # Start continuous collection if configured
        if self.collect_interval_ms > 0:
            self.continuous_collection = True
            self.collection_thread = threading.Thread(
                target=self._continuous_collection,
                daemon=True
            )
            self.collection_thread.start()
        
        logger.info(f"Started metrics collection on {self.metrics.device_type} with {self.metrics.device_count} devices")
    
    def stop_collection(self) -> SearchMetrics:
        """
        Stop metrics collection and return the final metrics.
        
        Returns:
            Collected metrics
        """
        # Stop continuous collection
        if self.continuous_collection:
            self.stop_collection.set()
            if self.collection_thread and self.collection_thread.is_alive():
                self.collection_thread.join(timeout=1.0)
            self.continuous_collection = False
        
        # Calculate final metrics
        self._finalize_metrics()
        
        logger.info(f"Stopped metrics collection. Collected data for {self.metrics.iterations} iterations")
        return self.metrics
    
    def record_iteration(self) -> None:
        """Record a search iteration."""
        self.iteration_timestamps.append(time.time())
        self.metrics.iterations += 1
    
    def record_simulation(self) -> None:
        """Record a simulation call."""
        self.simulation_timestamps.append(time.time())
        self.metrics.simulation_calls += 1
    
    def record_custom_event(self, event_type: str) -> None:
        """
        Record a custom event timestamp.
        
        Args:
            event_type: Type of event to record
        """
        if event_type not in self.custom_timestamps:
            self.custom_timestamps[event_type] = []
        
        self.custom_timestamps[event_type].append(time.time())
    
    def update_tree_metrics(self, tree: Tree) -> None:
        """
        Update metrics based on the current search tree.
        
        Args:
            tree: The MCTS tree
        """
        # Extract tree data
        node_visits = np.asarray(tree.node_visits)
        node_values = np.asarray(tree.node_values)
        children_index = tree.children_index
        
        # Basic tree metrics
        self.metrics.tree_size = len(node_visits)
        
        # Visit statistics
        if len(node_visits) > 0:
            self.metrics.max_visits = int(np.max(node_visits))
            self.metrics.avg_visits = float(np.mean(node_visits))
        
        # Value statistics
        if len(node_values) > 0:
            self.metrics.min_value = float(np.min(node_values))
            self.metrics.max_value = float(np.max(node_values))
            self.metrics.avg_value = float(np.mean(node_values))
            self.metrics.value_uncertainty = float(np.std(node_values))
        
        # Calculate tree depth and branching factor
        max_depth = 0
        total_branches = 0
        internal_nodes = 0
        leaf_count = 0
        
        for i, children in enumerate(children_index):
            if len(children) > 0:
                # Count non-negative children (actual children)
                actual_children = [c for c in children if c >= 0]
                branch_count = len(actual_children)
                
                if branch_count > 0:
                    internal_nodes += 1
                    total_branches += branch_count
                else:
                    leaf_count += 1
                
                # Calculate depth for this node's children
                for child in actual_children:
                    # Calculate depth by walking up the tree
                    depth = 1
                    current = child
                    while current != 0:  # Not root
                        # Find parent of current
                        for parent_idx, parent_children in enumerate(children_index):
                            if current in parent_children:
                                current = parent_idx
                                depth += 1
                                break
                    max_depth = max(max_depth, depth)
            else:
                leaf_count += 1
        
        self.metrics.max_depth = max_depth
        self.metrics.leaf_nodes = leaf_count
        
        if internal_nodes > 0:
            self.metrics.branching_factor = total_branches / internal_nodes
        
        # Calculate exploration rate
        explored_nodes = np.sum(node_visits > 0)
        if self.metrics.tree_size > 0:
            self.metrics.exploration_rate = explored_nodes / self.metrics.tree_size
    
    def update_custom_metric(self, name: str, value: Any) -> None:
        """
        Update a custom metric.
        
        Args:
            name: Metric name
            value: Metric value
        """
        self.metrics.custom_metrics[name] = value
    
    def record_search_time(self, start_time: float, end_time: float) -> None:
        """
        Record search time metrics.
        
        Args:
            start_time: Search start time
            end_time: Search end time
        """
        duration_ms = (end_time - start_time) * 1000
        self.metrics.search_time_ms = duration_ms
        
        # Calculate rates
        if duration_ms > 0:
            seconds = duration_ms / 1000
            self.metrics.iterations_per_second = self.metrics.iterations / seconds
            self.metrics.simulations_per_second = self.metrics.simulation_calls / seconds
    
    def record_time_breakdown(self, 
                              recurrent_fn_time: float,
                              tree_traversal_time: float,
                              node_selection_time: float,
                              backpropagation_time: float) -> None:
        """
        Record time breakdown for search components.
        
        Args:
            recurrent_fn_time: Time spent in recurrent function (ms)
            tree_traversal_time: Time spent in tree traversal (ms)
            node_selection_time: Time spent in node selection (ms)
            backpropagation_time: Time spent in backpropagation (ms)
        """
        self.metrics.recurrent_fn_time_ms = recurrent_fn_time
        self.metrics.tree_traversal_time_ms = tree_traversal_time
        self.metrics.node_selection_time_ms = node_selection_time
        self.metrics.backpropagation_time_ms = backpropagation_time
    
    def _continuous_collection(self) -> None:
        """Continuously collect metrics at specified intervals."""
        while not self.stop_collection.is_set():
            # Update performance metrics
            self._update_performance_metrics()
            
            # Sleep for the specified interval
            self.stop_collection.wait(self.collect_interval_ms / 1000)
    
    def _update_performance_metrics(self) -> None:
        """Update performance-related metrics."""
        # Calculate durations
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        if elapsed_time > 0:
            # Calculate rates
            self.metrics.iterations_per_second = self.metrics.iterations / elapsed_time
            self.metrics.simulations_per_second = self.metrics.simulation_calls / elapsed_time
            
            # Update memory usage if monitoring resources
            if self.resource_monitoring:
                try:
                    import psutil
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    self.metrics.memory_usage_mb = memory_info.rss / (1024 * 1024)
                except (ImportError, Exception) as e:
                    logger.warning(f"Failed to collect memory metrics: {e}")
            
            # Update GPU memory usage if applicable
            if self.device_monitoring and self.metrics.device_type == "gpu":
                try:
                    # Try to get GPU memory info from different sources
                    gpu_memory = self._get_gpu_memory_usage()
                    if gpu_memory is not None:
                        self.metrics.gpu_memory_usage_mb = gpu_memory
                except Exception as e:
                    logger.warning(f"Failed to collect GPU metrics: {e}")
    
    def _get_gpu_memory_usage(self) -> Optional[float]:
        """
        Attempt to get GPU memory usage.
        
        Returns:
            GPU memory usage in MB, or None if unavailable
        """
        try:
            # Try NVIDIA-SMI through pynvml
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return info.used / (1024 * 1024)
        except ImportError:
            pass
        
        try:
            # Try subprocess to call nvidia-smi
            import subprocess
            result = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
                encoding='utf-8'
            )
            return float(result.strip())
        except (subprocess.SubprocessError, FileNotFoundError, ValueError):
            pass
        
        # If we can't get GPU memory usage, return None
        return None
    
    def _finalize_metrics(self) -> None:
        """Calculate final metrics when collection stops."""
        # Record total search time
        end_time = time.time()
        self.record_search_time(self.start_time, end_time)
        
        # Update performance metrics one last time
        self._update_performance_metrics()


def with_metrics(metrics_collector: Optional[MCTSMetricsCollector] = None) -> Callable:
    """
    Decorator to automatically collect metrics for a search function.
    
    Args:
        metrics_collector: Optional metrics collector to use
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> R:
            # Create metrics collector if not provided
            nonlocal metrics_collector
            if metrics_collector is None:
                metrics_collector = MCTSMetricsCollector()
            
            # Extract tree if it's being passed to the function
            tree = None
            for arg in args:
                if isinstance(arg, Tree):
                    tree = arg
                    break
            
            if tree is None:
                for _, value in kwargs.items():
                    if isinstance(value, Tree):
                        tree = value
                        break
            
            # Start collection
            metrics_collector.start_collection()
            
            # Call the function
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
            finally:
                end_time = time.time()
                
                # Update tree metrics if we have a tree
                if tree is not None:
                    metrics_collector.update_tree_metrics(tree)
                
                # Record search time
                metrics_collector.record_search_time(start_time, end_time)
                
                # Stop collection
                metrics = metrics_collector.stop_collection()
                
                # Attach metrics to the result if it's a dict
                if isinstance(result, dict):
                    result['metrics'] = metrics.as_dict()
            
            return result
        
        return wrapper
    
    return decorator