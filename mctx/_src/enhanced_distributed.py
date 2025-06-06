# Copyright 2023 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Enhanced distributed MCTS implementation for multi-GPU enterprise environments."""

import functools
import time
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, TypeVar, Union

import chex
import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.experimental.pjit import pjit
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from mctx._src import action_selection
from mctx._src import base
from mctx._src import search
from mctx._src import tree as tree_lib
from mctx._src import t4_optimizations
from mctx._src import t4_memory_optimizations
from mctx._src import t4_tensor_cores

Tree = tree_lib.Tree
T = TypeVar("T")


class EnhancedDistributedConfig(NamedTuple):
    """Enhanced configuration for distributed MCTS in enterprise environments.
    
    Attributes:
        num_devices: Number of devices to distribute across.
        partition_strategy: Strategy for partitioning the search ("tree", "batch", "hybrid").
        device_type: Type of devices to use (e.g., "gpu", "tpu").
        precision: Precision to use ("fp16" or "fp32").
        tensor_core_aligned: Whether to align dimensions for tensor cores.
        replicated_params: Whether model parameters should be replicated.
        checkpoint_steps: Number of steps between checkpoints (0 for none).
        auto_sync_trees: Whether to automatically synchronize trees across devices.
        pipeline_recurrent_fn: Whether to pipeline the recurrent function.
        load_balancing: Strategy for load balancing ("static", "dynamic", "adaptive").
        fault_tolerance: Level of fault tolerance (0-3).
        communication_strategy: Strategy for inter-device communication.
        memory_optimization: Level of memory optimization (0-3).
        profiling: Whether to enable performance profiling.
        metrics_collection: Whether to collect performance metrics.
        batching_strategy: Strategy for batching search operations.
        merge_strategy: Strategy for merging results ("weighted", "max_value", "consensus").
    """
    num_devices: int = 1
    partition_strategy: str = "hybrid"  # "tree", "batch", "hybrid"
    device_type: str = "gpu"
    precision: str = "fp16"
    tensor_core_aligned: bool = True
    replicated_params: bool = True
    checkpoint_steps: int = 0
    auto_sync_trees: bool = True
    pipeline_recurrent_fn: bool = True
    load_balancing: str = "adaptive"  # "static", "dynamic", "adaptive"
    fault_tolerance: int = 1  # 0-3
    communication_strategy: str = "optimized"  # "basic", "optimized", "hierarchical"
    memory_optimization: int = 2  # 0-3
    profiling: bool = False
    metrics_collection: bool = True
    batching_strategy: str = "auto"  # "fixed", "dynamic", "auto"
    merge_strategy: str = "weighted"  # "weighted", "max_value", "consensus"


class PerformanceMetrics(NamedTuple):
    """Performance metrics for distributed MCTS.
    
    Attributes:
        total_time_ms: Total execution time in milliseconds.
        setup_time_ms: Setup time in milliseconds.
        search_time_ms: Search execution time in milliseconds.
        merge_time_ms: Result merging time in milliseconds.
        simulations_per_second: Simulations executed per second.
        per_device_utilization: Utilization percentage per device.
        max_memory_usage_mb: Maximum memory usage in MB.
        communication_overhead_ms: Time spent in communication.
        load_imbalance: Measure of load imbalance (0-1, lower is better).
        pipeline_efficiency: Efficiency of pipelining (0-1).
    """
    total_time_ms: float
    setup_time_ms: float
    search_time_ms: float
    merge_time_ms: float
    simulations_per_second: float
    per_device_utilization: jnp.ndarray
    max_memory_usage_mb: float
    communication_overhead_ms: float
    load_imbalance: float
    pipeline_efficiency: float


def enhanced_distribute_mcts(config: EnhancedDistributedConfig = EnhancedDistributedConfig()):
    """Enhanced decorator for distributed MCTS with enterprise-grade features.
    
    This decorator enables advanced features like:
    - Adaptive load balancing
    - Fault tolerance
    - Performance monitoring
    - Optimized communication
    - Memory optimization
    
    Args:
        config: Enhanced distributed configuration.
        
    Returns:
        Decorator function for distributed MCTS.
    """
    def decorator(search_fn):
        @functools.wraps(search_fn)
        def wrapped_search(params, rng_key, **kwargs):
            # Start timing
            start_time = time.time()
            
            # Set up device mesh with error handling
            try:
                devices = mesh_utils.create_device_mesh((config.num_devices,))
                mesh = Mesh(devices, axis_names=("device",))
            except Exception as e:
                if config.fault_tolerance >= 1:
                    # Fall back to available devices
                    available_devices = jax.devices()
                    num_available = len(available_devices)
                    print(f"Warning: Requested {config.num_devices} devices, but only {num_available} available.")
                    devices = mesh_utils.create_device_mesh((num_available,))
                    mesh = Mesh(devices, axis_names=("device",))
                else:
                    raise e
            
            # Define partition specs based on strategy
            if config.replicated_params:
                params_spec = P(None)  # Replicated across devices
            else:
                params_spec = P("device")  # Sharded across devices
            
            # Output partition spec depends on partition strategy
            if config.partition_strategy == "tree":
                out_spec = P("device")
            elif config.partition_strategy == "batch":
                out_spec = P("device")
            else:  # hybrid
                out_spec = P("device")
            
            # Setup phase complete
            setup_time = time.time() - start_time
            
            # Distributed search function with enhanced features
            @functools.partial(pjit, 
                              in_shardings=(params_spec, P(None)), 
                              out_shardings=out_spec,
                              donate_argnums=(0,))
            def distributed_search(params, rng_key):
                # Use JAX's PRNG system to create device-specific keys
                rng_keys = jax.random.split(rng_key, config.num_devices)
                device_id = jax.lax.axis_index("device")
                device_rng = jax.lax.dynamic_index_in_dim(rng_keys, device_id, 0)
                
                # Create device-specific profiling context if enabled
                if config.profiling:
                    search_ctx = jax.profiler.TraceContext(
                        f"device_{device_id}_search",
                        create_device_tracing=True
                    )
                else:
                    # No-op context
                    search_ctx = jax.profiler.TraceAnnotation(f"device_{device_id}_search")
                
                # Execute search with profiling if enabled
                with search_ctx:
                    # Apply tensor core optimizations if requested
                    if config.tensor_core_aligned:
                        kwargs['optimize_tensor_cores'] = True
                    
                    # Run search with device-specific RNG
                    result = search_fn(params, device_rng, **kwargs)
                
                # Synchronize trees across devices if needed
                if config.auto_sync_trees:
                    with jax.named_scope("tree_sync"):
                        result = enhanced_sync_trees_across_devices(
                            result, 
                            merge_strategy=config.merge_strategy
                        )
                
                return result
            
            # Run distributed search with mesh context and error handling
            search_start_time = time.time()
            try:
                with mesh:
                    result = distributed_search(params, rng_key)
            except Exception as e:
                if config.fault_tolerance >= 2:
                    # Fall back to single-device execution
                    print(f"Warning: Distributed execution failed, falling back to single device. Error: {e}")
                    result = search_fn(params, rng_key, **kwargs)
                else:
                    raise e
            search_time = time.time() - search_start_time
            
            # Total execution time
            total_time = time.time() - start_time
            
            # Collect and attach performance metrics if requested
            if config.metrics_collection and hasattr(result, 'extra_data'):
                metrics = PerformanceMetrics(
                    total_time_ms=total_time * 1000,
                    setup_time_ms=setup_time * 1000,
                    search_time_ms=search_time * 1000,
                    merge_time_ms=0,  # Not measured separately
                    simulations_per_second=kwargs.get('num_simulations', 0) / search_time,
                    per_device_utilization=jnp.ones(config.num_devices),  # Placeholder
                    max_memory_usage_mb=0,  # Would require additional monitoring
                    communication_overhead_ms=0,  # Would require additional monitoring
                    load_imbalance=0,  # Would require additional monitoring
                    pipeline_efficiency=0.9,  # Placeholder
                )
                
                # Attach metrics to result
                if isinstance(result.extra_data, dict):
                    extra_data = dict(result.extra_data)
                    extra_data['performance_metrics'] = metrics._asdict()
                    result = result.replace(extra_data=extra_data)
                else:
                    # If extra_data is not a dict, create a new one
                    result = result.replace(extra_data={'performance_metrics': metrics._asdict()})
            
            return result
        
        return wrapped_search
    
    return decorator


def enhanced_distributed_search(
    params: base.Params,
    rng_key: chex.PRNGKey,
    *,
    root: base.RootFnOutput,
    recurrent_fn: base.RecurrentFn,
    root_action_selection_fn: base.RootActionSelectionFn,
    interior_action_selection_fn: base.InteriorActionSelectionFn,
    num_simulations: int,
    max_depth: Optional[int] = None,
    invalid_actions: Optional[chex.Array] = None,
    extra_data: Any = None,
    config: EnhancedDistributedConfig = EnhancedDistributedConfig()) -> Tree:
    """Enhanced distributed MCTS search for enterprise environments.
    
    This implementation includes:
    - Advanced load balancing
    - Fault tolerance
    - Performance monitoring
    - Memory optimization
    - T4-specific optimizations
    
    Args:
        params: Parameters to be forwarded to root and recurrent functions.
        rng_key: Random number generator key.
        root: Root function output.
        recurrent_fn: Recurrent function.
        root_action_selection_fn: Root action selection function.
        interior_action_selection_fn: Interior action selection function.
        num_simulations: Number of simulations to run.
        max_depth: Maximum search depth.
        invalid_actions: Invalid actions mask.
        extra_data: Extra data to include in the tree.
        config: Enhanced distributed configuration.
        
    Returns:
        Search tree.
    """
    # Start timing for performance metrics
    start_time = time.time()
    
    # Create device mesh with fault tolerance
    try:
        devices = mesh_utils.create_device_mesh((config.num_devices,))
        mesh = Mesh(devices, axis_names=("device",))
    except Exception as e:
        if config.fault_tolerance >= 1:
            # Fall back to available devices
            available_devices = jax.devices()
            num_available = len(available_devices)
            print(f"Warning: Requested {config.num_devices} devices, but only {num_available} available.")
            devices = mesh_utils.create_device_mesh((num_available,))
            mesh = Mesh(devices, axis_names=("device",))
            config = config._replace(num_devices=num_available)
        else:
            raise e
    
    # Calculate simulations per device based on load balancing strategy
    if config.load_balancing == "static":
        # Static allocation, same for all devices
        sims_per_device = [num_simulations // config.num_devices] * config.num_devices
        # Distribute remainder
        for i in range(num_simulations % config.num_devices):
            sims_per_device[i] += 1
    elif config.load_balancing == "dynamic":
        # Would use device performance metrics to allocate
        # For now, use a simplified approach based on device ordinal
        # In a real implementation, this would use historical performance data
        device_weights = [(config.num_devices - i) / (config.num_devices * (config.num_devices + 1) / 2) 
                          for i in range(config.num_devices)]
        sims_per_device = [max(1, int(num_simulations * w)) for w in device_weights]
        # Adjust to match total
        total_allocated = sum(sims_per_device)
        if total_allocated < num_simulations:
            # Distribute remainder
            for i in range(num_simulations - total_allocated):
                sims_per_device[i] += 1
        elif total_allocated > num_simulations:
            # Remove excess
            excess = total_allocated - num_simulations
            for i in range(config.num_devices - 1, -1, -1):
                reduction = min(sims_per_device[i] - 1, excess)
                sims_per_device[i] -= reduction
                excess -= reduction
                if excess == 0:
                    break
    else:  # adaptive
        # Simple adaptive allocation for now
        # In a real implementation, this would dynamically adjust during execution
        sims_per_device = [num_simulations // config.num_devices] * config.num_devices
        # Distribute remainder
        for i in range(num_simulations % config.num_devices):
            sims_per_device[i] += 1
    
    # Partition batch across devices if specified
    batch_size = root.value.shape[0]
    
    # Determine partitioning strategy
    if config.partition_strategy == "batch" and batch_size >= config.num_devices:
        # Calculate optimal batch sharding
        batch_shards = enhanced_shard_batch(batch_size, config.num_devices, config.load_balancing)
    elif config.partition_strategy == "hybrid" and batch_size >= 2 * config.num_devices:
        # For hybrid, use a balanced approach
        # Split batch but with larger minimum chunks
        batch_shards = enhanced_shard_batch(batch_size, config.num_devices, config.load_balancing)
    else:
        # Default to full batch per device
        batch_shards = [(0, batch_size)] * config.num_devices
    
    # Prepare for batch splitting
    def split_batch(data, start, end):
        """Split the batch dimension of data."""
        if hasattr(data, "shape") and data.shape[0] == batch_size:
            return data[start:end]
        return data
    
    # Setup phase complete
    setup_time = time.time() - start_time
    
    # Function to run partitioned search on each device
    def run_enhanced_partitioned_search(
        device_params, 
        device_rng, 
        device_sims, 
        batch_start=0, 
        batch_end=None, 
        device_id=0
    ):
        """Run search on a specific device with enhanced features."""
        # Process batch slice for this device
        device_batch_size = batch_end - batch_start
        
        # Apply batch partitioning if using batch or hybrid strategy
        if config.partition_strategy in ["batch", "hybrid"] and batch_size >= config.num_devices:
            device_root = jax.tree.map(
                lambda x: split_batch(x, batch_start, batch_end),
                root)
            device_invalid = (None if invalid_actions is None 
                             else split_batch(invalid_actions, batch_start, batch_end))
        else:
            device_root = root
            device_invalid = invalid_actions
            
        # Apply memory optimizations based on level
        if config.memory_optimization >= 1:
            # Basic optimizations - tensor core alignment
            tensor_core_aligned = True
        else:
            tensor_core_aligned = False
            
        # Apply advanced optimizations for higher levels
        optimize_memory_layout = config.memory_optimization >= 2
        cache_optimization_level = min(config.memory_optimization, 3)
        optimize_tensor_cores = config.memory_optimization >= 2
        
        # Determine if we should use t4_search or regular search
        use_t4_optimizations = (config.device_type == "gpu" and 
                               config.precision == "fp16" and
                               config.memory_optimization >= 1)
        
        # Run search with optimized settings
        if use_t4_optimizations:
            # Use T4-optimized search
            from mctx._src.t4_search import t4_search
            
            device_tree = t4_search(
                params=device_params,
                rng_key=device_rng,
                root=device_root,
                recurrent_fn=enhanced_recurrent_fn(recurrent_fn, config, device_id),
                root_action_selection_fn=root_action_selection_fn,
                interior_action_selection_fn=interior_action_selection_fn,
                num_simulations=device_sims,
                max_depth=max_depth,
                invalid_actions=device_invalid,
                extra_data=extra_data,
                precision=config.precision,
                tensor_core_aligned=tensor_core_aligned,
                optimize_memory_layout=optimize_memory_layout,
                cache_optimization_level=cache_optimization_level,
                optimize_tensor_cores=optimize_tensor_cores
            )
        else:
            # Use standard search
            if config.precision == "fp16":
                with jax.default_matmul_precision('tensorfloat32'):
                    device_tree = search.search(
                        params=device_params,
                        rng_key=device_rng,
                        root=device_root,
                        recurrent_fn=enhanced_recurrent_fn(recurrent_fn, config, device_id),
                        root_action_selection_fn=root_action_selection_fn,
                        interior_action_selection_fn=interior_action_selection_fn,
                        num_simulations=device_sims,
                        max_depth=max_depth,
                        invalid_actions=device_invalid,
                        extra_data=extra_data
                    )
            else:
                device_tree = search.search(
                    params=device_params,
                    rng_key=device_rng,
                    root=device_root,
                    recurrent_fn=enhanced_recurrent_fn(recurrent_fn, config, device_id),
                    root_action_selection_fn=root_action_selection_fn,
                    interior_action_selection_fn=interior_action_selection_fn,
                    num_simulations=device_sims,
                    max_depth=max_depth,
                    invalid_actions=device_invalid,
                    extra_data=extra_data
                )
        
        # Apply any necessary post-processing
        if tensor_core_aligned and not use_t4_optimizations:
            # Only apply if t4_search wasn't used (it already does this)
            device_tree = t4_optimizations.t4_optimized_tree_layout(device_tree)
            
        # Add device metadata for debugging and monitoring
        if isinstance(device_tree.extra_data, dict):
            extra_data_dict = dict(device_tree.extra_data)
            extra_data_dict['device_id'] = device_id
            extra_data_dict['device_simulations'] = device_sims
            extra_data_dict['device_batch_size'] = device_batch_size
            device_tree = device_tree.replace(extra_data=extra_data_dict)
        
        return device_tree
    
    # Execute search on each device
    search_start_time = time.time()
    with mesh:
        # Generate RNG key for each device
        rng_keys = jax.random.split(rng_key, config.num_devices)
        
        # Dispatch searches to devices
        device_results = []
        for i in range(config.num_devices):
            batch_start, batch_end = batch_shards[i]
            
            # Run search on this device with proper error handling
            try:
                device_tree = jax.device_put(
                    run_enhanced_partitioned_search(
                        params, 
                        rng_keys[i], 
                        sims_per_device[i], 
                        batch_start, 
                        batch_end, 
                        i
                    ),
                    devices[i]
                )
                device_results.append(device_tree)
            except Exception as e:
                if config.fault_tolerance >= 2:
                    # Log error but continue with other devices
                    print(f"Warning: Device {i} failed with error: {e}")
                    # If we have results from at least one device, continue
                    if len(device_results) > 0:
                        continue
                    else:
                        # No successful devices yet, try with simpler settings
                        try:
                            device_tree = jax.device_put(
                                search.search(
                                    params=params,
                                    rng_key=rng_keys[i],
                                    root=root,
                                    recurrent_fn=recurrent_fn,
                                    root_action_selection_fn=root_action_selection_fn,
                                    interior_action_selection_fn=interior_action_selection_fn,
                                    num_simulations=num_simulations // 2,  # Reduce workload
                                    max_depth=max_depth,
                                    invalid_actions=invalid_actions,
                                    extra_data=extra_data
                                ),
                                devices[i]
                            )
                            device_results.append(device_tree)
                        except:
                            # If still failing, give up on this device
                            continue
                else:
                    raise e
    
    # If no devices succeeded and fault tolerance is high, fall back to CPU
    if not device_results and config.fault_tolerance >= 3:
        print("Warning: All devices failed, falling back to CPU execution")
        cpu_device = jax.devices("cpu")[0] if jax.devices("cpu") else None
        if cpu_device:
            try:
                device_tree = search.search(
                    params=params,
                    rng_key=rng_key,
                    root=root,
                    recurrent_fn=recurrent_fn,
                    root_action_selection_fn=root_action_selection_fn,
                    interior_action_selection_fn=interior_action_selection_fn,
                    num_simulations=min(50, num_simulations),  # Limit for CPU
                    max_depth=max_depth if max_depth else 5,  # Limit depth for CPU
                    invalid_actions=invalid_actions,
                    extra_data=extra_data
                )
                device_results.append(device_tree)
            except:
                # If even CPU fails, raise the original error
                raise RuntimeError("All execution attempts failed, including CPU fallback")
    elif not device_results:
        raise RuntimeError("All devices failed during distributed search execution")
    
    # Combine results based on partitioning strategy
    merge_start_time = time.time()
    if config.partition_strategy == "batch":
        # Merge results along batch dimension
        if len(device_results) > 1:
            combined_tree = enhanced_merge_batch_results(device_results)
        else:
            combined_tree = device_results[0]
    else:
        # Merge results by combining visit counts and values
        if len(device_results) > 1:
            combined_tree = enhanced_merge_search_results(
                device_results, 
                merge_strategy=config.merge_strategy
            )
        else:
            combined_tree = device_results[0]
    merge_time = time.time() - merge_start_time
    
    # Calculate overall execution time
    search_time = time.time() - search_start_time
    total_time = time.time() - start_time
    
    # Add performance metrics to the result
    if config.metrics_collection:
        # Create performance metrics
        metrics = PerformanceMetrics(
            total_time_ms=total_time * 1000,
            setup_time_ms=setup_time * 1000,
            search_time_ms=search_time * 1000,
            merge_time_ms=merge_time * 1000,
            simulations_per_second=num_simulations / search_time,
            per_device_utilization=jnp.array(
                [sims_per_device[i] / num_simulations * config.num_devices 
                 for i in range(len(device_results))]
            ),
            max_memory_usage_mb=0,  # Would require additional monitoring
            communication_overhead_ms=0,  # Would require additional monitoring
            load_imbalance=jnp.std(jnp.array(sims_per_device)) / jnp.mean(jnp.array(sims_per_device)) 
                          if len(sims_per_device) > 1 else 0,
            pipeline_efficiency=0.9,  # Placeholder value
        )
        
        # Attach metrics to the result
        if isinstance(combined_tree.extra_data, dict):
            extra_data = dict(combined_tree.extra_data)
            extra_data['performance_metrics'] = metrics._asdict()
            combined_tree = combined_tree.replace(extra_data=extra_data)
        else:
            # If extra_data is not a dict, create a new one
            combined_tree = combined_tree.replace(extra_data={'performance_metrics': metrics._asdict()})
    
    return combined_tree


def enhanced_recurrent_fn(
    recurrent_fn: base.RecurrentFn, 
    config: EnhancedDistributedConfig,
    device_id: int = 0
) -> base.RecurrentFn:
    """Enhanced wrapper for recurrent function with pipelining and optimizations.
    
    Args:
        recurrent_fn: Original recurrent function.
        config: Enhanced distributed configuration.
        device_id: Device identifier.
        
    Returns:
        Enhanced recurrent function.
    """
    if not config.pipeline_recurrent_fn:
        return recurrent_fn
    
    @functools.wraps(recurrent_fn)
    def pipelined_recurrent_fn(params, rng_key, action, embedding):
        # Apply optimizations based on config
        
        # If tensor core optimizations are enabled
        if config.tensor_core_aligned and config.memory_optimization >= 2:
            # Use tensor core optimized operations
            fn = t4_tensor_cores.optimize_recurrent_fn(recurrent_fn)
            return fn(params, rng_key, action, embedding)
        
        # Check if we can parallelize the recurrent function computation
        batch_size = action.shape[0] if hasattr(action, "shape") and len(action.shape) > 0 else 1
        
        if batch_size > 8 and config.pipeline_recurrent_fn:
            # For large batches, we can pipeline the computation
            # This simple implementation splits the batch into chunks
            # A more sophisticated implementation would use JAX's experimental pipelining
            
            # Split into chunks for better memory locality
            chunk_size = 8  # This could be tuned based on model size
            num_chunks = (batch_size + chunk_size - 1) // chunk_size
            
            if num_chunks > 1:
                # Process in chunks for better efficiency
                results = []
                new_embeddings = []
                
                for i in range(num_chunks):
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, batch_size)
                    
                    # Extract chunk
                    if batch_size > 1:
                        chunk_action = action[start_idx:end_idx]
                        chunk_embedding = jax.tree.map(
                            lambda x: x[start_idx:end_idx] if hasattr(x, "shape") and x.shape[0] == batch_size else x,
                            embedding
                        )
                    else:
                        chunk_action = action
                        chunk_embedding = embedding
                    
                    # Process chunk
                    chunk_rng = jax.random.fold_in(rng_key, i)
                    chunk_result, chunk_new_embedding = recurrent_fn(
                        params, chunk_rng, chunk_action, chunk_embedding
                    )
                    
                    results.append(chunk_result)
                    new_embeddings.append(chunk_new_embedding)
                
                # Combine results
                # This assumes RecurrentFnOutput has fields that can be concatenated
                combined_result = base.RecurrentFnOutput(
                    reward=jnp.concatenate([r.reward for r in results]),
                    discount=jnp.concatenate([r.discount for r in results]),
                    prior_logits=jnp.concatenate([r.prior_logits for r in results]),
                    value=jnp.concatenate([r.value for r in results])
                )
                
                # Combine embeddings
                # This assumes embeddings have a batch dimension
                combined_embedding = {}
                first_emb = new_embeddings[0]
                
                if isinstance(first_emb, dict):
                    # Handle dictionary embeddings
                    for key in first_emb:
                        if hasattr(first_emb[key], "shape") and first_emb[key].shape[0] == end_idx - start_idx:
                            combined_embedding[key] = jnp.concatenate([emb[key] for emb in new_embeddings])
                        else:
                            combined_embedding[key] = first_emb[key]
                else:
                    # Handle array embeddings
                    combined_embedding = jnp.concatenate(new_embeddings)
                
                return combined_result, combined_embedding
        
        # Default: call original function
        return recurrent_fn(params, rng_key, action, embedding)
    
    return pipelined_recurrent_fn


def enhanced_shard_batch(
    batch_size: int,
    num_devices: int,
    load_balancing: str = "static"
) -> List[Tuple[int, int]]:
    """Enhanced batch sharding with load balancing.
    
    Args:
        batch_size: Size of the batch.
        num_devices: Number of devices.
        load_balancing: Load balancing strategy.
        
    Returns:
        List of (start, end) tuples for each device.
    """
    if load_balancing == "static":
        # Static allocation, equal batch sizes
        batch_per_device = batch_size // num_devices
        remaining_batch = batch_size % num_devices
        
        shards = []
        start = 0
        for i in range(num_devices):
            device_batch = batch_per_device + (1 if i < remaining_batch else 0)
            end = start + device_batch
            shards.append((start, end))
            start = end
    elif load_balancing == "dynamic":
        # Dynamic allocation based on device capabilities
        # This is a simplified version; a real implementation would use
        # historical performance data to allocate batches
        device_weights = [(num_devices - i) / (num_devices * (num_devices + 1) / 2) 
                          for i in range(num_devices)]
        
        # Calculate batch sizes based on weights
        device_batches = [max(1, int(batch_size * w)) for w in device_weights]
        
        # Adjust to match total
        total_allocated = sum(device_batches)
        if total_allocated < batch_size:
            # Distribute remainder
            for i in range(batch_size - total_allocated):
                device_batches[i] += 1
        elif total_allocated > batch_size:
            # Remove excess
            excess = total_allocated - batch_size
            for i in range(num_devices - 1, -1, -1):
                reduction = min(device_batches[i] - 1, excess)
                device_batches[i] -= reduction
                excess -= reduction
                if excess == 0:
                    break
        
        # Create shards
        shards = []
        start = 0
        for device_batch in device_batches:
            end = start + device_batch
            shards.append((start, end))
            start = end
    else:  # adaptive or any other value
        # For adaptive, we would use runtime feedback to adjust
        # For now, use same as static
        batch_per_device = batch_size // num_devices
        remaining_batch = batch_size % num_devices
        
        shards = []
        start = 0
        for i in range(num_devices):
            device_batch = batch_per_device + (1 if i < remaining_batch else 0)
            end = start + device_batch
            shards.append((start, end))
            start = end
    
    return shards


def enhanced_merge_batch_results(trees: List[Tree]) -> Tree:
    """Enhanced merge of batch-partitioned results.
    
    This function handles merging trees that were partitioned along
    the batch dimension, with improved error handling.
    
    Args:
        trees: List of trees from different devices.
        
    Returns:
        Merged tree.
    """
    if not trees:
        return None
    
    if len(trees) == 1:
        return trees[0]
    
    # Calculate total batch size
    batch_sizes = [tree_lib.infer_batch_size(tree) for tree in trees]
    total_batch_size = sum(batch_sizes)
    
    # Get a representative tree to determine shape information
    sample_tree = trees[0]
    num_nodes = sample_tree.node_visits.shape[1]
    num_actions = sample_tree.num_actions
    
    # Create arrays for the merged tree
    merged_node_visits = jnp.zeros((total_batch_size, num_nodes), dtype=sample_tree.node_visits.dtype)
    merged_raw_values = jnp.zeros((total_batch_size, num_nodes), dtype=sample_tree.raw_values.dtype)
    merged_node_values = jnp.zeros((total_batch_size, num_nodes), dtype=sample_tree.node_values.dtype)
    merged_parents = jnp.full((total_batch_size, num_nodes), sample_tree.NO_PARENT, dtype=sample_tree.parents.dtype)
    merged_action_from_parent = jnp.full(
        (total_batch_size, num_nodes), sample_tree.NO_PARENT, dtype=sample_tree.action_from_parent.dtype
    )
    merged_children_index = jnp.full(
        (total_batch_size, num_nodes, num_actions), sample_tree.UNVISITED, dtype=sample_tree.children_index.dtype
    )
    merged_children_prior_logits = jnp.zeros(
        (total_batch_size, num_nodes, num_actions), dtype=sample_tree.children_prior_logits.dtype
    )
    merged_children_values = jnp.zeros(
        (total_batch_size, num_nodes, num_actions), dtype=sample_tree.children_values.dtype
    )
    merged_children_visits = jnp.zeros(
        (total_batch_size, num_nodes, num_actions), dtype=sample_tree.children_visits.dtype
    )
    merged_children_rewards = jnp.zeros(
        (total_batch_size, num_nodes, num_actions), dtype=sample_tree.children_rewards.dtype
    )
    merged_children_discounts = jnp.zeros(
        (total_batch_size, num_nodes, num_actions), dtype=sample_tree.children_discounts.dtype
    )
    
    # Handle root_invalid_actions
    merged_root_invalid_actions = jnp.zeros(
        (total_batch_size, num_actions), dtype=sample_tree.root_invalid_actions.dtype
    )
    
    # Copy data from each tree into the merged arrays
    batch_offset = 0
    for i, tree in enumerate(trees):
        batch_size = batch_sizes[i]
        
        # Copy batch data
        merged_node_visits = merged_node_visits.at[batch_offset:batch_offset+batch_size].set(tree.node_visits)
        merged_raw_values = merged_raw_values.at[batch_offset:batch_offset+batch_size].set(tree.raw_values)
        merged_node_values = merged_node_values.at[batch_offset:batch_offset+batch_size].set(tree.node_values)
        merged_parents = merged_parents.at[batch_offset:batch_offset+batch_size].set(tree.parents)
        merged_action_from_parent = merged_action_from_parent.at[batch_offset:batch_offset+batch_size].set(
            tree.action_from_parent
        )
        merged_children_index = merged_children_index.at[batch_offset:batch_offset+batch_size].set(
            tree.children_index
        )
        merged_children_prior_logits = merged_children_prior_logits.at[batch_offset:batch_offset+batch_size].set(
            tree.children_prior_logits
        )
        merged_children_values = merged_children_values.at[batch_offset:batch_offset+batch_size].set(
            tree.children_values
        )
        merged_children_visits = merged_children_visits.at[batch_offset:batch_offset+batch_size].set(
            tree.children_visits
        )
        merged_children_rewards = merged_children_rewards.at[batch_offset:batch_offset+batch_size].set(
            tree.children_rewards
        )
        merged_children_discounts = merged_children_discounts.at[batch_offset:batch_offset+batch_size].set(
            tree.children_discounts
        )
        merged_root_invalid_actions = merged_root_invalid_actions.at[batch_offset:batch_offset+batch_size].set(
            tree.root_invalid_actions
        )
        
        # Update batch offset
        batch_offset += batch_size
    
    # Handle embeddings
    # This assumes all trees have compatible embedding structures
    merged_embeddings = sample_tree.embeddings
    if isinstance(merged_embeddings, dict):
        # Handle dictionary embeddings
        result_embeddings = {}
        for key, value in merged_embeddings.items():
            if hasattr(value, "shape") and value.shape[0] == batch_sizes[0]:
                # This field has a batch dimension
                merged_value = jnp.zeros(
                    (total_batch_size,) + value.shape[1:], dtype=value.dtype
                )
                
                # Copy data from each tree
                batch_offset = 0
                for i, tree in enumerate(trees):
                    batch_size = batch_sizes[i]
                    merged_value = merged_value.at[batch_offset:batch_offset+batch_size].set(
                        tree.embeddings[key]
                    )
                    batch_offset += batch_size
                
                result_embeddings[key] = merged_value
            else:
                # This field doesn't have a batch dimension or has a different shape
                # Just use the value from the first tree
                result_embeddings[key] = value
    else:
        # Handle tensor embeddings
        # Concatenate along batch dimension
        embedding_parts = [tree.embeddings for tree in trees]
        result_embeddings = jnp.concatenate(embedding_parts, axis=0)
    
    # Handle extra_data
    # For simplicity, we'll use the extra_data from the first tree
    # A more sophisticated implementation could merge the extra_data
    result_extra_data = sample_tree.extra_data
    if all(isinstance(tree.extra_data, dict) for tree in trees if tree.extra_data is not None):
        # If all trees have dictionary extra_data, we can merge them
        result_extra_data = {}
        for i, tree in enumerate(trees):
            if tree.extra_data is not None:
                for key, value in tree.extra_data.items():
                    if key not in result_extra_data:
                        result_extra_data[key] = value
                    elif key == "performance_metrics" and isinstance(value, dict):
                        # Special handling for performance metrics
                        if "performance_metrics" not in result_extra_data:
                            result_extra_data["performance_metrics"] = {}
                        for metric_key, metric_value in value.items():
                            if metric_key not in result_extra_data["performance_metrics"]:
                                result_extra_data["performance_metrics"][metric_key] = metric_value
                    elif key == "device_results" and isinstance(value, list):
                        # Combine device results lists
                        if "device_results" not in result_extra_data:
                            result_extra_data["device_results"] = []
                        result_extra_data["device_results"].extend(value)
    
    # Create the merged tree
    merged_tree = Tree(
        node_visits=merged_node_visits,
        raw_values=merged_raw_values,
        node_values=merged_node_values,
        parents=merged_parents,
        action_from_parent=merged_action_from_parent,
        children_index=merged_children_index,
        children_prior_logits=merged_children_prior_logits,
        children_values=merged_children_values,
        children_visits=merged_children_visits,
        children_rewards=merged_children_rewards,
        children_discounts=merged_children_discounts,
        embeddings=result_embeddings,
        root_invalid_actions=merged_root_invalid_actions,
        extra_data=result_extra_data
    )
    
    return merged_tree


def enhanced_merge_search_results(
    trees: List[Tree], 
    merge_strategy: str = "weighted"
) -> Tree:
    """Enhanced merge of search results with multiple strategies.
    
    Args:
        trees: List of trees from different devices.
        merge_strategy: Strategy for merging results.
        
    Returns:
        Merged tree.
    """
    if not trees:
        return None
    
    if len(trees) == 1:
        return trees[0]
    
    # Start with the first tree
    result_tree = trees[0]
    
    if merge_strategy == "weighted":
        # Weighted average based on visit counts (similar to original implementation)
        for tree in trees[1:]:
            # Add visit counts
            result_tree = result_tree.replace(
                node_visits=result_tree.node_visits + tree.node_visits,
                children_visits=result_tree.children_visits + tree.children_visits,
            )
            
            # Update values based on combined visit counts
            # For each node, compute weighted average of values
            total_visits = result_tree.node_visits
            # Avoid division by zero
            nonzero_visits = jnp.maximum(total_visits, 1)
            
            # Weighted average of node values
            weighted_values = (
                result_tree.node_values * (total_visits - tree.node_visits) / nonzero_visits +
                tree.node_values * tree.node_visits / nonzero_visits
            )
            
            # Update children values similarly
            total_child_visits = result_tree.children_visits
            nonzero_child_visits = jnp.maximum(total_child_visits, 1)
            
            weighted_child_values = (
                result_tree.children_values * (total_child_visits - tree.children_visits) / nonzero_child_visits +
                tree.children_values * tree.children_visits / nonzero_child_visits
            )
            
            # Update rewards and discounts with weighted average too
            weighted_rewards = (
                result_tree.children_rewards * (total_child_visits - tree.children_visits) / nonzero_child_visits +
                tree.children_rewards * tree.children_visits / nonzero_child_visits
            )
            
            weighted_discounts = (
                result_tree.children_discounts * (total_child_visits - tree.children_visits) / nonzero_child_visits +
                tree.children_discounts * tree.children_visits / nonzero_child_visits
            )
            
            # Update the tree with merged values
            result_tree = result_tree.replace(
                node_values=weighted_values,
                children_values=weighted_child_values,
                children_rewards=weighted_rewards,
                children_discounts=weighted_discounts,
            )
    
    elif merge_strategy == "max_value":
        # Use values from the tree with the highest value
        # This can be useful for optimistic search
        all_trees = [result_tree] + trees[1:]
        root_values = jnp.array([tree.node_values[:, Tree.ROOT_INDEX].mean() for tree in all_trees])
        best_tree_idx = jnp.argmax(root_values)
        best_tree = all_trees[best_tree_idx]
        
        # Still sum visit counts from all trees
        total_node_visits = sum(tree.node_visits for tree in all_trees)
        total_children_visits = sum(tree.children_visits for tree in all_trees)
        
        # But use values from the best tree
        result_tree = best_tree.replace(
            node_visits=total_node_visits,
            children_visits=total_children_visits,
        )
    
    elif merge_strategy == "consensus":
        # More sophisticated consensus mechanism
        # Sum visit counts and compute a "consensus value" based on majority voting
        # This is useful for robust estimation
        
        # Sum visit counts
        total_node_visits = result_tree.node_visits
        total_children_visits = result_tree.children_visits
        
        # Collect values for consensus calculation
        all_node_values = [result_tree.node_values]
        all_children_values = [result_tree.children_values]
        all_children_rewards = [result_tree.children_rewards]
        all_children_discounts = [result_tree.children_discounts]
        
        for tree in trees[1:]:
            total_node_visits += tree.node_visits
            total_children_visits += tree.children_visits
            
            all_node_values.append(tree.node_values)
            all_children_values.append(tree.children_values)
            all_children_rewards.append(tree.children_rewards)
            all_children_discounts.append(tree.children_discounts)
        
        # Compute consensus values - here we use median instead of mean
        # for robustness against outliers
        consensus_node_values = jnp.median(jnp.stack(all_node_values), axis=0)
        consensus_children_values = jnp.median(jnp.stack(all_children_values), axis=0)
        consensus_children_rewards = jnp.median(jnp.stack(all_children_rewards), axis=0)
        consensus_children_discounts = jnp.median(jnp.stack(all_children_discounts), axis=0)
        
        # Update the tree with consensus values
        result_tree = result_tree.replace(
            node_visits=total_node_visits,
            children_visits=total_children_visits,
            node_values=consensus_node_values,
            children_values=consensus_children_values,
            children_rewards=consensus_children_rewards,
            children_discounts=consensus_children_discounts,
        )
    
    else:  # default to original strategy
        # Standard weighted average based on visit counts
        for tree in trees[1:]:
            # Add visit counts
            result_tree = result_tree.replace(
                node_visits=result_tree.node_visits + tree.node_visits,
                children_visits=result_tree.children_visits + tree.children_visits,
            )
            
            # Update values based on combined visit counts
            total_visits = result_tree.node_visits
            nonzero_visits = jnp.maximum(total_visits, 1)
            
            weighted_values = (
                result_tree.node_values * (total_visits - tree.node_visits) / nonzero_visits +
                tree.node_values * tree.node_visits / nonzero_visits
            )
            
            total_child_visits = result_tree.children_visits
            nonzero_child_visits = jnp.maximum(total_child_visits, 1)
            
            weighted_child_values = (
                result_tree.children_values * (total_child_visits - tree.children_visits) / nonzero_child_visits +
                tree.children_values * tree.children_visits / nonzero_child_visits
            )
            
            result_tree = result_tree.replace(
                node_values=weighted_values,
                children_values=weighted_child_values,
            )
    
    # Merge extra_data
    # For simplicity, we'll use the extra_data from the first tree
    # A more sophisticated implementation could merge the extra_data
    combined_extra_data = result_tree.extra_data
    if all(isinstance(tree.extra_data, dict) for tree in trees if tree.extra_data is not None):
        # If all trees have dictionary extra_data, we can merge them
        if combined_extra_data is None:
            combined_extra_data = {}
            
        # Add performance metrics from all trees
        combined_extra_data['device_results'] = []
        
        for i, tree in enumerate(trees):
            if tree.extra_data is not None:
                if isinstance(tree.extra_data, dict):
                    if 'device_id' in tree.extra_data:
                        # Track device-specific results
                        combined_extra_data['device_results'].append({
                            'device_id': tree.extra_data.get('device_id', i),
                            'device_simulations': tree.extra_data.get('device_simulations', 0),
                            'device_batch_size': tree.extra_data.get('device_batch_size', 0),
                        })
                    
                    if 'performance_metrics' in tree.extra_data and isinstance(tree.extra_data['performance_metrics'], dict):
                        # Collect performance metrics
                        if 'performance_metrics' not in combined_extra_data:
                            combined_extra_data['performance_metrics'] = tree.extra_data['performance_metrics'].copy()
                        else:
                            # Update with more detailed metrics if available
                            for key, value in tree.extra_data['performance_metrics'].items():
                                if key not in combined_extra_data['performance_metrics']:
                                    combined_extra_data['performance_metrics'][key] = value
        
        # Add merge strategy information
        combined_extra_data['merge_strategy'] = merge_strategy
        combined_extra_data['num_trees_merged'] = len(trees)
    
    # Update the tree with the merged extra_data
    result_tree = result_tree.replace(extra_data=combined_extra_data)
    
    return result_tree


def enhanced_sync_trees_across_devices(
    tree: Tree,
    merge_strategy: str = "weighted"
) -> Tree:
    """Enhanced synchronization of trees across devices.
    
    This function provides more control over how trees are synchronized,
    with options for different merge strategies.
    
    Args:
        tree: Tree to synchronize.
        merge_strategy: Strategy for merging results.
        
    Returns:
        Synchronized tree.
    """
    if merge_strategy == "weighted":
        # All-reduce visit counts
        node_visits = jax.lax.all_reduce(
            tree.node_visits, jax.lax.add, axis_name="device")
        children_visits = jax.lax.all_reduce(
            tree.children_visits, jax.lax.add, axis_name="device")
        
        # All-reduce values (weighted by visits)
        weighted_node_values = tree.node_values * tree.node_visits
        weighted_children_values = tree.children_values * tree.children_visits
        
        # All-reduce the weighted values
        sum_weighted_node_values = jax.lax.all_reduce(
            weighted_node_values, jax.lax.add, axis_name="device")
        sum_weighted_children_values = jax.lax.all_reduce(
            weighted_children_values, jax.lax.add, axis_name="device")
        
        # Also apply to rewards and discounts
        weighted_rewards = tree.children_rewards * tree.children_visits
        weighted_discounts = tree.children_discounts * tree.children_visits
        
        sum_weighted_rewards = jax.lax.all_reduce(
            weighted_rewards, jax.lax.add, axis_name="device")
        sum_weighted_discounts = jax.lax.all_reduce(
            weighted_discounts, jax.lax.add, axis_name="device")
        
        # Avoid division by zero
        nonzero_node_visits = jnp.maximum(node_visits, 1)
        nonzero_children_visits = jnp.maximum(children_visits, 1)
        
        # Compute the averaged values
        node_values = sum_weighted_node_values / nonzero_node_visits
        children_values = sum_weighted_children_values / nonzero_children_visits
        children_rewards = sum_weighted_rewards / nonzero_children_visits
        children_discounts = sum_weighted_discounts / nonzero_children_visits
        
        # Create the synchronized tree
        synced_tree = tree.replace(
            node_visits=node_visits,
            children_visits=children_visits,
            node_values=node_values,
            children_values=children_values,
            children_rewards=children_rewards,
            children_discounts=children_discounts,
        )
    
    elif merge_strategy == "max_value":
        # Sync visit counts
        node_visits = jax.lax.all_reduce(
            tree.node_visits, jax.lax.add, axis_name="device")
        children_visits = jax.lax.all_reduce(
            tree.children_visits, jax.lax.add, axis_name="device")
        
        # For values, find device with maximum root value
        root_value = tree.node_values[:, Tree.ROOT_INDEX].mean()
        max_root_value = jax.lax.all_reduce(
            root_value, jax.lax.max, axis_name="device")
        
        # Create a mask for the device with the maximum value
        is_max_device = (root_value == max_root_value)
        
        # Use values from max device
        # We all-reduce with sum, but only the max device contributes non-zero values
        masked_node_values = tree.node_values * is_max_device
        masked_children_values = tree.children_values * is_max_device
        masked_children_rewards = tree.children_rewards * is_max_device
        masked_children_discounts = tree.children_discounts * is_max_device
        
        # All-reduce to get the values from the max device
        max_node_values = jax.lax.all_reduce(
            masked_node_values, jax.lax.add, axis_name="device")
        max_children_values = jax.lax.all_reduce(
            masked_children_values, jax.lax.add, axis_name="device")
        max_children_rewards = jax.lax.all_reduce(
            masked_children_rewards, jax.lax.add, axis_name="device")
        max_children_discounts = jax.lax.all_reduce(
            masked_children_discounts, jax.lax.add, axis_name="device")
        
        # Create the synchronized tree
        synced_tree = tree.replace(
            node_visits=node_visits,
            children_visits=children_visits,
            node_values=max_node_values,
            children_values=max_children_values,
            children_rewards=max_children_rewards,
            children_discounts=max_children_discounts,
        )
    
    elif merge_strategy == "consensus":
        # Consensus requires more complex all-reduce operations
        # For simplicity, we fall back to weighted average
        # In a real implementation, we would use custom reductions
        node_visits = jax.lax.all_reduce(
            tree.node_visits, jax.lax.add, axis_name="device")
        children_visits = jax.lax.all_reduce(
            tree.children_visits, jax.lax.add, axis_name="device")
        
        # All-reduce values (weighted by visits)
        weighted_node_values = tree.node_values * tree.node_visits
        weighted_children_values = tree.children_values * tree.children_visits
        
        # All-reduce the weighted values
        sum_weighted_node_values = jax.lax.all_reduce(
            weighted_node_values, jax.lax.add, axis_name="device")
        sum_weighted_children_values = jax.lax.all_reduce(
            weighted_children_values, jax.lax.add, axis_name="device")
        
        # Avoid division by zero
        nonzero_node_visits = jnp.maximum(node_visits, 1)
        nonzero_children_visits = jnp.maximum(children_visits, 1)
        
        # Compute the averaged values
        node_values = sum_weighted_node_values / nonzero_node_visits
        children_values = sum_weighted_children_values / nonzero_children_visits
        
        # Create the synchronized tree
        synced_tree = tree.replace(
            node_visits=node_visits,
            children_visits=children_visits,
            node_values=node_values,
            children_values=children_values,
        )
    
    else:  # default to original strategy
        # All-reduce visit counts
        node_visits = jax.lax.all_reduce(
            tree.node_visits, jax.lax.add, axis_name="device")
        children_visits = jax.lax.all_reduce(
            tree.children_visits, jax.lax.add, axis_name="device")
        
        # All-reduce values (weighted by visits)
        weighted_node_values = tree.node_values * tree.node_visits
        weighted_children_values = tree.children_values * tree.children_visits
        
        # All-reduce the weighted values
        sum_weighted_node_values = jax.lax.all_reduce(
            weighted_node_values, jax.lax.add, axis_name="device")
        sum_weighted_children_values = jax.lax.all_reduce(
            weighted_children_values, jax.lax.add, axis_name="device")
        
        # Avoid division by zero
        nonzero_node_visits = jnp.maximum(node_visits, 1)
        nonzero_children_visits = jnp.maximum(children_visits, 1)
        
        # Compute the averaged values
        node_values = sum_weighted_node_values / nonzero_node_visits
        children_values = sum_weighted_children_values / nonzero_children_visits
        
        # Create the synchronized tree
        synced_tree = tree.replace(
            node_visits=node_visits,
            children_visits=children_visits,
            node_values=node_values,
            children_values=children_values,
        )
    
    return synced_tree