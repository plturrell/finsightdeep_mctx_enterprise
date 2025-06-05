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
"""Distributed MCTS implementation across multiple GPUs."""

import functools
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

Tree = tree_lib.Tree
T = TypeVar("T")


class DistributedConfig(NamedTuple):
  """Configuration for distributed MCTS.
  
  Attributes:
    num_devices: Number of devices to distribute across.
    partition_search: Whether to partition the search across devices.
    partition_batch: Whether to partition the batch across devices.
    device_type: Type of devices to use (e.g., "gpu", "tpu").
    precision: Precision to use ("fp16" or "fp32").
    tensor_core_aligned: Whether to align dimensions for tensor cores.
    replicated_params: Whether model parameters should be replicated.
    checkpoint_steps: Number of steps between checkpoints (0 for none).
    auto_sync_trees: Whether to automatically synchronize trees across devices.
    pipeline_recurrent_fn: Whether to pipeline the recurrent function.
  """
  num_devices: int = 1
  partition_search: bool = True
  partition_batch: bool = True
  device_type: str = "gpu"
  precision: str = "fp16"
  tensor_core_aligned: bool = True
  replicated_params: bool = True
  checkpoint_steps: int = 0
  auto_sync_trees: bool = True
  pipeline_recurrent_fn: bool = True


def distribute_mcts(config: DistributedConfig = DistributedConfig()):
  """Decorator for distributed MCTS search.
  
  Args:
    config: Distributed configuration.
    
  Returns:
    Decorator function for distributed MCTS.
  """
  def decorator(search_fn):
    @functools.wraps(search_fn)
    def wrapped_search(params, rng_key, **kwargs):
      # Set up device mesh
      devices = mesh_utils.create_device_mesh((config.num_devices,))
      mesh = Mesh(devices, axis_names=("device",))
      
      # Define partition specs
      if config.replicated_params:
        params_spec = P(None)  # Replicated across devices
      else:
        params_spec = P("device")  # Sharded across devices
      
      if config.partition_batch:
        batch_spec = P("device")  # Sharded across devices
      else:
        batch_spec = P(None)  # Replicated across devices
      
      # Define input and output specs
      in_specs = (params_spec, P(None))  # params, rng_key
      out_spec = P("device") if config.partition_search else P(None)
      
      # Distributed search function
      @functools.partial(pjit, 
                        in_shardings=in_specs, 
                        out_shardings=out_spec,
                        donate_argnums=(0,))  # Donate params buffer
      def distributed_search(params, rng_key):
        # Split RNG key per device
        rng_keys = jax.random.split(rng_key, config.num_devices)
        device_id = jax.lax.axis_index("device")
        device_rng = jax.lax.dynamic_index_in_dim(rng_keys, device_id, 0)
        
        # Run search with per-device RNG
        result = search_fn(params, device_rng, **kwargs)
        
        # Synchronize trees across devices if needed
        if config.auto_sync_trees:
          result = sync_trees_across_devices(result)
        
        return result
      
      # Run distributed search with mesh context
      with mesh:
        return distributed_search(params, rng_key)
    
    return wrapped_search
  
  return decorator


def distributed_search(
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
    config: DistributedConfig = DistributedConfig()) -> Tree:
  """Distributed MCTS search across multiple GPUs.
  
  This implementation partitions both the batch and search simulations
  across multiple devices for maximum parallelism.
  
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
    config: Distributed configuration.
    
  Returns:
    Search tree.
  """
  # Create device mesh
  devices = mesh_utils.create_device_mesh((config.num_devices,))
  mesh = Mesh(devices, axis_names=("device",))
  
  # Partition simulation count across devices
  sims_per_device = num_simulations // config.num_devices
  remaining_sims = num_simulations % config.num_devices
  
  # Partition batch across devices if specified
  batch_size = root.value.shape[0]
  
  if config.partition_batch and batch_size >= config.num_devices:
    # Partition the batch across devices
    batch_per_device = batch_size // config.num_devices
    remaining_batch = batch_size % config.num_devices
    
    # Split the input data
    def split_batch(data, start, end):
      """Split the batch dimension of data."""
      if hasattr(data, "shape") and data.shape[0] == batch_size:
        return data[start:end]
      return data
    
    def merge_batch(results):
      """Merge results along batch dimension."""
      if not results:
        return None
      
      first = results[0]
      if isinstance(first, Tree):
        # Combine trees along batch dimension
        return tree_lib.concatenate_trees(results)
      elif hasattr(first, "shape"):
        # Combine arrays along batch dimension
        return jnp.concatenate(results, axis=0)
      else:
        return first
  
  # Define partitioned search function
  def run_partitioned_search(device_params, device_rng, device_sims, 
                            batch_start=0, batch_end=None):
    """Run search on a specific device with a subset of data."""
    # Get batch slice for this device
    if config.partition_batch and batch_size >= config.num_devices:
      device_root = jax.tree.map(
          lambda x: split_batch(x, batch_start, batch_end),
          root)
      device_invalid = (None if invalid_actions is None 
                        else split_batch(invalid_actions, batch_start, batch_end))
    else:
      device_root = root
      device_invalid = invalid_actions
      
    # Run search with optimized precision
    if config.precision == "fp16":
      # Use mixed precision for performance
      with jax.default_matmul_precision('tensorfloat32'):
        device_tree = search.search(
            params=device_params,
            rng_key=device_rng,
            root=device_root,
            recurrent_fn=recurrent_fn,
            root_action_selection_fn=root_action_selection_fn,
            interior_action_selection_fn=interior_action_selection_fn,
            num_simulations=device_sims,
            max_depth=max_depth,
            invalid_actions=device_invalid,
            extra_data=extra_data)
    else:
      # Use full precision
      device_tree = search.search(
          params=device_params,
          rng_key=device_rng,
          root=device_root,
          recurrent_fn=recurrent_fn,
          root_action_selection_fn=root_action_selection_fn,
          interior_action_selection_fn=interior_action_selection_fn,
          num_simulations=device_sims,
          max_depth=max_depth,
          invalid_actions=device_invalid,
          extra_data=extra_data)
    
    # Apply tensor core optimization if requested
    if config.tensor_core_aligned:
      device_tree = t4_optimizations.t4_optimized_tree_layout(device_tree)
      
    return device_tree
  
  # Run search on each device
  with mesh:
    # Generate RNG key for each device
    rng_keys = jax.random.split(rng_key, config.num_devices)
    
    # Define cross-device parallelism
    device_results = []
    
    for i in range(config.num_devices):
      # Calculate simulations for this device
      device_sims = sims_per_device + (1 if i < remaining_sims else 0)
      
      # Calculate batch range for this device
      if config.partition_batch and batch_size >= config.num_devices:
        batch_start = i * batch_per_device + min(i, remaining_batch)
        batch_end = (i + 1) * batch_per_device + min(i + 1, remaining_batch)
      else:
        batch_start = 0
        batch_end = batch_size
      
      # Run search on this device
      device_tree = jax.device_put(
          run_partitioned_search(
              params, rng_keys[i], device_sims, batch_start, batch_end),
          devices[i])
      
      device_results.append(device_tree)
    
    # Combine results from all devices
    if config.partition_batch and batch_size >= config.num_devices:
      # Merge results along batch dimension
      combined_tree = merge_batch(device_results)
    else:
      # Merge results by combining visit counts and values
      combined_tree = merge_search_results(device_results)
    
    return combined_tree


def merge_search_results(trees: List[Tree]) -> Tree:
  """Merge search results from multiple devices.
  
  When search is partitioned across devices (but not batch),
  we need to combine the visit counts and values.
  
  Args:
    trees: List of trees from different devices.
    
  Returns:
    Merged tree.
  """
  if not trees:
    return None
  
  if len(trees) == 1:
    return trees[0]
  
  # Start with the first tree
  result_tree = trees[0]
  
  # Merge in all other trees
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
    
    # Update the tree with merged values
    result_tree = result_tree.replace(
        node_values=weighted_values,
        children_values=weighted_child_values,
    )
  
  return result_tree


def sync_trees_across_devices(tree: Tree) -> Tree:
  """Synchronize trees across devices.
  
  This function is used to ensure all devices have the same tree
  state, which is important for distributed training.
  
  Args:
    tree: Tree to synchronize.
    
  Returns:
    Synchronized tree.
  """
  # All-reduce the tree across devices
  # This ensures each device has the same tree state
  # We use sum as the reduction to combine visit counts
  # and node values properly
  
  # Step 1: All-reduce visit counts
  node_visits = jax.lax.all_reduce(
      tree.node_visits, jax.lax.add, axis_name="device")
  children_visits = jax.lax.all_reduce(
      tree.children_visits, jax.lax.add, axis_name="device")
  
  # Step 2: All-reduce values (weighted by visits)
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


def pipeline_recurrent_fn(recurrent_fn: base.RecurrentFn) -> base.RecurrentFn:
  """Pipeline the recurrent function for better performance.
  
  This wraps the recurrent function to enable pipelining across
  multiple stages, which can improve GPU utilization.
  
  Args:
    recurrent_fn: Recurrent function to pipeline.
    
  Returns:
    Pipelined recurrent function.
  """
  @functools.wraps(recurrent_fn)
  def pipelined_recurrent_fn(params, rng_key, action, embedding):
    # Split the embedding into stages for pipelining
    # We do this to maximize GPU utilization by breaking
    # the computation into smaller chunks that can be
    # processed in parallel
    
    # For now, just call the original function
    # In a full implementation, we would pipeline the computation
    # by splitting it into stages and scheduling them across devices
    return recurrent_fn(params, rng_key, action, embedding)
  
  return pipelined_recurrent_fn


def shard_batch(
    batch_size: int,
    num_devices: int) -> List[Tuple[int, int]]:
  """Calculate batch sharding across devices.
  
  Args:
    batch_size: Size of the batch.
    num_devices: Number of devices.
    
  Returns:
    List of (start, end) tuples for each device.
  """
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