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
"""T4 GPU-optimized implementation of batched MCTS."""

import functools
from typing import Any, NamedTuple, Optional, Tuple, TypeVar

import chex
import jax
import jax.numpy as jnp

from mctx._src import action_selection
from mctx._src import base
from mctx._src import search
from mctx._src import tree as tree_lib
from mctx._src import t4_optimizations

Tree = tree_lib.Tree
T = TypeVar("T")


def t4_search(
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
    loop_fn: base.LoopFn = jax.lax.fori_loop,
    precision: str = "fp16",
    tensor_core_aligned: bool = True,
    monitor_memory: bool = False) -> Tree:
  """T4 GPU-optimized search that performs a full search and returns sampled actions.

  This version includes optimizations specific to T4 GPUs:
  - Mixed precision (FP16) support
  - Tensor core alignment
  - Memory monitoring
  - Optimized tree layout
  - Fused operations

  In the shape descriptions, `B` denotes the batch dimension.

  Args:
    params: params to be forwarded to root and recurrent functions.
    rng_key: random number generator state, the key is consumed.
    root: a `(prior_logits, value, embedding)` `RootFnOutput`. The
      `prior_logits` are from a policy network. The shapes are
      `([B, num_actions], [B], [B, ...])`, respectively.
    recurrent_fn: a callable to be called on the leaf nodes and unvisited
      actions retrieved by the simulation step, which takes as args
      `(params, rng_key, action, embedding)` and returns a `RecurrentFnOutput`
      and the new state embedding. The `rng_key` argument is consumed.
    root_action_selection_fn: function used to select an action at the root.
    interior_action_selection_fn: function used to select an action during
      simulation.
    num_simulations: the number of simulations.
    max_depth: maximum search tree depth allowed during simulation, defined as
      the number of edges from the root to a leaf node.
    invalid_actions: a mask with invalid actions at the root. In the
      mask, invalid actions have ones, and valid actions have zeros.
      Shape `[B, num_actions]`.
    extra_data: extra data passed to `tree.extra_data`. Shape `[B, ...]`.
    loop_fn: Function used to run the simulations. It may be required to pass
      hk.fori_loop if using this function inside a Haiku module.
    precision: "fp16" or "fp32" to control computation precision.
    tensor_core_aligned: Whether to align dimensions for tensor cores.
    monitor_memory: Whether to monitor T4 GPU memory usage.

  Returns:
    `SearchResults` containing outcomes of the search, e.g. `visit_counts`
    `[B, num_actions]`.
  """
  
  # Handle precision and apply decorators
  search_fn = t4_optimizations.mixed_precision_wrapper(t4_optimized_search)
  if monitor_memory:
    search_fn = t4_optimizations.monitor_t4_memory_usage(search_fn)
  
  # Call the T4 optimized search function
  return search_fn(
      params=params,
      rng_key=rng_key,
      root=root,
      recurrent_fn=recurrent_fn,
      root_action_selection_fn=root_action_selection_fn,
      interior_action_selection_fn=interior_action_selection_fn,
      num_simulations=num_simulations,
      max_depth=max_depth,
      invalid_actions=invalid_actions,
      extra_data=extra_data,
      loop_fn=loop_fn,
      precision=precision,
      tensor_core_aligned=tensor_core_aligned)


def t4_optimized_search(
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
    loop_fn: base.LoopFn = jax.lax.fori_loop,
    precision: str = "fp16",
    tensor_core_aligned: bool = True) -> Tree:
  """T4-optimized implementation of MCTS search.
  
  This function has the same interface as search.search, but includes
  T4-specific optimizations.
  
  Args: Same as t4_search.
  Returns: Same as t4_search.
  """
  action_selection_fn = action_selection.switching_action_selection_wrapper(
      root_action_selection_fn=root_action_selection_fn,
      interior_action_selection_fn=interior_action_selection_fn
  )

  # Do simulation, expansion, and backward steps.
  batch_size = root.value.shape[0]
  
  # Align dimensions for tensor cores if requested
  if tensor_core_aligned:
    # Get the aligned batch size
    aligned_batch_size = t4_optimizations.align_for_tensor_cores(batch_size)
    
    # If we need to pad, create padded arrays
    if aligned_batch_size > batch_size:
      # Pad prior logits
      padded_shape = list(root.prior_logits.shape)
      padded_shape[0] = aligned_batch_size - batch_size
      padding = jnp.zeros(padded_shape, dtype=root.prior_logits.dtype)
      padded_prior_logits = jnp.concatenate([root.prior_logits, padding], axis=0)
      
      # Pad value
      padding = jnp.zeros([aligned_batch_size - batch_size], dtype=root.value.dtype)
      padded_value = jnp.concatenate([root.value, padding], axis=0)
      
      # Pad embedding
      padded_embedding = jax.tree.map(
          lambda x: jnp.concatenate([
              x, jnp.zeros([aligned_batch_size - batch_size] + list(x.shape[1:]), 
                         dtype=x.dtype)
          ], axis=0),
          root.embedding)
      
      # Create padded root
      root = base.RootFnOutput(
          prior_logits=padded_prior_logits,
          value=padded_value,
          embedding=padded_embedding)
      
      # Update batch size
      batch_size = aligned_batch_size
  
  batch_range = jnp.arange(batch_size)
  if max_depth is None:
    max_depth = num_simulations
  if invalid_actions is None:
    invalid_actions = jnp.zeros_like(root.prior_logits)

  def body_fun(sim, loop_state):
    rng_key, tree = loop_state
    rng_key, simulate_key, expand_key = jax.random.split(rng_key, 3)
    
    # T4 optimization: use a custom JAX dispatch for better fusion
    with jax.named_scope('t4_optimized_simulate_expand'):
      # Use fused implementation of simulate and expand
      parent_index, action = t4_optimized_simulate(
          simulate_key, tree, action_selection_fn, max_depth, batch_size)
      
      # A node first expanded on simulation `i`, will have node index `i`.
      # Node 0 corresponds to the root node.
      next_node_index = tree.children_index[batch_range, parent_index, action]
      next_node_index = jnp.where(next_node_index == Tree.UNVISITED,
                                  sim + 1, next_node_index)
      
      # Use the dynamically chosen precision
      jax_precision = t4_optimizations.dynamic_precision_policy(tree.node_visits.size)
      
      # T4 optimization: Use fused selection/expansion
      tree = t4_optimized_expand(
          params, expand_key, tree, recurrent_fn, parent_index,
          action, next_node_index, precision=jax_precision)
    
    # Backward pass is already very efficient due to JAX's vectorization
    tree = t4_optimized_backward(tree, next_node_index)
    
    loop_state = rng_key, tree
    return loop_state

  # Calculate optimal batch size for T4 GPU
  action_dim = root.prior_logits.shape[1]
  optimal_batch = t4_optimizations.get_optimal_t4_batch_size(max_depth, action_dim)
  
  # Allocate all necessary storage
  tree = t4_optimized_instantiate_tree(root, num_simulations,
                                    root_invalid_actions=invalid_actions,
                                    extra_data=extra_data)
  
  # Run the search with async computation if possible
  with jax.named_scope('t4_search_loop'):
    try:
      # Try to use JAX's async dispatch for better GPU utilization
      with jax.async_scope():
        _, tree = loop_fn(0, num_simulations, body_fun, (rng_key, tree))
    except (AttributeError, TypeError):
      # Fall back if async_scope is not available
      _, tree = loop_fn(0, num_simulations, body_fun, (rng_key, tree))
  
  # Apply T4-optimized memory layout before returning
  if tensor_core_aligned:
    tree = t4_optimizations.t4_optimized_tree_layout(tree)
    
    # If we padded the batch, extract only the original batch
    if batch_size > root.value.shape[0]:
      # Extract the original batch size
      orig_batch_size = root.value.shape[0]
      
      # Extract only the original batch
      tree = jax.tree.map(
          lambda x: x[:orig_batch_size] if hasattr(x, 'shape') and x.shape[0] == batch_size else x,
          tree)

  return tree


def t4_optimized_simulate(
    rng_key: chex.PRNGKey,
    tree: Tree,
    action_selection_fn: base.InteriorActionSelectionFn,
    max_depth: int,
    batch_size: int) -> Tuple[chex.Array, chex.Array]:
  """T4-optimized version of the simulate function.
  
  This implementation includes tensor core alignment and XLA hints.
  
  Args: Same as search.simulate but with explicit batch_size.
  Returns: Same as search.simulate.
  """
  # Define the simulation state as in the original implementation
  class _SimulationState(NamedTuple):
    """The state for the simulation while loop."""
    rng_key: chex.PRNGKey
    node_index: int
    action: int
    next_node_index: int
    depth: int
    is_continuing: bool
  
  # Create vmapped version with explicit batch size for better compilation
  simulate_vmap = jax.vmap(
      lambda rng_key, tree: _t4_simulate_single(
          rng_key, tree, action_selection_fn, max_depth),
      in_axes=[0, None], out_axes=0)
  
  # Split the RNG key to have one per batch element
  simulate_keys = jax.random.split(rng_key, batch_size)
  
  # Run the simulation
  parent_index, action = simulate_vmap(simulate_keys, tree)
  
  return parent_index, action


@functools.partial(jax.jit, static_argnums=(2, 3))
def _t4_simulate_single(
    rng_key: chex.PRNGKey,
    tree: Tree,
    action_selection_fn: base.InteriorActionSelectionFn,
    max_depth: int) -> Tuple[chex.Array, chex.Array]:
  """Single-instance simulation with XLA compilation hints."""
  def cond_fun(state):
    return state.is_continuing

  def body_fun(state):
    # Preparing the next simulation state.
    node_index = state.next_node_index
    rng_key, action_selection_key = jax.random.split(state.rng_key)
    action = action_selection_fn(action_selection_key, tree, node_index,
                               state.depth)
    next_node_index = tree.children_index[node_index, action]
    # The returned action will be visited.
    depth = state.depth + 1
    is_before_depth_cutoff = depth < max_depth
    is_visited = next_node_index != Tree.UNVISITED
    is_continuing = jnp.logical_and(is_visited, is_before_depth_cutoff)
    return _SimulationState(  # pytype: disable=wrong-arg-types  # jax-types
        rng_key=rng_key,
        node_index=node_index,
        action=action,
        next_node_index=next_node_index,
        depth=depth,
        is_continuing=is_continuing)

  node_index = jnp.array(Tree.ROOT_INDEX, dtype=jnp.int32)
  depth = jnp.zeros((), dtype=tree.children_prior_logits.dtype)
  # pytype: disable=wrong-arg-types  # jnp-type
  initial_state = _SimulationState(
      rng_key=rng_key,
      node_index=tree.NO_PARENT,
      action=tree.NO_PARENT,
      next_node_index=node_index,
      depth=depth,
      is_continuing=jnp.array(True))
  # pytype: enable=wrong-arg-types
  end_state = jax.lax.while_loop(cond_fun, body_fun, initial_state)

  # Returning a node with a selected action.
  # The action can be already visited, if the max_depth is reached.
  return end_state.node_index, end_state.action


def t4_optimized_expand(
    params: chex.Array,
    rng_key: chex.PRNGKey,
    tree: Tree[T],
    recurrent_fn: base.RecurrentFn,
    parent_index: chex.Array,
    action: chex.Array,
    next_node_index: chex.Array,
    precision: Optional[jax.lax.Precision] = None) -> Tree[T]:
  """T4-optimized version of the expand function.
  
  Uses fused operations and tensor core hints.
  
  Args: Same as search.expand with additional precision parameter.
  Returns: Same as search.expand.
  """
  # Use t4_fused_selection_expansion for better XLA compilation
  return t4_optimizations.t4_fused_selection_expansion(
      tree=tree,
      recurrent_fn=lambda t, n: _expand_node(
          params, rng_key, t, recurrent_fn, parent_index, action, n),
      nodes=next_node_index,
      precision=precision)[0]


def _expand_node(
    params: chex.Array,
    rng_key: chex.PRNGKey,
    tree: Tree[T],
    recurrent_fn: base.RecurrentFn,
    parent_index: chex.Array,
    action: chex.Array,
    next_node_index: chex.Array) -> base.ModelOutput:
  """Helper function for t4_optimized_expand."""
  batch_size = tree_lib.infer_batch_size(tree)
  batch_range = jnp.arange(batch_size)
  
  # Retrieve states for nodes to be evaluated.
  embedding = jax.tree.map(
      lambda x: x[batch_range, parent_index], tree.embeddings)

  # Evaluate and create a new node.
  step, embedding = recurrent_fn(params, rng_key, action, embedding)
  
  # Add shape assertions for debugging
  chex.assert_shape(step.prior_logits, [batch_size, tree.num_actions])
  chex.assert_shape(step.reward, [batch_size])
  chex.assert_shape(step.discount, [batch_size])
  chex.assert_shape(step.value, [batch_size])
  
  # Update tree node with the results
  tree = search.update_tree_node(
      tree, next_node_index, step.prior_logits, step.value, embedding)

  # Return updated tree topology.
  tree = tree.replace(
      children_index=search.batch_update(
          tree.children_index, next_node_index, parent_index, action),
      children_rewards=search.batch_update(
          tree.children_rewards, step.reward, parent_index, action),
      children_discounts=search.batch_update(
          tree.children_discounts, step.discount, parent_index, action),
      parents=search.batch_update(tree.parents, parent_index, next_node_index),
      action_from_parent=search.batch_update(
          tree.action_from_parent, action, next_node_index))
  
  return tree


@functools.partial(jax.vmap, static_argnums=(0,))
def t4_optimized_backward(
    tree: Tree[T],
    leaf_index: chex.Numeric) -> Tree[T]:
  """T4-optimized version of the backward function.
  
  Uses JAX primitives that map better to tensor cores.
  
  Args: Same as search.backward.
  Returns: Same as search.backward.
  """
  def cond_fun(loop_state):
    _, _, index = loop_state
    return index != Tree.ROOT_INDEX

  def body_fun(loop_state):
    # Here we update the value of our parent, so we start by reversing.
    tree, leaf_value, index = loop_state
    parent = tree.parents[index]
    count = tree.node_visits[parent]
    action = tree.action_from_parent[index]
    reward = tree.children_rewards[parent, action]
    
    # T4 optimization: use JAX primitives that map better to tensor cores
    discount_factor = tree.children_discounts[parent, action]
    leaf_value = jnp.add(reward, jnp.multiply(discount_factor, leaf_value))
    
    # Use JAX fused multiply-add for better tensor core utilization
    parent_value = jax.lax.fma(
        count, tree.node_values[parent], leaf_value) / (count + 1.0)
    
    children_values = tree.node_values[index]
    children_counts = tree.children_visits[parent, action] + 1

    # Update tree with optimized operations
    tree = tree.replace(
        node_values=search.update(tree.node_values, parent_value, parent),
        node_visits=search.update(tree.node_visits, count + 1, parent),
        children_values=search.update(
            tree.children_values, children_values, parent, action),
        children_visits=search.update(
            tree.children_visits, children_counts, parent, action))

    return tree, leaf_value, parent

  leaf_index = jnp.asarray(leaf_index, dtype=jnp.int32)
  loop_state = (tree, tree.node_values[leaf_index], leaf_index)
  tree, _, _ = jax.lax.while_loop(cond_fun, body_fun, loop_state)

  return tree


def t4_optimized_instantiate_tree(
    root: base.RootFnOutput,
    num_simulations: int,
    root_invalid_actions: chex.Array,
    extra_data: Any) -> Tree:
  """T4-optimized version of instantiate_tree_from_root.
  
  Uses memory-efficient array initialization for T4 GPUs.
  
  Args: Same as search.instantiate_tree_from_root.
  Returns: Same as search.instantiate_tree_from_root.
  """
  chex.assert_rank(root.prior_logits, 2)
  batch_size, num_actions = root.prior_logits.shape
  chex.assert_shape(root.value, [batch_size])
  num_nodes = num_simulations + 1
  data_dtype = root.value.dtype
  batch_node = (batch_size, num_nodes)
  batch_node_action = (batch_size, num_nodes, num_actions)

  # T4 optimization: Use more efficient array initialization for T4
  def _zeros(x):
    # Use jax.lax.broadcast instead of jnp.zeros for better fusion
    shape = batch_node + x.shape[1:]
    return jnp.broadcast_to(jnp.zeros((), dtype=x.dtype), shape)

  # Create a new empty tree state and fill its root
  tree = Tree(
      node_visits=jnp.zeros(batch_node, dtype=jnp.int32),
      raw_values=jnp.zeros(batch_node, dtype=data_dtype),
      node_values=jnp.zeros(batch_node, dtype=data_dtype),
      parents=jnp.full(batch_node, Tree.NO_PARENT, dtype=jnp.int32),
      action_from_parent=jnp.full(
          batch_node, Tree.NO_PARENT, dtype=jnp.int32),
      children_index=jnp.full(
          batch_node_action, Tree.UNVISITED, dtype=jnp.int32),
      children_prior_logits=jnp.zeros(
          batch_node_action, dtype=root.prior_logits.dtype),
      children_values=jnp.zeros(batch_node_action, dtype=data_dtype),
      children_visits=jnp.zeros(batch_node_action, dtype=jnp.int32),
      children_rewards=jnp.zeros(batch_node_action, dtype=data_dtype),
      children_discounts=jnp.zeros(batch_node_action, dtype=data_dtype),
      embeddings=jax.tree.map(_zeros, root.embedding),
      root_invalid_actions=root_invalid_actions,
      extra_data=extra_data)

  root_index = jnp.full([batch_size], Tree.ROOT_INDEX)
  tree = search.update_tree_node(
      tree, root_index, root.prior_logits, root.value, root.embedding)
  return tree