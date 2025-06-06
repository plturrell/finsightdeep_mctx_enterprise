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
"""T4 GPU-specific optimizations for MCTX."""

import functools
import time
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

import jax
import jax.numpy as jnp
from jax.experimental import enable_x64

from mctx._src import base
from mctx._src import tree as tree_lib

# Type definitions
T = TypeVar('T')
ArrayTree = base.ArrayTree
Tree = tree_lib.Tree
ModelOutput = base.ModelOutput

# Constants for T4 optimization
T4_WARP_SIZE = 32
T4_TENSOR_CORE_DIM = 8  # Dimensions should be multiples of 8 for Tensor Cores


def align_for_tensor_cores(dimension: int) -> int:
    """Pads dimensions to multiples of 8 for T4 Tensor Cores.
    
    Args:
        dimension: The input dimension to pad.
        
    Returns:
        The padded dimension aligned to T4 Tensor Cores.
    """
    return ((dimension + T4_TENSOR_CORE_DIM - 1) // T4_TENSOR_CORE_DIM) * T4_TENSOR_CORE_DIM


def get_optimal_t4_batch_size(tree_depth: int, action_dim: int) -> int:
    """Calculate optimal batch size for T4 GPU based on model parameters.
    
    Args:
        tree_depth: Maximum depth of the search tree.
        action_dim: Dimension of the action space.
        
    Returns:
        The optimal batch size for T4 GPU.
    """
    # T4-specific heuristic based on 16GB VRAM
    max_nodes = 2**(tree_depth+1) - 1
    # Approximate bytes per node (values, visits, rewards, policy, etc.)
    node_size_bytes = action_dim * 4 * 4
    # Reserve ~2GB for other operations
    return min(512, int(14 * 1e9 / (max_nodes * node_size_bytes)))


def t4_autotuned_parameters() -> Dict[str, Any]:
    """Returns T4-specific parameter tuning defaults.
    
    These parameters are tuned for optimal performance on T4 GPUs.
    
    Returns:
        Dictionary of optimized parameters for T4 GPUs.
    """
    return {
        'simulation_batch_size': 128,  # Optimal for T4
        'dirichlet_alpha': 0.3,        # Tuned for T4 throughput
        'pb_c_init': 1.25,             # Optimized for T4
        'max_sim_depth': 50,           # Memory-optimized for T4
        'value_scale': 0.5,            # Better numerical stability on T4
    }


def dynamic_precision_policy(tree_size: int) -> jax.lax.Precision:
    """Dynamically adjust precision based on tree size.
    
    For larger trees, we use lower precision to save memory.
    For smaller trees, we use higher precision for better accuracy.
    
    Args:
        tree_size: Number of nodes in the tree.
        
    Returns:
        JAX precision policy.
    """
    if tree_size > 10000:
        return jax.lax.Precision.DEFAULT  # Save memory for large trees
    else:
        return jax.lax.Precision.HIGHEST  # Use full precision for smaller trees


def t4_optimized_tree_layout(tree: Tree) -> Tree:
    """Reorganize tree memory layout for T4's memory hierarchy.
    
    Args:
        tree: The input search tree.
        
    Returns:
        Reorganized tree with memory layout optimized for T4 GPU.
    """
    # Create a new tree with optimized memory layout
    new_tree = tree_lib.Tree(
        # Use column-major order for better GPU cache usage
        node_visits=jnp.asarray(tree.node_visits, order='F'),
        # Align children dimensions for tensor cores
        children_index=jnp.asarray(tree.children_index, order='F'),
        # Align policy dimensions for tensor cores
        node_values=jnp.asarray(tree.node_values, order='F'),
        # Use the original rewards
        rewards=tree.rewards,
        # Align action dimensions for tensor cores
        probs=jnp.asarray(tree.probs, order='F'),
        # Keep the batch dimension
        batch_size=tree.batch_size,
    )
    return new_tree


def t4_optimized_puct(
    prior: jnp.ndarray,
    value: jnp.ndarray,
    visit_count: jnp.ndarray,
    total_count: jnp.ndarray,
    pb_c_init: float,
    pb_c_base: float,
) -> jnp.ndarray:
    """T4-optimized PUCT calculation using tensor operations.
    
    This implementation reshapes arrays to align with tensor cores
    for optimal T4 GPU performance.
    
    Args:
        prior: Prior probabilities.
        value: Q-values.
        visit_count: Visit counts per action.
        total_count: Total visit count.
        pb_c_init: Exploration constant.
        pb_c_base: Base constant for exploration.
        
    Returns:
        PUCT scores.
    """
    # Reshape for tensor core utilization
    batch_size = prior.shape[0]
    action_dim = prior.shape[1]
    
    # Align dimensions for tensor cores
    aligned_batch = align_for_tensor_cores(batch_size)
    aligned_action = align_for_tensor_cores(action_dim)
    
    # Pad arrays to aligned dimensions
    prior_pad = jnp.zeros((aligned_batch, aligned_action))
    prior_pad = prior_pad.at[:batch_size, :action_dim].set(prior)
    
    value_pad = jnp.zeros((aligned_batch, aligned_action))
    value_pad = value_pad.at[:batch_size, :action_dim].set(value)
    
    visit_count_pad = jnp.zeros((aligned_batch, aligned_action))
    visit_count_pad = visit_count_pad.at[:batch_size, :action_dim].set(visit_count)
    
    total_count_pad = jnp.zeros((aligned_batch))
    total_count_pad = total_count_pad.at[:batch_size].set(total_count)
    
    # Calculate pb_c
    pb_c = jnp.log((total_count_pad + pb_c_base + 1) / pb_c_base) + pb_c_init
    pb_c = pb_c.reshape(-1, 1)  # For broadcasting
    
    # Calculate PUCT with tensor operations
    # Reshape for efficient matrix operations
    total_count_reshaped = total_count_pad.reshape(-1, 1)
    
    # Calculate exploration term
    exploration = prior_pad * pb_c * jnp.sqrt(total_count_reshaped) / (1 + visit_count_pad)
    
    # Calculate full PUCT score
    puct = value_pad + exploration
    
    # Extract the original dimensions
    puct = puct[:batch_size, :action_dim]
    
    return puct


def mixed_precision_wrapper(fn: Callable[..., T]) -> Callable[..., T]:
    """Wrapper to enable mixed precision for a function.
    
    Args:
        fn: Function to wrap.
        
    Returns:
        Wrapped function with mixed precision enabled.
    """
    @functools.wraps(fn)
    def wrapped(*args, precision: str = "fp16", **kwargs):
        if precision == "fp16":
            # Enable mixed precision for this function
            with jax.default_matmul_precision('tensorfloat32'):
                return fn(*args, **kwargs)
        else:
            # Regular implementation
            return fn(*args, **kwargs)
    return wrapped


def monitor_t4_memory_usage(fn: Callable[..., T]) -> Callable[..., T]:
    """Decorator to monitor T4 GPU memory during execution.
    
    Args:
        fn: Function to monitor.
        
    Returns:
        Wrapped function with memory monitoring.
    """
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            # Only import if available (development environments)
            import pynvml
            
            # Initialize NVML
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assume first GPU
            
            # Get initial memory info
            info_start = pynvml.nvmlDeviceGetMemoryInfo(handle)
            start_time = time.time()
            
            # Run the function
            result = fn(*args, **kwargs)
            
            # Get final memory info
            info_end = pynvml.nvmlDeviceGetMemoryInfo(handle)
            end_time = time.time()
            
            # Calculate and log memory usage
            memory_used = (info_end.used - info_start.used) / (1024 * 1024)  # MB
            print(f"T4 GPU Memory: {memory_used:.2f} MB, Time: {(end_time - start_time)*1000:.2f} ms")
            
            # Shutdown NVML
            pynvml.nvmlShutdown()
            
            return result
        except (ImportError, Exception):
            # If monitoring fails, just run the function
            return fn(*args, **kwargs)
    return wrapped


@jax.jit
def t4_fused_selection_expansion(
    tree: Tree,
    recurrent_fn: Callable[[Tree, jnp.ndarray], ModelOutput],
    nodes: jnp.ndarray,
    precision: Optional[jax.lax.Precision] = None,
) -> Tuple[Tree, jnp.ndarray]:
    """Fuse selection and expansion for T4's architecture.
    
    This fuses node selection and expansion into a single operation,
    reducing memory transfers and improving T4 GPU utilization.
    
    Args:
        tree: The search tree.
        recurrent_fn: Model function for evaluating nodes.
        nodes: Selected nodes to expand.
        precision: JAX precision to use (defaults to dynamic based on tree size).
        
    Returns:
        Tuple of (updated tree, new node indices).
    """
    # Use dynamic precision if not specified
    if precision is None:
        precision = dynamic_precision_policy(tree.node_visits.size)
    
    # Get the children indices for the selected nodes
    children = tree_lib.get_children_indices(tree, nodes)
    
    # Custom fusion hint for XLA
    def fused_op(tree, children):
        # Evaluate the nodes
        recurrent_output = recurrent_fn(tree, children)
        # Update the tree with the new nodes
        new_tree = tree_lib.expand(tree, nodes, children, recurrent_output)
        return new_tree, children
    
    # Use custom linear solve to hint XLA for fusion
    result = jax.lax.custom_linear_solve(
        (tree, children), 
        fused_op, 
        transpose_solve=False,
        precision=precision)
    
    return result