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
"""Distributed GPU-optimized MCTS implementation for NVIDIA hardware."""

import functools
import time
import os
import yaml
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import jax
import jax.numpy as jnp
from jax.experimental import maps
from jax.experimental import multihost_utils
from jax.experimental.pjit import pjit
from jax.experimental.compile_cache import compilation_cache
from jax.sharding import Mesh, PartitionSpec

from mctx._src import base
from mctx._src import tree as tree_lib
from mctx._src import t4_optimizations
from mctx._src import t4_search
from mctx._src import t4_memory_optimizations
from mctx._src import t4_tensor_cores

Tree = tree_lib.Tree
T = TypeVar('T')


def load_distribution_config(config_path: str) -> Dict[str, Any]:
    """Load distribution configuration from YAML file.
    
    Args:
        config_path: Path to distribution configuration YAML file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def initialize_distributed_environment(config: Dict[str, Any]) -> Tuple[Mesh, Dict[str, Any]]:
    """Initialize the distributed execution environment.
    
    Args:
        config: Distribution configuration
        
    Returns:
        Tuple of (device mesh, process groups)
    """
    # Set up distributed compilation cache
    if jax.process_count() > 1:
        compilation_cache.initialize_cache("./jax_cache")
    
    # Get all devices
    devices = jax.devices()
    
    # Create global device mesh
    device_mesh = jnp.array(devices).reshape(jax.process_count(), jax.local_device_count())
    mesh = Mesh(device_mesh, ('process', 'device'))
    
    # Create process groups
    process_groups = {}
    
    # Global process group includes all devices
    process_groups['global'] = {'mesh': mesh, 'devices': devices}
    
    # Set up additional process groups from config
    for group_config in config.get('distribution', {}).get('process_groups', []):
        name = group_config.get('name')
        if name == 'global':
            continue  # Already created
        
        if group_config.get('devices') == 'all':
            # Use all devices
            process_groups[name] = {'mesh': mesh, 'devices': devices}
        elif 'devices_per_node' in group_config:
            # Use a subset of devices per node
            devices_per_node = group_config['devices_per_node']
            group_devices = []
            for i in range(jax.process_count()):
                node_devices = jax.devices(i)[:devices_per_node]
                group_devices.extend(node_devices)
            
            # Create mesh for this group
            group_mesh = Mesh(jnp.array(group_devices).reshape(-1, devices_per_node),
                           ('process', 'device'))
            process_groups[name] = {'mesh': group_mesh, 'devices': group_devices}
    
    return mesh, process_groups


def get_optimal_sharding(config: Dict[str, Any], 
                       batch_size: int, 
                       num_actions: int,
                       tree_size: int) -> Dict[str, PartitionSpec]:
    """Determine optimal array sharding for distributed execution.
    
    Args:
        config: Distribution configuration
        batch_size: Batch size for search
        num_actions: Number of actions
        tree_size: Size of the search tree
        
    Returns:
        Dictionary of partition specs for different array types
    """
    strategy = config.get('distribution', {}).get('strategy', 'data_parallel')
    
    # Define partition specs based on strategy
    if strategy == 'data_parallel':
        # Shard batch dimension across processes
        return {
            'batch': PartitionSpec('process', None),
            'batch_node': PartitionSpec('process', None),
            'batch_action': PartitionSpec('process', None),
            'batch_node_action': PartitionSpec('process', None, None),
            'embedding': PartitionSpec('process', None, None),  # Shard first dim of embeddings
        }
    elif strategy == 'model_parallel':
        # Shard model parameters across devices
        return {
            'batch': PartitionSpec(None),
            'batch_node': PartitionSpec(None, 'device'),  # Shard nodes across devices
            'batch_action': PartitionSpec(None, 'device'),  # Shard actions across devices
            'batch_node_action': PartitionSpec(None, 'device', None),
            'embedding': PartitionSpec(None, None, 'device'),  # Shard embedding dimension
        }
    elif strategy == 'hybrid':
        # Hybrid approach: shard batch across processes, model across devices
        return {
            'batch': PartitionSpec('process', None),
            'batch_node': PartitionSpec('process', 'device'),
            'batch_action': PartitionSpec('process', 'device'),
            'batch_node_action': PartitionSpec('process', 'device', None),
            'embedding': PartitionSpec('process', None, 'device'),
        }
    else:
        # Default to data parallel
        return {
            'batch': PartitionSpec('process', None),
            'batch_node': PartitionSpec('process', None),
            'batch_action': PartitionSpec('process', None),
            'batch_node_action': PartitionSpec('process', None, None),
            'embedding': PartitionSpec('process', None, None),
        }


def distributed_search(
    params: base.Params,
    rng_key: jnp.ndarray,
    *,
    root: base.RootFnOutput,
    recurrent_fn: base.RecurrentFn,
    root_action_selection_fn: base.RootActionSelectionFn,
    interior_action_selection_fn: base.InteriorActionSelectionFn,
    num_simulations: int,
    config: Dict[str, Any],
    max_depth: Optional[int] = None,
    invalid_actions: Optional[jnp.ndarray] = None,
    extra_data: Any = None) -> Tree:
    """Distributed GPU-optimized MCTS search.
    
    This function distributes the search across multiple GPUs according to the
    provided configuration. It supports data parallelism, model parallelism,
    and hybrid approaches.
    
    Args:
        params: Model parameters
        rng_key: Random key
        root: Root node output
        recurrent_fn: Recurrent function for node evaluation
        root_action_selection_fn: Action selection at root
        interior_action_selection_fn: Action selection for interior nodes
        num_simulations: Number of simulations to run
        config: Distribution configuration
        max_depth: Maximum search depth
        invalid_actions: Mask of invalid actions
        extra_data: Extra data to include
        
    Returns:
        Completed search tree
    """
    # Extract configuration options
    precision = config.get('general', {}).get('precision', 'fp16')
    enable_tensor_cores = config.get('general', {}).get('enable_tensor_cores', True)
    enable_t4_optimizations = config.get('general', {}).get('enable_t4_optimizations', True)
    memory_optimization_level = config.get('general', {}).get('memory_optimization_level', 2)
    
    # Set up distributed execution
    mesh, process_groups = initialize_distributed_environment(config)
    
    # Determine sharding strategy
    batch_size = root.value.shape[0]
    num_actions = root.prior_logits.shape[1]
    tree_size = num_simulations + 1
    sharding = get_optimal_sharding(config, batch_size, num_actions, tree_size)
    
    # Wrap recurrent_fn for distributed execution
    def distributed_recurrent_fn(p, k, a, e):
        # Add device communication for model parallelism if needed
        if config.get('distribution', {}).get('strategy') in ['model_parallel', 'hybrid']:
            # Communicate intermediate results between devices
            result, next_state = recurrent_fn(p, k, a, e)
            # Synchronize across devices if needed
            if config.get('distribution', {}).get('sync', {}).get('sync_weights_every', 0) > 0:
                result = multihost_utils.process_allgather(result)
            return result, next_state
        else:
            # Standard recurrent function for data parallelism
            return recurrent_fn(p, k, a, e)
    
    # Apply T4 optimizations if enabled
    if enable_t4_optimizations:
        # Use T4-optimized search
        search_fn = t4_search.t4_search
        
        # Optimize recurrent function
        distributed_recurrent_fn = t4_optimizations.mixed_precision_wrapper(distributed_recurrent_fn)
        
        if enable_tensor_cores:
            # Apply tensor core optimizations
            distributed_recurrent_fn = t4_tensor_cores.optimize_recurrent_fn(distributed_recurrent_fn)
            root_action_selection_fn = t4_tensor_cores.optimize_action_selection(root_action_selection_fn)
            interior_action_selection_fn = t4_tensor_cores.optimize_action_selection(interior_action_selection_fn)
    else:
        # Use standard search
        search_fn = jax.jit(mctx.search)
    
    # Distribute the search computation
    with mesh:
        # Partition inputs according to sharding strategy
        p_params = params  # Parameters are usually replicated
        p_rng_key = rng_key  # RNG key replicated or sharded based on strategy
        
        # Run sharded search
        if config.get('distribution', {}).get('strategy') == 'data_parallel':
            # In data parallel mode, we shard the batch dimension
            # Each device processes a subset of the batch
            local_batch_size = batch_size // jax.process_count()
            
            # Split the root for each process
            def get_local_slice(x):
                start_idx = jax.process_index() * local_batch_size
                end_idx = start_idx + local_batch_size
                return x[start_idx:end_idx]
            
            local_root = base.RootFnOutput(
                prior_logits=get_local_slice(root.prior_logits),
                value=get_local_slice(root.value),
                embedding=jax.tree.map(get_local_slice, root.embedding)
            )
            
            local_invalid_actions = None
            if invalid_actions is not None:
                local_invalid_actions = get_local_slice(invalid_actions)
            
            # Run search on local slice
            local_result = search_fn(
                params=p_params,
                rng_key=p_rng_key,
                root=local_root,
                recurrent_fn=distributed_recurrent_fn,
                root_action_selection_fn=root_action_selection_fn,
                interior_action_selection_fn=interior_action_selection_fn,
                num_simulations=num_simulations,
                max_depth=max_depth,
                invalid_actions=local_invalid_actions,
                extra_data=extra_data,
                precision=precision,
                tensor_core_aligned=enable_tensor_cores,
                optimize_memory_layout=True,
                cache_optimization_level=memory_optimization_level,
                optimize_tensor_cores=enable_tensor_cores
            )
            
            # Gather results from all processes
            result = multihost_utils.process_allgather(local_result)
            
        else:
            # For model parallel or hybrid, use pjit to handle the sharding
            p_search = pjit(
                lambda p, k, r, i: search_fn(
                    params=p,
                    rng_key=k,
                    root=r,
                    recurrent_fn=distributed_recurrent_fn,
                    root_action_selection_fn=root_action_selection_fn,
                    interior_action_selection_fn=interior_action_selection_fn,
                    num_simulations=num_simulations,
                    max_depth=max_depth,
                    invalid_actions=i,
                    extra_data=extra_data,
                    precision=precision,
                    tensor_core_aligned=enable_tensor_cores,
                    optimize_memory_layout=True,
                    cache_optimization_level=memory_optimization_level,
                    optimize_tensor_cores=enable_tensor_cores
                ),
                in_shardings=(None, None, sharding['batch'], sharding['batch_action']),
                out_shardings=None  # Output is replicated
            )
            
            result = p_search(p_params, p_rng_key, root, invalid_actions)
    
    return result


class DistributedRunner:
    """Runner for distributed MCTS search across multiple GPUs."""
    
    def __init__(self, config_path: str):
        """Initialize the distributed runner.
        
        Args:
            config_path: Path to distribution configuration YAML file
        """
        self.config = load_distribution_config(config_path)
        self.mesh, self.process_groups = initialize_distributed_environment(self.config)
        
        # Log initialization
        if jax.process_index() == 0:
            print(f"Initialized distributed runner with {jax.device_count()} devices "
                 f"across {jax.process_count()} processes")
            print(f"Using distribution strategy: {self.config.get('distribution', {}).get('strategy')}")
            print(f"Process groups: {list(self.process_groups.keys())}")
    
    def run_search(self, 
                  params: base.Params,
                  model_fn: Callable,
                  initial_states: jnp.ndarray,
                  num_simulations: Optional[int] = None) -> Tree:
        """Run distributed search across multiple GPUs.
        
        Args:
            params: Model parameters
            model_fn: Model function that returns (recurrent_fn, root_fn)
            initial_states: Initial states to search from (batch dimension)
            num_simulations: Number of simulations (overrides config if provided)
            
        Returns:
            Completed search tree
        """
        # Create RNG key
        seed = self.config.get('general', {}).get('seed', 42)
        rng_key = jax.random.PRNGKey(seed)
        
        # Get recurrent and root functions from model
        recurrent_fn, root_fn = model_fn(params)
        
        # Create root node
        root_key, search_key = jax.random.split(rng_key)
        root = root_fn(params, root_key, initial_states)
        
        # Get search parameters from config
        search_config = self.config.get('search', {})
        sim_count = num_simulations or search_config.get('num_simulations', 800)
        max_depth = search_config.get('max_depth', 50)
        
        # Get action selection functions
        action_selection = search_config.get('action_selection', 'muzero')
        if action_selection == 'muzero':
            root_action_selection_fn = mctx.muzero_action_selection
            interior_action_selection_fn = mctx.muzero_action_selection
        elif action_selection == 'puct':
            c_init = search_config.get('puct', {}).get('c_init', 1.25)
            c_base = search_config.get('puct', {}).get('c_base', 19652)
            root_action_selection_fn = functools.partial(mctx.puct_action_selection, 
                                                      c_init=c_init, c_base=c_base)
            interior_action_selection_fn = functools.partial(mctx.puct_action_selection,
                                                          c_init=c_init, c_base=c_base)
        elif action_selection == 'gumbel':
            root_action_selection_fn = mctx.gumbel_muzero_action_selection
            interior_action_selection_fn = mctx.muzero_action_selection
        else:
            raise ValueError(f"Unknown action selection: {action_selection}")
        
        # Run distributed search
        start_time = time.time()
        result = distributed_search(
            params=params,
            rng_key=search_key,
            root=root,
            recurrent_fn=recurrent_fn,
            root_action_selection_fn=root_action_selection_fn,
            interior_action_selection_fn=interior_action_selection_fn,
            num_simulations=sim_count,
            max_depth=max_depth,
            config=self.config
        )
        end_time = time.time()
        
        # Log performance on main process
        if jax.process_index() == 0:
            duration = end_time - start_time
            batch_size = initial_states.shape[0]
            print(f"Distributed search completed in {duration:.2f}s")
            print(f"Performance: {sim_count * batch_size / duration:.2f} simulations/second")
            print(f"Tree size: {result.node_visits.shape[1]} nodes")
        
        return result


def run_distributed_search_from_config(config_path: str):
    """Run distributed search from configuration file.
    
    This function is the main entry point for distributed search.
    
    Args:
        config_path: Path to distribution configuration YAML file
    """
    # Initialize runner
    runner = DistributedRunner(config_path)
    
    # Get configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get model parameters from environment or config
    model_path = os.environ.get('MCTX_MODEL_PATH', 
                               config.get('general', {}).get('model_path', None))
    
    if not model_path:
        raise ValueError("Model path not specified. Set MCTX_MODEL_PATH environment variable "
                        "or specify model_path in the configuration file.")
    
    # Load model parameters
    if model_path.endswith('.pkl'):
        import pickle
        with open(model_path, 'rb') as f:
            params = pickle.load(f)
    elif model_path.endswith('.npz'):
        params = dict(np.load(model_path))
    else:
        # Try to load using JAX's serialization
        params = jax.tree.load(model_path)
    
    # Define model functions based on architecture specified in config
    def model_fn(params):
        model_type = os.environ.get('MCTX_MODEL_TYPE', 
                                  config.get('general', {}).get('model_type', 'default'))
        
        if model_type == 'default':
            # Default model implementation
            def recurrent_fn(p, k, a, e):
                # Basic recurrent function
                next_state = jnp.roll(e, 1, axis=-1)
                next_state = next_state.at[..., 0].set(a)
                
                # Apply model to get outputs
                outputs = apply_model(p, next_state)
                
                return (mctx.RecurrentFnOutput(
                    reward=outputs['reward'],
                    discount=outputs['discount'],
                    prior_logits=outputs['policy'],
                    value=outputs['value']
                ), next_state)
            
            def root_fn(p, k, s):
                # Apply model to get outputs
                outputs = apply_model(p, s)
                
                return mctx.RootFnOutput(
                    prior_logits=outputs['policy'],
                    value=outputs['value'],
                    embedding=s
                )
        else:
            # Custom model implementation based on type
            # This should be extended for specific model architectures
            recurrent_fn, root_fn = get_model_fns(model_type, params)
        
        return recurrent_fn, root_fn
    
    # Create initial states
    batch_size = int(os.environ.get('MCTX_BATCH_SIZE', 
                                   config.get('parallelism', {}).get('per_device_batch_size', 64)))
    state_size = int(os.environ.get('MCTX_STATE_SIZE', 
                                   config.get('general', {}).get('state_size', 8)))
    
    # Use random states or load from file if specified
    state_path = os.environ.get('MCTX_STATE_PATH', 
                               config.get('general', {}).get('state_path', None))
    
    if state_path:
        try:
            initial_states = jnp.load(state_path)
        except:
            logging.warning(f"Failed to load states from {state_path}, using random states")
            initial_states = jax.random.normal(jax.random.PRNGKey(0), (batch_size, state_size))
    else:
        initial_states = jax.random.normal(jax.random.PRNGKey(0), (batch_size, state_size))
    
    # Run distributed search
    result = runner.run_search(params, model_fn, initial_states)
    
    # Process and save results
    output_dir = os.environ.get('MCTX_OUTPUT_DIR', 
                               config.get('general', {}).get('output_dir', './mctx_output'))
    
    os.makedirs(output_dir, exist_ok=True)
    actions = mctx.bv_action_selection(result)
    
    # Save results
    jnp.save(f"{output_dir}/actions.npy", actions)
    
    # Return success
    return 0


def apply_model(params, state):
    """Apply model to state, can be overridden with custom implementation."""
    # This is a simplified implementation
    # In a real application, this would use the actual model architecture
    
    # Get model dimensions from params
    policy_params, value_params = params
    num_actions = policy_params.shape[-1] if len(policy_params.shape) > 1 else 1
    
    # Simple linear model
    policy = jnp.dot(state, policy_params)
    value = jnp.dot(state, value_params).squeeze(-1)
    
    # Create dummy reward and discount
    reward = jnp.zeros_like(value)
    discount = jnp.ones_like(value) * 0.99
    
    return {
        'policy': policy,
        'value': value,
        'reward': reward,
        'discount': discount
    }


def get_model_fns(model_type, params):
    """Get model functions based on model type."""
    # This would be expanded with actual implementations
    # for different model architectures
    
    if model_type == 'muzero':
        # MuZero model functions
        def recurrent_fn(p, k, a, e):
            # Implementation for MuZero
            pass
            
        def root_fn(p, k, s):
            # Implementation for MuZero
            pass
            
    elif model_type == 'alphazero':
        # AlphaZero model functions
        def recurrent_fn(p, k, a, e):
            # Implementation for AlphaZero
            pass
            
        def root_fn(p, k, s):
            # Implementation for AlphaZero
            pass
    else:
        # Default to simple model
        def recurrent_fn(p, k, a, e):
            next_state = jnp.roll(e, 1, axis=-1)
            next_state = next_state.at[..., 0].set(a)
            outputs = apply_model(p, next_state)
            return (mctx.RecurrentFnOutput(
                reward=outputs['reward'],
                discount=outputs['discount'],
                prior_logits=outputs['policy'],
                value=outputs['value']
            ), next_state)
        
        def root_fn(p, k, s):
            outputs = apply_model(p, s)
            return mctx.RootFnOutput(
                prior_logits=outputs['policy'],
                value=outputs['value'],
                embedding=s
            )
    
    return recurrent_fn, root_fn


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run distributed MCTS search")
    parser.add_argument("--config", type=str, required=True,
                      help="Path to distribution configuration YAML file")
    args = parser.parse_args()
    
    run_distributed_search_from_config(args.config)