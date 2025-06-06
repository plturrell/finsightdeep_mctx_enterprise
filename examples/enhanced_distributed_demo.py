#!/usr/bin/env python3
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
"""Demonstration of enhanced distributed MCTS for enterprise environments."""

import argparse
import os
import time
import pprint
from typing import Dict, Any, List

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import mctx


def main():
    """Run a demo of enhanced distributed MCTS with fault tolerance and optimizations."""
    parser = argparse.ArgumentParser(description='Enhanced Distributed MCTS Demo')
    parser.add_argument('--num_gpus', type=int, default=1, 
                      help='Number of GPUs to use (default: 1)')
    parser.add_argument('--batch_size', type=int, default=128, 
                      help='Batch size for search (default: 128)')
    parser.add_argument('--simulations', type=int, default=200, 
                      help='Number of simulations (default: 200)')
    parser.add_argument('--partition_strategy', type=str, 
                      choices=['tree', 'batch', 'hybrid'], default='hybrid',
                      help='Strategy for partitioning (default: hybrid)')
    parser.add_argument('--precision', type=str, 
                      choices=['fp16', 'fp32'], default='fp16',
                      help='Computation precision (default: fp16)')
    parser.add_argument('--load_balancing', type=str, 
                      choices=['static', 'dynamic', 'adaptive'], default='adaptive',
                      help='Load balancing strategy (default: adaptive)')
    parser.add_argument('--merge_strategy', type=str, 
                      choices=['weighted', 'max_value', 'consensus'], default='weighted',
                      help='Strategy for merging results (default: weighted)')
    parser.add_argument('--fault_tolerance', type=int, choices=[0, 1, 2, 3], default=2,
                      help='Level of fault tolerance (default: 2)')
    parser.add_argument('--benchmark', action='store_true', 
                      help='Run performance benchmark')
    args = parser.parse_args()
    
    print("\n=== Enhanced Distributed MCTS Demo ===\n")
    
    # Print JAX and device information
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    num_devices = min(args.num_gpus, len(jax.devices()))
    print(f"Using {num_devices} device(s) for distributed MCTS\n")
    
    # Create environment and model
    env = BinaryTreeEnvironment()
    model = SimpleModel(env.num_actions)
    
    # Initialize model parameters
    params = model.init(jax.random.PRNGKey(42))
    
    # Initialize state
    state = jnp.zeros((args.batch_size, 8))  # 8-dimensional state
    
    # Create distributed configuration
    config = mctx.EnhancedDistributedConfig(
        num_devices=num_devices,
        partition_strategy=args.partition_strategy,
        precision=args.precision,
        tensor_core_aligned=True,
        load_balancing=args.load_balancing,
        fault_tolerance=args.fault_tolerance,
        memory_optimization=2,
        profiling=args.benchmark,
        metrics_collection=True,
        merge_strategy=args.merge_strategy
    )
    
    print("Configuration:")
    config_dict = config._asdict()
    for key, value in config_dict.items():
        print(f"  {key}: {value}")
    
    # Define root and recurrent functions
    def root_fn(params, rng_key, state):
        prior_logits, value = model.apply(params, state)
        return mctx.RootFnOutput(
            prior_logits=prior_logits,
            value=value,
            embedding=state)
    
    def recurrent_fn(params, rng_key, action, embedding):
        next_state = jnp.roll(embedding, 1, axis=-1)  # Simple state transition
        next_state = next_state.at[..., 0].set(action)
        prior_logits, value = model.apply(params, next_state)
        return (
            mctx.RecurrentFnOutput(
                reward=jnp.ones_like(value) * 0.1,  # Small positive reward
                discount=jnp.ones_like(value) * 0.99,  # Standard discount
                prior_logits=prior_logits,
                value=value),
            next_state)
    
    # Prepare for search
    print(f"\nRunning MCTS with {args.simulations} simulations on batch size {args.batch_size}...")
    
    # Define search function
    @mctx.enhanced_distribute_mcts(config=config)
    def run_search(params, rng_key):
        root = root_fn(params, rng_key, state)
        return mctx.search(
            params=params,
            rng_key=rng_key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=args.simulations,
            max_depth=50,
            root_action_selection_fn=mctx.muzero_action_selection,
            interior_action_selection_fn=mctx.muzero_action_selection)
    
    # Alternative approach using enhanced_distributed_search directly
    def run_direct_search(params, rng_key):
        root = root_fn(params, rng_key, state)
        return mctx.enhanced_distributed_search(
            params=params,
            rng_key=rng_key,
            root=root,
            recurrent_fn=recurrent_fn,
            root_action_selection_fn=mctx.muzero_action_selection,
            interior_action_selection_fn=mctx.muzero_action_selection,
            num_simulations=args.simulations,
            max_depth=50,
            config=config)
    
    # Run basic example
    start_time = time.time()
    result_tree = run_search(params, jax.random.PRNGKey(0))
    search_time = time.time() - start_time
    
    print(f"\nSearch completed in {search_time:.2f} seconds")
    print(f"Simulations per second: {args.simulations / search_time:.2f}")
    
    # Extract search results
    tree_summary = result_tree.summary()
    root_value = tree_summary.value[0]
    selected_action = jnp.argmax(tree_summary.visit_counts[0])
    print(f"Root value: {root_value:.4f}")
    print(f"Selected action: {selected_action}")
    print(f"Visit distribution: {tree_summary.visit_counts[0]}")
    
    # Show performance metrics if available
    if hasattr(result_tree, 'extra_data') and result_tree.extra_data is not None:
        if isinstance(result_tree.extra_data, dict) and 'performance_metrics' in result_tree.extra_data:
            print("\nPerformance Metrics:")
            metrics = result_tree.extra_data['performance_metrics']
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.2f}")
                elif isinstance(value, (list, tuple, np.ndarray, jnp.ndarray)):
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {value}")
    
    # Run benchmark if requested
    if args.benchmark:
        print("\n=== Performance Benchmark ===")
        
        # Compare against standard distributed search
        std_config = mctx.DistributedConfig(
            num_devices=num_devices,
            partition_search=True,
            partition_batch=True,
            precision=args.precision,
            tensor_core_aligned=True
        )
        
        # Define standard distributed search
        @mctx.distribute_mcts(config=std_config)
        def run_std_search(params, rng_key):
            root = root_fn(params, rng_key, state)
            return mctx.search(
                params=params,
                rng_key=rng_key,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=args.simulations,
                max_depth=50,
                root_action_selection_fn=mctx.muzero_action_selection,
                interior_action_selection_fn=mctx.muzero_action_selection)
        
        # Define direct enhanced search
        def run_direct_enhanced(params, rng_key):
            root = root_fn(params, rng_key, state)
            return mctx.enhanced_distributed_search(
                params=params,
                rng_key=rng_key,
                root=root,
                recurrent_fn=recurrent_fn,
                root_action_selection_fn=mctx.muzero_action_selection,
                interior_action_selection_fn=mctx.muzero_action_selection,
                num_simulations=args.simulations,
                max_depth=50,
                config=config)
        
        # Benchmark functions
        def benchmark_search(search_fn, name, num_runs=3):
            print(f"\nBenchmarking {name}...")
            times = []
            for i in range(num_runs):
                rng_key = jax.random.PRNGKey(i)
                start_time = time.time()
                tree = search_fn(params, rng_key)
                jax.block_until_ready(tree)
                end_time = time.time()
                times.append(end_time - start_time)
                print(f"  Run {i+1}: {times[-1]:.4f}s")
            
            avg_time = sum(times) / len(times)
            print(f"  Average time: {avg_time:.4f}s")
            print(f"  Simulations per second: {args.simulations / avg_time:.2f}")
            return avg_time, args.simulations / avg_time
        
        # Warm up
        print("Warming up...")
        _ = run_std_search(params, jax.random.PRNGKey(99))
        _ = run_search(params, jax.random.PRNGKey(98))
        _ = run_direct_enhanced(params, jax.random.PRNGKey(97))
        
        # Run benchmarks
        std_time, std_sps = benchmark_search(run_std_search, "Standard Distributed")
        enh_time, enh_sps = benchmark_search(run_search, "Enhanced Distributed (Decorator)")
        dir_time, dir_sps = benchmark_search(run_direct_enhanced, "Enhanced Distributed (Direct)")
        
        # Plot results
        methods = ["Standard", "Enhanced\n(Decorator)", "Enhanced\n(Direct)"]
        times = [std_time, enh_time, dir_time]
        speedups = [1.0, std_time / enh_time, std_time / dir_time]
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.bar(methods, times)
        plt.ylabel("Time (s)")
        plt.title("Execution Time")
        plt.grid(axis='y')
        
        for i, time_val in enumerate(times):
            plt.text(i, time_val + 0.05, f"{time_val:.2f}s", 
                    ha='center', va='bottom')
        
        plt.subplot(1, 2, 2)
        plt.bar(methods, speedups)
        plt.ylabel("Speedup")
        plt.title("Speedup vs. Standard Distributed")
        plt.grid(axis='y')
        
        for i, speedup in enumerate(speedups):
            plt.text(i, speedup + 0.05, f"{speedup:.2f}x", 
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig("enhanced_distributed_performance.png")
        print("\nPerformance chart saved to 'enhanced_distributed_performance.png'")
    
    print("\nDemo completed successfully.")


class BinaryTreeEnvironment:
    """Simple binary tree environment for testing."""
    
    def __init__(self):
        self.num_actions = 2
        self.state_size = 8
        
    def step(self, state, action):
        """Take a step in the environment."""
        next_state = jnp.roll(state, 1)
        next_state = next_state.at[0].set(action)
        reward = 0.0
        done = False
        return next_state, reward, done


class SimpleModel:
    """Simple neural network model for testing."""
    
    def __init__(self, num_actions):
        self.num_actions = num_actions
    
    def init(self, rng_key):
        """Initialize model parameters."""
        key1, key2 = jax.random.split(rng_key)
        policy_params = jax.random.normal(key1, (8, self.num_actions))
        value_params = jax.random.normal(key2, (8, 1))
        return (policy_params, value_params)
    
    def apply(self, params, state):
        """Apply the model to a state."""
        policy_params, value_params = params
        
        # Simple linear model
        policy_logits = jnp.dot(state, policy_params)
        value = jnp.dot(state, value_params).squeeze(-1)
        
        return policy_logits, value


if __name__ == "__main__":
    # Set environment variables for XLA
    os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'
    
    main()