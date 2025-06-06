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
"""Demonstration of distributed MCTS across multiple GPUs."""

import time
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

import mctx
from mctx._src.distributed import DistributedConfig, distribute_mcts, distributed_search


def main():
    """Run a demo of distributed MCTS."""
    parser = argparse.ArgumentParser(description='Distributed MCTS Demo')
    parser.add_argument('--num_gpus', type=int, default=1, 
                        help='Number of GPUs to use')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='Batch size')
    parser.add_argument('--simulations', type=int, default=100, 
                        help='Number of simulations')
    parser.add_argument('--tree_depth', type=int, default=10, 
                        help='Maximum tree depth')
    parser.add_argument('--precision', type=str, default='fp16', 
                        choices=['fp16', 'fp32'], help='Precision to use')
    parser.add_argument('--repeat', type=int, default=5, 
                        help='Number of runs to average')
    args = parser.parse_args()
    
    print("\n=== Distributed MCTS Demo ===\n")
    
    # Print JAX configuration
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print(f"Using {args.num_gpus} GPU(s) for distributed MCTS\n")
    
    # Create environment and model
    env = BinaryTreeEnvironment()
    model = SimpleModel(env.num_actions)
    
    # Get model parameters
    params = model.init(jax.random.PRNGKey(42))
    
    # Run performance tests
    print("Running performance tests...\n")
    
    # Initialize state
    state = jnp.zeros((args.batch_size, 8))  # 8-dimensional state
    
    # Configure standard search
    def run_standard_search(params, rng_key, root, num_simulations):
        return mctx.search(
            params=params,
            rng_key=rng_key,
            root=root,
            recurrent_fn=recurrent_fn,
            root_action_selection_fn=mctx.muzero_action_selection,
            interior_action_selection_fn=mctx.muzero_action_selection,
            num_simulations=num_simulations,
            max_depth=args.tree_depth)
    
    # Configure distributed search
    dist_config = DistributedConfig(
        num_devices=args.num_gpus,
        partition_search=True,
        partition_batch=True,
        device_type="gpu",
        precision=args.precision,
        tensor_core_aligned=True,
        replicated_params=True)
    
    # Decorate the standard search with distributed wrapper
    distributed_search_fn = distribute_mcts(dist_config)(run_standard_search)
    
    # Define model functions
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
    
    # Benchmark functions
    def benchmark_standard(key, repeat=args.repeat):
        times = []
        for i in range(repeat):
            run_key = jax.random.fold_in(key, i)
            root = root_fn(params, run_key, state)
            
            start_time = time.time()
            tree = run_standard_search(params, run_key, root, args.simulations)
            jax.block_until_ready(tree)
            end_time = time.time()
            
            times.append((end_time - start_time) * 1000)  # ms
        return np.mean(times), np.std(times)
    
    def benchmark_distributed(key, repeat=args.repeat):
        times = []
        for i in range(repeat):
            run_key = jax.random.fold_in(key, i)
            root = root_fn(params, run_key, state)
            
            start_time = time.time()
            tree = distributed_search_fn(params, run_key, root, args.simulations)
            jax.block_until_ready(tree)
            end_time = time.time()
            
            times.append((end_time - start_time) * 1000)  # ms
        return np.mean(times), np.std(times)
    
    # Run benchmarks
    key = jax.random.PRNGKey(42)
    
    # Warmup
    print("Warming up...")
    _ = benchmark_standard(key, repeat=2)
    _ = benchmark_distributed(key, repeat=2)
    
    # Benchmark
    print(f"Running benchmarks with batch_size={args.batch_size}, "
          f"simulations={args.simulations}, tree_depth={args.tree_depth}...")
    
    standard_mean, standard_std = benchmark_standard(key)
    dist_mean, dist_std = benchmark_distributed(key)
    
    # Calculate speedup
    speedup = standard_mean / dist_mean
    
    # Print results
    print("\n=== Results ===")
    print(f"Standard MCTS:   {standard_mean:.2f} ± {standard_std:.2f} ms")
    print(f"Distributed MCTS: {dist_mean:.2f} ± {dist_std:.2f} ms")
    print(f"Speedup: {speedup:.2f}x\n")
    
    if args.num_gpus > 1:
        expected_speedup = min(args.num_gpus, 
                            args.simulations / (args.simulations // args.num_gpus))
        scaling_efficiency = speedup / expected_speedup * 100
        print(f"Scaling efficiency: {scaling_efficiency:.2f}%")
        print(f"(100% would mean perfect linear scaling)")
    
    # Scaling test with different batch sizes
    if args.batch_size >= 16:
        print("\n=== Scaling Test ===")
        
        batch_sizes = [16, 32, 64, 128, 256, 512]
        batch_sizes = [b for b in batch_sizes if b <= args.batch_size]
        
        standard_times = []
        distributed_times = []
        speedups = []
        
        for bs in batch_sizes:
            print(f"Testing batch size {bs}...")
            
            # Create state with this batch size
            test_state = jnp.zeros((bs, 8))
            
            # Standard search
            std_times = []
            for i in range(3):  # Fewer repeats for quicker results
                run_key = jax.random.fold_in(key, i)
                root = root_fn(params, run_key, test_state)
                
                start_time = time.time()
                tree = run_standard_search(params, run_key, root, args.simulations)
                jax.block_until_ready(tree)
                end_time = time.time()
                
                std_times.append((end_time - start_time) * 1000)
            
            # Distributed search
            dist_times = []
            for i in range(3):  # Fewer repeats for quicker results
                run_key = jax.random.fold_in(key, i + 100)
                root = root_fn(params, run_key, test_state)
                
                start_time = time.time()
                tree = distributed_search_fn(params, run_key, root, args.simulations)
                jax.block_until_ready(tree)
                end_time = time.time()
                
                dist_times.append((end_time - start_time) * 1000)
            
            # Calculate means
            std_mean = np.mean(std_times)
            dist_mean = np.mean(dist_times)
            bs_speedup = std_mean / dist_mean
            
            standard_times.append(std_mean)
            distributed_times.append(dist_mean)
            speedups.append(bs_speedup)
            
            print(f"  Batch {bs}: Standard {std_mean:.2f}ms, "
                  f"Distributed {dist_mean:.2f}ms, Speedup {bs_speedup:.2f}x")
        
        # Plot results
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(batch_sizes, standard_times, 'o-', label='Standard MCTS')
        plt.plot(batch_sizes, distributed_times, 's-', label='Distributed MCTS')
        plt.xlabel('Batch Size')
        plt.ylabel('Execution Time (ms)')
        plt.title('MCTS Performance vs Batch Size')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(batch_sizes, speedups, 'o-')
        plt.axhline(y=args.num_gpus, color='r', linestyle='--', 
                   label=f'Perfect Scaling ({args.num_gpus}x)')
        plt.xlabel('Batch Size')
        plt.ylabel('Speedup')
        plt.title('Scaling Efficiency')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('distributed_mcts_scaling.png')
        print("\nScaling chart saved to 'distributed_mcts_scaling.png'")


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