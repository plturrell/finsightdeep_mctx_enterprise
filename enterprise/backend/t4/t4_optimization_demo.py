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
"""Demonstration of T4-optimized MCTS search."""

import time
from typing import Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import mctx


def main():
  """Run a demo comparing regular MCTS to T4-optimized MCTS."""
  print("T4-optimized MCTS Demo")
  print("======================")
  
  # Set up JAX
  print("\nJAX device info:")
  print(jax.devices())
  
  # Create a simple environment for testing
  env = BinaryTreeEnvironment()
  
  # Set up search parameters
  batch_size = 16
  num_actions = 2
  num_simulations = 50
  
  # Create a simple neural network model
  model = SimpleModel(num_actions)
  
  # Get the autotuned parameters for T4 GPUs
  t4_params = mctx.t4_autotuned_parameters()
  print(f"\nT4 autotuned parameters: {t4_params}")
  
  # Calculate optimal batch size for T4
  optimal_batch = mctx.get_optimal_t4_batch_size(
      tree_depth=10, action_dim=num_actions)
  print(f"Optimal T4 batch size for this model: {optimal_batch}")
  
  # Run performance tests
  print("\nRunning performance comparison...")
  
  # Test batch sizes for comparison
  batch_sizes = [8, 16, 32, 64, 128]
  standard_times = []
  t4_fp32_times = []
  t4_fp16_times = []
  
  for bs in batch_sizes:
    print(f"\nBatch size: {bs}")
    
    # Get model parameters
    params = model.init(jax.random.PRNGKey(42))
    
    # Initialize dummy state
    dummy_state = jnp.zeros((bs, 8))  # 8-dimensional state
    
    # Standard MCTS
    standard_time = benchmark_search(
        bs, params, model, dummy_state, env.num_actions, 
        num_simulations, use_t4=False)
    standard_times.append(standard_time)
    
    # T4-optimized MCTS with FP32
    t4_fp32_time = benchmark_search(
        bs, params, model, dummy_state, env.num_actions, 
        num_simulations, use_t4=True, precision="fp32")
    t4_fp32_times.append(t4_fp32_time)
    
    # T4-optimized MCTS with FP16
    t4_fp16_time = benchmark_search(
        bs, params, model, dummy_state, env.num_actions, 
        num_simulations, use_t4=True, precision="fp16")
    t4_fp16_times.append(t4_fp16_time)
    
    # Print speedup
    print(f"  Standard MCTS:       {standard_time:.2f} ms")
    print(f"  T4-optimized (FP32): {t4_fp32_time:.2f} ms " +
          f"({standard_time/t4_fp32_time:.2f}x speedup)")
    print(f"  T4-optimized (FP16): {t4_fp16_time:.2f} ms " + 
          f"({standard_time/t4_fp16_time:.2f}x speedup)")
  
  # Plot performance comparison
  plot_performance(batch_sizes, standard_times, t4_fp32_times, t4_fp16_times)


def benchmark_search(batch_size, params, model, state, num_actions, 
                     num_simulations, use_t4=False, precision="fp32"):
  """Benchmark a search function and return time in milliseconds."""
  # Create root function
  def root_fn(params, rng_key, state):
    prior_logits, value = model.apply(params, state)
    return mctx.RootFnOutput(
        prior_logits=prior_logits,
        value=value,
        embedding=state)
  
  # Create recurrent function
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
  
  # JIT compile the search
  if use_t4:
    # Use T4-optimized search
    search_fn = jax.jit(lambda p, k, s: mctx.t4_search(
        params=p,
        rng_key=k,
        root=root_fn(p, k, s),
        recurrent_fn=recurrent_fn,
        num_simulations=num_simulations,
        max_depth=10,
        root_action_selection_fn=mctx.muzero_action_selection,
        interior_action_selection_fn=mctx.muzero_action_selection,
        precision=precision,
        tensor_core_aligned=True,
        monitor_memory=False))
  else:
    # Use standard search
    search_fn = jax.jit(lambda p, k, s: mctx.search(
        params=p,
        rng_key=k,
        root=root_fn(p, k, s),
        recurrent_fn=recurrent_fn,
        num_simulations=num_simulations,
        max_depth=10,
        root_action_selection_fn=mctx.muzero_action_selection,
        interior_action_selection_fn=mctx.muzero_action_selection))
  
  # Compile the function
  key = jax.random.PRNGKey(42)
  _ = search_fn(params, key, state)
  
  # Benchmark
  runs = 5
  times = []
  for i in range(runs):
    key = jax.random.PRNGKey(i)
    start_time = time.time()
    _ = search_fn(params, key, state)
    jax.block_until_ready(_)  # Wait for result
    end_time = time.time()
    times.append((end_time - start_time) * 1000)  # Convert to ms
  
  # Return average time
  return sum(times) / len(times)


def plot_performance(batch_sizes, standard_times, t4_fp32_times, t4_fp16_times):
  """Plot performance comparison."""
  plt.figure(figsize=(10, 6))
  
  plt.plot(batch_sizes, standard_times, 'o-', label='Standard MCTS')
  plt.plot(batch_sizes, t4_fp32_times, 's-', label='T4-optimized (FP32)')
  plt.plot(batch_sizes, t4_fp16_times, '^-', label='T4-optimized (FP16)')
  
  plt.xlabel('Batch Size')
  plt.ylabel('Execution Time (ms)')
  plt.title('MCTS Performance Comparison on T4 GPU')
  plt.grid(True)
  plt.legend()
  
  # Add speedup annotations
  for i, bs in enumerate(batch_sizes):
    fp32_speedup = standard_times[i] / t4_fp32_times[i]
    fp16_speedup = standard_times[i] / t4_fp16_times[i]
    
    plt.annotate(f'{fp32_speedup:.2f}x', 
                 (bs, t4_fp32_times[i]),
                 textcoords="offset points",
                 xytext=(0, -15),
                 ha='center')
                 
    plt.annotate(f'{fp16_speedup:.2f}x', 
                 (bs, t4_fp16_times[i]),
                 textcoords="offset points",
                 xytext=(0, -15),
                 ha='center')
  
  plt.savefig('t4_mcts_performance.png')
  print("\nPerformance chart saved to 't4_mcts_performance.png'")


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
  main()