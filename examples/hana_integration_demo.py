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
"""Demonstration of SAP HANA integration with MCTX."""

import argparse
import os
import time
from typing import Dict, Any

import jax
import jax.numpy as jnp
import numpy as np

import mctx
from mctx.enterprise import (
    HanaConfig,
    connect_to_hana,
    save_tree_to_hana,
    load_tree_from_hana,
    save_model_to_hana,
    load_model_from_hana,
    save_simulation_results,
    load_simulation_results,
    batch_tree_operations,
)


def main():
    """Run a demo of SAP HANA integration with MCTX."""
    parser = argparse.ArgumentParser(description='SAP HANA Integration Demo')
    parser.add_argument('--host', type=str, default="localhost",
                      help='SAP HANA host')
    parser.add_argument('--port', type=int, default=30015,
                      help='SAP HANA port')
    parser.add_argument('--user', type=str, default="SYSTEM",
                      help='SAP HANA user')
    parser.add_argument('--password', type=str, default="",
                      help='SAP HANA password')
    parser.add_argument('--schema', type=str, default="MCTX",
                      help='SAP HANA schema')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size for MCTS')
    parser.add_argument('--num_simulations', type=int, default=50,
                      help='Number of simulations')
    args = parser.parse_args()
    
    # Check if hdbcli is available
    try:
        import hdbcli
        print("SAP HANA client library (hdbcli) found.")
    except ImportError:
        print("SAP HANA client library (hdbcli) not found.")
        print("This demo requires the hdbcli package. Install with:")
        print("  pip install hdbcli")
        return
    
    print("\n=== SAP HANA Integration Demo ===\n")
    
    # Read password from environment variable if not provided
    if not args.password:
        args.password = os.environ.get("HANA_PASSWORD", "")
        if not args.password:
            print("Error: SAP HANA password not provided.")
            print("Either use --password or set the HANA_PASSWORD environment variable.")
            return
    
    # Create HANA configuration
    hana_config = HanaConfig(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        schema=args.schema,
        use_compression=True,
        enable_caching=True
    )
    
    # Connect to SAP HANA
    print(f"Connecting to SAP HANA at {args.host}:{args.port}...")
    try:
        hana_connection = connect_to_hana(hana_config)
        print("Connected to SAP HANA successfully!")
    except Exception as e:
        print(f"Error connecting to SAP HANA: {e}")
        return
    
    # Create environment and model
    print("\nCreating environment and model...")
    env = BinaryTreeEnvironment()
    model = SimpleModel(env.num_actions)
    
    # Run MCTS
    print("\nRunning MCTS...")
    tree, params, summary = run_mcts(
        model, args.batch_size, env.num_actions, args.num_simulations)
    
    # Save model to HANA
    print("\nSaving model to SAP HANA...")
    model_id = save_model_to_hana(
        hana_connection,
        params,
        name="SimpleModel",
        model_type="binary_tree",
        metadata={
            "num_actions": env.num_actions,
            "embedding_size": 8,
            "creation_time": time.time()
        }
    )
    print(f"Model saved with ID: {model_id}")
    
    # Save tree to HANA
    print("\nSaving tree to SAP HANA...")
    tree_id = save_tree_to_hana(
        hana_connection,
        tree,
        name="BinaryTree",
        metadata={
            "model_id": model_id,
            "environment": "binary_tree",
            "creation_time": time.time()
        }
    )
    print(f"Tree saved with ID: {tree_id}")
    
    # Save simulation results
    print("\nSaving simulation results to SAP HANA...")
    result_id = save_simulation_results(
        hana_connection,
        tree_id,
        model_id,
        0,  # batch index
        summary,
        metadata={
            "simulation_params": {
                "num_simulations": args.num_simulations,
                "batch_size": args.batch_size
            }
        }
    )
    print(f"Simulation results saved with ID: {result_id}")
    
    # Load data from HANA
    print("\nLoading data from SAP HANA...")
    
    # Load model
    loaded_params = load_model_from_hana(hana_connection, model_id)
    if loaded_params is not None:
        print(f"Successfully loaded model with ID: {model_id}")
    else:
        print(f"Failed to load model with ID: {model_id}")
    
    # Load tree
    loaded_tree_result = load_tree_from_hana(hana_connection, tree_id)
    if loaded_tree_result is not None:
        loaded_tree, loaded_metadata = loaded_tree_result
        print(f"Successfully loaded tree with ID: {tree_id}")
        print(f"Tree metadata: {loaded_metadata}")
    else:
        print(f"Failed to load tree with ID: {tree_id}")
    
    # Load simulation results
    loaded_results = load_simulation_results(
        hana_connection, tree_id=tree_id)
    if loaded_results:
        print(f"Successfully loaded {len(loaded_results)} simulation results")
        result_id, result_summary, result_metadata = loaded_results[0]
        print(f"Result ID: {result_id}")
        print(f"Result metadata: {result_metadata}")
    else:
        print("Failed to load simulation results")
    
    # Test batch operations
    print("\nTesting batch operations...")
    batch_results = batch_tree_operations(
        hana_connection,
        [
            {
                "operation": "load_tree",
                "tree_id": tree_id
            },
            {
                "operation": "load_model",
                "model_id": model_id
            }
        ]
    )
    print(f"Batch operations completed with {len(batch_results)} results")
    
    # Close connection
    hana_connection.close_all()
    print("\nConnection to SAP HANA closed.")


def run_mcts(model, batch_size, num_actions, num_simulations):
    """Run MCTS and return the tree, params, and summary."""
    # Initialize model
    params = model.init(jax.random.PRNGKey(42))
    
    # Initialize state
    state = jnp.zeros((batch_size, 8))  # 8-dimensional state
    
    # Define root function
    def root_fn(params, rng_key, state):
        prior_logits, value = model.apply(params, state)
        return mctx.RootFnOutput(
            prior_logits=prior_logits,
            value=value,
            embedding=state)
    
    # Define recurrent function
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
    
    # Create JIT-compiled search function
    search_fn = jax.jit(lambda p, k, s: mctx.search(
        params=p,
        rng_key=k,
        root=root_fn(p, k, s),
        recurrent_fn=recurrent_fn,
        num_simulations=num_simulations,
        max_depth=10,
        root_action_selection_fn=mctx.muzero_action_selection,
        interior_action_selection_fn=mctx.muzero_action_selection))
    
    # Run search
    key = jax.random.PRNGKey(0)
    tree = search_fn(params, key, state)
    
    # Get summary
    summary = tree.summary()
    
    return tree, params, summary


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