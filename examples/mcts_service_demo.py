#!/usr/bin/env python3
"""
Demo script for using the MCTS service with T4 and distributed optimizations.

This script demonstrates how to construct requests to the MCTS service
to utilize both T4-optimized and distributed MCTS search capabilities.
"""

import argparse
import json
import numpy as np
import sys
import time
from typing import Dict, List, Any, Optional

# Add the project root to the Python path
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.app.models.mcts_models import MCTSRequest, SearchParams, RootInput
from api.app.services.mcts_service import MCTSService


def create_mock_root_input(batch_size: int = 4, num_actions: int = 32) -> RootInput:
    """Create a mock root input for testing."""
    # Create prior logits
    prior_logits = [np.random.normal(0, 1, num_actions).tolist() for _ in range(batch_size)]
    
    # Create value estimates
    value = np.random.normal(0, 1, batch_size).tolist()
    
    # Create mock embeddings (just a single feature per batch element)
    embedding = [np.zeros(10).tolist() for _ in range(batch_size)]
    
    return RootInput(
        prior_logits=prior_logits,
        value=value,
        embedding=embedding,
        batch_size=batch_size,
        num_actions=num_actions
    )


def run_search_benchmark(
    batch_size: int = 4,
    num_actions: int = 32,
    num_simulations: int = 32,
    use_t4: bool = False,
    distributed: bool = False,
    num_devices: int = 1,
    search_type: str = "gumbel_muzero",
    precision: str = "fp16"
) -> Dict[str, Any]:
    """Run a benchmark of the MCTS service with different configurations."""
    # Create the service
    service = MCTSService()
    
    # Create the root input
    root_input = create_mock_root_input(batch_size, num_actions)
    
    # Create search parameters
    search_params = SearchParams(
        num_simulations=num_simulations,
        max_depth=None,
        max_num_considered_actions=16,
        dirichlet_fraction=0.25,
        dirichlet_alpha=0.3,
        use_t4_optimizations=use_t4,
        precision=precision,
        tensor_core_aligned=True,
        distributed=distributed,
        num_devices=num_devices,
        partition_batch=True
    )
    
    # Create the request
    request = MCTSRequest(
        root_input=root_input,
        search_params=search_params,
        search_type=search_type,
        device_type="gpu"
    )
    
    # Run the search
    start_time = time.time()
    result = service.run_search(request, user_id="benchmark_user")
    elapsed_time = time.time() - start_time
    
    # Create benchmark results
    benchmark_result = {
        "configuration": {
            "batch_size": batch_size,
            "num_actions": num_actions,
            "num_simulations": num_simulations,
            "search_type": search_type,
            "use_t4": use_t4,
            "distributed": distributed,
            "num_devices": num_devices,
            "precision": precision,
        },
        "performance": {
            "duration_ms": result.search_statistics["duration_ms"],
            "elapsed_time_ms": elapsed_time * 1000,
            "num_expanded_nodes": result.search_statistics["num_expanded_nodes"],
            "max_depth_reached": result.search_statistics["max_depth_reached"],
        },
        "distributed_stats": result.distributed_stats
    }
    
    return benchmark_result


def print_table(results: List[Dict[str, Any]]) -> None:
    """Print benchmark results as a formatted table."""
    # Print header
    print("\n" + "=" * 100)
    print(f"{'Configuration':<50} | {'Duration (ms)':<12} | {'Expanded Nodes':<15} | {'Max Depth':<10}")
    print("-" * 100)
    
    # Print results
    for result in results:
        config = result["configuration"]
        perf = result["performance"]
        
        config_str = (
            f"B={config['batch_size']}, A={config['num_actions']}, "
            f"S={config['num_simulations']}, {config['search_type']}, "
            f"T4={config['use_t4']}, Dist={config['distributed']}({config['num_devices']})"
        )
        
        print(
            f"{config_str:<50} | {perf['duration_ms']:<12.2f} | "
            f"{perf['num_expanded_nodes']:<15} | {perf['max_depth_reached']:<10}"
        )
    
    print("=" * 100 + "\n")


def main():
    """Run the MCTS service benchmark."""
    parser = argparse.ArgumentParser(description="MCTS Service Benchmark")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_actions", type=int, default=32, help="Number of actions")
    parser.add_argument("--num_simulations", type=int, default=32, help="Number of simulations")
    parser.add_argument("--search_type", type=str, default="gumbel_muzero", 
                        choices=["muzero", "gumbel_muzero", "stochastic_muzero"],
                        help="Search algorithm to use")
    parser.add_argument("--run_all", action="store_true", help="Run all benchmark configurations")
    
    args = parser.parse_args()
    
    # Configure benchmark parameters
    batch_size = args.batch_size
    num_actions = args.num_actions
    num_simulations = args.num_simulations
    search_type = args.search_type
    
    results = []
    
    if args.run_all:
        # Run standard search
        results.append(run_search_benchmark(
            batch_size=batch_size,
            num_actions=num_actions,
            num_simulations=num_simulations,
            search_type=search_type,
            use_t4=False,
            distributed=False
        ))
        
        # Run T4-optimized search with FP16
        results.append(run_search_benchmark(
            batch_size=batch_size,
            num_actions=num_actions,
            num_simulations=num_simulations,
            search_type=search_type,
            use_t4=True,
            distributed=False,
            precision="fp16"
        ))
        
        # Run T4-optimized search with FP32
        results.append(run_search_benchmark(
            batch_size=batch_size,
            num_actions=num_actions,
            num_simulations=num_simulations,
            search_type=search_type,
            use_t4=True,
            distributed=False,
            precision="fp32"
        ))
        
        # Run distributed search with 2 devices
        results.append(run_search_benchmark(
            batch_size=batch_size,
            num_actions=num_actions,
            num_simulations=num_simulations,
            search_type=search_type,
            use_t4=False,
            distributed=True,
            num_devices=2,
            precision="fp16"
        ))
        
        # Run distributed search with 4 devices
        results.append(run_search_benchmark(
            batch_size=batch_size,
            num_actions=num_actions,
            num_simulations=num_simulations,
            search_type=search_type,
            use_t4=False,
            distributed=True,
            num_devices=4,
            precision="fp16"
        ))
    else:
        # Run with default settings
        results.append(run_search_benchmark(
            batch_size=batch_size,
            num_actions=num_actions,
            num_simulations=num_simulations,
            search_type=search_type,
            use_t4=True,
            distributed=False,
            precision="fp16"
        ))
    
    # Print results
    print_table(results)
    
    # Print detailed JSON results
    print("Detailed results:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()