#!/usr/bin/env python3
"""
MCTX Monitoring and Visualization Demo

This example demonstrates the comprehensive monitoring and visualization
capabilities for Monte Carlo Tree Search algorithms.
"""

import os
import sys
import time
import argparse
import logging
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from mctx import PolicyType, RootFnOutput, SearchResults, search, gumbel_muzero_policy

# Import monitoring components
from mctx.monitoring import (
    MCTSMetricsCollector,
    MCTSMonitor,
    TreeVisualizer,
    PerformanceProfiler,
    ResourceMonitor
)
from mctx.monitoring.visualization import VisualizationConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mctx.examples.monitoring_demo")


def create_root_fn_output(batch_size: int, num_actions: int, key) -> RootFnOutput:
    """
    Create mock root function output.
    
    Args:
        batch_size: Batch size
        num_actions: Number of actions
        key: Random key
        
    Returns:
        Root function output
    """
    # Create random key for value
    value_key, prior_logits_key = jax.random.split(key)
    
    # Generate random value between -1 and 1
    value = jax.random.uniform(
        value_key, shape=(batch_size,), minval=-1.0, maxval=1.0
    )
    
    # Generate random policy logits
    prior_logits = jax.random.normal(
        prior_logits_key, shape=(batch_size, num_actions)
    )
    
    # Create recurrent state (embedding) - zeros for simplicity
    embedding = jnp.zeros((batch_size, 16))
    
    return RootFnOutput(
        prior_logits=prior_logits,
        value=value,
        embedding=embedding
    )


def create_recurrent_fn(key):
    """
    Create a mock recurrent function.
    
    Args:
        key: Random key
        
    Returns:
        Recurrent function
    """
    def recurrent_fn(embeddings, actions):
        # Create random keys
        value_key, prior_logits_key, embedding_key = jax.random.split(key, 3)
        
        # Get batch size and num_actions from shapes
        batch_size = embeddings.shape[0]
        num_actions = 16  # Fixed for this example
        
        # Generate random values between -1 and 1
        value = jax.random.uniform(
            value_key, shape=(batch_size,), minval=-1.0, maxval=1.0
        )
        
        # Generate random policy logits
        prior_logits = jax.random.normal(
            prior_logits_key, shape=(batch_size, num_actions)
        )
        
        # Create new embedding (random for variety)
        new_embedding = jax.random.normal(
            embedding_key, shape=(batch_size, 16)
        ) * 0.1 + embeddings
        
        # Add reward of 0.1 for each step
        reward = jnp.ones((batch_size,)) * 0.1
        
        # Always valid actions
        is_terminal = jnp.zeros((batch_size,), dtype=jnp.bool_)
        
        return RootFnOutput(
            prior_logits=prior_logits,
            value=value,
            embedding=new_embedding,
            reward=reward,
            is_terminal=is_terminal
        )
    
    return recurrent_fn


def run_mcts_with_monitoring(
    batch_size: int = 1,
    num_actions: int = 16,
    num_simulations: int = 50,
    max_depth: Optional[int] = None,
    random_seed: int = 42,
    enable_metrics: bool = True,
    enable_profiling: bool = True,
    enable_resource_monitoring: bool = True,
    save_visualizations: bool = False,
    visualization_dir: str = 'mcts_visualizations',
    jupyter_mode: bool = False
) -> Tuple[SearchResults, MCTSMonitor]:
    """
    Run MCTS search with comprehensive monitoring.
    
    Args:
        batch_size: Batch size for search
        num_actions: Number of actions
        num_simulations: Number of simulations to run
        max_depth: Maximum search depth
        random_seed: Random seed
        enable_metrics: Whether to enable metrics collection
        enable_profiling: Whether to enable performance profiling
        enable_resource_monitoring: Whether to enable resource monitoring
        save_visualizations: Whether to save visualizations
        visualization_dir: Directory to save visualizations in
        jupyter_mode: Whether running in Jupyter (for interactive visualizations)
        
    Returns:
        Tuple of (search results, monitor)
    """
    # Create random key
    key = jax.random.PRNGKey(random_seed)
    
    # Initialize monitoring components
    metrics_collector = MCTSMetricsCollector(
        collect_interval_ms=500,
        resource_monitoring=enable_resource_monitoring
    ) if enable_metrics else None
    
    profiler = PerformanceProfiler(
        track_memory=enable_resource_monitoring,
        track_gpu_memory=enable_resource_monitoring and 'gpu' in str(jax.devices()[0])
    ) if enable_profiling else None
    
    resource_monitor = ResourceMonitor(
        collect_interval_ms=1000,
        track_cpu=enable_resource_monitoring,
        track_memory=enable_resource_monitoring,
        track_gpu='gpu' in str(jax.devices()[0]) and enable_resource_monitoring
    ) if enable_resource_monitoring else None
    
    # Initialize visualization components
    viz_config = VisualizationConfig(
        width=900,
        height=700,
        theme='light',
        layout_type='radial',
        animation_duration=500,
        save_path=os.path.join(visualization_dir, 'mcts_tree.html') if save_visualizations else None
    )
    
    monitor = MCTSMonitor(
        config=viz_config,
        metrics_collector=metrics_collector,
        auto_update_interval=1000
    )
    
    # Create directory for visualizations if needed
    if save_visualizations:
        os.makedirs(visualization_dir, exist_ok=True)
    
    # Create root and recurrent functions
    root_fn_output = create_root_fn_output(batch_size, num_actions, key)
    recurrent_fn = create_recurrent_fn(key)
    
    # Start resource monitoring if enabled
    if resource_monitor:
        resource_monitor.start_collection()
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Profile the search if profiling is enabled
    if profiler:
        with profiler.profile_section("mcts_search"):
            # Run search
            search_results = search(
                root=root_fn_output,
                recurrent_fn=recurrent_fn,
                num_simulations=num_simulations,
                max_depth=max_depth,
                max_num_considered_actions=num_actions,
                policy_type=PolicyType.GUMBEL_MUZERO,
                dirichlet_fraction=0.25,
                dirichlet_alpha=0.3,
                pb_c_init=1.25,
                temperature=1.0
            )
    else:
        # Run search without profiling
        search_results = search(
            root=root_fn_output,
            recurrent_fn=recurrent_fn,
            num_simulations=num_simulations,
            max_depth=max_depth,
            max_num_considered_actions=num_actions,
            policy_type=PolicyType.GUMBEL_MUZERO,
            dirichlet_fraction=0.25,
            dirichlet_alpha=0.3,
            pb_c_init=1.25,
            temperature=1.0
        )
    
    # Update monitor with search tree
    monitor.update_tree(search_results.search_tree)
    
    # Stop monitoring
    final_metrics = monitor.stop_monitoring()
    
    # Stop resource monitoring
    if resource_monitor:
        snapshots = resource_monitor.stop_collection()
        
        # Log peak resource usage
        peak_cpu = resource_monitor.get_peak_usage(resource_monitor.ResourceType.CPU)
        peak_memory = resource_monitor.get_peak_usage(resource_monitor.ResourceType.MEMORY)
        memory_increase = resource_monitor.get_memory_increase()
        
        logger.info(f"Peak CPU usage: {peak_cpu:.1f}%")
        logger.info(f"Peak memory usage: {peak_memory:.1f}%")
        logger.info(f"Memory increase: {memory_increase:.2f} MB")
        
        # Plot resource usage if not in Jupyter mode
        if not jupyter_mode:
            fig = resource_monitor.plot_resource_usage()
            if fig and save_visualizations:
                fig.savefig(os.path.join(visualization_dir, 'resource_usage.png'))
    
    # Print profiler summary
    if profiler:
        profiler.print_profile_summary()
        
        # Plot profile summary if not in Jupyter mode
        if not jupyter_mode:
            fig = profiler.plot_profile_summary()
            if fig and save_visualizations:
                fig.savefig(os.path.join(visualization_dir, 'profile_summary.png'))
    
    # Print metrics summary
    if metrics_collector:
        logger.info(f"Search metrics summary:")
        logger.info(f"- Search time: {final_metrics.search_time_ms:.1f} ms")
        logger.info(f"- Tree size: {final_metrics.tree_size} nodes")
        logger.info(f"- Max depth: {final_metrics.max_depth}")
        logger.info(f"- Branching factor: {final_metrics.branching_factor:.2f}")
        logger.info(f"- Iterations per second: {final_metrics.iterations_per_second:.1f}")
    
    # Create and save visualizations if not in Jupyter mode
    if not jupyter_mode and save_visualizations:
        # Create tree visualizer
        visualizer = TreeVisualizer(viz_config)
        
        # Convert tree to visualization data
        tree_data = visualizer.tree_to_visualization_data(search_results.search_tree)
        
        # Create visualizations
        tree_fig = visualizer.visualize_tree(tree_data)
        metrics_fig = visualizer.create_metrics_panel(tree_data)
        
        # Create metrics history visualization if we have history
        if monitor.metrics_history:
            history_fig = visualizer.visualize_metrics_over_time(monitor.metrics_history)
    
    return search_results, monitor


def jupyter_interactive_demo(batch_size: int = 1,
                           num_actions: int = 16,
                           num_simulations: int = 50):
    """
    Run an interactive demo in Jupyter.
    
    This function creates a dashboard for real-time monitoring
    of MCTS processes.
    
    Args:
        batch_size: Batch size for search
        num_actions: Number of actions
        num_simulations: Number of simulations to run
    """
    # Check if running in Jupyter
    try:
        from IPython import get_ipython
        if get_ipython() is None:
            print("This function is designed to run in a Jupyter notebook.")
            return
    except ImportError:
        print("IPython not available. This function is designed to run in a Jupyter notebook.")
        return
    
    # Run MCTS with monitoring
    search_results, monitor = run_mcts_with_monitoring(
        batch_size=batch_size,
        num_actions=num_actions,
        num_simulations=num_simulations,
        jupyter_mode=True
    )
    
    # Create interactive dashboard
    monitor.create_dashboard(height=1000)
    
    # Return the monitor so users can interact with it
    return monitor


def main():
    """Run the MCTS monitoring demo."""
    parser = argparse.ArgumentParser(description="MCTX Monitoring Demo")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--num-actions", type=int, default=16, help="Number of actions")
    parser.add_argument("--num-simulations", type=int, default=50, help="Number of simulations")
    parser.add_argument("--max-depth", type=int, default=None, help="Maximum search depth")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed")
    parser.add_argument("--disable-metrics", action="store_true", help="Disable metrics collection")
    parser.add_argument("--disable-profiling", action="store_true", help="Disable performance profiling")
    parser.add_argument("--disable-resource-monitoring", action="store_true", help="Disable resource monitoring")
    parser.add_argument("--save-visualizations", action="store_true", help="Save visualizations")
    parser.add_argument("--visualization-dir", type=str, default="mcts_visualizations", help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    logger.info("Starting MCTX Monitoring Demo")
    logger.info(f"JAX devices: {jax.devices()}")
    
    # Run MCTS with monitoring
    search_results, monitor = run_mcts_with_monitoring(
        batch_size=args.batch_size,
        num_actions=args.num_actions,
        num_simulations=args.num_simulations,
        max_depth=args.max_depth,
        random_seed=args.random_seed,
        enable_metrics=not args.disable_metrics,
        enable_profiling=not args.disable_profiling,
        enable_resource_monitoring=not args.disable_resource_monitoring,
        save_visualizations=args.save_visualizations,
        visualization_dir=args.visualization_dir
    )
    
    logger.info("MCTS search completed")
    logger.info(f"Search tree size: {len(search_results.search_tree.node_values)}")
    logger.info(f"Selected actions: {search_results.action}")
    
    if args.save_visualizations:
        logger.info(f"Visualizations saved to {args.visualization_dir}")


if __name__ == "__main__":
    main()