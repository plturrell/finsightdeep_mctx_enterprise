#!/usr/bin/env python3
"""
MCTX Memory Optimization Demo for Large Trees

This script demonstrates the memory optimizations for handling large MCTS trees,
including batched serialization and incremental loading with SAP HANA.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple

# Add the project root to the path so we can import the MCTX modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mctx.enterprise.hana_integration import HanaConnection, HanaConnectionConfig
from mctx.enterprise.batched_serialization import BatchedTreeSerializer, batch_serialize_tree, batch_deserialize_tree
from mctx.enterprise.incremental_loader import (
    IncrementalTreeLoader, 
    load_tree_by_pages, 
    load_tree_by_depth,
    load_subtree,
    load_path_to_node,
    load_high_value_nodes
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("memory_optimization_demo")


def generate_large_tree(node_count: int, branching_factor: int = 4, max_depth: int = 20) -> Dict[str, Any]:
    """Generate a large synthetic tree for testing.
    
    Args:
        node_count: Target number of nodes to generate.
        branching_factor: Number of children per node.
        max_depth: Maximum depth of the tree.
        
    Returns:
        The generated tree structure.
    """
    logger.info(f"Generating synthetic tree with ~{node_count} nodes")
    start_time = time.time()
    
    tree_id = str(uuid.uuid4())
    nodes = []
    
    # Create root node
    root_id = str(uuid.uuid4())
    root_node = {
        'id': root_id,
        'tree_id': tree_id,
        'parent_id': '',
        'visit_count': 1000,
        'value': 0.5,
        'state': {'position': 'root'},
        'action': None
    }
    nodes.append(root_node)
    
    # Track node IDs by level for easier parent assignment
    nodes_by_level: List[List[str]] = [[root_id]]
    
    # Generate tree level by level
    current_level = 0
    while len(nodes) < node_count and current_level < max_depth:
        level_nodes = []
        parent_level = nodes_by_level[current_level]
        
        for parent_id in parent_level:
            # Generate children for this parent
            child_count = min(branching_factor, (node_count - len(nodes)) // max(1, len(parent_level)))
            if child_count <= 0:
                break
                
            for _ in range(child_count):
                node_id = str(uuid.uuid4())
                visit_count = random.randint(1, 100)
                value = random.random()
                
                node = {
                    'id': node_id,
                    'tree_id': tree_id,
                    'parent_id': parent_id,
                    'visit_count': visit_count,
                    'value': value,
                    'state': {'position': f'level_{current_level+1}_node', 'depth': current_level+1},
                    'action': {'move': f'move_to_{node_id}'}
                }
                nodes.append(node)
                level_nodes.append(node_id)
                
                if len(nodes) >= node_count:
                    break
        
        nodes_by_level.append(level_nodes)
        current_level += 1
        
        if not level_nodes:
            break
    
    # Create the tree structure
    tree = {
        'tree_id': tree_id,
        'name': f'Synthetic_Tree_{node_count}_nodes',
        'batch_size': 64,
        'num_actions': branching_factor,
        'num_simulations': node_count,
        'metadata': json.dumps({
            'synthetic': True,
            'node_count': len(nodes),
            'branching_factor': branching_factor,
            'max_depth': max_depth,
            'actual_depth': len(nodes_by_level),
            'generated_at': time.time()
        }),
        'nodes': nodes
    }
    
    generation_time = time.time() - start_time
    logger.info(f"Generated tree with {len(nodes)} nodes in {generation_time:.2f}s")
    
    return tree


def profile_memory_usage(func, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
    """Profile memory usage of a function.
    
    Args:
        func: Function to profile.
        *args: Arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.
        
    Returns:
        Tuple of (function result, memory stats).
    """
    import tracemalloc
    import gc
    
    # Force garbage collection before starting
    gc.collect()
    
    # Start memory tracking
    tracemalloc.start()
    start_time = time.time()
    
    # Run the function
    result = func(*args, **kwargs)
    
    # Capture memory stats
    current, peak = tracemalloc.get_traced_memory()
    end_time = time.time()
    
    # Stop tracking
    tracemalloc.stop()
    
    # Compile stats
    stats = {
        'current_memory_mb': current / (1024 * 1024),
        'peak_memory_mb': peak / (1024 * 1024),
        'execution_time_sec': end_time - start_time
    }
    
    return result, stats


def test_batched_serialization(connection: HanaConnection, 
                               node_count: int, 
                               use_batching: bool = True) -> None:
    """Test batched serialization with a large tree.
    
    Args:
        connection: SAP HANA connection.
        node_count: Number of nodes to generate.
        use_batching: Whether to use batched serialization.
    """
    logger.info(f"{'=' * 40}")
    logger.info(f"Testing {'batched' if use_batching else 'standard'} serialization with {node_count} nodes")
    logger.info(f"{'=' * 40}")
    
    # Generate a large tree
    tree = generate_large_tree(node_count)
    
    # Test serialization
    if use_batching:
        # Use batched serialization
        serializer = BatchedTreeSerializer(connection)
        
        # Profile memory usage
        _, serialize_stats = profile_memory_usage(
            serializer.batch_serialize_tree, tree
        )
        
        tree_id = tree['tree_id']
        
        # Test deserialization
        _, deserialize_stats = profile_memory_usage(
            serializer.batch_deserialize_tree, tree_id
        )
    else:
        # Use standard serialization (from hana_integration)
        from mctx.enterprise.hana_integration import serialize_tree, deserialize_tree
        
        # Profile memory usage
        _, serialize_stats = profile_memory_usage(
            serialize_tree, connection, tree
        )
        
        tree_id = tree['tree_id']
        
        # Test deserialization
        _, deserialize_stats = profile_memory_usage(
            deserialize_tree, connection, tree_id, True
        )
    
    # Print stats
    logger.info(f"Serialization stats:")
    logger.info(f"  Peak memory: {serialize_stats['peak_memory_mb']:.2f} MB")
    logger.info(f"  Execution time: {serialize_stats['execution_time_sec']:.2f}s")
    
    logger.info(f"Deserialization stats:")
    logger.info(f"  Peak memory: {deserialize_stats['peak_memory_mb']:.2f} MB")
    logger.info(f"  Execution time: {deserialize_stats['execution_time_sec']:.2f}s")


def test_incremental_loading(connection: HanaConnection, tree_id: str) -> None:
    """Test incremental loading with a large tree.
    
    Args:
        connection: SAP HANA connection.
        tree_id: ID of the tree to use.
    """
    logger.info(f"{'=' * 40}")
    logger.info(f"Testing incremental loading with tree {tree_id}")
    logger.info(f"{'=' * 40}")
    
    loader = IncrementalTreeLoader(connection)
    
    # Get tree metadata
    tree_metadata = loader.get_tree_metadata(tree_id)
    metadata = json.loads(tree_metadata.get('metadata', '{}'))
    node_count = metadata.get('node_count', 0)
    
    logger.info(f"Tree {tree_id} has {node_count} nodes")
    
    # Test different loading strategies
    
    # 1. Page-based loading
    logger.info(f"\nTesting page-based loading:")
    page_size = 1000
    
    # Get first page
    _, stats = profile_memory_usage(
        loader.load_tree_by_pages, tree_id, None
    )
    
    logger.info(f"  First page loading:")
    logger.info(f"    Peak memory: {stats['peak_memory_mb']:.2f} MB")
    logger.info(f"    Execution time: {stats['execution_time_sec']:.2f}s")
    
    # 2. Depth-based loading
    logger.info(f"\nTesting depth-based loading:")
    max_depth = 3
    
    _, stats = profile_memory_usage(
        loader.load_tree_by_depth, tree_id, max_depth
    )
    
    logger.info(f"  Depth-based loading (depth={max_depth}):")
    logger.info(f"    Peak memory: {stats['peak_memory_mb']:.2f} MB")
    logger.info(f"    Execution time: {stats['execution_time_sec']:.2f}s")
    
    # 3. Subtree loading
    logger.info(f"\nTesting subtree loading:")
    
    # Get a non-root node
    root_nodes = loader._get_root_nodes(tree_id)
    if root_nodes:
        root_id = root_nodes[0]['id']
        children = loader._get_node_children(tree_id, root_id)
        
        if children:
            child_id = children[0]['id']
            max_depth = 2
            max_nodes = 100
            
            _, stats = profile_memory_usage(
                loader.load_subtree, tree_id, child_id, max_depth, max_nodes
            )
            
            logger.info(f"  Subtree loading (node={child_id}, depth={max_depth}, max_nodes={max_nodes}):")
            logger.info(f"    Peak memory: {stats['peak_memory_mb']:.2f} MB")
            logger.info(f"    Execution time: {stats['execution_time_sec']:.2f}s")
    
    # 4. High-value nodes loading
    logger.info(f"\nTesting high-value nodes loading:")
    max_nodes = 50
    
    _, stats = profile_memory_usage(
        loader.load_high_value_nodes, tree_id, max_nodes
    )
    
    logger.info(f"  High-value nodes loading (max_nodes={max_nodes}):")
    logger.info(f"    Peak memory: {stats['peak_memory_mb']:.2f} MB")
    logger.info(f"    Execution time: {stats['execution_time_sec']:.2f}s")


def test_comparison(connection: HanaConnection, node_count: int) -> None:
    """Compare standard loading vs optimized loading.
    
    Args:
        connection: SAP HANA connection.
        node_count: Number of nodes to generate.
    """
    logger.info(f"{'=' * 40}")
    logger.info(f"Comparing standard vs optimized loading with {node_count} nodes")
    logger.info(f"{'=' * 40}")
    
    # Generate a large tree
    tree = generate_large_tree(node_count)
    tree_id = tree['tree_id']
    
    # Save the tree using batched serialization
    serializer = BatchedTreeSerializer(connection)
    serializer.batch_serialize_tree(tree)
    
    # Test standard loading
    from mctx.enterprise.hana_integration import deserialize_tree
    
    logger.info(f"\nStandard loading (entire tree at once):")
    _, standard_stats = profile_memory_usage(
        deserialize_tree, connection, tree_id, True
    )
    
    logger.info(f"  Peak memory: {standard_stats['peak_memory_mb']:.2f} MB")
    logger.info(f"  Execution time: {standard_stats['execution_time_sec']:.2f}s")
    
    # Test optimized loading (by depth)
    logger.info(f"\nOptimized loading (by depth, max_depth=3):")
    loader = IncrementalTreeLoader(connection)
    _, optimized_stats = profile_memory_usage(
        loader.load_tree_by_depth, tree_id, 3
    )
    
    logger.info(f"  Peak memory: {optimized_stats['peak_memory_mb']:.2f} MB")
    logger.info(f"  Execution time: {optimized_stats['execution_time_sec']:.2f}s")
    
    # Calculate improvements
    memory_improvement = (standard_stats['peak_memory_mb'] - optimized_stats['peak_memory_mb']) / standard_stats['peak_memory_mb'] * 100
    time_improvement = (standard_stats['execution_time_sec'] - optimized_stats['execution_time_sec']) / standard_stats['execution_time_sec'] * 100
    
    logger.info(f"\nImprovements:")
    logger.info(f"  Memory usage: {memory_improvement:.1f}% reduction")
    logger.info(f"  Execution time: {time_improvement:.1f}% {'reduction' if time_improvement > 0 else 'increase'}")


def main():
    """Main function to run the demo."""
    parser = argparse.ArgumentParser(description='MCTX Memory Optimization Demo')
    parser.add_argument('--host', required=True, help='SAP HANA host')
    parser.add_argument('--port', type=int, default=30015, help='SAP HANA port')
    parser.add_argument('--user', required=True, help='SAP HANA username')
    parser.add_argument('--password', required=True, help='SAP HANA password')
    parser.add_argument('--schema', default='MCTX', help='Schema name')
    parser.add_argument('--node-count', type=int, default=10000, help='Number of nodes for test trees')
    parser.add_argument('--test', choices=['serialization', 'loading', 'comparison', 'all'], 
                        default='all', help='Test to run')
    parser.add_argument('--tree-id', help='Tree ID to use for loading tests (optional)')
    
    args = parser.parse_args()
    
    # Create connection config
    config = HanaConnectionConfig(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        schema=args.schema
    )
    
    # Create connection
    connection = HanaConnection(config)
    
    try:
        # Test connection
        if not connection.test_connection():
            logger.error("Could not connect to SAP HANA")
            return
        
        logger.info("Connected to SAP HANA successfully")
        
        # Run selected tests
        if args.test in ('serialization', 'all'):
            # Test with and without batching
            test_batched_serialization(connection, args.node_count, use_batching=True)
            test_batched_serialization(connection, args.node_count, use_batching=False)
        
        if args.test in ('loading', 'all'):
            # Use provided tree ID or get one from serialization test
            tree_id = args.tree_id
            if not tree_id:
                # Generate and serialize a tree
                tree = generate_large_tree(args.node_count)
                serializer = BatchedTreeSerializer(connection)
                tree_id = serializer.batch_serialize_tree(tree)
            
            test_incremental_loading(connection, tree_id)
        
        if args.test in ('comparison', 'all'):
            test_comparison(connection, args.node_count)
        
    except Exception as e:
        logger.error(f"Error in demo: {e}")
        raise


if __name__ == "__main__":
    main()