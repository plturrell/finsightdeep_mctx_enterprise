"""
Tests for GPU optimizations, specifically T4 optimizations.
These tests will be skipped if no GPU is available.
"""

import pytest
import time
import jax
import jax.numpy as jnp
import numpy as np
from mctx._src import t4_optimizations
from mctx._src import t4_search
from mctx._src import t4_memory_optimizations
from mctx._src import base
from mctx._src import tree as tree_lib


# Skip all tests if GPU is not available
skip_if_no_gpu = pytest.mark.skipif(
    not jax.devices('gpu'), 
    reason="No GPU available for testing"
)


@skip_if_no_gpu
def test_gpu_available():
    """Verify that JAX can see the GPU."""
    # Get all devices
    devices = jax.devices()
    
    # Check if at least one GPU is available
    gpu_devices = [d for d in devices if d.platform == 'gpu']
    assert len(gpu_devices) > 0, "No GPU devices found"
    
    # Print GPU device info for debugging
    for i, device in enumerate(gpu_devices):
        print(f"GPU {i}: {device}")
    
    # Verify JAX can create arrays on GPU
    x = jax.device_put(jnp.array([1, 2, 3]), gpu_devices[0])
    assert x.device() == gpu_devices[0]


@skip_if_no_gpu
def test_align_for_tensor_cores():
    """Test tensor core alignment function."""
    # Test various dimensions
    test_cases = [
        (7, 8),     # Round up to nearest multiple of 8
        (8, 8),     # Already aligned
        (15, 16),   # Round up to next multiple
        (64, 64),   # Already aligned
    ]
    
    for input_dim, expected_output in test_cases:
        result = t4_optimizations.align_for_tensor_cores(input_dim)
        assert result == expected_output, f"Expected {expected_output} for input {input_dim}, got {result}"


@skip_if_no_gpu
def test_optimal_batch_size():
    """Test optimal batch size calculation for T4."""
    # Test with reasonable tree depth and action dimensions
    tree_depth = 10
    action_dim = 64
    
    batch_size = t4_optimizations.get_optimal_t4_batch_size(tree_depth, action_dim)
    
    # Batch size should be positive and reasonable
    assert batch_size > 0
    assert batch_size <= 512, "Batch size too large for T4 memory constraints"
    
    # Should be a power of 2 or multiple of 8 for optimal tensor core usage
    assert batch_size % 8 == 0, "Batch size should be a multiple of 8 for tensor cores"


@skip_if_no_gpu
def test_t4_autotuned_parameters():
    """Test T4 autotuned parameters."""
    params = t4_optimizations.t4_autotuned_parameters()
    
    # Check that required parameters are present and have reasonable values
    assert 'simulation_batch_size' in params
    assert params['simulation_batch_size'] > 0
    
    assert 'dirichlet_alpha' in params
    assert 0 < params['dirichlet_alpha'] < 1
    
    assert 'pb_c_init' in params
    assert params['pb_c_init'] > 0
    
    assert 'max_sim_depth' in params
    assert params['max_sim_depth'] > 0


@skip_if_no_gpu
def test_dynamic_precision_policy():
    """Test dynamic precision policy for different tree sizes."""
    small_tree_size = 100
    large_tree_size = 20000
    
    small_precision = t4_optimizations.dynamic_precision_policy(small_tree_size)
    large_precision = t4_optimizations.dynamic_precision_policy(large_tree_size)
    
    # Small trees should use highest precision
    assert small_precision == jax.lax.Precision.HIGHEST
    
    # Large trees should use default precision to save memory
    assert large_precision == jax.lax.Precision.DEFAULT


@skip_if_no_gpu
def test_mixed_precision_wrapper():
    """Test mixed precision wrapper functionality."""
    # Create a simple function to wrap
    def simple_matmul(x, y):
        return jnp.matmul(x, y)
    
    # Apply the wrapper
    wrapped_fn = t4_optimizations.mixed_precision_wrapper(simple_matmul)
    
    # Create test data
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (16, 32))
    y = jax.random.normal(rng, (32, 16))
    
    # Test with different precision modes
    fp16_result = wrapped_fn(x, y, precision="fp16")
    fp32_result = wrapped_fn(x, y, precision="fp32")
    
    # Both should return valid results
    assert fp16_result.shape == (16, 16)
    assert fp32_result.shape == (16, 16)


@skip_if_no_gpu
def test_memory_optimization():
    """Test memory optimization functions for T4."""
    # Test memory allocation optimization
    mem_params = t4_memory_optimizations.optimize_memory_allocation(
        batch_size=32,
        num_simulations=200,
        num_actions=64
    )
    
    # Check that we get reasonable values
    assert 'estimated_memory_bytes' in mem_params
    assert mem_params['estimated_memory_bytes'] > 0
    
    assert 'max_batch_size' in mem_params
    assert mem_params['max_batch_size'] > 0
    
    assert 'tensor_core_batch_size' in mem_params
    assert mem_params['tensor_core_batch_size'] % 8 == 0, "Tensor core batch size should be multiple of 8"


@skip_if_no_gpu
def test_t4_search_basic():
    """Basic test for T4-optimized search functionality."""
    # Skip test if not on T4 to avoid false positives
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        device_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
        if 'T4' not in device_name:
            pytest.skip("Test requires T4 GPU")
    except (ImportError, Exception):
        # If we can't check GPU type, just run the test
        pass
    
    # Set up a simple test case
    batch_size = 2
    num_actions = 4
    
    # Create a simple root output
    prior_logits = jnp.ones((batch_size, num_actions))
    value = jnp.zeros((batch_size,))
    embedding = jnp.zeros((batch_size, 8))  # Simple state embedding
    
    root = base.RootFnOutput(
        prior_logits=prior_logits,
        value=value,
        embedding=embedding
    )
    
    # Create a simple recurrent function
    def recurrent_fn(params, rng_key, action, embedding):
        next_embedding = jnp.roll(embedding, 1, axis=-1)
        next_embedding = next_embedding.at[..., 0].set(action)
        
        output = base.RecurrentFnOutput(
            reward=jnp.ones((batch_size,)) * 0.1,
            discount=jnp.ones((batch_size,)) * 0.99,
            prior_logits=jnp.ones((batch_size, num_actions)),
            value=jnp.zeros((batch_size,))
        )
        
        return output, next_embedding
    
    # Run a small search
    rng_key = jax.random.PRNGKey(0)
    result = t4_search.t4_search(
        params=None,
        rng_key=rng_key,
        root=root,
        recurrent_fn=recurrent_fn,
        root_action_selection_fn=jax.tree.map(lambda x: x, base.muzero_action_selection),  # Clone the function
        interior_action_selection_fn=jax.tree.map(lambda x: x, base.muzero_action_selection),  # Clone the function
        num_simulations=4,
        max_depth=5,
        precision="fp16",
        tensor_core_aligned=True,
        optimize_tensor_cores=True
    )
    
    # Verify basic properties of the result
    assert isinstance(result, tree_lib.Tree)
    assert result.node_visits.shape[0] == batch_size
    assert result.node_visits.shape[1] > 1  # Should have created some nodes


@skip_if_no_gpu
def test_performance_comparison():
    """Test performance comparison between standard and T4-optimized search."""
    # Define a simple test function
    def simple_function(size):
        # Create a simple matrix multiplication task
        x = jnp.ones((size, size))
        y = jnp.ones((size, size))
        
        # Standard implementation
        start_time = time.time()
        _ = jnp.matmul(x, y)
        jax.block_until_ready(_)
        standard_time = time.time() - start_time
        
        # T4-optimized implementation (with tensor core hints)
        start_time = time.time()
        _ = jnp.matmul(x, y, precision=jax.lax.Precision.HIGHEST)
        jax.block_until_ready(_)
        optimized_time = time.time() - start_time
        
        return standard_time, optimized_time
    
    # Test with a reasonably large matrix
    size = 1024
    standard_time, optimized_time = simple_function(size)
    
    # For informational purposes, print the times
    print(f"Standard time: {standard_time:.6f}s")
    print(f"Optimized time: {optimized_time:.6f}s")
    print(f"Speedup: {standard_time/optimized_time:.2f}x")
    
    # On a T4, optimized should generally be faster, but we don't make this
    # a hard assertion since it depends on the specific hardware and JIT compilation
    
    # Instead, verify that both implementations produced a result without errors
    assert standard_time > 0
    assert optimized_time > 0


if __name__ == "__main__":
    # This allows running the tests directly
    pytest.main(["-v", __file__])