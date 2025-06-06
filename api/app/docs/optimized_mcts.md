# Optimized MCTS Service

This document describes the optimized MCTS service capabilities, including T4 GPU optimizations and distributed MCTS across multiple devices.

## T4 GPU Optimizations

The MCTS service now supports optimizations specifically for NVIDIA T4 GPUs. These optimizations include:

- Mixed precision (FP16) computation
- Tensor core alignment
- Memory usage monitoring
- Optimized matrix operations
- Efficient memory layout

### Using T4 Optimizations

To use T4 optimizations, set the following parameters in your `SearchParams`:

```python
search_params = SearchParams(
    # Standard search parameters
    num_simulations=128,
    max_depth=50,
    
    # T4 optimization parameters
    use_t4_optimizations=True,
    precision="fp16",  # Use "fp16" or "fp32"
    tensor_core_aligned=True,  # Align dimensions for tensor cores
)
```

## Distributed MCTS

The service now supports distributed MCTS across multiple GPUs. This enables:

- Scaling to larger batch sizes
- Parallelizing simulations across devices
- Reduced search time for large models
- Automatic result synchronization

### Using Distributed MCTS

To use distributed MCTS, set the following parameters:

```python
search_params = SearchParams(
    # Standard search parameters
    num_simulations=128,
    max_depth=50,
    
    # Distributed parameters
    distributed=True,
    num_devices=4,  # Number of GPUs to use
    partition_batch=True,  # Partition batch across devices
    precision="fp16",  # Use "fp16" for best performance
)

request = MCTSRequest(
    root_input=root_input,
    search_params=search_params,
    search_type="gumbel_muzero",
    device_type="gpu",  # Can be "gpu", "tpu", or "cpu"
)
```

## Performance Monitoring

Both T4-optimized and distributed MCTS implementations include detailed performance metrics that are saved to the HANA database. These metrics include:

- Duration in milliseconds
- Number of expanded nodes
- Maximum depth reached
- Memory usage (for T4)
- Device utilization (for distributed)

You can query these metrics using the HANA connector:

```python
# Get statistics for T4-optimized searches
t4_stats = hana_manager.get_search_history(
    limit=100,
    search_type="gumbel_muzero",
    config_filter="$.optimizations.use_t4 = 'true'"
)

# Get statistics for distributed searches
distributed_stats = hana_manager.get_search_history(
    limit=100,
    search_type="gumbel_muzero",
    config_filter="$.optimizations.distributed = 'true'"
)
```

## Daily Statistics

The system maintains daily statistics on search performance, now including information about optimized searches:

- Total searches
- Average duration
- Average expanded nodes
- Maximum batch size
- Maximum simulation count
- T4-optimized search count
- Distributed search count
- Average number of devices used

These statistics are updated automatically after each search and can be accessed through the HANA connector.

## Example Usage

See the `examples/mcts_service_demo.py` script for a complete example of using both T4-optimized and distributed MCTS through the service API.

## Implementation Details

- T4 optimizations are implemented in `mctx._src.t4_search.py` and `mctx._src.t4_optimizations.py`
- Distributed MCTS is implemented in `mctx._src.distributed.py`
- Service integration is in `api/app/services/mcts_service.py`
- Database schema in `api/app/db/hana_connector.py` includes fields for tracking optimization metrics