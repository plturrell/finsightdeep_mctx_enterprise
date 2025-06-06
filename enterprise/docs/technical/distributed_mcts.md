# Distributed MCTS

This guide covers the distributed Monte Carlo Tree Search (MCTS) implementation in the MCTX library, which allows scaling MCTS across multiple devices for larger and more complex search problems.

## Overview

The distributed MCTS implementation in MCTX enables:
- Scaling search across multiple GPUs or TPUs
- Handling larger batch sizes and more simulations
- Reducing wall-clock time for intensive search operations
- Supporting both data and model parallelism strategies

## Core Components

The distributed implementation is built around the following key components:

### DistributedConfig

```python
from mctx import DistributedConfig

# Create a configuration for 8 devices
config = DistributedConfig(
    num_devices=8,
    batch_split_strategy="even",
    result_merge_strategy="value_sum",
    simulation_allocation="proportional",
    communication_mode="sync"
)
```

Configuration options include:

| Parameter | Description | Options |
|-----------|-------------|---------|
| `num_devices` | Number of devices to distribute across | Integer (default: available device count) |
| `batch_split_strategy` | How to split batches across devices | "even", "proportional", "dynamic" |
| `result_merge_strategy` | How to merge search results | "value_sum", "visit_weighted", "max_value" |
| `simulation_allocation` | How to allocate simulations | "equal", "proportional", "adaptive" |
| `communication_mode` | Communication pattern | "sync", "async" |

### distribute_mcts Decorator

The primary interface is the `distribute_mcts` decorator:

```python
import mctx

@mctx.distribute_mcts(config=DistributedConfig(num_devices=4))
def run_distributed_search(params, rng_key, root, recurrent_fn):
    return mctx.muzero_policy(
        params, 
        rng_key, 
        root, 
        recurrent_fn,
        num_simulations=1024
    )

# Run distributed search
policy_output = run_distributed_search(params, rng_key, root, recurrent_fn)
```

The decorator:
1. Splits the search batch across available devices
2. Manages PRNG key splitting for randomness
3. Coordinates the search process across devices
4. Merges results into a consistent policy output

## Implementation Details

### Data Partitioning

MCTX supports different strategies for partitioning data across devices:

#### Even Split

Divides batches equally across all devices:

```python
config = DistributedConfig(batch_split_strategy="even")
```

This works well when all devices have equal computational capacity.

#### Proportional Split

Divides batches based on device performance characteristics:

```python
config = DistributedConfig(
    batch_split_strategy="proportional",
    device_weights=[1.0, 1.2, 0.8, 1.0]  # Optional custom weights
)
```

This is useful for heterogeneous device setups.

#### Dynamic Split

Adjusts batch allocation during computation based on device utilization:

```python
config = DistributedConfig(
    batch_split_strategy="dynamic",
    rebalance_interval=10  # Rebalance every 10 batches
)
```

Provides maximum efficiency but with additional coordination overhead.

### Result Merging

Different strategies for merging search results from multiple devices:

#### Value Sum

```python
config = DistributedConfig(result_merge_strategy="value_sum")
```

Aggregates value estimates by summing, suitable for most search scenarios.

#### Visit Weighted

```python
config = DistributedConfig(result_merge_strategy="visit_weighted")
```

Weights values by visit counts, providing more emphasis on heavily explored branches.

#### Max Value

```python
config = DistributedConfig(result_merge_strategy="max_value")
```

Takes the maximum value across devices, useful for optimistic search policies.

### Communication Patterns

#### Synchronous

```python
config = DistributedConfig(communication_mode="sync")
```

All devices synchronize at predefined points, ensuring consistent search trees.

#### Asynchronous

```python
config = DistributedConfig(
    communication_mode="async",
    sync_interval=16  # Optional sync every N simulations
)
```

Devices operate independently with periodic synchronization, maximizing throughput.

## Integration with PJIT (Partitioned JIT)

For very large models, MCTX's distributed implementation can leverage JAX's PJIT for model parallelism:

```python
import jax
from jax.experimental import pjit
from jax.sharding import Mesh, PartitionSpec as P
import mctx

# Create device mesh
devices = jax.devices()
mesh = Mesh(devices, axis_names=('data', 'model'))

# Define partitioning strategy
partition_spec = {
    'embedding': P('model', None),
    'logits': P(None, 'data'),
    'value_head': P('model', None)
}

# Configure distributed search with PJIT integration
config = DistributedConfig(
    num_devices=len(devices),
    use_pjit=True,
    pjit_mesh=mesh,
    pjit_partition_spec=partition_spec
)

@mctx.distribute_mcts(config=config)
def run_distributed_search(params, rng_key, root, recurrent_fn):
    return mctx.muzero_policy(
        params, 
        rng_key, 
        root, 
        recurrent_fn,
        num_simulations=2048
    )
```

This enables:
- Distribution of very large models across multiple devices
- Combination of data and model parallelism
- Efficient handling of models too large for a single device

## Performance Monitoring

The distributed implementation includes built-in performance monitoring:

```python
from mctx.distributed import DistributedPerformanceMonitor

# Create monitor
monitor = DistributedPerformanceMonitor()

# Configure distributed search with monitoring
config = DistributedConfig(
    num_devices=8,
    performance_monitor=monitor
)

@mctx.distribute_mcts(config=config)
def run_distributed_search(params, rng_key, root, recurrent_fn):
    return mctx.muzero_policy(
        params, 
        rng_key, 
        root, 
        recurrent_fn,
        num_simulations=1024
    )

# Run search
policy_output = run_distributed_search(params, rng_key, root, recurrent_fn)

# Print performance metrics
print(f"Total time: {monitor.total_time:.2f}s")
print(f"Communication overhead: {monitor.communication_overhead:.2f}s")
print(f"Load balance ratio: {monitor.load_balance_ratio:.2f}")
print(f"Simulation throughput: {monitor.simulation_throughput:.2f} sims/second")

# Get per-device metrics
device_metrics = monitor.get_device_metrics()
for i, metrics in enumerate(device_metrics):
    print(f"Device {i}: {metrics.simulation_count} simulations, "
          f"{metrics.computation_time:.2f}s compute time")
```

## Fault Tolerance

The distributed implementation includes fault tolerance features:

```python
config = DistributedConfig(
    num_devices=8,
    fault_tolerance=True,
    fault_tolerance_strategy="skip_failed",
    fallback_to_single_device=True
)
```

Options include:
- `skip_failed`: Continue with available devices if some fail
- `retry_failed`: Attempt to retry operations on failed devices
- `fallback_to_single_device`: Revert to single-device execution if distributed mode fails

## Example: Large-Scale AlphaZero

Here's a complete example for running a large-scale AlphaZero-style search across multiple devices:

```python
import jax
import mctx
from mctx.distributed import DistributedConfig, DistributedPerformanceMonitor

# Setup parameters, root state, and recurrent function
params = ...
root = ...
recurrent_fn = ...

# Create performance monitor
monitor = DistributedPerformanceMonitor()

# Configure distributed search
config = DistributedConfig(
    num_devices=jax.device_count(),
    batch_split_strategy="proportional",
    result_merge_strategy="visit_weighted",
    simulation_allocation="adaptive",
    communication_mode="sync",
    performance_monitor=monitor
)

# Define distributed search function
@mctx.distribute_mcts(config=config)
def run_alphazero_search(params, rng_key, root, recurrent_fn):
    return mctx.muzero_policy(
        params, 
        rng_key, 
        root, 
        recurrent_fn,
        num_simulations=2048,
        dirichlet_fraction=0.25,
        dirichlet_alpha=0.3,
        pb_c_init=1.25,
        pb_c_base=19652,
        temperature=1.0
    )

# Run distributed search
rng_key = jax.random.PRNGKey(0)
policy_output = run_alphazero_search(params, rng_key, root, recurrent_fn)

# Use the policy output
action = policy_output.action
action_weights = policy_output.action_weights

# Print performance metrics
print(f"Total time: {monitor.total_time:.2f}s")
print(f"Simulation throughput: {monitor.simulation_throughput:.2f} sims/second")
print(f"Speedup vs. single device: {monitor.speedup_vs_single_device:.2f}x")
```

## Multi-Host Distribution

For distribution across multiple hosts (e.g., multiple machines in a cluster):

```python
from mctx.distributed import MultiHostConfig

# Configure multi-host distribution
multi_host_config = MultiHostConfig(
    num_hosts=4,
    host_communication="grpc",
    host_addresses=["host1:8470", "host2:8470", "host3:8470", "host4:8470"]
)

# Include in distributed config
config = DistributedConfig(
    num_devices=8,  # Devices per host
    multi_host=multi_host_config
)
```

This enables:
- Scaling across multiple machines
- Coordinating search across a cluster
- Handling very large search spaces

For more examples, see the [`examples/distributed_mcts_demo.py`](https://github.com/google-deepmind/mctx/blob/main/examples/distributed_mcts_demo.py) file.