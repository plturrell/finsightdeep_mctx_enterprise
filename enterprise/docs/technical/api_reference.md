# MCTX API Reference

This document provides a comprehensive reference for the MCTX API, covering all key functions, classes, and parameters.

## Core MCTS Components

### search

```python
mctx.search(
    params,
    rng_key,
    root,
    recurrent_fn,
    num_simulations,
    max_depth=None,
    max_num_considered_actions=None,
    dirichlet_fraction=None,
    dirichlet_alpha=None,
    pb_c_init=1.25,
    pb_c_base=19652,
    temperature=1.0,
    gumbel_scale=None,
    qtransform=None,
    value_scale=None,
    value_min=None,
    value_max=None,
    clip_value=True,
    maximum_value_scale=0.01,
    batch_size=None
)
```

Low-level generic MCTS search function.

**Parameters:**
- `params`: Parameters for the recurrent function.
- `rng_key`: JAX PRNGKey.
- `root`: Instance of `RootFnOutput` representing the root node.
- `recurrent_fn`: Function with signature `recurrent_fn(params, rng_key, action, embedding)` returning a tuple `(RecurrentFnOutput, new_embedding)`.
- `num_simulations`: Number of simulations to perform.
- `max_depth`: Maximum search depth (None for unlimited).
- `max_num_considered_actions`: Maximum number of actions to consider (None for all).
- `dirichlet_fraction`: Fraction of Dirichlet noise to add to root prior.
- `dirichlet_alpha`: Alpha parameter for Dirichlet distribution.
- `pb_c_init`: PUCT exploration constant init.
- `pb_c_base`: PUCT exploration constant base.
- `temperature`: Temperature for action selection.
- `gumbel_scale`: Scale for Gumbel noise (None for no Gumbel noise).
- `qtransform`: Q-value transform function.
- `value_scale`: Scale for value function.
- `value_min`: Minimum value for clipping.
- `value_max`: Maximum value for clipping.
- `clip_value`: Whether to clip values.
- `maximum_value_scale`: Scale for maximum value.
- `batch_size`: Batch size for parallelized search (None for auto).

**Returns:**
- `PolicyOutput`: Object containing the search results.

### muzero_policy

```python
mctx.muzero_policy(
    params,
    rng_key,
    root,
    recurrent_fn,
    num_simulations,
    max_depth=None,
    max_num_considered_actions=None,
    dirichlet_fraction=None,
    dirichlet_alpha=None,
    pb_c_init=1.25,
    pb_c_base=19652,
    temperature=1.0,
    value_scale=None,
    value_min=None,
    value_max=None,
    clip_value=True,
    maximum_value_scale=0.01,
    batch_size=None
)
```

Runs a MuZero-style MCTS search.

**Parameters:**
- Same as `search` function, with appropriate defaults for MuZero.

**Returns:**
- `PolicyOutput`: Object containing the MuZero-style search results.

### gumbel_muzero_policy

```python
mctx.gumbel_muzero_policy(
    params,
    rng_key,
    root,
    recurrent_fn,
    num_simulations,
    max_depth=None,
    max_num_considered_actions=None,
    dirichlet_fraction=None,
    dirichlet_alpha=None,
    gumbel_scale=1.0,
    value_scale=None,
    value_min=None,
    value_max=None,
    clip_value=True,
    maximum_value_scale=0.01,
    batch_size=None
)
```

Runs a Gumbel MuZero-style MCTS search.

**Parameters:**
- Same as `search` function, with appropriate defaults for Gumbel MuZero.
- `gumbel_scale`: Scale for Gumbel noise (defaults to 1.0).

**Returns:**
- `PolicyOutput`: Object containing the Gumbel MuZero-style search results.

### t4_optimized_search

```python
mctx.t4_optimized_search(
    params,
    rng_key,
    root,
    recurrent_fn,
    num_simulations,
    max_depth=None,
    max_num_considered_actions=None,
    dirichlet_fraction=None,
    dirichlet_alpha=None,
    pb_c_init=1.25,
    pb_c_base=19652,
    temperature=1.0,
    gumbel_scale=None,
    qtransform=None,
    value_scale=None,
    value_min=None,
    value_max=None,
    clip_value=True,
    maximum_value_scale=0.01,
    batch_size=None,
    use_mixed_precision=True,
    precision_policy=None
)
```

Runs an MCTS search optimized for NVIDIA T4 GPUs.

**Parameters:**
- Same as `search` function, plus:
- `use_mixed_precision`: Whether to use mixed precision (FP16) computation.
- `precision_policy`: Optional `T4PrecisionPolicy` object for fine-grained control.

**Returns:**
- `PolicyOutput`: Object containing the search results.

## Distributed MCTS

### DistributedConfig

```python
mctx.DistributedConfig(
    num_devices=None,
    batch_split_strategy="even",
    result_merge_strategy="value_sum",
    simulation_allocation="proportional",
    communication_mode="sync",
    device_weights=None,
    rebalance_interval=None,
    sync_interval=None,
    use_pjit=False,
    pjit_mesh=None,
    pjit_partition_spec=None,
    fault_tolerance=False,
    fault_tolerance_strategy="skip_failed",
    fallback_to_single_device=False,
    performance_monitor=None
)
```

Configuration for distributed MCTS.

**Parameters:**
- `num_devices`: Number of devices to distribute across.
- `batch_split_strategy`: How to split batches ("even", "proportional", "dynamic").
- `result_merge_strategy`: How to merge results ("value_sum", "visit_weighted", "max_value").
- `simulation_allocation`: How to allocate simulations ("equal", "proportional", "adaptive").
- `communication_mode`: Communication pattern ("sync", "async").
- `device_weights`: Optional weights for devices if using "proportional" split.
- `rebalance_interval`: How often to rebalance in "dynamic" mode.
- `sync_interval`: How often to sync in "async" mode.
- `use_pjit`: Whether to use PJIT for model parallelism.
- `pjit_mesh`: Device mesh for PJIT.
- `pjit_partition_spec`: Partition spec for PJIT.
- `fault_tolerance`: Whether to enable fault tolerance.
- `fault_tolerance_strategy`: Strategy for handling failures.
- `fallback_to_single_device`: Whether to fall back to single device on failure.
- `performance_monitor`: Optional performance monitoring object.

### distribute_mcts

```python
@mctx.distribute_mcts(config=None)
def search_function(params, rng_key, root, recurrent_fn, ...):
    ...
```

Decorator for distributing MCTS across multiple devices.

**Parameters:**
- `config`: `DistributedConfig` object or None to use defaults.

**Returns:**
- Decorated function that runs distributed MCTS.

## Data Structures

### RootFnOutput

```python
mctx.RootFnOutput(
    prior_logits,
    value,
    embedding
)
```

Output from the root function, representing the root node of the search tree.

**Parameters:**
- `prior_logits`: Prior logits for action selection.
- `value`: Value estimate for the root state.
- `embedding`: Embedding representing the root state.

### RecurrentFnOutput

```python
mctx.RecurrentFnOutput(
    reward,
    discount,
    prior_logits,
    value
)
```

Output from the recurrent function, representing a node in the search tree.

**Parameters:**
- `reward`: Reward for the transition.
- `discount`: Discount factor for the transition.
- `prior_logits`: Prior logits for action selection.
- `value`: Value estimate for the state.

### PolicyOutput

```python
mctx.PolicyOutput(
    action,
    action_weights,
    search_tree,
    root_value,
    root_embedding,
    q_values,
    visit_counts,
    search_path
)
```

Output from the search functions.

**Parameters:**
- `action`: Selected action.
- `action_weights`: Action probabilities.
- `search_tree`: The search tree.
- `root_value`: Value estimate for the root state.
- `root_embedding`: Embedding for the root state.
- `q_values`: Q-values for each action.
- `visit_counts`: Visit counts for each action.
- `search_path`: Path taken during search.

### Tree

```python
mctx.Tree(
    node_visits,
    node_values,
    node_rewards,
    node_discounts,
    children_indices,
    children_values,
    children_visits,
    children_rewards,
    children_discounts,
    children_prior_logits,
    children_actions,
    embeddings,
    action_space_size
)
```

Represents a search tree.

**Parameters:**
- `node_visits`: Visit counts for each node.
- `node_values`: Value estimates for each node.
- `node_rewards`: Rewards for each node.
- `node_discounts`: Discount factors for each node.
- `children_indices`: Indices of children for each node.
- `children_values`: Value estimates for children.
- `children_visits`: Visit counts for children.
- `children_rewards`: Rewards for children.
- `children_discounts`: Discount factors for children.
- `children_prior_logits`: Prior logits for children.
- `children_actions`: Actions for children.
- `embeddings`: Embeddings for each node.
- `action_space_size`: Size of the action space.

## T4 GPU Optimizations

### T4PrecisionPolicy

```python
mctx.T4PrecisionPolicy(
    use_fp16_for_embeddings=True,
    use_fp16_for_values=True,
    use_fp16_for_logits=False,
    use_fp16_for_rewards=True
)
```

Policy for controlling precision in T4-optimized search.

**Parameters:**
- `use_fp16_for_embeddings`: Whether to use FP16 for embeddings.
- `use_fp16_for_values`: Whether to use FP16 for values.
- `use_fp16_for_logits`: Whether to use FP16 for logits.
- `use_fp16_for_rewards`: Whether to use FP16 for rewards.

### align_for_tensor_cores

```python
mctx.align_for_tensor_cores(matrix, alignment=8)
```

Aligns matrix dimensions for optimal tensor core utilization.

**Parameters:**
- `matrix`: The matrix to align.
- `alignment`: Alignment factor (8 for FP16, 16 for INT8).

**Returns:**
- Aligned matrix.

### get_optimal_t4_batch_size

```python
mctx.get_optimal_t4_batch_size(
    embedding_size,
    recurrent_fn_params_size,
    available_memory=None
)
```

Calculates the optimal batch size for T4 GPUs.

**Parameters:**
- `embedding_size`: Size of state embeddings.
- `recurrent_fn_params_size`: Size of recurrent function parameters.
- `available_memory`: Available GPU memory in bytes (None for auto-detect).

**Returns:**
- Optimal batch size.

### profile_t4_memory_usage

```python
mctx.profile_t4_memory_usage()
```

Context manager for profiling T4 memory usage.

**Returns:**
- Context manager that provides `peak_usage`, `average_utilization`, and `throughput` attributes.

## Q-Transforms

### qtransform_by_parent_and_siblings

```python
mctx.qtransform_by_parent_and_siblings(
    parent_q_value,
    parent_visit_count,
    sibling_q_values,
    sibling_visit_counts,
    child_index,
    total_siblings,
    rescale_values=True
)
```

Q-transform used by MuZero, based on parent and sibling values.

**Parameters:**
- `parent_q_value`: Q-value of parent node.
- `parent_visit_count`: Visit count of parent node.
- `sibling_q_values`: Q-values of sibling nodes.
- `sibling_visit_counts`: Visit counts of sibling nodes.
- `child_index`: Index of the child node.
- `total_siblings`: Total number of siblings.
- `rescale_values`: Whether to rescale values.

**Returns:**
- Transformed Q-value.

### qtransform_completed_by_mix_value

```python
mctx.qtransform_completed_by_mix_value(
    q_value,
    value,
    visit_count,
    total_simulations,
    mix_value_scale=0.5,
    use_softmax=False
)
```

Q-transform that mixes Q-value with a value estimate.

**Parameters:**
- `q_value`: Original Q-value.
- `value`: Value estimate.
- `visit_count`: Visit count.
- `total_simulations`: Total number of simulations.
- `mix_value_scale`: Scale for mixing.
- `use_softmax`: Whether to use softmax for mixing.

**Returns:**
- Transformed Q-value.

## Action Selection

### compute_gumbel_action_weights

```python
mctx.compute_gumbel_action_weights(
    key,
    root_value,
    q_values,
    visit_counts,
    max_num_considered_actions=None,
    gumbel_scale=1.0
)
```

Computes action weights using Gumbel noise, as in Gumbel MuZero.

**Parameters:**
- `key`: JAX random key.
- `root_value`: Value estimate for root state.
- `q_values`: Q-values for actions.
- `visit_counts`: Visit counts for actions.
- `max_num_considered_actions`: Maximum number of actions to consider.
- `gumbel_scale`: Scale for Gumbel noise.

**Returns:**
- Action weights.

### compute_muzero_action_weights

```python
mctx.compute_muzero_action_weights(
    root_value,
    q_values,
    visit_counts,
    max_num_considered_actions=None,
    temperature=1.0
)
```

Computes action weights as in MuZero.

**Parameters:**
- `root_value`: Value estimate for root state.
- `q_values`: Q-values for actions.
- `visit_counts`: Visit counts for actions.
- `max_num_considered_actions`: Maximum number of actions to consider.
- `temperature`: Temperature for action selection.

**Returns:**
- Action weights.

### t4_optimized_puct

```python
mctx.t4_optimized_puct(
    q_values,
    visit_counts,
    prior_logits,
    exploration_weight=1.0
)
```

T4-optimized PUCT score computation.

**Parameters:**
- `q_values`: Q-values for actions.
- `visit_counts`: Visit counts for actions.
- `prior_logits`: Prior logits for actions.
- `exploration_weight`: Weight for exploration term.

**Returns:**
- PUCT scores.

## Visualization

### visualize_tree

```python
mctx.visualization.visualize_tree(
    tree,
    root_state=None,
    show_values=True,
    show_visit_counts=True,
    show_prior_probabilities=True,
    highlight_path=None,
    color_scheme="value",
    layout="radial",
    max_depth=None,
    width=1000,
    height=700,
    node_size="visits",
    edge_width="visits",
    include_controls=True,
    include_metrics=True,
    title="MCTS Visualization",
    color_function=None,
    custom_css=None,
    custom_js=None,
    use_webgl=False,
    load_on_demand=False,
    optimize_for_size=False,
    max_nodes=None,
    simplify_threshold=None
)
```

Visualizes a search tree.

**Parameters:**
- `tree`: The search tree to visualize.
- `root_state`: Optional root state label.
- `show_values`: Whether to show value estimates.
- `show_visit_counts`: Whether to show visit counts.
- `show_prior_probabilities`: Whether to show prior probabilities.
- `highlight_path`: Path to highlight.
- `color_scheme`: Color scheme to use.
- `layout`: Tree layout style.
- `max_depth`: Maximum depth to display.
- `width`: Width in pixels.
- `height`: Height in pixels.
- `node_size`: How to size nodes.
- `edge_width`: How to size edges.
- `include_controls`: Whether to include interactive controls.
- `include_metrics`: Whether to include metrics panel.
- `title`: Visualization title.
- `color_function`: Custom coloring function.
- `custom_css`: Custom CSS.
- `custom_js`: Custom JavaScript.
- `use_webgl`: Whether to use WebGL for rendering.
- `load_on_demand`: Whether to load subtrees on demand.
- `optimize_for_size`: Whether to optimize for size.
- `max_nodes`: Maximum number of nodes to display.
- `simplify_threshold`: Threshold for simplifying subtrees.

**Returns:**
- HTML string containing the visualization.

### MCTSDashboard

```python
mctx.visualization.MCTSDashboard(
    title=None,
    description=None,
    width=1200,
    height=800
)
```

Creates an interactive dashboard for MCTS visualization.

**Methods:**
- `add_tree(tree, name=None, metadata=None)`: Adds a tree to the dashboard.
- `add_metrics_panel(metrics)`: Adds a metrics panel.
- `add_panel(html, title=None)`: Adds a custom panel.
- `add_metrics_table(metrics, columns, column_names=None)`: Adds a metrics table.
- `generate_html()`: Generates the dashboard HTML.

### create_heatmap

```python
mctx.visualization.create_heatmap(
    data,
    title=None,
    x_labels=None,
    y_labels=None,
    colorscale="RdBu",
    min_value=None,
    max_value=None,
    width=600,
    height=400
)
```

Creates a heatmap visualization.

**Parameters:**
- `data`: 2D array of values.
- `title`: Heatmap title.
- `x_labels`: Labels for x-axis.
- `y_labels`: Labels for y-axis.
- `colorscale`: Color scale to use.
- `min_value`: Minimum value for color scale.
- `max_value`: Maximum value for color scale.
- `width`: Width in pixels.
- `height`: Height in pixels.

**Returns:**
- HTML string containing the heatmap.

### animate_search_process

```python
mctx.visualization.animate_search_process(
    trees,
    framerate=1,
    loop=True,
    include_controls=True,
    width=1000,
    height=600
)
```

Creates an animation of the search process.

**Parameters:**
- `trees`: List of trees at different stages.
- `framerate`: Frames per second.
- `loop`: Whether to loop the animation.
- `include_controls`: Whether to include playback controls.
- `width`: Width in pixels.
- `height`: Height in pixels.

**Returns:**
- HTML string containing the animation.

### compare_trees

```python
mctx.visualization.compare_trees(
    trees,
    names=None,
    metrics=None,
    highlight_differences=True,
    width=1200,
    height=800
)
```

Creates a comparative visualization of multiple trees.

**Parameters:**
- `trees`: List of trees to compare.
- `names`: Names for the trees.
- `metrics`: Metrics to compare.
- `highlight_differences`: Whether to highlight differences.
- `width`: Width in pixels.
- `height`: Height in pixels.

**Returns:**
- HTML string containing the comparison.

### export_visualization

```python
mctx.visualization.export_visualization(
    visualization,
    format="png",
    filename=None,
    width=None,
    height=None,
    scale=1
)
```

Exports a visualization to a file.

**Parameters:**
- `visualization`: The visualization HTML.
- `format`: Export format ("png", "svg", "pdf").
- `filename`: Output filename.
- `width`: Width in pixels.
- `height`: Height in pixels.
- `scale`: Scale factor for resolution.

**Returns:**
- Path to the exported file.

## SAP HANA Integration

### HanaConnector

```python
mctx.integrations.hana_connector.HanaConnector(
    host,
    port,
    user,
    password,
    use_ssl=True,
    ssl_validate_cert=False,
    ssl_ca_file=None,
    schema_name="MCTX_DATA",
    enable_performance_logging=False,
    max_tree_size_mb=None,
    compression_level=5
)
```

Connector for SAP HANA database.

**Methods:**
- `test_connection()`: Tests the connection.
- `initialize_schema(schema_name=None, drop_existing=False)`: Initializes the schema.
- `store_search_results(search_id=None, policy_output=None, metadata=None)`: Stores search results.
- `get_search_results(search_id)`: Gets search results by ID.
- `query_search_results(experiment=None, min_timestamp=None, max_timestamp=None, metadata_filters=None, limit=None, order_by=None, order_direction=None)`: Queries search results.
- `store_performance_metrics(search_id, metrics)`: Stores performance metrics.
- `query_performance_trend(experiment, metric, group_by, start_date=None, end_date=None)`: Queries performance trend.
- `analyze_search_trees(experiment, metrics)`: Analyzes search trees.
- `compare_experiments(experiment_ids, metrics)`: Compares experiments.
- `store_game_result(experiment_id, iteration, trajectory, total_reward, metadata=None)`: Stores game results.

**Static Methods:**
- `from_credential_file(file_path, use_ssl=True)`: Creates connector from credential file.
- `from_credential_manager(credential_name, use_ssl=True)`: Creates connector from credential manager.

### batch_operation

```python
mctx.integrations.hana_connector.HanaConnector.batch_operation(batch_size=100)
```

Context manager for batch operations.

**Parameters:**
- `batch_size`: Number of operations per batch.

**Returns:**
- Context manager with `add_search_result` method.

## Examples

### Basic MuZero Policy

```python
import jax
import mctx

# Setup parameters, root state, and recurrent function
params = ...
root = mctx.RootFnOutput(
    prior_logits=prior_logits,
    value=value,
    embedding=embedding
)
recurrent_fn = ...

# Run MuZero policy
policy_output = mctx.muzero_policy(
    params, 
    jax.random.PRNGKey(0), 
    root, 
    recurrent_fn,
    num_simulations=64,
    dirichlet_fraction=0.25,
    dirichlet_alpha=0.3,
    temperature=1.0
)

# Use policy output
action = policy_output.action
action_weights = policy_output.action_weights
```

### T4-Optimized Search

```python
import jax
import mctx
from mctx.t4_optimizations import profile_t4_memory_usage, get_optimal_t4_batch_size

# Setup parameters, root state, and recurrent function
params = ...
root = ...
recurrent_fn = ...

# Get optimal batch size
optimal_batch = get_optimal_t4_batch_size(
    embedding_size=embedding.shape[-1],
    recurrent_fn_params_size=jax.tree_util.tree_reduce(
        lambda x, y: x + y.size * y.dtype.itemsize,
        params,
        0
    )
)

# Run T4-optimized search with performance monitoring
with profile_t4_memory_usage() as memory_profile:
    policy_output = mctx.t4_optimized_search(
        params, 
        jax.random.PRNGKey(0), 
        root, 
        recurrent_fn,
        num_simulations=optimal_batch * 8,
        batch_size=optimal_batch,
        use_mixed_precision=True
    )

print(f"Peak memory usage: {memory_profile.peak_usage / 1024**2:.2f} MB")
print(f"Simulation throughput: {memory_profile.throughput:.2f} sims/second")
```

### Distributed MCTS

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
def run_distributed_search(params, rng_key, root, recurrent_fn):
    return mctx.muzero_policy(
        params, 
        rng_key, 
        root, 
        recurrent_fn,
        num_simulations=2048
    )

# Run distributed search
policy_output = run_distributed_search(params, jax.random.PRNGKey(0), root, recurrent_fn)

# Print performance metrics
print(f"Total time: {monitor.total_time:.2f}s")
print(f"Simulation throughput: {monitor.simulation_throughput:.2f} sims/second")
```

### SAP HANA Integration

```python
import jax
import mctx
from mctx.integrations import hana_connector

# Connect to SAP HANA
connector = hana_connector.HanaConnector(
    host="your_hana_host.example.com",
    port=443,
    user="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    use_ssl=True
)

# Initialize schema if needed
connector.initialize_schema()

# Run a search
params = ...
root = ...
recurrent_fn = ...

policy_output = mctx.muzero_policy(
    params, 
    jax.random.PRNGKey(0), 
    root, 
    recurrent_fn,
    num_simulations=128
)

# Store the search results
search_id = connector.store_search_results(
    search_id=None,
    policy_output=policy_output,
    metadata={"experiment": "example_experiment"}
)

# Retrieve results
result = connector.get_search_results(search_id)
```

### Tree Visualization

```python
import jax
import mctx
from mctx.visualization import visualize_tree

# Run a search
params = ...
root = ...
recurrent_fn = ...

policy_output = mctx.muzero_policy(
    params, 
    jax.random.PRNGKey(0), 
    root, 
    recurrent_fn,
    num_simulations=64
)

# Visualize the tree
html = visualize_tree(
    policy_output.search_tree,
    show_values=True,
    highlight_path=policy_output.search_path,
    color_scheme="value",
    layout="radial"
)

# Save to file
with open("tree_visualization.html", "w") as f:
    f.write(html)
```

For more detailed examples, see the [`examples/`](https://github.com/google-deepmind/mctx/blob/main/examples/) directory.