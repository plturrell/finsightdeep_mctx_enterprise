# MCTX Visualization and Monitoring

This document describes the comprehensive visualization and monitoring components available in MCTX for analyzing Monte Carlo Tree Search algorithms.

## Overview

The MCTX monitoring framework provides tools for:

1. **Metrics Collection**: Gathering comprehensive statistics about MCTS processes
2. **Performance Profiling**: Analyzing computational performance and resource usage
3. **Tree Visualization**: Creating sophisticated visual representations of search trees
4. **Interactive Dashboards**: Real-time monitoring and exploration of MCTS processes
5. **CLI Tools**: Command-line utilities for visualization and analysis

These components are designed to work with all MCTX search algorithms, including MuZero, Gumbel MuZero, and the T4-optimized versions.

## Key Components

### MCTSMetricsCollector

The `MCTSMetricsCollector` provides detailed metrics about MCTS processes, including:

- **Search Statistics**: Iterations, simulation calls, search time
- **Tree Metrics**: Size, depth, branching factor, leaf nodes
- **Visit Statistics**: Maximum, average, distribution
- **Value Statistics**: Min, max, average, uncertainty
- **Performance Metrics**: Iterations/second, simulations/second
- **Resource Usage**: Memory usage, GPU memory usage
- **Time Breakdown**: Tree traversal, node selection, backpropagation

Usage example:

```python
from mctx.monitoring import MCTSMetricsCollector

# Initialize metrics collector
metrics_collector = MCTSMetricsCollector(
    collect_interval_ms=500,  # Interval for continuous collection
    resource_monitoring=True   # Track system resources
)

# Start collection
metrics_collector.start_collection()

# Run search...
search_results = search(...)

# Update tree metrics
metrics_collector.update_tree_metrics(search_results.search_tree)

# Stop collection and get metrics
metrics = metrics_collector.stop_collection()

# Access metrics
print(f"Search time: {metrics.search_time_ms} ms")
print(f"Tree size: {metrics.tree_size} nodes")
print(f"Max depth: {metrics.max_depth}")
print(f"Branching factor: {metrics.branching_factor}")
```

### TreeVisualizer

The `TreeVisualizer` creates sophisticated visualizations of MCTS trees, with:

- **Multiple Layout Options**: Radial, hierarchical, or force-directed
- **Node Styling**: Size based on visits, color by state
- **Edge Styling**: Curved edges with refined aesthetics
- **Interactive Features**: Zooming, panning, tooltips
- **Metrics Panel**: Visit distribution, value distribution, depth metrics

Usage example:

```python
from mctx.monitoring import TreeVisualizer, VisualizationConfig

# Create visualization config
config = VisualizationConfig(
    width=900,
    height=700,
    theme='light',         # 'light' or 'dark'
    layout_type='radial',  # 'radial', 'hierarchical', or 'force'
    save_path='tree.html'  # Path to save visualization
)

# Create visualizer
visualizer = TreeVisualizer(config)

# Convert tree to visualization data
tree_data = visualizer.tree_to_visualization_data(search_results.search_tree)

# Create tree visualization
fig = visualizer.visualize_tree(tree_data)

# Create metrics panel
metrics_fig = visualizer.create_metrics_panel(tree_data)
```

### MCTSMonitor

The `MCTSMonitor` provides real-time monitoring of MCTS processes with:

- **Live Metrics Tracking**: Continuous collection of statistics
- **Tree Visualization**: Real-time updates of the search tree
- **Interactive Dashboard**: Jupyter integration for interactive exploration
- **Animation**: Tree evolution over time
- **Export Capabilities**: Save visualizations and metrics

Usage example:

```python
from mctx.monitoring import MCTSMonitor, VisualizationConfig

# Create visualization config
config = VisualizationConfig(
    width=900,
    height=700,
    theme='light',
    layout_type='radial'
)

# Create monitor
monitor = MCTSMonitor(
    config=config,
    auto_update_interval=1000  # Update interval in ms
)

# Start monitoring
monitor.start_monitoring()

# Run search...
search_results = search(...)

# Update monitor with search tree
monitor.update_tree(search_results.search_tree)

# Stop monitoring
final_metrics = monitor.stop_monitoring()

# Create interactive dashboard (in Jupyter)
monitor.create_dashboard()

# Save visualizations
monitor.save_visualizations(save_dir='mcts_visualizations')
```

### PerformanceProfiler

The `PerformanceProfiler` analyzes computational performance of MCTS processes:

- **Function Timing**: Detailed timing of functions and code sections
- **Memory Tracking**: Memory usage over time
- **GPU Memory Tracking**: GPU memory usage for GPU-accelerated searches
- **Profile Summaries**: Identify bottlenecks and optimization opportunities

Usage example:

```python
from mctx.monitoring import PerformanceProfiler

# Create profiler
profiler = PerformanceProfiler(
    track_memory=True,
    track_gpu_memory=True
)

# Profile a function
@profiler.profile
def my_function():
    # Function code...
    pass

# Profile a section of code
with profiler.profile_section("critical_section"):
    # Code to profile...
    pass

# Get profile summary
profiler.print_profile_summary()

# Plot profile summary
fig = profiler.plot_profile_summary()
```

### ResourceMonitor

The `ResourceMonitor` tracks system resources during MCTS execution:

- **CPU Usage**: Processor utilization
- **Memory Usage**: RAM consumption
- **GPU Usage**: GPU utilization and memory
- **I/O Tracking**: Disk and network activity
- **Resource Visualization**: Plots of resource usage over time

Usage example:

```python
from mctx.monitoring import ResourceMonitor
from mctx.monitoring.profiler import ResourceType

# Create monitor
monitor = ResourceMonitor(
    collect_interval_ms=1000,  # Collect every second
    track_cpu=True,
    track_memory=True,
    track_gpu=True
)

# Start collection
monitor.start_collection()

# Run search...
search_results = search(...)

# Stop collection
snapshots = monitor.stop_collection()

# Get peak usage
peak_cpu = monitor.get_peak_usage(ResourceType.CPU)
peak_memory = monitor.get_peak_usage(ResourceType.MEMORY)
memory_increase = monitor.get_memory_increase()

# Plot resource usage
fig = monitor.plot_resource_usage()
```

## Command-Line Interface

The monitoring framework includes a command-line interface for visualization and analysis:

```bash
# Visualize a tree from a JSON file
python -m mctx.monitoring.cli visualize --tree-file tree.json --output visualizations

# Create a mock tree for testing
python -m mctx.monitoring.cli mock --nodes 100 --branching 3 --depth 5 --output mock_viz

# Run an interactive visualization server
python -m mctx.monitoring.cli server --tree-file tree.json --port 8050
```

## Monitoring Demo

The `monitoring_demo.py` example demonstrates the comprehensive monitoring capabilities:

```bash
# Run the monitoring demo
python examples/monitoring_demo.py --num-simulations 100 --save-visualizations
```

For interactive usage in Jupyter notebooks:

```python
from examples.monitoring_demo import jupyter_interactive_demo

# Run interactive demo
monitor = jupyter_interactive_demo(num_simulations=100)
```

## Best Practices

1. **Real-time Monitoring**: Use `MCTSMonitor` for continuous tracking during development and debugging.
2. **Resource Analysis**: Use `ResourceMonitor` to identify memory leaks and bottlenecks.
3. **Performance Optimization**: Use `PerformanceProfiler` to find slow functions and sections.
4. **Visualization**: Use `TreeVisualizer` to understand search behavior and debug issues.
5. **Metrics Collection**: Use `MCTSMetricsCollector` to gather statistics for tuning hyperparameters.

## Integration with T4 Optimizations

The monitoring components are fully compatible with T4-optimized MCTS implementations:

```python
from mctx import t4_search
from mctx.monitoring import MCTSMonitor

# Create monitor
monitor = MCTSMonitor()

# Start monitoring
monitor.start_monitoring()

# Run T4-optimized search
search_results = t4_search(
    root=root_fn_output,
    recurrent_fn=recurrent_fn,
    num_simulations=100,
    tensor_core_aligned=True,
    optimize_memory=True
)

# Update monitor with search tree
monitor.update_tree(search_results.search_tree)

# Stop monitoring and analyze results
metrics = monitor.stop_monitoring()
```

## Integration with Distributed MCTS

The monitoring framework supports distributed MCTS implementations:

```python
from mctx import enhanced_distributed_search, EnhancedDistributedConfig
from mctx.monitoring import MCTSMetricsCollector, MCTSMonitor

# Create metrics collector
metrics_collector = MCTSMetricsCollector()

# Create distributed config
config = EnhancedDistributedConfig(
    num_devices=4,
    batch_split=True,
    tree_split=False,
    collect_metrics=True  # Enable metrics collection
)

# Run distributed search with metrics
search_results = enhanced_distributed_search(
    root=root_fn_output,
    recurrent_fn=recurrent_fn,
    num_simulations=100,
    config=config
)

# Access metrics from search results
metrics = search_results.metrics
```

## Requirements

The monitoring framework has the following optional dependencies:

- **Visualization**: `plotly`, `dash`, `dash-bootstrap-components`
- **Interactive Dashboards**: `ipywidgets` (for Jupyter integration)
- **Resource Monitoring**: `psutil`, `pynvml` (for GPU monitoring)
- **Plotting**: `matplotlib`, `pandas`

Install all dependencies with:

```bash
pip install plotly dash dash-bootstrap-components ipywidgets psutil pynvml matplotlib pandas
```