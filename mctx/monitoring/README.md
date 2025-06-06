# MCTX Monitoring and Visualization

A comprehensive suite of monitoring, visualization, and profiling tools for Monte Carlo Tree Search algorithms in JAX.

## Features

- **Real-time Metrics Collection**: Track detailed statistics about MCTS processes
- **Interactive Tree Visualization**: Explore search trees with sophisticated visualizations
- **Performance Profiling**: Analyze computational efficiency and identify bottlenecks
- **Resource Monitoring**: Track CPU, memory, and GPU usage during search
- **Command-line Interface**: Visualize trees and metrics from the command line
- **Jupyter Integration**: Interactive dashboards for exploration in notebooks

## Components

### Metrics Collection

The `MCTSMetricsCollector` provides comprehensive metrics about MCTS processes:

```python
from mctx.monitoring import MCTSMetricsCollector

# Initialize
collector = MCTSMetricsCollector(collect_interval_ms=500)

# Start collection
collector.start_collection()

# Record events
collector.record_iteration()
collector.record_simulation()
collector.update_tree_metrics(tree)

# Stop collection and get metrics
metrics = collector.stop_collection()

# Decorator for automatic collection
@with_metrics(collector)
def run_search():
    # Search code...
```

### Tree Visualization

The `TreeVisualizer` creates sophisticated visualizations of MCTS trees:

```python
from mctx.monitoring import TreeVisualizer, VisualizationConfig

# Configure visualization
config = VisualizationConfig(
    width=900,
    height=700,
    theme='light',
    layout_type='radial'
)

# Create visualizer
visualizer = TreeVisualizer(config)

# Create visualization
tree_data = visualizer.tree_to_visualization_data(tree)
fig = visualizer.visualize_tree(tree_data)

# Create metrics panel
metrics_fig = visualizer.create_metrics_panel(tree_data)
```

### Real-time Monitoring

The `MCTSMonitor` provides real-time monitoring of MCTS processes:

```python
from mctx.monitoring import MCTSMonitor

# Create monitor
monitor = MCTSMonitor()

# Start monitoring
monitor.start_monitoring()

# Update with new tree
monitor.update_tree(tree)

# Create dashboard (in Jupyter)
monitor.create_dashboard()

# Stop monitoring
metrics = monitor.stop_monitoring()

# Save visualizations
monitor.save_visualizations()
```

### Performance Profiling

The `PerformanceProfiler` analyzes computational performance:

```python
from mctx.monitoring import PerformanceProfiler

# Create profiler
profiler = PerformanceProfiler()

# Profile a function
@profiler.profile
def my_function():
    # Function code...

# Profile a code section
with profiler.profile_section("critical_section"):
    # Code to profile...

# Get results
profiler.print_profile_summary()
```

### Resource Monitoring

The `ResourceMonitor` tracks system resources:

```python
from mctx.monitoring import ResourceMonitor

# Create monitor
monitor = ResourceMonitor()

# Start collection
monitor.start_collection()

# Run search...

# Stop collection
snapshots = monitor.stop_collection()

# Plot resource usage
fig = monitor.plot_resource_usage()
```

## Command-line Interface

The monitoring framework includes a command-line interface:

```bash
# Visualize a tree from JSON
python -m mctx.monitoring.cli visualize --tree-file tree.json

# Create a mock tree
python -m mctx.monitoring.cli mock --nodes 100

# Run visualization server
python -m mctx.monitoring.cli server --tree-file tree.json
```

## Demo

Try the comprehensive demo:

```bash
python examples/monitoring_demo.py --save-visualizations
```

For Jupyter notebooks:

```python
from examples.monitoring_demo import jupyter_interactive_demo
monitor = jupyter_interactive_demo()
```

## Requirements

The monitoring framework has the following optional dependencies:

- **Visualization**: plotly, dash, dash-bootstrap-components
- **Interactive Dashboards**: ipywidgets (for Jupyter)
- **Resource Monitoring**: psutil, pynvml (for GPU)
- **Plotting**: matplotlib, pandas