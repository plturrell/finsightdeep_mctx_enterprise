# MCTX Visualization Guide

This guide covers the visualization capabilities of the MCTX library, which provides powerful tools for visualizing Monte Carlo Tree Search (MCTS) trees, policies, and search processes.

## Overview

The MCTX visualization system enables:
- Interactive visualization of search trees
- Value and policy heatmaps
- Search path animation and highlighting
- Performance metrics dashboards
- Comparative visualization of multiple trees

## Basic Tree Visualization

The simplest way to visualize a search tree is with the `visualize_tree` function:

```python
import mctx
from mctx.visualization import visualize_tree

# Run a search
policy_output = mctx.muzero_policy(
    params, 
    rng_key, 
    root, 
    recurrent_fn,
    num_simulations=64
)

# Generate visualization
html = visualize_tree(
    policy_output.search_tree,
    root_state="Initial State",
    show_values=True,
    highlight_path=policy_output.search_path
)

# Save to file
with open("mcts_visualization.html", "w") as f:
    f.write(html)
```

This generates an interactive HTML visualization that can be opened in any web browser.

## Visualization Options

The `visualize_tree` function supports numerous options:

```python
html = visualize_tree(
    tree=policy_output.search_tree,
    root_state="Initial State",
    show_values=True,                # Show value estimates
    show_visit_counts=True,          # Show visit counts
    show_prior_probabilities=True,   # Show prior probabilities
    highlight_path=policy_output.search_path,  # Highlight specific path
    color_scheme="value",            # Color nodes by value
    layout="radial",                 # Tree layout style
    max_depth=5,                     # Maximum depth to display
    width=1200,                      # Width in pixels
    height=800,                      # Height in pixels
    node_size="visits",              # Size nodes by visit count
    edge_width="visits",             # Size edges by visit count
    include_controls=True,           # Include interactive controls
    include_metrics=True,            # Include metrics panel
    title="MCTS Visualization"       # Visualization title
)
```

### Color Schemes

Available color schemes:

| Scheme | Description |
|--------|-------------|
| `"value"` | Colors nodes based on value (blue=low, red=high) |
| `"visits"` | Colors nodes based on visit count (light=low, dark=high) |
| `"depth"` | Colors nodes based on tree depth |
| `"q_value"` | Colors nodes based on Q-value |
| `"prior"` | Colors nodes based on prior probability |
| `"custom"` | Uses the provided `color_function` |

Example with custom coloring:

```python
def custom_color(node):
    """Custom coloring based on node properties."""
    value = node.value
    visits = node.visit_count
    if visits > 100:
        return "#ff5500"  # Orange for heavily visited
    elif value > 0.7:
        return "#22cc88"  # Green for high value
    elif value < -0.3:
        return "#cc2288"  # Pink for low value
    else:
        return "#2288cc"  # Blue for others

html = visualize_tree(
    tree=policy_output.search_tree,
    color_scheme="custom",
    color_function=custom_color
)
```

### Layouts

Available tree layouts:

| Layout | Description |
|--------|-------------|
| `"radial"` | Circular layout with root at center |
| `"horizontal"` | Left-to-right tree layout |
| `"vertical"` | Top-to-bottom tree layout |
| `"cluster"` | Dendrogram-style clustered layout |
| `"force"` | Force-directed graph layout |

## Interactive Dashboards

For more advanced visualizations, you can create interactive dashboards:

```python
from mctx.visualization import MCTSDashboard

# Create dashboard
dashboard = MCTSDashboard(
    title="AlphaZero Search Analysis",
    description="Analysis of search trees from AlphaZero training"
)

# Add trees to the dashboard
dashboard.add_tree(
    tree=policy_output1.search_tree,
    name="Game 1, Move 10",
    metadata={"player": "white", "position": "mid-game"}
)

dashboard.add_tree(
    tree=policy_output2.search_tree,
    name="Game 1, Move 20",
    metadata={"player": "black", "position": "end-game"}
)

# Add metrics panel
dashboard.add_metrics_panel([
    {"name": "Average Depth", "value": 4.2},
    {"name": "Average Branching Factor", "value": 8.7},
    {"name": "Max Value", "value": 0.92},
    {"name": "Min Value", "value": -0.86},
    {"name": "Total Nodes", "value": 512}
])

# Generate the dashboard
html = dashboard.generate_html()

# Save to file
with open("mcts_dashboard.html", "w") as f:
    f.write(html)
```

## Animation and Time Series

For visualizing the search process over time:

```python
from mctx.visualization import animate_search_process

# Collect trees at different stages of search
trees = []
for num_sims in [16, 32, 64, 128, 256]:
    policy_output = mctx.muzero_policy(
        params, 
        rng_key, 
        root, 
        recurrent_fn,
        num_simulations=num_sims
    )
    trees.append({
        "tree": policy_output.search_tree,
        "step": num_sims,
        "metrics": {
            "value": policy_output.root_value,
            "selected_action": policy_output.action
        }
    })

# Generate animation
html = animate_search_process(
    trees=trees,
    framerate=1,  # Frames per second
    loop=True,    # Loop animation
    include_controls=True,  # Include playback controls
    width=1000,
    height=600
)

# Save to file
with open("search_animation.html", "w") as f:
    f.write(html)
```

## Heatmaps

For visualizing values and policies as heatmaps:

```python
from mctx.visualization import create_heatmap

# Generate a value heatmap
html = create_heatmap(
    data=policy_output.q_values,
    title="Q-Values Heatmap",
    x_labels=["Action 1", "Action 2", "Action 3", "Action 4"],
    colorscale="RdBu",  # Red-White-Blue scale
    min_value=-1.0,
    max_value=1.0,
    width=600,
    height=400
)

# Save to file
with open("q_values_heatmap.html", "w") as f:
    f.write(html)
```

For game-specific visualizations:

```python
from mctx.visualization.games import visualize_chess_policy

# Visualize a chess policy
html = visualize_chess_policy(
    board_fen="rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    policy=policy_output.action_weights,
    move_mapping=chess_move_mapping,  # Mapping from actions to UCI moves
    show_arrows=True,
    arrow_scale="probability",
    min_probability=0.01
)

# Save to file
with open("chess_policy.html", "w") as f:
    f.write(html)
```

## Comparative Visualization

For comparing multiple trees:

```python
from mctx.visualization import compare_trees

# Compare two trees
html = compare_trees(
    trees=[policy_output1.search_tree, policy_output2.search_tree],
    names=["AlphaZero", "MuZero"],
    metrics=["value", "visits", "depth"],
    highlight_differences=True,
    width=1200,
    height=800
)

# Save to file
with open("tree_comparison.html", "w") as f:
    f.write(html)
```

## Embedding in Applications

### Flask Web Application

```python
from flask import Flask, render_template_string
import mctx
from mctx.visualization import visualize_tree

app = Flask(__name__)

@app.route('/')
def index():
    # Run a search
    policy_output = mctx.muzero_policy(...)
    
    # Generate visualization
    vis_html = visualize_tree(
        policy_output.search_tree,
        show_values=True,
        include_controls=True
    )
    
    # Embed in web page
    template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MCTS Visualization</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
            h1 { color: #333; }
        </style>
    </head>
    <body>
        <h1>Monte Carlo Tree Search Visualization</h1>
        <div id="visualization">
            {{ visualization|safe }}
        </div>
    </body>
    </html>
    """
    
    return render_template_string(template, visualization=vis_html)

if __name__ == '__main__':
    app.run(debug=True)
```

### Jupyter Notebook Integration

```python
from IPython.display import HTML
import mctx
from mctx.visualization import visualize_tree

# Run a search
policy_output = mctx.muzero_policy(...)

# Generate and display visualization
vis_html = visualize_tree(
    policy_output.search_tree,
    show_values=True,
    width=800,
    height=600
)

# Display in notebook
HTML(vis_html)
```

## Customization with CSS and JavaScript

The visualizations can be customized with CSS and JavaScript:

```python
from mctx.visualization import visualize_tree

# Custom CSS
custom_css = """
.node circle {
    stroke: #fff;
    stroke-width: 2px;
}
.node text {
    font-family: 'Courier New', monospace;
    font-size: 12px;
}
.link {
    fill: none;
    stroke: #ccc;
    stroke-width: 1.5px;
}
"""

# Custom JavaScript
custom_js = """
// Add click handler to nodes
d3.selectAll(".node").on("click", function(d) {
    console.log("Node clicked:", d);
    d3.select(this).select("circle")
        .transition()
        .duration(500)
        .attr("r", d => d.data.visit_count / 20 + 10);
});
"""

# Generate visualization with custom CSS and JS
html = visualize_tree(
    policy_output.search_tree,
    custom_css=custom_css,
    custom_js=custom_js
)
```

## Performance Considerations

For visualizing very large trees:

```python
from mctx.visualization import visualize_tree

# Optimize for large trees
html = visualize_tree(
    policy_output.search_tree,
    max_nodes=1000,           # Limit number of displayed nodes
    simplify_threshold=5,     # Simplify subtrees with fewer than 5 visits
    use_webgl=True,           # Use WebGL for rendering (faster)
    load_on_demand=True,      # Load subtrees on demand
    optimize_for_size=True    # Optimize SVG size
)
```

## Export Formats

Export visualizations in different formats:

```python
from mctx.visualization import visualize_tree, export_visualization

# Generate visualization
vis = visualize_tree(
    policy_output.search_tree,
    show_values=True
)

# Export as PNG
export_visualization(
    visualization=vis,
    format="png",
    filename="tree_visualization.png",
    width=1200,
    height=800,
    scale=2  # For higher resolution
)

# Export as SVG
export_visualization(
    visualization=vis,
    format="svg",
    filename="tree_visualization.svg"
)

# Export as PDF
export_visualization(
    visualization=vis,
    format="pdf",
    filename="tree_visualization.pdf"
)
```

## Integration with Design System

The visualization components follow a cohesive design system:

```python
from mctx.visualization import MCTSDesignSystem, visualize_tree

# Create or customize design system
design_system = MCTSDesignSystem(
    primary_color="#0066cc",
    secondary_color="#6600cc",
    text_color="#333333",
    background_color="#ffffff",
    font_family="'Helvetica Neue', Arial, sans-serif",
    border_radius=4,
    animation_duration=300
)

# Use design system for visualization
html = visualize_tree(
    policy_output.search_tree,
    design_system=design_system
)
```

## Example: Complete Visualization Dashboard

Here's a complete example of creating a comprehensive visualization dashboard:

```python
import mctx
import jax
from mctx.visualization import (
    MCTSDashboard, 
    visualize_tree, 
    create_heatmap, 
    animate_search_process
)

# Run multiple searches with different parameters
searches = []
for c_puct in [1.0, 1.5, 2.0]:
    policy_output = mctx.muzero_policy(
        params, 
        jax.random.PRNGKey(0), 
        root, 
        recurrent_fn,
        num_simulations=128,
        pb_c_init=c_puct
    )
    searches.append({
        "name": f"c_puct={c_puct}",
        "policy_output": policy_output,
        "params": {"c_puct": c_puct}
    })

# Create dashboard
dashboard = MCTSDashboard(
    title="MuZero Parameter Study",
    description="Analyzing the effect of exploration parameter c_puct"
)

# Add trees to the dashboard
for search in searches:
    dashboard.add_tree(
        tree=search["policy_output"].search_tree,
        name=search["name"],
        metadata=search["params"]
    )

# Add value heatmap
q_values = [search["policy_output"].q_values for search in searches]
dashboard.add_panel(
    create_heatmap(
        data=q_values,
        title="Q-Values Comparison",
        x_labels=["Action 1", "Action 2", "Action 3", "Action 4"],
        y_labels=[search["name"] for search in searches],
        colorscale="RdBu"
    ),
    title="Q-Values by Exploration Parameter"
)

# Add policy comparison
action_weights = [search["policy_output"].action_weights for search in searches]
dashboard.add_panel(
    create_heatmap(
        data=action_weights,
        title="Action Probability Comparison",
        x_labels=["Action 1", "Action 2", "Action 3", "Action 4"],
        y_labels=[search["name"] for search in searches],
        colorscale="Viridis"
    ),
    title="Policy by Exploration Parameter"
)

# Add search metrics
metrics = []
for search in searches:
    po = search["policy_output"]
    metrics.append({
        "name": search["name"],
        "root_value": po.root_value,
        "selected_action": po.action,
        "max_depth": po.search_tree.max_depth,
        "node_count": po.search_tree.node_count
    })

dashboard.add_metrics_table(
    metrics=metrics,
    columns=["name", "root_value", "selected_action", "max_depth", "node_count"],
    column_names=["Configuration", "Root Value", "Selected Action", "Max Depth", "Node Count"]
)

# Generate the dashboard
html = dashboard.generate_html()

# Save to file
with open("parameter_study_dashboard.html", "w") as f:
    f.write(html)
```

For more examples, see the [`examples/visualization_demo.py`](https://github.com/google-deepmind/mctx/blob/main/examples/visualization_demo.py) file.