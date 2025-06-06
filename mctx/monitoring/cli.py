#!/usr/bin/env python3
"""
MCTX Monitoring CLI

Command-line interface for MCTX monitoring and visualization.
Provides tools for real-time visualization, performance profiling,
and analysis of Monte Carlo Tree Search processes.
"""

import os
import sys
import time
import json
import argparse
import logging
import webbrowser
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from mctx._src.tree import Tree
from mctx.monitoring.visualization import (
    TreeVisualizer,
    VisualizationConfig,
    MCTSMonitor
)
from mctx.monitoring.metrics import (
    MCTSMetricsCollector,
    SearchMetrics
)
from mctx.monitoring.profiler import (
    PerformanceProfiler,
    ResourceMonitor
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mctx.monitoring.cli")


def load_tree_from_file(filepath: str) -> Dict[str, Any]:
    """
    Load a tree from a file.
    
    Args:
        filepath: Path to the tree file
        
    Returns:
        Tree data
    """
    with open(filepath, 'r') as f:
        try:
            tree_data = json.load(f)
            logger.info(f"Loaded tree data from {filepath}")
            return tree_data
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing tree file: {e}")
            sys.exit(1)


def load_tree_checkpoint(filepath: str) -> Tree:
    """
    Load a Tree object from a checkpoint.
    
    Args:
        filepath: Path to the checkpoint file
        
    Returns:
        Tree object
    """
    try:
        from flax import serialization
        with open(filepath, 'rb') as f:
            tree_bytes = f.read()
        
        # Create empty tree structure
        empty_tree = Tree(
            node_values=jnp.zeros((1,)),
            node_visits=jnp.zeros((1,), dtype=jnp.int32),
            children_index=jnp.zeros((1, 0), dtype=jnp.int32),
            parents=jnp.zeros((1,), dtype=jnp.int32),
        )
        
        # Deserialize
        tree = serialization.from_bytes(empty_tree, tree_bytes)
        logger.info(f"Loaded Tree checkpoint from {filepath}")
        return tree
    except ImportError as e:
        logger.error(f"Error importing required packages: {e}")
        logger.error("Make sure flax is installed: pip install flax")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading tree checkpoint: {e}")
        sys.exit(1)


def convert_json_tree_to_viz_data(tree_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert tree data from JSON to visualization format.
    
    Args:
        tree_data: Tree data from JSON
        
    Returns:
        Tree data in visualization format
    """
    # Check if it's already in visualization format
    if all(k in tree_data for k in ["node_count", "visits", "values", "parents", "children", "states"]):
        return tree_data
    
    # Handle MuZero/Gumbel MuZero format
    if "node_values" in tree_data and "node_visits" in tree_data:
        node_values = tree_data.get("node_values", [])
        node_visits = tree_data.get("node_visits", [])
        children_index = tree_data.get("children_index", [])
        
        # Create parents mapping
        parents = {}
        for i, children in enumerate(children_index):
            for j, child in enumerate(children):
                if child != -1:  # -1 means no child
                    parents[child] = i
        
        # Determine node states
        states = []
        node_visits_np = np.array(node_visits)
        for i in range(len(node_visits)):
            if node_visits[i] == 0:
                states.append("unexplored")
            elif i == 0:  # Root
                states.append("selected")
            elif node_visits[i] > np.percentile(node_visits_np[node_visits_np > 0], 90):
                states.append("optimal")
            else:
                states.append("explored")
        
        # Create children mapping
        children = {}
        for parent, children_arr in enumerate(children_index):
            parent_children = []
            for child in children_arr:
                if child != -1:  # -1 means no child
                    parent_children.append(child)
            children[parent] = parent_children
        
        # Construct visualization data
        vis_data = {
            "node_count": len(node_values),
            "visits": node_visits,
            "values": node_values,
            "parents": parents,
            "children": children,
            "states": states
        }
        
        return vis_data
    
    # If we can't determine the format, return an empty tree
    logger.error("Could not determine tree format")
    return {
        "node_count": 1,
        "visits": [1],
        "values": [0],
        "parents": {},
        "children": {0: []},
        "states": ["explored"]
    }


def create_mock_tree(num_nodes: int = 50,
                   branching_factor: int = 3,
                   max_depth: int = 4) -> Dict[str, Any]:
    """
    Create a mock tree for visualization.
    
    Args:
        num_nodes: Number of nodes
        branching_factor: Average branching factor
        max_depth: Maximum depth
        
    Returns:
        Mock tree data
    """
    # Create mock node data
    visits = np.zeros(num_nodes)
    values = np.zeros(num_nodes)
    parents = {}
    children = {}
    states = []
    
    # Set root node values
    visits[0] = 50
    values[0] = 0.5
    children[0] = []
    states.append("explored")
    
    # Create tree structure
    node_counter = 1
    
    def create_subtree(parent, depth):
        nonlocal node_counter
        if depth >= max_depth or node_counter >= num_nodes:
            return
            
        # Determine number of children for this node
        num_children = min(
            branching_factor, 
            num_nodes - node_counter
        )
        
        # Create children
        for _ in range(num_children):
            if node_counter >= num_nodes:
                break
                
            # Create child node
            child = node_counter
            node_counter += 1
            
            # Set up parent-child relationship
            parents[child] = parent
            children[parent].append(child)
            
            # Initialize child values
            if depth == max_depth - 1:
                # Leaf nodes
                visits[child] = np.random.randint(1, 10)
                values[child] = np.random.uniform(-1, 1)
                states.append("explored")
                children[child] = []
            else:
                # Internal nodes
                visits[child] = np.random.randint(10, 30)
                values[child] = np.random.uniform(-0.5, 0.8)
                states.append("explored")
                children[child] = []
                
                # Recursively create subtree
                create_subtree(child, depth + 1)
    
    # Create the full tree
    create_subtree(0, 1)
    
    # Mark some nodes as unexplored, optimal, or selected
    for i in range(1, num_nodes):
        if np.random.random() < 0.1:
            visits[i] = 0
            states[i] = "unexplored"
        elif np.random.random() < 0.05:
            states[i] = "optimal"
        elif np.random.random() < 0.05:
            states[i] = "selected"
    
    # Construct visualization data
    vis_data = {
        "node_count": num_nodes,
        "visits": visits.tolist(),
        "values": values.tolist(),
        "parents": parents,
        "children": children,
        "states": states
    }
    
    return vis_data


def visualize_tree(tree_data: Dict[str, Any],
                 output_path: str,
                 width: int = 800,
                 height: int = 600,
                 theme: str = 'light',
                 layout: str = 'radial',
                 include_metrics: bool = True,
                 open_browser: bool = False) -> None:
    """
    Create visualizations for a tree.
    
    Args:
        tree_data: Tree data to visualize
        output_path: Directory to save visualizations
        width: Visualization width
        height: Visualization height
        theme: Visualization theme ('light' or 'dark')
        layout: Layout type ('radial', 'hierarchical', or 'force')
        include_metrics: Whether to include metrics panels
        open_browser: Whether to open visualizations in browser
    """
    # Prepare output directory
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualization config
    config = VisualizationConfig(
        width=width,
        height=height,
        theme=theme,
        layout_type=layout,
        save_path=str(output_dir / "tree.html")
    )
    
    # Create visualizer
    visualizer = TreeVisualizer(config)
    
    # Create tree visualization
    tree_fig = visualizer.visualize_tree(tree_data)
    
    # Create metrics visualization if requested
    if include_metrics:
        metrics_fig = visualizer.create_metrics_panel(tree_data)
    
    # Print output paths
    logger.info(f"Saved tree visualization to {output_dir / 'tree.html'}")
    if include_metrics:
        logger.info(f"Saved metrics visualization to {output_dir / 'tree_metrics.html'}")
    
    # Open in browser if requested
    if open_browser:
        tree_path = output_dir / "tree.html"
        try:
            webbrowser.open(f"file://{tree_path.absolute()}")
            logger.info(f"Opened visualization in browser")
        except Exception as e:
            logger.error(f"Error opening browser: {e}")


def run_web_server(tree_data: Dict[str, Any], port: int = 8050) -> None:
    """
    Run a web server for interactive visualization.
    
    Args:
        tree_data: Tree data to visualize
        port: Port to run server on
    """
    try:
        import dash
        from dash import dcc, html
        import dash_bootstrap_components as dbc
        from dash.dependencies import Input, Output, State
    except ImportError:
        logger.error("Required packages not installed. Please install:")
        logger.error("pip install dash dash-bootstrap-components plotly")
        sys.exit(1)
    
    # Create visualization config
    config = VisualizationConfig(
        width=900,
        height=700,
        theme='light',
        layout_type='radial'
    )
    
    # Create visualizer
    visualizer = TreeVisualizer(config)
    
    # Create initial visualizations
    tree_fig = visualizer.visualize_tree(tree_data)
    metrics_fig = visualizer.create_metrics_panel(tree_data)
    
    # Create app
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        meta_tags=[
            {"name": "viewport", "content": "width=device-width, initial-scale=1"}
        ],
        title="MCTS Visualization"
    )
    
    # Create layout
    app.layout = html.Div([
        html.Div([
            html.H1("Monte Carlo Tree Search Visualization", className="header-title"),
            html.P("Interactive visualization of MCTS processes", className="header-subtitle")
        ], className="header"),
        
        dbc.Container([
            dbc.Row([
                # Controls column
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Visualization Controls"),
                        dbc.CardBody([
                            dbc.Form([
                                dbc.FormGroup([
                                    dbc.Label("Layout Type"),
                                    dbc.Select(
                                        id="layout-select",
                                        options=[
                                            {"label": "Radial", "value": "radial"},
                                            {"label": "Hierarchical", "value": "hierarchical"},
                                            {"label": "Force", "value": "force"}
                                        ],
                                        value="radial"
                                    )
                                ]),
                                dbc.FormGroup([
                                    dbc.Label("Node Size"),
                                    dbc.Input(
                                        id="node-size-input",
                                        type="number",
                                        min=0.5,
                                        max=2.0,
                                        step=0.1,
                                        value=1.0
                                    )
                                ]),
                                dbc.FormGroup([
                                    dbc.Label("Theme"),
                                    dbc.RadioItems(
                                        id="theme-radio",
                                        options=[
                                            {"label": "Light", "value": "light"},
                                            {"label": "Dark", "value": "dark"}
                                        ],
                                        value="light",
                                        inline=True
                                    )
                                ]),
                                dbc.Button(
                                    "Reset View",
                                    id="reset-button",
                                    color="primary",
                                    className="mt-3"
                                )
                            ])
                        ])
                    ], className="mb-4"),
                    
                    dbc.Card([
                        dbc.CardHeader("Tree Statistics"),
                        dbc.CardBody([
                            html.Div([
                                html.Div([
                                    html.H5("Node Count", className="stat-label"),
                                    html.Div(str(tree_data["node_count"]), id="node-count", className="stat-value")
                                ], className="stat-item"),
                                html.Div([
                                    html.H5("Max Visits", className="stat-label"),
                                    html.Div(str(max(tree_data["visits"])), id="max-visits", className="stat-value")
                                ], className="stat-item"),
                                html.Div([
                                    html.H5("Max Value", className="stat-label"),
                                    html.Div(f"{max(tree_data['values']):.3f}", id="max-value", className="stat-value")
                                ], className="stat-item"),
                                html.Div([
                                    html.H5("Min Value", className="stat-label"),
                                    html.Div(f"{min(tree_data['values']):.3f}", id="min-value", className="stat-value")
                                ], className="stat-item")
                            ], className="stat-grid")
                        ])
                    ])
                ], md=3),
                
                # Visualization column
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            dbc.Tabs([
                                dbc.Tab(label="Tree Visualization", tab_id="tree-tab"),
                                dbc.Tab(label="Metrics Analysis", tab_id="metrics-tab")
                            ], id="tabs", active_tab="tree-tab")
                        ]),
                        dbc.CardBody([
                            # Tree visualization
                            html.Div([
                                dcc.Graph(
                                    id="tree-graph",
                                    figure=tree_fig,
                                    config={
                                        "displayModeBar": True,
                                        "scrollZoom": True,
                                        "responsive": True
                                    },
                                    style={"height": "75vh"}
                                )
                            ], id="tree-content"),
                            
                            # Metrics panel
                            html.Div([
                                dcc.Graph(
                                    id="metrics-graph",
                                    figure=metrics_fig,
                                    config={
                                        "displayModeBar": True,
                                        "responsive": True
                                    },
                                    style={"height": "75vh"}
                                )
                            ], id="metrics-content", style={"display": "none"})
                        ])
                    ])
                ], md=9)
            ])
        ], fluid=True),
        
        # Store for tree data
        dcc.Store(id="tree-data-store", data=tree_data)
    ])
    
    # Add custom CSS
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                    background-color: #f8fafc;
                    color: #334155;
                }
                
                .header {
                    background-color: #0f172a;
                    color: #f8fafc;
                    padding: 2rem;
                    margin-bottom: 2rem;
                    text-align: center;
                }
                
                .header-title {
                    font-weight: 600;
                    margin-bottom: 0.5rem;
                }
                
                .header-subtitle {
                    opacity: 0.8;
                    font-weight: 300;
                    margin-bottom: 0;
                }
                
                .card {
                    border-radius: 0.5rem;
                    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.1), 0 1px 2px rgba(15, 23, 42, 0.06);
                    overflow: hidden;
                    margin-bottom: 1.5rem;
                    background-color: white;
                    transition: box-shadow 0.2s ease, transform 0.2s ease;
                }
                
                .card:hover {
                    box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.1), 0 2px 4px -1px rgba(15, 23, 42, 0.06);
                    transform: translateY(-2px);
                }
                
                .card-header {
                    background-color: #f1f5f9;
                    border-bottom: 1px solid #e2e8f0;
                    padding: 1rem 1.25rem;
                    font-weight: 600;
                    color: #0f172a;
                }
                
                .stat-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
                    gap: 1rem;
                }
                
                .stat-item {
                    text-align: center;
                    padding: 0.5rem;
                    border-radius: 0.375rem;
                    background-color: #f8fafc;
                    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
                }
                
                .stat-label {
                    font-size: 0.875rem;
                    color: #64748b;
                    margin-bottom: 0.25rem;
                }
                
                .stat-value {
                    font-size: 1.25rem;
                    font-weight: 600;
                    color: #334155;
                }
                
                .btn-primary {
                    background-color: #6366f1;
                    border-color: #6366f1;
                    transition: all 0.2s ease;
                }
                
                .btn-primary:hover {
                    background-color: #4f46e5;
                    border-color: #4f46e5;
                    transform: translateY(-1px);
                    box-shadow: 0 4px 6px -1px rgba(99, 102, 241, 0.5);
                }
                
                .form-group {
                    margin-bottom: 1rem;
                }
                
                .form-control:focus {
                    border-color: #6366f1;
                    box-shadow: 0 0 0 0.2rem rgba(99, 102, 241, 0.25);
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''
    
    # Define callbacks
    @app.callback(
        [Output("tree-content", "style"),
         Output("metrics-content", "style")],
        [Input("tabs", "active_tab")]
    )
    def update_active_tab(active_tab):
        """Switch between tree and metrics visualization."""
        if active_tab == "tree-tab":
            return {"display": "block"}, {"display": "none"}
        else:
            return {"display": "none"}, {"display": "block"}
    
    @app.callback(
        [Output("tree-graph", "figure"),
         Output("metrics-graph", "figure")],
        [Input("layout-select", "value"),
         Input("node-size-input", "value"),
         Input("theme-radio", "value"),
         Input("reset-button", "n_clicks")],
        [State("tree-data-store", "data")]
    )
    def update_visualization(layout_type, node_size, theme, reset_clicks, tree_data):
        """Update visualization based on controls."""
        # Update config
        config.layout_type = layout_type
        config.node_size_factor = node_size
        config.theme = theme
        
        # Update visualizer config
        visualizer.config = config
        visualizer.node_colors = visualizer._get_color_palette()
        
        # Create visualizations
        tree_fig = visualizer.visualize_tree(tree_data)
        metrics_fig = visualizer.create_metrics_panel(tree_data)
        
        return tree_fig, metrics_fig
    
    # Run server
    logger.info(f"Starting visualization server on port {port}")
    logger.info(f"Open http://localhost:{port} in your browser")
    app.run_server(debug=False, host="0.0.0.0", port=port)


def main():
    """Run the CLI."""
    parser = argparse.ArgumentParser(description="MCTX Monitoring and Visualization CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Visualize command
    visualize_parser = subparsers.add_parser("visualize", help="Visualize a tree")
    visualize_parser.add_argument("--tree-file", "-t", type=str, help="Path to tree file (JSON)")
    visualize_parser.add_argument("--tree-checkpoint", "-c", type=str, help="Path to tree checkpoint (Flax)")
    visualize_parser.add_argument("--output", "-o", type=str, default="mctx_visualizations", help="Output directory")
    visualize_parser.add_argument("--width", "-w", type=int, default=800, help="Visualization width")
    visualize_parser.add_argument("--height", "-h", type=int, default=600, help="Visualization height")
    visualize_parser.add_argument("--theme", choices=["light", "dark"], default="light", help="Visualization theme")
    visualize_parser.add_argument("--layout", choices=["radial", "hierarchical", "force"], default="radial", help="Layout type")
    visualize_parser.add_argument("--no-metrics", action="store_true", help="Disable metrics panels")
    visualize_parser.add_argument("--open", action="store_true", help="Open visualization in browser")
    
    # Mock tree command
    mock_parser = subparsers.add_parser("mock", help="Generate a mock tree for visualization")
    mock_parser.add_argument("--nodes", "-n", type=int, default=50, help="Number of nodes")
    mock_parser.add_argument("--branching", "-b", type=int, default=3, help="Branching factor")
    mock_parser.add_argument("--depth", "-d", type=int, default=4, help="Maximum depth")
    mock_parser.add_argument("--output", "-o", type=str, default="mctx_visualizations", help="Output directory")
    mock_parser.add_argument("--width", "-w", type=int, default=800, help="Visualization width")
    mock_parser.add_argument("--height", "-h", type=int, default=600, help="Visualization height")
    mock_parser.add_argument("--theme", choices=["light", "dark"], default="light", help="Visualization theme")
    mock_parser.add_argument("--layout", choices=["radial", "hierarchical", "force"], default="radial", help="Layout type")
    mock_parser.add_argument("--no-metrics", action="store_true", help="Disable metrics panels")
    mock_parser.add_argument("--open", action="store_true", help="Open visualization in browser")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Run visualization server")
    server_parser.add_argument("--tree-file", "-t", type=str, help="Path to tree file (JSON)")
    server_parser.add_argument("--tree-checkpoint", "-c", type=str, help="Path to tree checkpoint (Flax)")
    server_parser.add_argument("--port", "-p", type=int, default=8050, help="Server port")
    server_parser.add_argument("--nodes", "-n", type=int, default=50, help="Number of nodes (for mock tree)")
    server_parser.add_argument("--branching", "-b", type=int, default=3, help="Branching factor (for mock tree)")
    server_parser.add_argument("--depth", "-d", type=int, default=4, help="Maximum depth (for mock tree)")
    
    args = parser.parse_args()
    
    # Handle commands
    if args.command == "visualize":
        # Load tree data
        if args.tree_file:
            tree_data = load_tree_from_file(args.tree_file)
            tree_data = convert_json_tree_to_viz_data(tree_data)
        elif args.tree_checkpoint:
            tree = load_tree_checkpoint(args.tree_checkpoint)
            visualizer = TreeVisualizer()
            tree_data = visualizer.tree_to_visualization_data(tree)
        else:
            logger.error("No tree file or checkpoint specified")
            visualize_parser.print_help()
            sys.exit(1)
        
        # Create visualization
        visualize_tree(
            tree_data=tree_data,
            output_path=args.output,
            width=args.width,
            height=args.height,
            theme=args.theme,
            layout=args.layout,
            include_metrics=not args.no_metrics,
            open_browser=args.open
        )
    
    elif args.command == "mock":
        # Create mock tree
        tree_data = create_mock_tree(
            num_nodes=args.nodes,
            branching_factor=args.branching,
            max_depth=args.depth
        )
        
        # Create visualization
        visualize_tree(
            tree_data=tree_data,
            output_path=args.output,
            width=args.width,
            height=args.height,
            theme=args.theme,
            layout=args.layout,
            include_metrics=not args.no_metrics,
            open_browser=args.open
        )
    
    elif args.command == "server":
        # Load tree data
        if args.tree_file:
            tree_data = load_tree_from_file(args.tree_file)
            tree_data = convert_json_tree_to_viz_data(tree_data)
        elif args.tree_checkpoint:
            tree = load_tree_checkpoint(args.tree_checkpoint)
            visualizer = TreeVisualizer()
            tree_data = visualizer.tree_to_visualization_data(tree)
        else:
            logger.info("No tree file specified, generating mock tree")
            tree_data = create_mock_tree(
                num_nodes=args.nodes,
                branching_factor=args.branching,
                max_depth=args.depth
            )
        
        # Run server
        run_web_server(tree_data, port=args.port)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()