#!/usr/bin/env python3
"""
Interactive MCTS Visualization

A sophisticated visualization interface for exploring Monte Carlo Tree Search
with an elegant, information-rich design system inspired by Jony Ive's
principles of clarity, purpose, and refinement.
"""

import os
import sys
import argparse
import numpy as np
from typing import Dict, List, Any
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import MCTS service and visualization components
from api.app.models.mcts_models import MCTSRequest, SearchParams, RootInput
from api.app.services.mcts_service import MCTSService
from api.app.front_end.mcts_visualization import MCTSVisualization, NodeState
from api.app.front_end.onboarding import create_mcts_guided_tour


def convert_mcts_result_to_visualization_data(result: Dict[str, Any], search_type: str) -> Dict[str, Any]:
    """
    Convert MCTS search result to visualization-friendly format.
    
    Args:
        result: The search result from the MCTS service
        search_type: Type of search algorithm used
        
    Returns:
        Dictionary containing visualization data
    """
    # Extract search tree from result
    search_tree = result.get("search_tree", {})
    
    # Extract node data
    node_visits = search_tree.get("node_visits", [])
    node_values = search_tree.get("node_values", [])
    children_index = search_tree.get("children_index", [])
    parents = search_tree.get("parents", {})
    
    # Calculate node states
    states = []
    for i in range(len(node_visits)):
        if node_visits[i] == 0:
            states.append(NodeState.UNEXPLORED)
        elif i in result.get("selected_path", []):
            states.append(NodeState.SELECTED)
        elif node_visits[i] > np.percentile(node_visits, 90):
            states.append(NodeState.OPTIMAL)
        else:
            states.append(NodeState.EXPLORED)
    
    # Create children mapping
    children = {}
    for parent, actions in enumerate(children_index):
        parent_children = []
        for action, child in enumerate(actions):
            if child != -1:  # -1 represents unvisited
                parent_children.append(child)
        children[parent] = parent_children
    
    # Construct visualization data
    vis_data = {
        "node_count": len(node_visits),
        "visits": node_visits,
        "values": node_values,
        "parents": parents,
        "children": children,
        "states": states,
        "search_type": search_type
    }
    
    return vis_data


def create_interactive_dashboard(service: MCTSService) -> dash.Dash:
    """
    Create an interactive dashboard for MCTS visualization.
    
    Args:
        service: The MCTS service to use for searches
        
    Returns:
        A Dash app with the interactive dashboard
    """
    # Initialize app with sophisticated styling
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        meta_tags=[
            {"name": "viewport", "content": "width=device-width, initial-scale=1"}
        ],
        title="MCTS Interactive Visualization"
    )
    
    # Initialize visualization
    vis = MCTSVisualization()
    
    # Create layout
    app.layout = html.Div([
        html.Div([
            html.H1("Monte Carlo Tree Search Visualization", className="header-title"),
            html.P("Explore and understand MCTS algorithms through interactive visualization",
                   className="header-subtitle")
        ], className="header"),
        
        dbc.Container([
            dbc.Row([
                # Configuration column
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Search Configuration"),
                        dbc.CardBody([
                            dbc.Form([
                                dbc.FormGroup([
                                    dbc.Label("Search Algorithm"),
                                    dbc.Select(
                                        id="algorithm-select",
                                        options=[
                                            {"label": "MuZero", "value": "muzero"},
                                            {"label": "Gumbel MuZero", "value": "gumbel_muzero"},
                                            {"label": "Stochastic MuZero", "value": "stochastic_muzero"}
                                        ],
                                        value="gumbel_muzero"
                                    )
                                ]),
                                dbc.FormGroup([
                                    dbc.Label("Batch Size"),
                                    dbc.Input(
                                        id="batch-size-input",
                                        type="number",
                                        min=1,
                                        max=32,
                                        step=1,
                                        value=4
                                    )
                                ]),
                                dbc.FormGroup([
                                    dbc.Label("Number of Actions"),
                                    dbc.Input(
                                        id="num-actions-input",
                                        type="number",
                                        min=2,
                                        max=64,
                                        step=1,
                                        value=16
                                    )
                                ]),
                                dbc.FormGroup([
                                    dbc.Label("Number of Simulations"),
                                    dbc.Input(
                                        id="num-simulations-input",
                                        type="number",
                                        min=10,
                                        max=1000,
                                        step=10,
                                        value=50
                                    )
                                ]),
                                dbc.FormGroup([
                                    dbc.Label("Optimization"),
                                    dbc.Checklist(
                                        id="optimization-checklist",
                                        options=[
                                            {"label": "Use T4 Optimizations", "value": "t4"},
                                            {"label": "Use Distributed MCTS", "value": "distributed"}
                                        ],
                                        value=[]
                                    )
                                ]),
                                dbc.FormGroup([
                                    dbc.Label("Precision"),
                                    dbc.RadioItems(
                                        id="precision-radio",
                                        options=[
                                            {"label": "FP16 (half precision)", "value": "fp16"},
                                            {"label": "FP32 (full precision)", "value": "fp32"}
                                        ],
                                        value="fp16",
                                        inline=True
                                    )
                                ]),
                                dbc.Button(
                                    "Run Search",
                                    id="run-search-button",
                                    color="primary",
                                    className="mt-3 w-100"
                                )
                            ])
                        ])
                    ], className="mb-4"),
                    
                    dbc.Card([
                        dbc.CardHeader("Visualization Options"),
                        dbc.CardBody([
                            dbc.Form([
                                dbc.FormGroup([
                                    dbc.Label("Layout Type"),
                                    dbc.Select(
                                        id="layout-select",
                                        options=[
                                            {"label": "Radial", "value": "radial"},
                                            {"label": "Hierarchical", "value": "hierarchical"}
                                        ],
                                        value="radial"
                                    )
                                ]),
                                dbc.FormGroup([
                                    dbc.Label("Node Size By"),
                                    dbc.RadioItems(
                                        id="size-select",
                                        options=[
                                            {"label": "Visit Count", "value": "visits"},
                                            {"label": "Node Value", "value": "values"}
                                        ],
                                        value="visits",
                                        inline=True
                                    )
                                ])
                            ])
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
                                html.Div(
                                    id="visualization-placeholder",
                                    className="visualization-placeholder",
                                    children=[
                                        html.Div([
                                            html.I(className="fas fa-tree"),
                                            html.H3("Run a search to visualize the MCTS tree"),
                                            html.P("Configure your search parameters and click 'Run Search'")
                                        ], className="placeholder-content")
                                    ]
                                ),
                                dcc.Graph(
                                    id="tree-graph",
                                    config={
                                        "displayModeBar": True,
                                        "scrollZoom": True,
                                        "responsive": True
                                    },
                                    style={"height": "70vh", "display": "none"}
                                )
                            ], id="tree-content"),
                            
                            # Metrics panel
                            html.Div([
                                html.Div(
                                    id="metrics-placeholder",
                                    className="metrics-placeholder",
                                    children=[
                                        html.Div([
                                            html.I(className="fas fa-chart-bar"),
                                            html.H3("Run a search to see metrics"),
                                            html.P("Configure your search parameters and click 'Run Search'")
                                        ], className="placeholder-content")
                                    ]
                                ),
                                dcc.Graph(
                                    id="metrics-graph",
                                    config={
                                        "displayModeBar": True,
                                        "responsive": True
                                    },
                                    style={"height": "70vh", "display": "none"}
                                )
                            ], id="metrics-content", style={"display": "none"})
                        ])
                    ]),
                    
                    # Results info card
                    dbc.Card([
                        dbc.CardHeader("Search Results"),
                        dbc.CardBody([
                            html.Div([
                                html.Div([
                                    html.H5("Duration", className="metric-label"),
                                    html.Div(id="duration-value", className="metric-value", children="--")
                                ], className="result-metric"),
                                html.Div([
                                    html.H5("Expanded Nodes", className="metric-label"),
                                    html.Div(id="expanded-nodes-value", className="metric-value", children="--")
                                ], className="result-metric"),
                                html.Div([
                                    html.H5("Max Depth", className="metric-label"),
                                    html.Div(id="max-depth-value", className="metric-value", children="--")
                                ], className="result-metric"),
                                html.Div([
                                    html.H5("Optimization", className="metric-label"),
                                    html.Div(id="optimization-value", className="metric-value", children="--")
                                ], className="result-metric")
                            ], className="result-metrics-container")
                        ])
                    ], className="mt-4")
                ], md=9)
            ])
        ], fluid=True),
        
        # Store components for state management
        dcc.Store(id="search-result-store"),
        dcc.Store(id="visualization-data-store"),
        
        # Loading indicator
        dbc.Spinner(html.Div(id="loading-output"), color="primary", fullscreen=True, type="grow")
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
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
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
                
                .visualization-placeholder, .metrics-placeholder {
                    height: 70vh;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    background-color: #f8fafc;
                    border-radius: 0.25rem;
                }
                
                .placeholder-content {
                    text-align: center;
                    color: #64748b;
                }
                
                .placeholder-content i {
                    font-size: 3rem;
                    margin-bottom: 1rem;
                }
                
                .result-metrics-container {
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: space-between;
                    margin: -0.5rem;
                }
                
                .result-metric {
                    flex: 1 1 calc(25% - 2rem);
                    min-width: 150px;
                    margin: 0.5rem;
                    text-align: center;
                }
                
                .metric-label {
                    font-size: 0.875rem;
                    color: #64748b;
                    margin-bottom: 0.25rem;
                    font-weight: 500;
                }
                
                .metric-value {
                    font-size: 1.5rem;
                    font-weight: 600;
                    color: #334155;
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
        [Output("search-result-store", "data"),
         Output("loading-output", "children")],
        [Input("run-search-button", "n_clicks")],
        [State("algorithm-select", "value"),
         State("batch-size-input", "value"),
         State("num-actions-input", "value"),
         State("num-simulations-input", "value"),
         State("optimization-checklist", "value"),
         State("precision-radio", "value")]
    )
    def run_search(n_clicks, algorithm, batch_size, num_actions, num_simulations, 
                  optimizations, precision):
        """Run MCTS search with the specified parameters."""
        if not n_clicks:
            return None, ""
            
        # Create mock root input
        prior_logits = [np.random.normal(0, 1, num_actions).tolist() for _ in range(batch_size)]
        value = np.random.normal(0, 1, batch_size).tolist()
        embedding = [np.zeros(10).tolist() for _ in range(batch_size)]
        
        root_input = RootInput(
            prior_logits=prior_logits,
            value=value,
            embedding=embedding,
            batch_size=batch_size,
            num_actions=num_actions
        )
        
        # Create search parameters
        search_params = SearchParams(
            num_simulations=num_simulations,
            max_depth=None,
            max_num_considered_actions=min(16, num_actions),
            dirichlet_fraction=0.25 if algorithm == "muzero" else None,
            dirichlet_alpha=0.3 if algorithm == "muzero" else None,
            use_t4_optimizations="t4" in optimizations,
            precision=precision,
            tensor_core_aligned=True,
            distributed="distributed" in optimizations,
            num_devices=2 if "distributed" in optimizations else 1,
            partition_batch=True
        )
        
        # Create the request
        request = MCTSRequest(
            root_input=root_input,
            search_params=search_params,
            search_type=algorithm,
            device_type="gpu"
        )
        
        # Run the search
        result = service.run_search(request, user_id="interactive_visualization")
        
        # Return the result
        return result.dict(), ""
    
    @app.callback(
        [Output("visualization-data-store", "data"),
         Output("duration-value", "children"),
         Output("expanded-nodes-value", "children"),
         Output("max-depth-value", "children"),
         Output("optimization-value", "children")],
        [Input("search-result-store", "data")]
    )
    def process_search_result(result):
        """Process the search result for visualization."""
        if not result:
            return None, "--", "--", "--", "--"
            
        # Extract statistics
        stats = result.get("search_statistics", {})
        duration = f"{stats.get('duration_ms', 0):.2f} ms"
        expanded_nodes = str(stats.get("num_expanded_nodes", 0))
        max_depth = str(stats.get("max_depth_reached", 0))
        
        # Determine optimization type
        optimized = stats.get("optimized", False)
        if optimized:
            if result.get("distributed_stats"):
                optimization = "Distributed"
            else:
                optimization = "T4"
        else:
            optimization = "None"
        
        # Convert result to visualization data
        vis_data = convert_mcts_result_to_visualization_data(
            result, 
            result.get("search_type", "unknown")
        )
        
        return vis_data, duration, expanded_nodes, max_depth, optimization
    
    @app.callback(
        [Output("visualization-placeholder", "style"),
         Output("metrics-placeholder", "style"),
         Output("tree-graph", "figure"),
         Output("tree-graph", "style"),
         Output("metrics-graph", "figure"),
         Output("metrics-graph", "style")],
        [Input("visualization-data-store", "data"),
         Input("layout-select", "value")]
    )
    def update_visualization(vis_data, layout_type):
        """Update the visualization with the search result."""
        if not vis_data:
            return (
                {"display": "flex"}, {"display": "flex"},
                {}, {"display": "none"},
                {}, {"display": "none"}
            )
            
        # Create tree visualization
        tree_fig = vis.visualize_tree(vis_data, layout_type=layout_type)
        
        # Create metrics visualization
        metrics_fig = vis.create_metrics_panel(vis_data)
        
        return (
            {"display": "none"}, {"display": "none"},
            tree_fig, {"display": "block", "height": "70vh"},
            metrics_fig, {"display": "block", "height": "70vh"}
        )
    
    # Add guided tour
    create_mcts_guided_tour(app)
    
    return app


def main():
    """Run the interactive visualization app."""
    parser = argparse.ArgumentParser(description="Interactive MCTS Visualization")
    parser.add_argument("--host", default="localhost", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8050, help="Port to run the server on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    # Create MCTS service
    service = MCTSService()
    
    # Create and run app
    app = create_interactive_dashboard(service)
    app.run_server(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()