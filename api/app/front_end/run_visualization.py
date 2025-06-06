#!/usr/bin/env python3
"""
MCTS Visualization Server

Standalone server that runs the MCTS visualization interface.
"""

import os
import sys
import argparse
import time
import logging
from typing import Dict, List, Any, Optional

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mctx.visualization")

# Make sure the module is in the path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
sys.path.append(project_root)

# Import MCTS and visualization components
try:
    from api.app.services.mcts_service import MCTSService
    from api.app.front_end.mcts_visualization import MCTSVisualization
    from api.app.front_end.onboarding import create_mcts_guided_tour
    from api.app.front_end.visualization_service import VisualizationService
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error(f"Make sure the project is in the Python path: {project_root}")
    sys.exit(1)


def create_dashboard() -> dash.Dash:
    """
    Create the MCTS visualization dashboard.
    
    Returns:
        A configured Dash app
    """
    # Initialize services
    try:
        mcts_service = MCTSService()
        vis_service = VisualizationService(mcts_service)
        vis = MCTSVisualization()
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        # Create empty visualization
        vis = MCTSVisualization()
    
    # Create mock data for initial visualization
    try:
        mock_data = vis_service.create_mock_visualization_data(
            num_nodes=75,
            branching_factor=3,
            max_depth=4
        )
    except Exception as e:
        logger.error(f"Failed to create mock data: {e}")
        # Use minimal mock data
        mock_data = {
            "node_count": 1,
            "visits": [1],
            "values": [0],
            "parents": {},
            "children": {0: []},
            "states": ["explored"],
            "search_type": "mock"
        }
    
    # Initialize app
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        meta_tags=[
            {"name": "viewport", "content": "width=device-width, initial-scale=1"}
        ],
        title="MCTS Visualization"
    )
    
    # Generate initial visualizations
    try:
        tree_fig = vis.visualize_tree(mock_data)
        metrics_fig = vis.create_metrics_panel(mock_data)
    except Exception as e:
        logger.error(f"Failed to create initial visualizations: {e}")
        # Create empty figures
        import plotly.graph_objects as go
        tree_fig = go.Figure()
        metrics_fig = go.Figure()
    
    # Create app layout
    app.layout = html.Div([
        html.Div([
            html.H1("Monte Carlo Tree Search Visualization", className="header-title"),
            html.P("An elegant, information-rich exploration of MCTS algorithms", 
                   className="header-subtitle")
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
                                ]),
                                html.Div([
                                    dbc.Button(
                                        "Reset View", 
                                        id="reset-button",
                                        color="primary",
                                        className="mr-2"
                                    ),
                                    dbc.Button(
                                        "Generate New Tree", 
                                        id="generate-button",
                                        color="secondary"
                                    )
                                ], className="d-flex justify-content-between mt-3")
                            ])
                        ])
                    ], className="mb-4"),
                    
                    # Tree settings
                    dbc.Card([
                        dbc.CardHeader("Tree Generation"),
                        dbc.CardBody([
                            dbc.Form([
                                dbc.FormGroup([
                                    dbc.Label("Number of Nodes"),
                                    dbc.Input(
                                        id="num-nodes-input",
                                        type="number",
                                        min=10,
                                        max=200,
                                        step=10,
                                        value=75
                                    )
                                ]),
                                dbc.FormGroup([
                                    dbc.Label("Branching Factor"),
                                    dbc.Input(
                                        id="branching-factor-input",
                                        type="number",
                                        min=2,
                                        max=5,
                                        step=1,
                                        value=3
                                    )
                                ]),
                                dbc.FormGroup([
                                    dbc.Label("Maximum Depth"),
                                    dbc.Input(
                                        id="max-depth-input",
                                        type="number",
                                        min=2,
                                        max=8,
                                        step=1,
                                        value=4
                                    )
                                ])
                            ])
                        ])
                    ], className="mb-4")
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
                                    style={"height": "70vh"}
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
                                    style={"height": "70vh"}
                                )
                            ], id="metrics-content", style={"display": "none"})
                        ])
                    ])
                ], md=9)
            ])
        ], fluid=True),
        
        # Store for tree data
        dcc.Store(id="tree-data-store", data=mock_data),
        
        # Loading indicator
        dbc.Spinner(html.Div(id="loading-output"), color="primary", spinner_style={"width": "3rem", "height": "3rem"})
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
                    box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.1), 0 2px 4px -1px rgba(15, 23, 42, 0.06);
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
                
                .btn-secondary {
                    background-color: #64748b;
                    border-color: #64748b;
                    transition: all 0.2s ease;
                }
                
                .btn-secondary:hover {
                    background-color: #475569;
                    border-color: #475569;
                    transform: translateY(-1px);
                    box-shadow: 0 4px 6px -1px rgba(100, 116, 139, 0.5);
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
        [Output("tree-data-store", "data"),
         Output("loading-output", "children")],
        [Input("generate-button", "n_clicks")],
        [State("num-nodes-input", "value"),
         State("branching-factor-input", "value"),
         State("max-depth-input", "value")]
    )
    def generate_new_tree(n_clicks, num_nodes, branching_factor, max_depth):
        """Generate a new tree based on the input parameters."""
        if not n_clicks:
            # Initial load - return the default data
            return dash.no_update, ""
        
        try:
            # Generate new mock tree data
            new_data = vis_service.create_mock_visualization_data(
                num_nodes=num_nodes,
                branching_factor=branching_factor,
                max_depth=max_depth
            )
            return new_data, ""
        except Exception as e:
            logger.error(f"Failed to generate new tree: {e}")
            return dash.no_update, f"Error: {str(e)}"
    
    @app.callback(
        [Output("tree-graph", "figure"),
         Output("metrics-graph", "figure")],
        [Input("tree-data-store", "data"),
         Input("layout-select", "value"),
         Input("size-select", "value")]
    )
    def update_visualization(tree_data, layout_type, size_by):
        """Update the visualization with new data or settings."""
        if not tree_data:
            return dash.no_update, dash.no_update
        
        try:
            # Generate tree visualization
            tree_fig = vis.visualize_tree(tree_data, layout_type=layout_type)
            
            # Generate metrics visualization
            metrics_fig = vis.create_metrics_panel(tree_data)
            
            return tree_fig, metrics_fig
        except Exception as e:
            logger.error(f"Failed to update visualization: {e}")
            # Return empty figures on error
            import plotly.graph_objects as go
            empty_fig = go.Figure()
            return empty_fig, empty_fig
    
    # Add guided tour
    try:
        create_mcts_guided_tour(app)
    except Exception as e:
        logger.error(f"Failed to create guided tour: {e}")
    
    return app


def main():
    """Run the visualization server."""
    parser = argparse.ArgumentParser(description="MCTS Visualization Server")
    parser.add_argument("--host", default="localhost", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8050, help="Port to run the server on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    # Create and run the dashboard
    try:
        app = create_dashboard()
        logger.info(f"Starting server on {args.host}:{args.port}")
        app.run_server(host=args.host, port=args.port, debug=args.debug)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()