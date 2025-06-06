"""
MCTX Tree Visualization

An elegant, information-rich visualization of Monte Carlo Tree Search processes.
Follows Jony Ive's principles of clarity, purpose, and refinement in every detail.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from .design_system import (
    Colors, Typography, Spacing, Shadows, BorderRadius, 
    Animation, InteractionPatterns, ZIndex, Layout
)


class NodeState:
    """Tree node states with associated visual properties."""
    UNEXPLORED = "unexplored"
    EXPLORED = "explored"
    SELECTED = "selected"
    PRUNED = "pruned"
    OPTIMAL = "optimal"


class MCTSVisualization:
    """
    Monte Carlo Tree Search visualization with exquisite attention to detail.
    
    This class renders MCTS trees with a focus on clarity, information density,
    and refined aesthetics that would meet Jony Ive's exacting standards.
    """
    
    def __init__(self):
        """Initialize the visualization module."""
        # Create a color palette for node states
        self.node_colors = {
            NodeState.UNEXPLORED: Colors.SLATE_300,
            NodeState.EXPLORED: Colors.INDIGO_500,
            NodeState.SELECTED: Colors.AMBER_500,
            NodeState.PRUNED: Colors.SLATE_200,
            NodeState.OPTIMAL: Colors.EMERALD_500,
        }
        
        # Initialize tracking for animations
        self.animation_frames = []
        self.current_tree = None
        
        # Create sequential color scales for metrics
        self.visit_color_scale = Colors.get_sequential_palette(Colors.INDIGO_500, 9)
        self.value_color_scale = Colors.get_sequential_palette(Colors.EMERALD_500, 9)
        
    def _normalize_values(self, values: np.ndarray) -> np.ndarray:
        """
        Normalize values to [0, 1] range with a pleasing distribution.
        
        Uses an advanced normalization that preserves important differences
        while creating visual harmony.
        """
        if values.size == 0:
            return np.array([])
            
        # Apply sigmoid normalization for a more natural distribution
        min_val, max_val = values.min(), values.max()
        if min_val == max_val:
            return np.zeros_like(values)
            
        # Rescale to [-3, 3] range for sigmoid function
        rescaled = -3 + 6 * (values - min_val) / (max_val - min_val)
        # Apply sigmoid to create a pleasing distribution
        sigmoid = 1 / (1 + np.exp(-rescaled))
        return sigmoid
        
    def _create_node_trace(self, positions: np.ndarray, visits: np.ndarray, 
                          values: np.ndarray, states: List[str]) -> go.Scatter:
        """
        Create a visually refined node trace.
        
        Applies sophisticated styling to nodes based on their metrics
        and states, with subtle visual cues for information hierarchy.
        """
        # Normalize metrics for visual encoding
        norm_visits = self._normalize_values(visits)
        norm_values = self._normalize_values(values)
        
        # Calculate sizes with logarithmic scaling for perceptual accuracy
        sizes = 10 + 15 * np.log1p(norm_visits * 9) / np.log(10)
        
        # Create color array
        colors = [self.node_colors[state] for state in states]
        
        # Create node trace with refined styling
        node_trace = go.Scatter(
            x=positions[:, 0],
            y=positions[:, 1],
            mode='markers',
            marker=dict(
                size=sizes,
                color=colors,
                line=dict(
                    width=1,
                    color=Colors.SLATE_100,
                ),
                opacity=0.9,
            ),
            hoverinfo='text',
            hovertext=[
                f"Visits: {v}<br>Value: {val:.3f}<br>State: {s}"
                for v, val, s in zip(visits, values, states)
            ],
        )
        
        return node_trace
        
    def _create_edge_trace(self, edges: List[Tuple[int, int]], 
                          positions: np.ndarray) -> go.Scatter:
        """
        Create a visually refined edge trace.
        
        Applies sophisticated styling to edges with subtle
        thickness variation and perfect curvature.
        """
        # Create arrays for edge lines
        edge_x = []
        edge_y = []
        
        # Create lines for each edge with subtle curves
        for edge in edges:
            src, tgt = edge
            x0, y0 = positions[src]
            x1, y1 = positions[tgt]
            
            # Calculate control points for subtle curves
            ctrl_x = (x0 + x1) / 2
            ctrl_y = (y0 + y1) / 2 - 0.05
            
            # Add points for cubic bezier curve
            points = 10
            for i in range(points + 1):
                t = i / points
                # Quadratic bezier formula
                x = (1-t)**2 * x0 + 2*(1-t)*t * ctrl_x + t**2 * x1
                y = (1-t)**2 * y0 + 2*(1-t)*t * ctrl_y + t**2 * y1
                edge_x.append(x)
                edge_y.append(y)
            
            # Add None to create breaks between edges
            edge_x.append(None)
            edge_y.append(None)
        
        # Create edge trace with refined styling
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(
                width=1,
                color=Colors.SLATE_300,
                shape='spline',
            ),
            hoverinfo='none',
            mode='lines',
        )
        
        return edge_trace
        
    def _calculate_positions(self, tree_data: Dict[str, Any], 
                            layout_type: str = 'radial') -> np.ndarray:
        """
        Calculate visually pleasing node positions.
        
        Uses advanced layout algorithms with perfect spacing
        and visual balance.
        """
        num_nodes = tree_data['node_count']
        positions = np.zeros((num_nodes, 2))
        
        if layout_type == 'radial':
            # Radial layout with perfect circular harmony
            self._calculate_radial_layout(tree_data, positions)
        elif layout_type == 'hierarchical':
            # Hierarchical layout with perfect vertical rhythm
            self._calculate_hierarchical_layout(tree_data, positions)
        
        return positions
        
    def _calculate_radial_layout(self, tree_data: Dict[str, Any], 
                                positions: np.ndarray) -> None:
        """
        Calculate a radial layout with perfect circular harmony.
        
        Places nodes in concentric circles with mathematically
        precise angular distribution.
        """
        parents = tree_data['parents']
        children = tree_data.get('children', {})
        
        # Set root at center
        positions[0] = [0, 0]
        
        # Group nodes by their depth
        nodes_by_depth = {0: [0]}
        max_depth = 0
        
        # Calculate depths
        for i in range(1, len(parents)):
            if i in parents:  # Skip nodes that don't exist
                parent = parents[i]
                depth = 1
                current = parent
                while current != 0:
                    current = parents[current]
                    depth += 1
                
                if depth not in nodes_by_depth:
                    nodes_by_depth[depth] = []
                nodes_by_depth[depth].append(i)
                max_depth = max(max_depth, depth)
        
        # Golden ratio for radius growth - aesthetically perfect
        radius_factor = 1.618
        
        # Place nodes at each depth
        for depth in range(1, max_depth + 1):
            if depth not in nodes_by_depth:
                continue
                
            nodes = nodes_by_depth[depth]
            radius = depth * radius_factor
            
            # Place nodes in a circle at this depth
            for i, node in enumerate(nodes):
                angle = 2 * np.pi * i / len(nodes)
                positions[node] = [
                    radius * np.cos(angle),
                    radius * np.sin(angle)
                ]
                
            # Adjust positions to avoid overlap with parent nodes
            self._adjust_positions_for_parent_children(positions, parents, children, nodes)
    
    def _calculate_hierarchical_layout(self, tree_data: Dict[str, Any], 
                                      positions: np.ndarray) -> None:
        """
        Calculate a hierarchical layout with perfect vertical rhythm.
        
        Places nodes in layers with mathematically precise
        horizontal distribution.
        """
        parents = tree_data['parents']
        children = tree_data.get('children', {})
        
        # Set root at top
        positions[0] = [0, 0]
        
        # Group nodes by their depth
        nodes_by_depth = {0: [0]}
        max_depth = 0
        
        # Calculate depths
        for i in range(1, len(parents)):
            if i in parents:  # Skip nodes that don't exist
                parent = parents[i]
                depth = 1
                current = parent
                while current != 0:
                    current = parents[current]
                    depth += 1
                
                if depth not in nodes_by_depth:
                    nodes_by_depth[depth] = []
                nodes_by_depth[depth].append(i)
                max_depth = max(max_depth, depth)
        
        # Place nodes at each depth
        vertical_spacing = 2.0
        
        for depth in range(1, max_depth + 1):
            if depth not in nodes_by_depth:
                continue
                
            nodes = nodes_by_depth[depth]
            y = -depth * vertical_spacing  # Negative to grow downward
            
            # Group nodes by their parent
            nodes_by_parent = {}
            for node in nodes:
                parent = parents[node]
                if parent not in nodes_by_parent:
                    nodes_by_parent[parent] = []
                nodes_by_parent[parent].append(node)
            
            # Place nodes for each parent
            for parent, parent_nodes in nodes_by_parent.items():
                parent_x = positions[parent][0]
                width = len(parent_nodes) - 1
                width = max(width, 1)  # Ensure minimum width
                
                for i, node in enumerate(parent_nodes):
                    # Calculate offset from parent
                    if len(parent_nodes) == 1:
                        offset = 0
                    else:
                        offset = -width/2 + i * width/(len(parent_nodes)-1)
                    
                    x = parent_x + offset
                    positions[node] = [x, y]
    
    def _adjust_positions_for_parent_children(self, positions: np.ndarray, 
                                             parents: Dict[int, int], 
                                             children: Dict[int, List[int]], 
                                             nodes: List[int]) -> None:
        """
        Adjust positions to create visual harmony between parent and child nodes.
        
        Applies subtle position refinements to emphasize parent-child
        relationships while maintaining overall layout clarity.
        """
        # For each node, slightly adjust position toward its parent
        attraction_factor = 0.15
        
        for node in nodes:
            parent = parents[node]
            parent_pos = positions[parent]
            node_pos = positions[node]
            
            # Vector from node to parent
            vec = parent_pos - node_pos
            
            # Apply subtle attraction toward parent
            positions[node] = node_pos + vec * attraction_factor
            
        # Apply repulsion between siblings
        repulsion_factor = 0.05
        for parent, parent_children in children.items():
            for i, node1 in enumerate(parent_children):
                if node1 not in nodes:
                    continue
                    
                for node2 in parent_children[i+1:]:
                    if node2 not in nodes:
                        continue
                        
                    # Calculate repulsion between siblings
                    pos1 = positions[node1]
                    pos2 = positions[node2]
                    
                    # Vector from node2 to node1
                    vec = pos1 - pos2
                    dist = np.linalg.norm(vec)
                    
                    if dist < 0.001:  # Avoid division by zero
                        continue
                        
                    # Normalize and apply repulsion
                    vec = vec / dist
                    force = repulsion_factor / (dist ** 2)
                    
                    positions[node1] += vec * force
                    positions[node2] -= vec * force
    
    def visualize_tree(self, tree_data: Dict[str, Any], 
                      layout_type: str = 'radial',
                      width: int = 800, 
                      height: int = 600) -> go.Figure:
        """
        Create an elegant visualization of an MCTS tree.
        
        Args:
            tree_data: Dictionary containing tree structure:
                - 'parents': Dict mapping node ID to parent ID
                - 'visits': Array of visit counts
                - 'values': Array of node values
                - 'states': List of node states
                - 'children': Dict mapping node ID to list of children
                - 'node_count': Total number of nodes
            layout_type: 'radial' or 'hierarchical'
            width: Figure width
            height: Figure height
            
        Returns:
            A plotly Figure object with the tree visualization
        """
        # Store current tree for later use
        self.current_tree = tree_data
        
        # Calculate node positions
        positions = self._calculate_positions(tree_data, layout_type)
        
        # Create edge list
        edges = []
        for node, parent in tree_data['parents'].items():
            if node != 0:  # Skip root node
                edges.append((parent, node))
        
        # Create traces
        edge_trace = self._create_edge_trace(edges, positions)
        node_trace = self._create_node_trace(
            positions, 
            tree_data['visits'], 
            tree_data['values'], 
            tree_data['states']
        )
        
        # Create figure with refined styling
        fig = go.Figure(data=[edge_trace, node_trace])
        
        # Apply sophisticated styling to layout
        fig.update_layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            plot_bgcolor=Colors.SLATE_100,
            paper_bgcolor=Colors.SLATE_100,
            font=dict(
                family=Typography.FONT_FAMILY["system"],
                size=14,
                color=Colors.SLATE_700
            ),
            title=dict(
                text="Monte Carlo Tree Search Visualization",
                font=dict(
                    family=Typography.FONT_FAMILY["system"],
                    size=18,
                    color=Colors.SLATE_900
                ),
                x=0.5,
                y=0.98
            ),
            width=width,
            height=height,
            transition_duration=500,
        )
        
        return fig
    
    def create_metrics_panel(self, tree_data: Dict[str, Any]) -> go.Figure:
        """
        Create an elegant panel of MCTS metrics.
        
        Displays key metrics with sophisticated data visualization
        and perfect visual harmony.
        """
        # Create subplots for metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Visit Distribution", 
                "Value Distribution",
                "Visit Count Over Depth", 
                "Exploration vs. Exploitation"
            ),
            specs=[
                [{"type": "histogram"}, {"type": "histogram"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ]
        )
        
        # Visit distribution
        fig.add_trace(
            go.Histogram(
                x=tree_data['visits'],
                marker_color=Colors.INDIGO_500,
                opacity=0.75,
                name="Visits",
                nbinsx=20,
                hovertemplate="Visits: %{x}<br>Count: %{y}"
            ),
            row=1, col=1
        )
        
        # Value distribution
        fig.add_trace(
            go.Histogram(
                x=tree_data['values'],
                marker_color=Colors.EMERALD_500,
                opacity=0.75,
                name="Values",
                nbinsx=20,
                hovertemplate="Value: %{x}<br>Count: %{y}"
            ),
            row=1, col=2
        )
        
        # Calculate metrics by depth
        parents = tree_data['parents']
        depths = {}
        for node in range(len(tree_data['visits'])):
            if node == 0:  # Root node
                depths[node] = 0
            else:
                parent = parents.get(node)
                if parent is not None:
                    depths[node] = depths.get(parent, 0) + 1
        
        # Group metrics by depth
        depth_df = pd.DataFrame({
            'node': range(len(tree_data['visits'])),
            'depth': [depths.get(n, 0) for n in range(len(tree_data['visits']))],
            'visits': tree_data['visits'],
            'values': tree_data['values']
        })
        
        depth_stats = depth_df.groupby('depth').agg({
            'visits': ['mean', 'sum', 'count'],
            'values': ['mean']
        }).reset_index()
        
        depth_stats.columns = ['depth', 'mean_visits', 'sum_visits', 'count', 'mean_value']
        
        # Visits by depth
        fig.add_trace(
            go.Scatter(
                x=depth_stats['depth'],
                y=depth_stats['mean_visits'],
                mode='lines+markers',
                marker=dict(
                    color=Colors.INDIGO_500,
                    size=8
                ),
                line=dict(
                    color=Colors.INDIGO_500,
                    width=2
                ),
                name="Mean Visits",
                hovertemplate="Depth: %{x}<br>Mean Visits: %{y:.2f}"
            ),
            row=2, col=1
        )
        
        # Exploration vs. exploitation
        # Calculate UCB1 score components
        c = 1.41  # Exploration constant
        max_visits = np.max(tree_data['visits'])
        
        exploration = []
        exploitation = []
        
        for node in range(len(tree_data['visits'])):
            if tree_data['visits'][node] > 0:
                exploit = tree_data['values'][node]
                explore = c * np.sqrt(np.log(max_visits) / tree_data['visits'][node])
                exploration.append(explore)
                exploitation.append(exploit)
        
        fig.add_trace(
            go.Scatter(
                x=exploitation,
                y=exploration,
                mode='markers',
                marker=dict(
                    color=Colors.AMBER_500,
                    size=8,
                    opacity=0.7
                ),
                name="Nodes",
                hovertemplate="Exploitation: %{x:.3f}<br>Exploration: %{y:.3f}"
            ),
            row=2, col=2
        )
        
        # Apply sophisticated styling to layout
        fig.update_layout(
            height=600,
            showlegend=False,
            plot_bgcolor=Colors.SLATE_100,
            paper_bgcolor=Colors.SLATE_100,
            font=dict(
                family=Typography.FONT_FAMILY["system"],
                size=12,
                color=Colors.SLATE_700
            ),
            margin=dict(l=50, r=20, t=60, b=50),
            hovermode="closest",
        )
        
        # Update axes
        fig.update_xaxes(
            title_text="Visit Count",
            row=1, col=1,
            title_font=dict(size=12, color=Colors.SLATE_700),
            showgrid=True,
            gridwidth=1,
            gridcolor=Colors.SLATE_200,
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor=Colors.SLATE_300,
        )
        
        fig.update_xaxes(
            title_text="Node Value",
            row=1, col=2,
            title_font=dict(size=12, color=Colors.SLATE_700),
            showgrid=True,
            gridwidth=1,
            gridcolor=Colors.SLATE_200,
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor=Colors.SLATE_300,
        )
        
        fig.update_xaxes(
            title_text="Tree Depth",
            row=2, col=1,
            title_font=dict(size=12, color=Colors.SLATE_700),
            showgrid=True,
            gridwidth=1,
            gridcolor=Colors.SLATE_200,
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor=Colors.SLATE_300,
        )
        
        fig.update_xaxes(
            title_text="Exploitation (Value)",
            row=2, col=2,
            title_font=dict(size=12, color=Colors.SLATE_700),
            showgrid=True,
            gridwidth=1,
            gridcolor=Colors.SLATE_200,
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor=Colors.SLATE_300,
        )
        
        fig.update_yaxes(
            title_text="Count",
            row=1, col=1,
            title_font=dict(size=12, color=Colors.SLATE_700),
            showgrid=True,
            gridwidth=1,
            gridcolor=Colors.SLATE_200,
        )
        
        fig.update_yaxes(
            title_text="Count",
            row=1, col=2,
            title_font=dict(size=12, color=Colors.SLATE_700),
            showgrid=True,
            gridwidth=1,
            gridcolor=Colors.SLATE_200,
        )
        
        fig.update_yaxes(
            title_text="Mean Visits",
            row=2, col=1,
            title_font=dict(size=12, color=Colors.SLATE_700),
            showgrid=True,
            gridwidth=1,
            gridcolor=Colors.SLATE_200,
        )
        
        fig.update_yaxes(
            title_text="Exploration (UCB)",
            row=2, col=2,
            title_font=dict(size=12, color=Colors.SLATE_700),
            showgrid=True,
            gridwidth=1,
            gridcolor=Colors.SLATE_200,
        )
        
        return fig
    
    def create_dashboard(self, tree_data: Dict[str, Any]) -> dash.Dash:
        """
        Create an exquisite dashboard for MCTS visualization.
        
        Combines tree visualization, metrics, and interactive controls
        with perfect visual harmony and intentional user experience.
        """
        # Initialize app with sophisticated styling
        app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            meta_tags=[
                {"name": "viewport", "content": "width=device-width, initial-scale=1"}
            ]
        )
        
        # Create custom CSS for refined styling
        app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>MCTS Visualization</title>
                {%favicon%}
                {%css%}
                <style>
                    body {
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                        background-color: ''' + Colors.SLATE_100 + ''';
                        color: ''' + Colors.SLATE_700 + ''';
                        line-height: 1.5;
                    }
                    
                    .header {
                        background-color: ''' + Colors.SLATE_900 + ''';
                        color: ''' + Colors.SLATE_100 + ''';
                        padding: 1rem 2rem;
                        margin-bottom: 2rem;
                        box-shadow: ''' + Shadows.ELEVATIONS["md"] + ''';
                    }
                    
                    h1, h2, h3, h4, h5, h6 {
                        font-weight: 600;
                        color: ''' + Colors.SLATE_900 + ''';
                    }
                    
                    .card {
                        border-radius: ''' + BorderRadius.RADIUS["lg"] + ''';
                        box-shadow: ''' + Shadows.ELEVATIONS["sm"] + ''';
                        transition: ''' + Animation.transition("all") + ''';
                        overflow: hidden;
                        height: 100%;
                    }
                    
                    .card:hover {
                        box-shadow: ''' + Shadows.ELEVATIONS["md"] + ''';
                        transform: translateY(-2px);
                    }
                    
                    .card-header {
                        background-color: ''' + Colors.SLATE_200 + ''';
                        border-bottom: 1px solid ''' + Colors.SLATE_300 + ''';
                        padding: 0.75rem 1.25rem;
                        font-weight: 600;
                    }
                    
                    .btn-primary {
                        background-color: ''' + Colors.INDIGO_500 + ''';
                        border-color: ''' + Colors.INDIGO_500 + ''';
                        transition: ''' + Animation.transition("all") + ''';
                    }
                    
                    .btn-primary:hover {
                        background-color: ''' + Colors.INDIGO_600 + ''';
                        border-color: ''' + Colors.INDIGO_600 + ''';
                        transform: translateY(-1px);
                        box-shadow: ''' + Shadows.ELEVATIONS["sm"] + ''';
                    }
                    
                    .btn-primary:active {
                        background-color: ''' + Colors.INDIGO_600 + ''';
                        border-color: ''' + Colors.INDIGO_600 + ''';
                        transform: translateY(1px);
                    }
                    
                    .form-control:focus {
                        border-color: ''' + Colors.INDIGO_500 + ''';
                        box-shadow: ''' + Shadows.ELEVATIONS["focus"] + ''';
                    }
                    
                    .nav-pills .nav-link.active {
                        background-color: ''' + Colors.INDIGO_500 + ''';
                    }
                    
                    .nav-link {
                        color: ''' + Colors.SLATE_700 + ''';
                    }
                    
                    .nav-link:hover {
                        color: ''' + Colors.INDIGO_600 + ''';
                    }
                    
                    .metrics-card {
                        display: flex;
                        flex-direction: column;
                        justify-content: center;
                        align-items: center;
                        padding: 1.5rem;
                        text-align: center;
                    }
                    
                    .metric-value {
                        font-size: 2rem;
                        font-weight: 600;
                        color: ''' + Colors.INDIGO_500 + ''';
                        margin-bottom: 0.5rem;
                    }
                    
                    .metric-label {
                        font-size: 0.875rem;
                        color: ''' + Colors.SLATE_500 + ''';
                        text-transform: uppercase;
                        letter-spacing: 0.05em;
                    }
                    
                    .legend-item {
                        display: flex;
                        align-items: center;
                        margin-bottom: 0.5rem;
                    }
                    
                    .legend-color {
                        width: 16px;
                        height: 16px;
                        border-radius: 4px;
                        margin-right: 0.5rem;
                    }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Monte Carlo Tree Search Visualization</h1>
                    <p class="lead mb-0">Explore the search process with sophisticated visual precision</p>
                </div>
                {%app_entry%}
                <footer>
                    {%config%}
                    {%scripts%}
                    {%renderer%}
                </footer>
            </body>
        </html>
        '''
        
        # Generate initial visualizations
        tree_fig = self.visualize_tree(tree_data)
        metrics_fig = self.create_metrics_panel(tree_data)
        
        # Calculate key metrics
        max_visits = np.max(tree_data['visits'])
        avg_value = np.mean(tree_data['values'])
        explored_nodes = np.sum(tree_data['visits'] > 0)
        exploration_rate = explored_nodes / len(tree_data['visits'])
        
        # Create legend items
        legend_items = []
        for state, color in self.node_colors.items():
            legend_items.append(
                html.Div([
                    html.Div(className="legend-color", style={"backgroundColor": color}),
                    html.Div(state.capitalize())
                ], className="legend-item")
            )
        
        # Layout with visual refinement and perfect spacing
        app.layout = html.Div([
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
                                            "Animate Search", 
                                            id="animate-button",
                                            color="secondary"
                                        )
                                    ], className="d-flex justify-content-between mt-3")
                                ])
                            ])
                        ], className="mb-4"),
                        
                        # Legend card
                        dbc.Card([
                            dbc.CardHeader("Node Legend"),
                            dbc.CardBody(legend_items)
                        ], className="mb-4"),
                        
                        # Key metrics cards
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    html.Div([
                                        html.Div(f"{max_visits}", className="metric-value"),
                                        html.Div("Max Visits", className="metric-label")
                                    ], className="metrics-card")
                                ])
                            ], width=6, className="mb-3"),
                            dbc.Col([
                                dbc.Card([
                                    html.Div([
                                        html.Div(f"{avg_value:.3f}", className="metric-value"),
                                        html.Div("Avg Value", className="metric-label")
                                    ], className="metrics-card")
                                ])
                            ], width=6, className="mb-3")
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    html.Div([
                                        html.Div(f"{explored_nodes}", className="metric-value"),
                                        html.Div("Explored Nodes", className="metric-label")
                                    ], className="metrics-card")
                                ])
                            ], width=6, className="mb-3"),
                            dbc.Col([
                                dbc.Card([
                                    html.Div([
                                        html.Div(f"{exploration_rate:.1%}", className="metric-value"),
                                        html.Div("Exploration Rate", className="metric-label")
                                    ], className="metrics-card")
                                ])
                            ], width=6, className="mb-3")
                        ])
                    ], md=3),
                    
                    # Main visualization column
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
            ], fluid=True)
        ])
        
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
            Output("tree-graph", "figure"),
            [Input("layout-select", "value"),
             Input("size-select", "value"),
             Input("reset-button", "n_clicks"),
             Input("animate-button", "n_clicks")]
        )
        def update_visualization(layout_type, size_by, reset_clicks, animate_clicks):
            """Update tree visualization based on controls."""
            ctx = dash.callback_context
            trigger = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else ""
            
            if trigger == "animate-button":
                # Start animation sequence
                # In a real implementation, this would generate frames based on search steps
                pass
            
            # Update visualization
            return self.visualize_tree(tree_data, layout_type=layout_type)
        
        return app
    
    def create_animation_frames(self, search_steps: List[Dict[str, Any]]) -> None:
        """
        Create animation frames from search steps.
        
        Generates a sequence of frames showing the tree's evolution,
        with sophisticated transitions between states.
        """
        self.animation_frames = []
        
        for step in search_steps:
            # Create tree visualization for this step
            frame = self.visualize_tree(step)
            self.animation_frames.append(frame)
            
    def play_animation(self, interval: int = 500) -> dash.Dash:
        """
        Create an animation player for the search process.
        
        Args:
            interval: Time between frames in ms
            
        Returns:
            A Dash app with animation controls
        """
        # Initialize app
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        # Layout
        app.layout = html.Div([
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.H1("MCTS Search Animation", className="mt-4 mb-4"),
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Graph(
                                    id="animation-graph",
                                    figure=self.animation_frames[0] if self.animation_frames else go.Figure(),
                                    config={
                                        "displayModeBar": True,
                                        "scrollZoom": True,
                                    },
                                    style={"height": "60vh"}
                                ),
                            ])
                        ]),
                        html.Div([
                            dbc.Button("Play", id="play-button", color="primary", className="mr-2"),
                            dbc.Button("Pause", id="pause-button", color="secondary", className="mr-2"),
                            dbc.Button("Reset", id="reset-animation", color="secondary", className="mr-2"),
                            html.Span("Frame: ", className="mr-2"),
                            html.Span("0", id="frame-counter")
                        ], className="d-flex align-items-center mt-3")
                    ])
                ])
            ], fluid=True),
            dcc.Interval(id="animation-interval", interval=interval, disabled=True),
            dcc.Store(id="current-frame", data=0)
        ])
        
        @app.callback(
            Output("animation-interval", "disabled"),
            [Input("play-button", "n_clicks"),
             Input("pause-button", "n_clicks"),
             Input("reset-animation", "n_clicks")],
            [State("animation-interval", "disabled")]
        )
        def toggle_animation(play, pause, reset, disabled):
            """Toggle animation playback."""
            ctx = dash.callback_context
            if not ctx.triggered:
                return True
                
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            
            if button_id == "play-button":
                return False
            elif button_id == "pause-button":
                return True
            elif button_id == "reset-animation":
                return True
            
            return disabled
            
        @app.callback(
            [Output("animation-graph", "figure"),
             Output("current-frame", "data"),
             Output("frame-counter", "children")],
            [Input("animation-interval", "n_intervals"),
             Input("reset-animation", "n_clicks")],
            [State("current-frame", "data")]
        )
        def update_animation_frame(n_intervals, reset_clicks, current_frame):
            """Update animation to the next frame."""
            ctx = dash.callback_context
            trigger = ctx.triggered[0]["prop_id"].split(".")[0]
            
            if trigger == "reset-animation":
                return self.animation_frames[0], 0, "0"
                
            if not self.animation_frames:
                return go.Figure(), 0, "0"
                
            next_frame = (current_frame + 1) % len(self.animation_frames)
            return self.animation_frames[next_frame], next_frame, str(next_frame)
            
        return app