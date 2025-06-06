"""
MCTX Visualization

Sophisticated visualization components for Monte Carlo Tree Search.
Provides real-time visualization of search trees, metrics, and performance data.
"""

import os
import time
import json
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from enum import Enum
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

from mctx._src.tree import Tree
from mctx.monitoring.metrics import SearchMetrics, MCTSMetricsCollector

# Configure logging
logger = logging.getLogger("mctx.monitoring.visualization")


class NodeState(str, Enum):
    """States for tree visualization nodes."""
    UNEXPLORED = "unexplored"
    EXPLORED = "explored"
    SELECTED = "selected"
    PRUNED = "pruned"
    OPTIMAL = "optimal"


@dataclass
class VisualizationConfig:
    """Configuration options for visualizations."""
    width: int = 800
    height: int = 600
    theme: str = "light"  # 'light' or 'dark'
    node_size_factor: float = 1.0
    edge_width_factor: float = 1.0
    animation_duration: int = 500  # ms
    color_palette: str = "default"  # 'default', 'colorblind', 'pastel'
    max_nodes_to_render: int = 500
    include_tooltips: bool = True
    include_animation: bool = True
    layout_type: str = "radial"  # 'radial', 'hierarchical', 'force'
    save_path: Optional[str] = None


class TreeVisualizer:
    """
    Sophisticated visualization for Monte Carlo search trees.
    
    Creates elegant, information-rich visualizations of MCTS trees
    with attention to detail and aesthetic refinement.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize the tree visualizer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        
        # Initialize color palettes
        self._init_color_palettes()
        
        # Track visualization history for animation
        self.visualization_history = []
        self.current_tree_data = None
    
    def _init_color_palettes(self) -> None:
        """Initialize color palettes for different themes and states."""
        # Default palette for light theme
        self.light_colors = {
            NodeState.UNEXPLORED: "#CBD5E1",  # Slate 300
            NodeState.EXPLORED: "#6366F1",    # Indigo 500
            NodeState.SELECTED: "#F59E0B",    # Amber 500
            NodeState.PRUNED: "#94A3B8",      # Slate 400
            NodeState.OPTIMAL: "#10B981",     # Emerald 500
        }
        
        # Default palette for dark theme
        self.dark_colors = {
            NodeState.UNEXPLORED: "#475569",  # Slate 600
            NodeState.EXPLORED: "#818CF8",    # Indigo 400
            NodeState.SELECTED: "#FBBF24",    # Amber 400
            NodeState.PRUNED: "#334155",      # Slate 700
            NodeState.OPTIMAL: "#34D399",     # Emerald 400
        }
        
        # Colorblind-friendly palette
        self.colorblind_colors = {
            NodeState.UNEXPLORED: "#A9A9A9",  # Gray
            NodeState.EXPLORED: "#0072B2",    # Blue
            NodeState.SELECTED: "#E69F00",    # Orange
            NodeState.PRUNED: "#56B4E9",      # Light blue
            NodeState.OPTIMAL: "#009E73",     # Green
        }
        
        # Pastel palette
        self.pastel_colors = {
            NodeState.UNEXPLORED: "#D3D3D3",  # Light gray
            NodeState.EXPLORED: "#B5D3E7",    # Pastel blue
            NodeState.SELECTED: "#FFD8B1",    # Pastel orange
            NodeState.PRUNED: "#E5E5E5",      # Silver
            NodeState.OPTIMAL: "#B1E5C2",     # Pastel green
        }
        
        # Set active color palette based on configuration
        self.node_colors = self._get_color_palette()
    
    def _get_color_palette(self) -> Dict[str, str]:
        """
        Get the appropriate color palette based on configuration.
        
        Returns:
            Dictionary mapping node states to colors
        """
        if self.config.color_palette == "colorblind":
            return self.colorblind_colors
        elif self.config.color_palette == "pastel":
            return self.pastel_colors
        else:  # default
            return self.light_colors if self.config.theme == "light" else self.dark_colors
    
    def _normalize_values(self, values: np.ndarray) -> np.ndarray:
        """
        Normalize values to [0, 1] range with pleasing distribution.
        
        Args:
            values: Array of values to normalize
            
        Returns:
            Normalized values
        """
        if values.size == 0:
            return np.array([])
            
        min_val, max_val = values.min(), values.max()
        if min_val == max_val:
            return np.zeros_like(values)
            
        # Use a sigmoid normalization for more aesthetic distribution
        rescaled = -3 + 6 * (values - min_val) / (max_val - min_val)
        sigmoid = 1 / (1 + np.exp(-rescaled))
        return sigmoid
    
    def _calculate_positions(self, tree_data: Dict[str, Any], 
                            layout_type: Optional[str] = None) -> np.ndarray:
        """
        Calculate node positions for visualization.
        
        Args:
            tree_data: Tree data for visualization
            layout_type: Type of layout to use
            
        Returns:
            Array of node positions
        """
        layout = layout_type or self.config.layout_type
        num_nodes = tree_data['node_count']
        positions = np.zeros((num_nodes, 2))
        
        if layout == "radial":
            self._calculate_radial_layout(tree_data, positions)
        elif layout == "hierarchical":
            self._calculate_hierarchical_layout(tree_data, positions)
        elif layout == "force":
            self._calculate_force_layout(tree_data, positions)
        else:
            # Default to radial layout
            self._calculate_radial_layout(tree_data, positions)
        
        return positions
    
    def _calculate_radial_layout(self, tree_data: Dict[str, Any], 
                                positions: np.ndarray) -> None:
        """
        Calculate a radial layout with perfect circular harmony.
        
        Args:
            tree_data: Tree data for visualization
            positions: Array to fill with node positions
        """
        parents = tree_data['parents']
        children = tree_data.get('children', {})
        
        # Set root at center
        positions[0] = [0, 0]
        
        # Group nodes by their depth
        nodes_by_depth = {0: [0]}
        max_depth = 0
        
        # Calculate depths
        for i in range(1, len(positions)):
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
        
        Args:
            tree_data: Tree data for visualization
            positions: Array to fill with node positions
        """
        parents = tree_data['parents']
        children = tree_data.get('children', {})
        
        # Set root at top
        positions[0] = [0, 0]
        
        # Group nodes by their depth
        nodes_by_depth = {0: [0]}
        max_depth = 0
        
        # Calculate depths
        for i in range(1, len(positions)):
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
    
    def _calculate_force_layout(self, tree_data: Dict[str, Any], 
                              positions: np.ndarray) -> None:
        """
        Calculate a force-directed layout.
        
        Args:
            tree_data: Tree data for visualization
            positions: Array to fill with node positions
        """
        parents = tree_data['parents']
        
        # Initialize with random positions
        for i in range(len(positions)):
            if i == 0:  # Root at center
                positions[i] = [0, 0]
            else:
                # Random position near parent
                if i in parents:
                    parent = parents[i]
                    parent_pos = positions[parent]
                    angle = np.random.uniform(0, 2 * np.pi)
                    distance = np.random.uniform(0.5, 1.5)
                    positions[i] = [
                        parent_pos[0] + distance * np.cos(angle),
                        parent_pos[1] + distance * np.sin(angle)
                    ]
        
        # Create edges
        edges = []
        for node, parent in parents.items():
            edges.append((parent, node))
        
        # Force-directed layout simulation
        iterations = 50
        k = 1.0  # Optimal distance
        temperature = 0.1 * len(positions)  # Initial temperature
        cooling_factor = 0.95  # Temperature cooling
        
        for _ in range(iterations):
            # Calculate repulsive forces
            forces = np.zeros_like(positions)
            
            # Node-node repulsion
            for i in range(len(positions)):
                for j in range(len(positions)):
                    if i != j:
                        delta = positions[i] - positions[j]
                        distance = np.linalg.norm(delta)
                        if distance < 0.01:  # Avoid division by zero
                            distance = 0.01
                            delta = np.array([0.01, 0.01])
                        
                        # Repulsive force
                        force = k*k / distance
                        forces[i] += force * delta / distance
            
            # Edge attraction
            for edge in edges:
                source, target = edge
                delta = positions[source] - positions[target]
                distance = np.linalg.norm(delta)
                if distance < 0.01:
                    continue
                
                # Attractive force
                force = distance*distance / k
                forces[source] -= force * delta / distance
                forces[target] += force * delta / distance
            
            # Apply forces with temperature limiting
            for i in range(len(positions)):
                force_mag = np.linalg.norm(forces[i])
                if force_mag > 0:
                    # Limit displacement to temperature
                    limit = min(force_mag, temperature)
                    positions[i] += forces[i] / force_mag * limit
            
            # Cool temperature
            temperature *= cooling_factor
    
    def _adjust_positions_for_parent_children(self, positions: np.ndarray, 
                                            parents: Dict[int, int], 
                                            children: Dict[int, List[int]], 
                                            nodes: List[int]) -> None:
        """
        Adjust positions to create visual harmony between nodes.
        
        Args:
            positions: Node positions to adjust
            parents: Mapping of nodes to their parents
            children: Mapping of nodes to their children
            nodes: List of nodes to adjust
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
    
    def _create_node_trace(self, positions: np.ndarray, visits: np.ndarray, 
                         values: np.ndarray, states: List[str]) -> go.Scatter:
        """
        Create a refined visualization of tree nodes.
        
        Args:
            positions: Node positions
            visits: Visit counts for nodes
            values: Values for nodes
            states: States for nodes
            
        Returns:
            A plotly trace for the nodes
        """
        # Normalize metrics for visual encoding
        norm_visits = self._normalize_values(visits)
        norm_values = self._normalize_values(values)
        
        # Calculate sizes with logarithmic scaling for perceptual accuracy
        size_factor = self.config.node_size_factor
        sizes = 10 + 15 * np.log1p(norm_visits * 9) / np.log(10)
        sizes = sizes * size_factor
        
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
                    color="#E2E8F0" if self.config.theme == "light" else "#1E293B",
                ),
                opacity=0.9,
            ),
            hoverinfo='text' if self.config.include_tooltips else 'none',
            hovertext=[
                f"Node {i}<br>Visits: {v}<br>Value: {val:.3f}<br>State: {s}"
                for i, (v, val, s) in enumerate(zip(visits, values, states))
            ] if self.config.include_tooltips else None,
        )
        
        return node_trace
    
    def _create_edge_trace(self, edges: List[Tuple[int, int]], 
                         positions: np.ndarray) -> go.Scatter:
        """
        Create a refined visualization of tree edges.
        
        Args:
            edges: List of edges as (parent, child) pairs
            positions: Node positions
            
        Returns:
            A plotly trace for the edges
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
        edge_width = max(1, self.config.edge_width_factor)
        edge_color = "#CBD5E1" if self.config.theme == "light" else "#475569"
        
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(
                width=edge_width,
                color=edge_color,
                shape='spline',
            ),
            hoverinfo='none',
            mode='lines',
        )
        
        return edge_trace
    
    def tree_to_visualization_data(self, tree: Tree) -> Dict[str, Any]:
        """
        Convert an MCTS tree to visualization data.
        
        Args:
            tree: The MCTS tree to visualize
            
        Returns:
            Dictionary with visualization data
        """
        # Extract tree data
        node_visits = np.asarray(tree.node_visits)
        node_values = np.asarray(tree.node_values)
        children_index = tree.children_index
        
        # Create parents mapping
        parents = {}
        for i, children in enumerate(children_index):
            for j, child in enumerate(children):
                if child >= 0:  # Valid child
                    parents[child] = i
        
        # Determine node states
        states = []
        for i in range(len(node_visits)):
            if node_visits[i] == 0:
                states.append(NodeState.UNEXPLORED)
            elif i == 0:  # Root
                states.append(NodeState.SELECTED)
            elif node_visits[i] > np.percentile(node_visits, 90):
                states.append(NodeState.OPTIMAL)
            else:
                states.append(NodeState.EXPLORED)
        
        # Create children mapping
        children = {}
        for parent, children_arr in enumerate(children_index):
            parent_children = []
            for child in children_arr:
                if child >= 0:  # Valid child
                    parent_children.append(child)
            children[parent] = parent_children
        
        # Construct visualization data
        vis_data = {
            "node_count": len(node_visits),
            "visits": node_visits.tolist(),
            "values": node_values.tolist(),
            "parents": parents,
            "children": children,
            "states": states
        }
        
        return vis_data
    
    def visualize_tree(self, tree_data: Dict[str, Any],
                      layout_type: Optional[str] = None,
                      width: Optional[int] = None,
                      height: Optional[int] = None,
                      title: Optional[str] = None) -> go.Figure:
        """
        Create a sophisticated visualization of an MCTS tree.
        
        Args:
            tree_data: Tree data for visualization
            layout_type: Type of layout to use
            width: Figure width
            height: Figure height
            title: Figure title
            
        Returns:
            A plotly Figure with the tree visualization
        """
        # Store current tree for animation
        self.current_tree_data = tree_data
        
        # Sample tree if it's too large
        max_nodes = self.config.max_nodes_to_render
        if tree_data["node_count"] > max_nodes:
            logger.info(f"Tree has {tree_data['node_count']} nodes, sampling to {max_nodes}")
            tree_data = self._sample_large_tree(tree_data, max_nodes)
        
        # Calculate node positions
        positions = self._calculate_positions(tree_data, layout_type)
        
        # Create edge list
        edges = []
        for node, parent in tree_data['parents'].items():
            edges.append((parent, node))
        
        # Create traces
        edge_trace = self._create_edge_trace(edges, positions)
        node_trace = self._create_node_trace(
            positions, 
            np.array(tree_data['visits']), 
            np.array(tree_data['values']), 
            tree_data['states']
        )
        
        # Create figure with refined styling
        fig = go.Figure(data=[edge_trace, node_trace])
        
        # Set width and height
        width = width or self.config.width
        height = height or self.config.height
        
        # Set background colors based on theme
        bg_color = "#F8FAFC" if self.config.theme == "light" else "#0F172A"
        text_color = "#334155" if self.config.theme == "light" else "#E2E8F0"
        
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
            plot_bgcolor=bg_color,
            paper_bgcolor=bg_color,
            font=dict(
                family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
                size=14,
                color=text_color
            ),
            title=dict(
                text=title or "Monte Carlo Tree Search Visualization",
                font=dict(
                    family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
                    size=18,
                    color=text_color
                ),
                x=0.5,
                y=0.98
            ),
            width=width,
            height=height,
            transition_duration=self.config.animation_duration,
        )
        
        # Add to visualization history if animation is enabled
        if self.config.include_animation:
            self.visualization_history.append(fig)
        
        # Save if path is provided
        if self.config.save_path:
            self._save_visualization(fig)
        
        return fig
    
    def _sample_large_tree(self, tree_data: Dict[str, Any], 
                         max_nodes: int) -> Dict[str, Any]:
        """
        Sample a large tree to reduce visualization complexity.
        
        Args:
            tree_data: The full tree data
            max_nodes: Maximum number of nodes to include
            
        Returns:
            Sampled tree data
        """
        # Always include root and high-value nodes
        visits = np.array(tree_data['visits'])
        values = np.array(tree_data['values'])
        
        # Calculate importance score (combination of visits and values)
        importance = visits * (1 + np.abs(values))
        
        # Nodes to definitely include
        must_include = {0}  # Always include root
        
        # Add nodes with highest importance
        top_indices = np.argsort(-importance)[:max_nodes//2]
        must_include.update(top_indices)
        
        # Add random sample of remaining nodes
        remaining = set(range(tree_data['node_count'])) - must_include
        if len(remaining) > max_nodes - len(must_include):
            sample_size = max_nodes - len(must_include)
            random_sample = np.random.choice(list(remaining), 
                                            size=sample_size, 
                                            replace=False)
            selected = must_include.union(random_sample)
        else:
            selected = must_include.union(remaining)
        
        # Create new tree data with selected nodes
        sampled_data = {
            "node_count": len(selected),
            "visits": [tree_data['visits'][i] for i in selected],
            "values": [tree_data['values'][i] for i in selected],
            "states": [tree_data['states'][i] for i in selected],
            "parents": {},
            "children": {}
        }
        
        # Create mapping from old to new indices
        old_to_new = {old: new for new, old in enumerate(sorted(selected))}
        
        # Update parents and children
        for old_idx in selected:
            if old_idx in tree_data['parents']:
                parent = tree_data['parents'][old_idx]
                if parent in old_to_new:  # Only include if parent is selected
                    sampled_data['parents'][old_to_new[old_idx]] = old_to_new[parent]
            
            if old_idx in tree_data['children']:
                children = [c for c in tree_data['children'][old_idx] if c in old_to_new]
                if children:
                    sampled_data['children'][old_to_new[old_idx]] = [old_to_new[c] for c in children]
                else:
                    sampled_data['children'][old_to_new[old_idx]] = []
        
        return sampled_data
    
    def create_metrics_panel(self, tree_data: Dict[str, Any],
                           width: Optional[int] = None,
                           height: Optional[int] = None) -> go.Figure:
        """
        Create a comprehensive panel of MCTS metrics.
        
        Args:
            tree_data: Tree data for visualization
            width: Figure width
            height: Figure height
            
        Returns:
            A plotly Figure with metrics visualizations
        """
        # Set width and height
        width = width or self.config.width
        height = height or self.config.height
        
        # Set colors based on theme
        bg_color = "#F8FAFC" if self.config.theme == "light" else "#0F172A"
        text_color = "#334155" if self.config.theme == "light" else "#E2E8F0"
        grid_color = "#E2E8F0" if self.config.theme == "light" else "#1E293B"
        
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
        visits = np.array(tree_data['visits'])
        primary_color = "#6366F1" if self.config.theme == "light" else "#818CF8"
        
        fig.add_trace(
            go.Histogram(
                x=visits,
                marker_color=primary_color,
                opacity=0.75,
                name="Visits",
                nbinsx=20,
                hovertemplate="Visits: %{x}<br>Count: %{y}"
            ),
            row=1, col=1
        )
        
        # Value distribution
        values = np.array(tree_data['values'])
        secondary_color = "#10B981" if self.config.theme == "light" else "#34D399"
        
        fig.add_trace(
            go.Histogram(
                x=values,
                marker_color=secondary_color,
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
        for node in range(len(visits)):
            if node == 0:  # Root node
                depths[node] = 0
            else:
                parent = parents.get(node)
                if parent is not None:
                    depths[node] = depths.get(parent, 0) + 1
        
        # Group metrics by depth
        depth_df = pd.DataFrame({
            'node': range(len(visits)),
            'depth': [depths.get(n, 0) for n in range(len(visits))],
            'visits': visits,
            'values': values
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
                    color=primary_color,
                    size=8
                ),
                line=dict(
                    color=primary_color,
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
        max_visits = np.max(visits)
        
        exploration = []
        exploitation = []
        
        for node in range(len(visits)):
            if visits[node] > 0:
                exploit = values[node]
                explore = c * np.sqrt(np.log(max_visits) / visits[node])
                exploration.append(explore)
                exploitation.append(exploit)
        
        highlight_color = "#F59E0B" if self.config.theme == "light" else "#FBBF24"
        
        fig.add_trace(
            go.Scatter(
                x=exploitation,
                y=exploration,
                mode='markers',
                marker=dict(
                    color=highlight_color,
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
            height=height,
            width=width,
            showlegend=False,
            plot_bgcolor=bg_color,
            paper_bgcolor=bg_color,
            font=dict(
                family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
                size=12,
                color=text_color
            ),
            margin=dict(l=50, r=20, t=60, b=50),
            hovermode="closest",
        )
        
        # Update axes with consistent styling
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor=grid_color,
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor=grid_color,
            title_font=dict(size=12, color=text_color),
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor=grid_color,
            title_font=dict(size=12, color=text_color),
        )
        
        # Set specific axis titles
        fig.update_xaxes(title_text="Visit Count", row=1, col=1)
        fig.update_xaxes(title_text="Node Value", row=1, col=2)
        fig.update_xaxes(title_text="Tree Depth", row=2, col=1)
        fig.update_xaxes(title_text="Exploitation (Value)", row=2, col=2)
        
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.update_yaxes(title_text="Mean Visits", row=2, col=1)
        fig.update_yaxes(title_text="Exploration (UCB)", row=2, col=2)
        
        # Save if path is provided
        if self.config.save_path:
            metrics_path = self.config.save_path.replace('.html', '_metrics.html')
            self._save_visualization(fig, metrics_path)
        
        return fig
    
    def visualize_metrics_over_time(self, metrics_history: List[SearchMetrics],
                                  width: Optional[int] = None,
                                  height: Optional[int] = None) -> go.Figure:
        """
        Create a visualization of MCTS metrics evolving over time.
        
        Args:
            metrics_history: List of metrics snapshots
            width: Figure width
            height: Figure height
            
        Returns:
            A plotly Figure with metrics visualizations
        """
        # Set width and height
        width = width or self.config.width
        height = height or self.config.height
        
        # Set colors based on theme
        bg_color = "#F8FAFC" if self.config.theme == "light" else "#0F172A"
        text_color = "#334155" if self.config.theme == "light" else "#E2E8F0"
        grid_color = "#E2E8F0" if self.config.theme == "light" else "#1E293B"
        
        # Create subplots for time-series metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Tree Growth", 
                "Visit Distribution",
                "Performance Metrics", 
                "Value Evolution"
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ]
        )
        
        # Convert metrics history to dataframe
        metrics_data = []
        for i, metrics in enumerate(metrics_history):
            data = metrics.as_dict()
            data['timestamp'] = i
            metrics_data.append(data)
        
        df = pd.DataFrame(metrics_data)
        
        # Set colors
        primary_color = "#6366F1" if self.config.theme == "light" else "#818CF8"
        secondary_color = "#10B981" if self.config.theme == "light" else "#34D399"
        tertiary_color = "#F59E0B" if self.config.theme == "light" else "#FBBF24"
        quaternary_color = "#EF4444" if self.config.theme == "light" else "#F87171"
        
        # Tree growth (tree size over time)
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['tree_size'],
                mode='lines+markers',
                name="Tree Size",
                line=dict(color=primary_color, width=2),
                marker=dict(color=primary_color, size=6),
                hovertemplate="Time: %{x}<br>Tree Size: %{y}"
            ),
            row=1, col=1
        )
        
        # Add max depth on the same plot
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['max_depth'],
                mode='lines+markers',
                name="Max Depth",
                line=dict(color=secondary_color, width=2, dash='dash'),
                marker=dict(color=secondary_color, size=6),
                hovertemplate="Time: %{x}<br>Max Depth: %{y}"
            ),
            row=1, col=1
        )
        
        # Visit distribution over time (max visits and avg visits)
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['max_visits'],
                mode='lines+markers',
                name="Max Visits",
                line=dict(color=tertiary_color, width=2),
                marker=dict(color=tertiary_color, size=6),
                hovertemplate="Time: %{x}<br>Max Visits: %{y}"
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['avg_visits'],
                mode='lines+markers',
                name="Avg Visits",
                line=dict(color=quaternary_color, width=2, dash='dash'),
                marker=dict(color=quaternary_color, size=6),
                hovertemplate="Time: %{x}<br>Avg Visits: %{y:.2f}"
            ),
            row=1, col=2
        )
        
        # Performance metrics (iterations and simulations per second)
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['iterations_per_second'],
                mode='lines+markers',
                name="Iterations/s",
                line=dict(color=primary_color, width=2),
                marker=dict(color=primary_color, size=6),
                hovertemplate="Time: %{x}<br>Iterations/s: %{y:.2f}"
            ),
            row=2, col=1
        )
        
        # Add memory usage on secondary axis if available
        if 'memory_usage_mb' in df.columns and not df['memory_usage_mb'].isna().all():
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['memory_usage_mb'],
                    mode='lines+markers',
                    name="Memory (MB)",
                    line=dict(color=quaternary_color, width=2, dash='dot'),
                    marker=dict(color=quaternary_color, size=6),
                    hovertemplate="Time: %{x}<br>Memory: %{y:.1f} MB",
                    yaxis="y3"
                ),
                row=2, col=1
            )
            
            # Add secondary y-axis for memory
            fig.update_layout(
                yaxis3=dict(
                    title="Memory Usage (MB)",
                    overlaying="y2",
                    side="right",
                    showgrid=False,
                    title_font=dict(color=quaternary_color)
                )
            )
        
        # Value evolution (min, max, avg value)
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['min_value'],
                mode='lines',
                name="Min Value",
                line=dict(color=quaternary_color, width=1, dash='dot'),
                hovertemplate="Time: %{x}<br>Min Value: %{y:.3f}"
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['max_value'],
                mode='lines',
                name="Max Value",
                line=dict(color=tertiary_color, width=1, dash='dot'),
                hovertemplate="Time: %{x}<br>Max Value: %{y:.3f}"
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['avg_value'],
                mode='lines+markers',
                name="Avg Value",
                line=dict(color=secondary_color, width=2),
                marker=dict(color=secondary_color, size=6),
                hovertemplate="Time: %{x}<br>Avg Value: %{y:.3f}"
            ),
            row=2, col=2
        )
        
        # Apply sophisticated styling to layout
        fig.update_layout(
            height=height,
            width=width,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=10)
            ),
            plot_bgcolor=bg_color,
            paper_bgcolor=bg_color,
            font=dict(
                family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
                size=12,
                color=text_color
            ),
            margin=dict(l=50, r=50, t=60, b=50),
            hovermode="x unified",
        )
        
        # Update axes with consistent styling
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor=grid_color,
            title_font=dict(size=12, color=text_color),
            title_text="Time Step"
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor=grid_color,
            title_font=dict(size=12, color=text_color),
        )
        
        # Set specific axis titles
        fig.update_yaxes(title_text="Node Count", row=1, col=1)
        fig.update_yaxes(title_text="Visit Count", row=1, col=2)
        fig.update_yaxes(title_text="Iterations/s", row=2, col=1)
        fig.update_yaxes(title_text="Node Value", row=2, col=2)
        
        # Save if path is provided
        if self.config.save_path:
            time_path = self.config.save_path.replace('.html', '_time_series.html')
            self._save_visualization(fig, time_path)
        
        return fig
    
    def create_animation(self, width: Optional[int] = None,
                       height: Optional[int] = None) -> str:
        """
        Create an animation of tree evolution.
        
        Args:
            width: Figure width
            height: Figure height
            
        Returns:
            Path to the saved animation file
        """
        if not self.visualization_history or len(self.visualization_history) < 2:
            logger.warning("Not enough frames for animation")
            return ""
        
        if not self.config.save_path:
            logger.warning("No save path specified for animation")
            return ""
        
        # Set width and height
        width = width or self.config.width
        height = height or self.config.height
        
        # Create animation path
        animation_path = self.config.save_path.replace('.html', '_animation.html')
        
        # Create figure with animation frames
        frames = self.visualization_history
        
        # Create a slider for animation control
        steps = []
        for i, _ in enumerate(frames):
            step = {
                "args": [
                    [i],
                    {
                        "frame": {"duration": 300, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 300}
                    }
                ],
                "label": str(i),
                "method": "animate"
            }
            steps.append(step)
        
        sliders = [{
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 16},
                "prefix": "Frame: ",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 300, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": steps
        }]
        
        # Create animation
        fig = frames[0]
        fig.update(frames=[go.Frame(data=frame.data) for frame in frames])
        fig.update_layout(
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [
                                None, 
                                {
                                    "frame": {"duration": 300, "redraw": True},
                                    "fromcurrent": True, 
                                    "transition": {"duration": 300}
                                }
                            ],
                            "label": "Play",
                            "method": "animate"
                        },
                        {
                            "args": [
                                [None], 
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}
                                }
                            ],
                            "label": "Pause",
                            "method": "animate"
                        }
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top"
                }
            ],
            sliders=sliders,
            width=width,
            height=height
        )
        
        # Save animation
        fig.write_html(animation_path)
        logger.info(f"Saved animation to {animation_path}")
        
        return animation_path
    
    def _save_visualization(self, fig: go.Figure, 
                          path: Optional[str] = None) -> None:
        """
        Save visualization to file.
        
        Args:
            fig: The figure to save
            path: Path to save to (default: self.config.save_path)
        """
        save_path = path or self.config.save_path
        if not save_path:
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        # Determine file type and save
        if save_path.endswith('.html'):
            fig.write_html(save_path)
        elif save_path.endswith('.png'):
            fig.write_image(save_path)
        elif save_path.endswith('.jpg') or save_path.endswith('.jpeg'):
            fig.write_image(save_path)
        elif save_path.endswith('.svg'):
            fig.write_image(save_path)
        elif save_path.endswith('.pdf'):
            fig.write_image(save_path)
        else:
            # Default to HTML
            fig.write_html(save_path)
        
        logger.info(f"Saved visualization to {save_path}")


class MCTSMonitor:
    """
    Real-time monitoring and visualization for MCTS processes.
    
    Provides a comprehensive suite of tools for tracking, visualizing,
    and analyzing Monte Carlo Tree Search algorithms.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None,
                 metrics_collector: Optional[MCTSMetricsCollector] = None,
                 auto_update_interval: int = 1000):
        """
        Initialize the MCTS monitor.
        
        Args:
            config: Visualization configuration
            metrics_collector: Metrics collector to use
            auto_update_interval: Interval for automatic updates (ms)
        """
        self.config = config or VisualizationConfig()
        self.metrics_collector = metrics_collector or MCTSMetricsCollector()
        self.auto_update_interval = auto_update_interval
        
        self.visualizer = TreeVisualizer(config)
        self.metrics_history = []
        
        self.is_monitoring = False
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        
        self.current_tree = None
        self.current_metrics = None
        
        # For real-time updates in notebooks
        self.is_jupyter = self._check_if_jupyter()
    
    def _check_if_jupyter(self) -> bool:
        """
        Check if running in a Jupyter environment.
        
        Returns:
            True if in Jupyter, False otherwise
        """
        try:
            from IPython import get_ipython
            if get_ipython() is None:
                return False
            if 'IPKernelApp' in get_ipython().config:
                return True
            return False
        except ImportError:
            return False
    
    def start_monitoring(self, tree: Optional[Tree] = None) -> None:
        """
        Start monitoring an MCTS process.
        
        Args:
            tree: The MCTS tree to monitor
        """
        if self.is_monitoring:
            logger.warning("Monitoring is already active")
            return
        
        self.current_tree = tree
        self.metrics_history = []
        self.stop_monitoring.clear()
        self.is_monitoring = True
        
        # Start metrics collection
        self.metrics_collector.start_collection()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info(f"Started MCTS monitoring with update interval of {self.auto_update_interval}ms")
    
    def stop_monitoring(self) -> SearchMetrics:
        """
        Stop monitoring and return final metrics.
        
        Returns:
            Final collected metrics
        """
        if not self.is_monitoring:
            logger.warning("Monitoring is not active")
            return self.current_metrics or SearchMetrics()
        
        # Signal thread to stop
        self.stop_monitoring.set()
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=1.0)
        
        self.is_monitoring = False
        
        # Stop metrics collection
        final_metrics = self.metrics_collector.stop_collection()
        self.current_metrics = final_metrics
        
        # Add final metrics to history
        self.metrics_history.append(final_metrics)
        
        logger.info("Stopped MCTS monitoring")
        return final_metrics
    
    def update_tree(self, tree: Tree) -> None:
        """
        Update the monitored tree.
        
        Args:
            tree: New tree to monitor
        """
        self.current_tree = tree
        
        # Update metrics if monitoring is active
        if self.is_monitoring:
            self.metrics_collector.update_tree_metrics(tree)
    
    def _monitoring_loop(self) -> None:
        """Monitoring loop that updates metrics and visualizations."""
        while not self.stop_monitoring.is_set():
            # Update metrics if we have a tree
            if self.current_tree is not None:
                self.metrics_collector.update_tree_metrics(self.current_tree)
                current_metrics = self.metrics_collector.metrics
                
                # Clone metrics to add to history
                metrics_copy = SearchMetrics()
                metrics_copy.update(current_metrics)
                self.metrics_history.append(metrics_copy)
                
                # Update current metrics
                self.current_metrics = current_metrics
            
            # Wait for next update
            self.stop_monitoring.wait(self.auto_update_interval / 1000)
    
    def visualize_current_tree(self) -> go.Figure:
        """
        Visualize the current tree.
        
        Returns:
            Plotly figure with tree visualization
        """
        if self.current_tree is None:
            logger.warning("No tree available for visualization")
            return go.Figure()
        
        # Convert tree to visualization data
        tree_data = self.visualizer.tree_to_visualization_data(self.current_tree)
        
        # Create visualization
        return self.visualizer.visualize_tree(tree_data)
    
    def visualize_current_metrics(self) -> go.Figure:
        """
        Visualize the current metrics.
        
        Returns:
            Plotly figure with metrics visualization
        """
        if self.current_tree is None:
            logger.warning("No tree available for metrics visualization")
            return go.Figure()
        
        # Convert tree to visualization data
        tree_data = self.visualizer.tree_to_visualization_data(self.current_tree)
        
        # Create metrics panel
        return self.visualizer.create_metrics_panel(tree_data)
    
    def visualize_metrics_history(self) -> go.Figure:
        """
        Visualize metrics history.
        
        Returns:
            Plotly figure with metrics history visualization
        """
        if not self.metrics_history:
            logger.warning("No metrics history available")
            return go.Figure()
        
        # Create time-series visualization
        return self.visualizer.visualize_metrics_over_time(self.metrics_history)
    
    def create_dashboard(self, height: int = 1000) -> None:
        """
        Create an interactive dashboard for monitoring.
        
        Args:
            height: Height of the dashboard in pixels
        """
        if not self.is_jupyter:
            logger.warning("Dashboard is only available in Jupyter environment")
            return
        
        try:
            from IPython.display import display, HTML, clear_output
            import ipywidgets as widgets
            
            # Create tabs for different visualizations
            tab = widgets.Tab()
            
            # Tree visualization tab
            tree_output = widgets.Output()
            with tree_output:
                fig = self.visualize_current_tree()
                display(fig)
            
            # Metrics tab
            metrics_output = widgets.Output()
            with metrics_output:
                fig = self.visualize_current_metrics()
                display(fig)
            
            # History tab
            history_output = widgets.Output()
            with history_output:
                if self.metrics_history:
                    fig = self.visualize_metrics_history()
                    display(fig)
                else:
                    print("No metrics history available yet")
            
            # Settings tab
            settings_output = widgets.Output()
            with settings_output:
                # Create settings widgets
                theme_dropdown = widgets.Dropdown(
                    options=['light', 'dark'],
                    value=self.config.theme,
                    description='Theme:',
                    disabled=False,
                )
                
                layout_dropdown = widgets.Dropdown(
                    options=['radial', 'hierarchical', 'force'],
                    value=self.config.layout_type,
                    description='Layout:',
                    disabled=False,
                )
                
                node_size_slider = widgets.FloatSlider(
                    value=self.config.node_size_factor,
                    min=0.5,
                    max=2.0,
                    step=0.1,
                    description='Node Size:',
                    disabled=False,
                    continuous_update=False,
                    orientation='horizontal',
                    readout=True,
                    readout_format='.1f',
                )
                
                update_interval_slider = widgets.IntSlider(
                    value=self.auto_update_interval,
                    min=500,
                    max=5000,
                    step=500,
                    description='Update (ms):',
                    disabled=False,
                    continuous_update=False,
                    orientation='horizontal',
                    readout=True,
                    readout_format='d',
                )
                
                # Function to update settings
                def on_change(change):
                    if change['owner'] == theme_dropdown:
                        self.config.theme = change['new']
                        self.visualizer.config.theme = change['new']
                        self.visualizer.node_colors = self.visualizer._get_color_palette()
                    elif change['owner'] == layout_dropdown:
                        self.config.layout_type = change['new']
                        self.visualizer.config.layout_type = change['new']
                    elif change['owner'] == node_size_slider:
                        self.config.node_size_factor = change['new']
                        self.visualizer.config.node_size_factor = change['new']
                    elif change['owner'] == update_interval_slider:
                        self.auto_update_interval = change['new']
                
                # Register callbacks
                theme_dropdown.observe(on_change, names='value')
                layout_dropdown.observe(on_change, names='value')
                node_size_slider.observe(on_change, names='value')
                update_interval_slider.observe(on_change, names='value')
                
                # Display widgets
                display(widgets.VBox([
                    widgets.HTML("<h3>Visualization Settings</h3>"),
                    theme_dropdown,
                    layout_dropdown,
                    node_size_slider,
                    update_interval_slider
                ]))
            
            # Add tabs
            tab.children = [tree_output, metrics_output, history_output, settings_output]
            tab.set_title(0, 'Tree')
            tab.set_title(1, 'Metrics')
            tab.set_title(2, 'History')
            tab.set_title(3, 'Settings')
            
            # Create buttons
            refresh_button = widgets.Button(
                description='Refresh',
                disabled=False,
                button_style='primary',
                tooltip='Refresh visualizations',
                icon='refresh'
            )
            
            save_button = widgets.Button(
                description='Save',
                disabled=False,
                button_style='success',
                tooltip='Save visualizations',
                icon='save'
            )
            
            # Button callbacks
            def on_refresh_clicked(b):
                with tree_output:
                    clear_output(wait=True)
                    fig = self.visualize_current_tree()
                    display(fig)
                
                with metrics_output:
                    clear_output(wait=True)
                    fig = self.visualize_current_metrics()
                    display(fig)
                
                with history_output:
                    clear_output(wait=True)
                    if self.metrics_history:
                        fig = self.visualize_metrics_history()
                        display(fig)
                    else:
                        print("No metrics history available yet")
            
            def on_save_clicked(b):
                # Set save path and save visualizations
                save_dir = 'mcts_visualizations'
                os.makedirs(save_dir, exist_ok=True)
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                
                # Save tree visualization
                save_path = f"{save_dir}/tree_{timestamp}.html"
                self.config.save_path = save_path
                self.visualizer.config.save_path = save_path
                
                # Save current visualizations
                if self.current_tree is not None:
                    tree_data = self.visualizer.tree_to_visualization_data(self.current_tree)
                    self.visualizer.visualize_tree(tree_data)
                    self.visualizer.create_metrics_panel(tree_data)
                
                # Save metrics history
                if self.metrics_history:
                    self.visualizer.visualize_metrics_over_time(self.metrics_history)
                
                # Create animation if possible
                if len(self.visualizer.visualization_history) > 1:
                    self.visualizer.create_animation()
                
                # Reset save path
                self.config.save_path = None
                self.visualizer.config.save_path = None
                
                with settings_output:
                    print(f"Visualizations saved to {save_dir}")
            
            refresh_button.on_click(on_refresh_clicked)
            save_button.on_click(on_save_clicked)
            
            # Create auto-refresh checkbox
            auto_refresh = widgets.Checkbox(
                value=True,
                description='Auto-refresh',
                disabled=False,
                indent=False
            )
            
            # Create status text
            status_text = widgets.HTML(
                value="<div style='color: green;'>Status: Ready</div>"
            )
            
            # Auto-refresh function
            def auto_refresh_visualizations():
                while auto_refresh.value and self.is_monitoring:
                    on_refresh_clicked(None)
                    time.sleep(self.auto_update_interval / 1000)
                    
                    # Update status
                    if self.current_metrics:
                        metrics = self.current_metrics
                        status_text.value = (
                            f"<div style='color: green;'>Status: Active | "
                            f"Tree size: {metrics.tree_size} | "
                            f"Depth: {metrics.max_depth} | "
                            f"Iterations: {metrics.iterations}/s</div>"
                        )
            
            # Start auto-refresh thread
            auto_refresh_thread = threading.Thread(
                target=auto_refresh_visualizations,
                daemon=True
            )
            auto_refresh_thread.start()
            
            # Create main dashboard
            dashboard = widgets.VBox([
                widgets.HBox([refresh_button, save_button, auto_refresh, status_text]),
                tab
            ])
            
            # Set height
            dashboard.layout.height = f"{height}px"
            
            # Display dashboard
            display(dashboard)
            
        except ImportError as e:
            logger.error(f"Required package not installed: {e}")
            print("To use the dashboard, install required packages:")
            print("pip install ipywidgets")
    
    def save_visualizations(self, save_dir: str = 'mcts_visualizations') -> None:
        """
        Save all visualizations to files.
        
        Args:
            save_dir: Directory to save visualizations in
        """
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Set save paths
        tree_path = f"{save_dir}/tree_{timestamp}.html"
        metrics_path = f"{save_dir}/metrics_{timestamp}.html"
        history_path = f"{save_dir}/history_{timestamp}.html"
        animation_path = f"{save_dir}/animation_{timestamp}.html"
        
        # Set save path on visualizer
        self.config.save_path = tree_path
        self.visualizer.config.save_path = tree_path
        
        # Save current tree visualization
        if self.current_tree is not None:
            tree_data = self.visualizer.tree_to_visualization_data(self.current_tree)
            self.visualizer.visualize_tree(tree_data)
            
            # Save metrics panel
            self.config.save_path = metrics_path
            self.visualizer.config.save_path = metrics_path
            self.visualizer.create_metrics_panel(tree_data)
        
        # Save metrics history
        if self.metrics_history:
            self.config.save_path = history_path
            self.visualizer.config.save_path = history_path
            self.visualizer.visualize_metrics_over_time(self.metrics_history)
        
        # Create animation if possible
        if len(self.visualizer.visualization_history) > 1:
            self.config.save_path = animation_path
            self.visualizer.config.save_path = animation_path
            self.visualizer.create_animation()
        
        # Reset save path
        self.config.save_path = None
        self.visualizer.config.save_path = None
        
        logger.info(f"Saved visualizations to {save_dir}")
    
    def export_metrics_history(self, filepath: str) -> None:
        """
        Export metrics history to a file.
        
        Args:
            filepath: Path to save metrics history
        """
        if not self.metrics_history:
            logger.warning("No metrics history to export")
            return
        
        # Convert metrics to list of dicts
        metrics_dicts = []
        for i, metrics in enumerate(self.metrics_history):
            data = metrics.as_dict()
            data['timestamp'] = i
            metrics_dicts.append(data)
        
        # Determine file type
        if filepath.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(metrics_dicts, f, indent=2)
        elif filepath.endswith('.csv'):
            df = pd.DataFrame(metrics_dicts)
            df.to_csv(filepath, index=False)
        else:
            # Default to JSON
            with open(filepath, 'w') as f:
                json.dump(metrics_dicts, f, indent=2)
        
        logger.info(f"Exported metrics history to {filepath}")