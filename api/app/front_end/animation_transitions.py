"""
Animation and Microinteraction System

A sophisticated animation system for MCTS visualization interfaces
with delicate, purposeful transitions that enhance understanding.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import plotly.graph_objects as go

from .design_system import Animation


class AnimationSystem:
    """
    Sophisticated animation system for MCTS visualizations.
    
    Creates delicate, meaningful transitions between states with
    proper easing functions and meticulous timing.
    """
    
    def __init__(self):
        """Initialize the animation system."""
        # Define easing functions
        self.easings = {
            "linear": lambda t: t,
            "ease-in": lambda t: t * t,
            "ease-out": lambda t: t * (2 - t),
            "ease-in-out": lambda t: t * t * (3 - 2 * t),
            "elastic": lambda t: 1 - np.cos(t * np.pi * 4.5) * np.exp(-t * 6)
        }
        
    def interpolate_positions(self, start_pos: np.ndarray, end_pos: np.ndarray, 
                             progress: float, easing: str = "ease-in-out") -> np.ndarray:
        """
        Interpolate node positions with perfect easing.
        
        Applies sophisticated easing functions to create natural movement
        between visualization states.
        
        Args:
            start_pos: Starting positions array
            end_pos: Ending positions array
            progress: Animation progress (0.0 to 1.0)
            easing: Type of easing function to apply
            
        Returns:
            Interpolated positions array
        """
        # Apply easing function to progress
        easing_fn = self.easings.get(easing, self.easings["ease-in-out"])
        eased_progress = easing_fn(progress)
        
        # Interpolate positions
        return start_pos * (1 - eased_progress) + end_pos * eased_progress
    
    def interpolate_colors(self, start_colors: List[str], end_colors: List[str],
                          progress: float, easing: str = "ease-in-out") -> List[str]:
        """
        Interpolate node colors with perfect transitions.
        
        Creates smooth color transitions with perceptually uniform
        color interpolation.
        
        Args:
            start_colors: Starting color list
            end_colors: Ending color list
            progress: Animation progress (0.0 to 1.0)
            easing: Type of easing function to apply
            
        Returns:
            Interpolated color list
        """
        # Apply easing function to progress
        easing_fn = self.easings.get(easing, self.easings["ease-in-out"])
        eased_progress = easing_fn(progress)
        
        # Convert hex colors to RGB for interpolation
        start_rgb = [self._hex_to_rgb(color) for color in start_colors]
        end_rgb = [self._hex_to_rgb(color) for color in end_colors]
        
        # Interpolate RGB values
        interpolated_rgb = []
        for i in range(len(start_rgb)):
            r = start_rgb[i][0] * (1 - eased_progress) + end_rgb[i][0] * eased_progress
            g = start_rgb[i][1] * (1 - eased_progress) + end_rgb[i][1] * eased_progress
            b = start_rgb[i][2] * (1 - eased_progress) + end_rgb[i][2] * eased_progress
            interpolated_rgb.append((r, g, b))
        
        # Convert back to hex
        return [self._rgb_to_hex(rgb) for rgb in interpolated_rgb]
    
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def _rgb_to_hex(self, rgb: Tuple[float, float, float]) -> str:
        """Convert RGB tuple to hex color string."""
        return f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"
    
    def create_entrance_animation(self, figure: go.Figure, duration: int = 1000,
                                 easing: str = "elastic") -> List[go.Figure]:
        """
        Create an elegant entrance animation for a visualization.
        
        Args:
            figure: The figure to animate
            duration: Animation duration in milliseconds
            easing: Easing function to use
            
        Returns:
            List of animation frames
        """
        frames = []
        steps = 20  # Number of animation frames
        
        # Extract node trace data
        node_trace = None
        for trace in figure.data:
            if trace.mode == 'markers':
                node_trace = trace
                break
                
        if not node_trace:
            return [figure]  # No nodes to animate
        
        # Store original data
        original_x = node_trace.x
        original_y = node_trace.y
        original_marker_size = node_trace.marker.size
        
        for i in range(steps + 1):
            progress = i / steps
            easing_fn = self.easings.get(easing, self.easings["ease-in-out"])
            eased_progress = easing_fn(progress)
            
            # Create a copy of the figure
            frame = go.Figure(figure)
            
            # Update node positions and sizes
            for trace in frame.data:
                if trace.mode == 'markers':
                    # Scale nodes from center
                    center_x = np.mean(original_x)
                    center_y = np.mean(original_y)
                    
                    # Calculate scaled positions
                    scaled_x = center_x + (original_x - center_x) * eased_progress
                    scaled_y = center_y + (original_y - center_y) * eased_progress
                    
                    # Update trace data
                    trace.x = scaled_x
                    trace.y = scaled_y
                    
                    # Scale marker sizes
                    if hasattr(trace.marker, 'size'):
                        trace.marker.size = original_marker_size * eased_progress
            
            frames.append(frame)
        
        return frames
    
    def create_node_highlight_animation(self, figure: go.Figure, node_indices: List[int],
                                       highlight_color: str = "#FF5500",
                                       duration: int = 800) -> List[go.Figure]:
        """
        Create an elegant highlight animation for specific nodes.
        
        Args:
            figure: The figure to animate
            node_indices: Indices of nodes to highlight
            highlight_color: Color to use for highlighting
            duration: Animation duration in milliseconds
            
        Returns:
            List of animation frames
        """
        frames = []
        steps = 15  # Number of animation frames
        
        # Extract node trace data
        node_trace = None
        for trace in figure.data:
            if trace.mode == 'markers':
                node_trace = trace
                break
                
        if not node_trace:
            return [figure]  # No nodes to animate
        
        # Store original marker data
        original_colors = node_trace.marker.color
        original_sizes = node_trace.marker.size
        
        for i in range(steps + 1):
            # Calculate progress with triangle wave pattern for pulsing effect
            progress = i / steps
            pulse_progress = 1 - abs(2 * progress - 1)  # Triangle wave 0->1->0
            
            # Create a copy of the figure
            frame = go.Figure(figure)
            
            # Update node colors and sizes
            for trace in frame.data:
                if trace.mode == 'markers':
                    # Create new color and size arrays
                    new_colors = original_colors.copy() if isinstance(original_colors, list) else [original_colors] * len(trace.x)
                    new_sizes = original_sizes.copy() if isinstance(original_sizes, list) else [original_sizes] * len(trace.x)
                    
                    # Update highlighted nodes
                    for idx in node_indices:
                        if idx < len(new_colors):
                            # Interpolate between original and highlight color
                            orig_color = original_colors[idx] if isinstance(original_colors, list) else original_colors
                            orig_rgb = self._hex_to_rgb(orig_color)
                            highlight_rgb = self._hex_to_rgb(highlight_color)
                            
                            # Interpolate RGB values
                            r = orig_rgb[0] * (1 - pulse_progress) + highlight_rgb[0] * pulse_progress
                            g = orig_rgb[1] * (1 - pulse_progress) + highlight_rgb[1] * pulse_progress
                            b = orig_rgb[2] * (1 - pulse_progress) + highlight_rgb[2] * pulse_progress
                            
                            new_colors[idx] = self._rgb_to_hex((r, g, b))
                            
                            # Scale node size
                            orig_size = original_sizes[idx] if isinstance(original_sizes, list) else original_sizes
                            new_sizes[idx] = orig_size * (1 + 0.5 * pulse_progress)
                    
                    # Update trace data
                    trace.marker.color = new_colors
                    trace.marker.size = new_sizes
            
            frames.append(frame)
        
        return frames
    
    def create_path_highlight_animation(self, figure: go.Figure, node_path: List[int],
                                       highlight_color: str = "#FF5500",
                                       duration: int = 1200) -> List[go.Figure]:
        """
        Create an elegant path highlight animation.
        
        Highlights a path through the tree with perfect timing and
        visual cues that draw attention to the path.
        
        Args:
            figure: The figure to animate
            node_path: List of node indices forming the path to highlight
            highlight_color: Color to use for highlighting
            duration: Animation duration in milliseconds
            
        Returns:
            List of animation frames
        """
        frames = []
        steps_per_node = 8  # Steps to spend on each node
        steps = steps_per_node * len(node_path)
        
        # Extract node and edge traces
        node_trace = None
        edge_trace = None
        for trace in figure.data:
            if trace.mode == 'markers':
                node_trace = trace
            elif trace.mode == 'lines':
                edge_trace = trace
        
        if not node_trace or not edge_trace:
            return [figure]  # Missing required traces
        
        # Store original data
        original_node_colors = node_trace.marker.color
        original_node_sizes = node_trace.marker.size
        original_edge_color = edge_trace.line.color
        original_edge_width = edge_trace.line.width
        
        # Calculate node-to-node edges
        edge_map = {}  # Maps (node1, node2) to edge indices in edge_trace
        
        # Assume edge_trace.x and edge_trace.y contain edge segments with None separators
        edge_segments = []
        current_segment = []
        
        for i in range(len(edge_trace.x)):
            if edge_trace.x[i] is None:
                if current_segment:
                    edge_segments.append(current_segment)
                    current_segment = []
            else:
                current_segment.append((edge_trace.x[i], edge_trace.y[i]))
        
        if current_segment:
            edge_segments.append(current_segment)
        
        # Create animation frames
        for i in range(steps + 1):
            # Calculate which node we're currently highlighting
            current_node_idx = min(int(i / steps_per_node), len(node_path) - 1)
            node_progress = (i % steps_per_node) / steps_per_node
            
            # Get the subset of the path to highlight
            highlight_path = node_path[:current_node_idx + 1]
            
            # Create a copy of the figure
            frame = go.Figure(figure)
            
            # Update node colors and sizes
            for trace in frame.data:
                if trace.mode == 'markers':
                    # Create new color and size arrays
                    new_colors = original_node_colors.copy() if isinstance(original_node_colors, list) else [original_node_colors] * len(trace.x)
                    new_sizes = original_node_sizes.copy() if isinstance(original_node_sizes, list) else [original_node_sizes] * len(trace.x)
                    
                    # Update all nodes in the highlight path
                    for idx in highlight_path[:-1]:
                        if idx < len(new_colors):
                            new_colors[idx] = highlight_color
                            orig_size = original_node_sizes[idx] if isinstance(original_node_sizes, list) else original_node_sizes
                            new_sizes[idx] = orig_size * 1.3
                    
                    # Animate the current node with pulsing
                    if highlight_path:
                        idx = highlight_path[-1]
                        if idx < len(new_colors):
                            # Pulse the current node
                            pulse = 0.7 + 0.3 * np.sin(node_progress * np.pi * 2)
                            
                            new_colors[idx] = highlight_color
                            orig_size = original_node_sizes[idx] if isinstance(original_node_sizes, list) else original_node_sizes
                            new_sizes[idx] = orig_size * (1.3 + 0.3 * pulse)
                    
                    # Update trace data
                    trace.marker.color = new_colors
                    trace.marker.size = new_sizes
            
            frames.append(frame)
        
        return frames
    
    def create_value_propagation_animation(self, figure: go.Figure, node_path: List[int],
                                          start_value: float, end_value: float,
                                          duration: int = 1500) -> List[go.Figure]:
        """
        Create an elegant value propagation animation.
        
        Visualizes the backpropagation of values through the tree
        with subtle visual cues and perfect timing.
        
        Args:
            figure: The figure to animate
            node_path: List of node indices forming the path
            start_value: Initial value
            end_value: Final value after propagation
            duration: Animation duration in milliseconds
            
        Returns:
            List of animation frames
        """
        frames = []
        steps = 30  # Number of animation frames
        
        # Extract node trace
        node_trace = None
        for trace in figure.data:
            if trace.mode == 'markers':
                node_trace = trace
                break
        
        if not node_trace:
            return [figure]  # No nodes to animate
        
        # Store original marker data
        original_colors = node_trace.marker.color
        original_sizes = node_trace.marker.size
        
        # Create color scale for values
        value_colors = ["#4B5563", "#10B981"]  # Slate to Emerald
        
        # Reverse the path for backpropagation
        backprop_path = node_path[::-1]
        
        for i in range(steps + 1):
            progress = i / steps
            
            # Calculate which node we're currently updating
            node_idx = min(int(progress * len(backprop_path)), len(backprop_path) - 1)
            
            # Create a copy of the figure
            frame = go.Figure(figure)
            
            # Update node colors based on propagated values
            for trace in frame.data:
                if trace.mode == 'markers':
                    # Create new color and size arrays
                    new_colors = original_colors.copy() if isinstance(original_colors, list) else [original_colors] * len(trace.x)
                    new_sizes = original_sizes.copy() if isinstance(original_sizes, list) else [original_sizes] * len(trace.x)
                    
                    # Update nodes that have been affected by backpropagation
                    for j, idx in enumerate(backprop_path[:node_idx + 1]):
                        if idx < len(new_colors):
                            # Calculate node value by interpolating
                            node_progress = 1.0 if j < node_idx else (progress * len(backprop_path) - node_idx)
                            node_value = start_value + (end_value - start_value) * node_progress
                            
                            # Map value to color
                            value_progress = (node_value - start_value) / (end_value - start_value)
                            value_progress = max(0, min(1, value_progress))  # Clamp to [0,1]
                            
                            # Interpolate between value colors
                            start_rgb = self._hex_to_rgb(value_colors[0])
                            end_rgb = self._hex_to_rgb(value_colors[1])
                            
                            r = start_rgb[0] * (1 - value_progress) + end_rgb[0] * value_progress
                            g = start_rgb[1] * (1 - value_progress) + end_rgb[1] * value_progress
                            b = start_rgb[2] * (1 - value_progress) + end_rgb[2] * value_progress
                            
                            new_colors[idx] = self._rgb_to_hex((r, g, b))
                            
                            # Pulse size for the active node
                            if j == node_idx:
                                pulse = 0.2 * np.sin(node_progress * np.pi * 4)
                                orig_size = original_sizes[idx] if isinstance(original_sizes, list) else original_sizes
                                new_sizes[idx] = orig_size * (1.2 + pulse)
                    
                    # Update trace data
                    trace.marker.color = new_colors
                    trace.marker.size = new_sizes
            
            frames.append(frame)
        
        return frames
    
    def create_microinteraction(self, interaction_type: str, figure: go.Figure, 
                               **kwargs) -> List[go.Figure]:
        """
        Create a delicate microinteraction animation.
        
        Args:
            interaction_type: Type of microinteraction to create
            figure: The figure to animate
            **kwargs: Additional parameters for the specific interaction
            
        Returns:
            List of animation frames
        """
        if interaction_type == "hover":
            return self._create_hover_interaction(figure, **kwargs)
        elif interaction_type == "select":
            return self._create_select_interaction(figure, **kwargs)
        elif interaction_type == "expand":
            return self._create_expand_interaction(figure, **kwargs)
        elif interaction_type == "focus":
            return self._create_focus_interaction(figure, **kwargs)
        else:
            return [figure]  # Unknown interaction type
    
    def _create_hover_interaction(self, figure: go.Figure, 
                                 node_index: int,
                                 duration: int = 300) -> List[go.Figure]:
        """Create a subtle hover interaction for a node."""
        frames = []
        steps = 10
        
        # Extract node trace
        node_trace = None
        for trace in figure.data:
            if trace.mode == 'markers':
                node_trace = trace
                break
        
        if not node_trace or node_index >= len(node_trace.x):
            return [figure]
        
        # Store original marker size for the node
        original_size = node_trace.marker.size[node_index] if isinstance(node_trace.marker.size, list) else node_trace.marker.size
        
        for i in range(steps + 1):
            progress = i / steps
            eased_progress = self.easings["ease-out"](progress)
            
            # Create a copy of the figure
            frame = go.Figure(figure)
            
            # Update the node size
            for trace in frame.data:
                if trace.mode == 'markers':
                    new_sizes = trace.marker.size.copy() if isinstance(trace.marker.size, list) else [trace.marker.size] * len(trace.x)
                    new_sizes[node_index] = original_size * (1 + 0.2 * eased_progress)
                    trace.marker.size = new_sizes
            
            frames.append(frame)
        
        return frames
    
    def _create_select_interaction(self, figure: go.Figure,
                                  node_index: int,
                                  duration: int = 400) -> List[go.Figure]:
        """Create a subtle selection interaction for a node."""
        frames = []
        steps = 12
        
        # Extract node trace
        node_trace = None
        for trace in figure.data:
            if trace.mode == 'markers':
                node_trace = trace
                break
        
        if not node_trace or node_index >= len(node_trace.x):
            return [figure]
        
        # Store original marker properties for the node
        original_size = node_trace.marker.size[node_index] if isinstance(node_trace.marker.size, list) else node_trace.marker.size
        original_color = node_trace.marker.color[node_index] if isinstance(node_trace.marker.color, list) else node_trace.marker.color
        
        # Selection color
        select_color = "#F59E0B"  # Amber
        
        for i in range(steps + 1):
            progress = i / steps
            eased_progress = self.easings["elastic"](progress)
            
            # Create a copy of the figure
            frame = go.Figure(figure)
            
            # Update the node properties
            for trace in frame.data:
                if trace.mode == 'markers':
                    # Update size with elastic effect
                    new_sizes = trace.marker.size.copy() if isinstance(trace.marker.size, list) else [trace.marker.size] * len(trace.x)
                    new_sizes[node_index] = original_size * (1 + 0.3 * eased_progress)
                    trace.marker.size = new_sizes
                    
                    # Update color
                    new_colors = trace.marker.color.copy() if isinstance(trace.marker.color, list) else [trace.marker.color] * len(trace.x)
                    
                    # Interpolate between original and selection color
                    orig_rgb = self._hex_to_rgb(original_color)
                    select_rgb = self._hex_to_rgb(select_color)
                    
                    r = orig_rgb[0] * (1 - progress) + select_rgb[0] * progress
                    g = orig_rgb[1] * (1 - progress) + select_rgb[1] * progress
                    b = orig_rgb[2] * (1 - progress) + select_rgb[2] * progress
                    
                    new_colors[node_index] = self._rgb_to_hex((r, g, b))
                    trace.marker.color = new_colors
            
            frames.append(frame)
        
        return frames
    
    def _create_expand_interaction(self, figure: go.Figure,
                                  parent_index: int,
                                  child_indices: List[int],
                                  duration: int = 600) -> List[go.Figure]:
        """Create an elegant expansion interaction for a node's children."""
        frames = []
        steps = 15
        
        # Extract node trace
        node_trace = None
        for trace in figure.data:
            if trace.mode == 'markers':
                node_trace = trace
                break
        
        if not node_trace:
            return [figure]
        
        # Store parent position
        parent_x = node_trace.x[parent_index] if parent_index < len(node_trace.x) else 0
        parent_y = node_trace.y[parent_index] if parent_index < len(node_trace.y) else 0
        
        # Store original positions and sizes for children
        child_positions = []
        child_sizes = []
        
        for idx in child_indices:
            if idx < len(node_trace.x):
                child_positions.append((node_trace.x[idx], node_trace.y[idx]))
                size = node_trace.marker.size[idx] if isinstance(node_trace.marker.size, list) else node_trace.marker.size
                child_sizes.append(size)
        
        for i in range(steps + 1):
            progress = i / steps
            eased_progress = self.easings["ease-out"](progress)
            
            # Create a copy of the figure
            frame = go.Figure(figure)
            
            # Update positions and sizes of children
            for trace in frame.data:
                if trace.mode == 'markers':
                    new_x = trace.x.copy()
                    new_y = trace.y.copy()
                    new_sizes = trace.marker.size.copy() if isinstance(trace.marker.size, list) else [trace.marker.size] * len(trace.x)
                    
                    # Update each child
                    for j, idx in enumerate(child_indices):
                        if idx < len(new_x) and j < len(child_positions):
                            # Interpolate position from parent to final position
                            final_x, final_y = child_positions[j]
                            new_x[idx] = parent_x + (final_x - parent_x) * eased_progress
                            new_y[idx] = parent_y + (final_y - parent_y) * eased_progress
                            
                            # Scale size from 0 to final
                            new_sizes[idx] = child_sizes[j] * eased_progress
                    
                    # Update trace data
                    trace.x = new_x
                    trace.y = new_y
                    trace.marker.size = new_sizes
            
            frames.append(frame)
        
        return frames
    
    def _create_focus_interaction(self, figure: go.Figure,
                                 focus_indices: List[int],
                                 duration: int = 800) -> List[go.Figure]:
        """Create a focus interaction that highlights nodes of interest."""
        frames = []
        steps = 20
        
        # Extract node trace
        node_trace = None
        for trace in figure.data:
            if trace.mode == 'markers':
                node_trace = trace
                break
        
        if not node_trace:
            return [figure]
        
        # Calculate center of focus
        focus_x = np.mean([node_trace.x[idx] for idx in focus_indices if idx < len(node_trace.x)])
        focus_y = np.mean([node_trace.y[idx] for idx in focus_indices if idx < len(node_trace.y)])
        
        # Store original positions and opacities
        original_x = node_trace.x.copy()
        original_y = node_trace.y.copy()
        
        # Create opacity mask (1 for focus nodes, 0.3 for others)
        opacities = []
        for i in range(len(original_x)):
            opacities.append(1.0 if i in focus_indices else 0.3)
        
        # Scale factor for non-focus nodes
        scale_factor = 1.2  # Zoom in by 20%
        
        for i in range(steps + 1):
            progress = i / steps
            eased_progress = self.easings["ease-in-out"](progress)
            
            # Create a copy of the figure
            frame = go.Figure(figure)
            
            # Update positions to create zoom effect
            for trace in frame.data:
                if hasattr(trace, 'x') and hasattr(trace, 'y'):
                    new_x = trace.x.copy() if hasattr(trace.x, 'copy') else trace.x
                    new_y = trace.y.copy() if hasattr(trace.y, 'copy') else trace.y
                    
                    # Apply zoom transform
                    for j in range(len(new_x)):
                        if new_x[j] is not None and new_y[j] is not None:
                            # Scale from center of focus
                            dx = new_x[j] - focus_x
                            dy = new_y[j] - focus_y
                            
                            # Apply scale factor based on progress
                            scale = 1.0 + (scale_factor - 1.0) * eased_progress
                            new_x[j] = focus_x + dx / scale
                            new_y[j] = focus_y + dy / scale
                    
                    # Update trace data
                    trace.x = new_x
                    trace.y = new_y
                
                # Update opacity for node trace
                if trace.mode == 'markers':
                    # Calculate opacity based on progress
                    new_opacities = []
                    for j in range(len(trace.x)):
                        target_opacity = opacities[j] if j < len(opacities) else 1.0
                        current_opacity = 1.0 + (target_opacity - 1.0) * eased_progress
                        new_opacities.append(current_opacity)
                    
                    trace.marker.opacity = new_opacities
            
            frames.append(frame)
        
        return frames