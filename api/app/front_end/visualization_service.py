"""
MCTS Visualization Service

Connects the front-end visualization system with the MCTS backend service.
"""

import json
import logging
from typing import Dict, List, Any, Optional

import numpy as np
from api.app.models.mcts_models import SearchResult
from api.app.services.mcts_service import MCTSService
from .mcts_visualization import NodeState

logger = logging.getLogger("mctx.visualization")


class VisualizationService:
    """
    Service that connects the MCTS backend with the visualization front-end.
    
    Handles data transformation, caching, and communication between components.
    """
    
    def __init__(self, mcts_service: Optional[MCTSService] = None):
        """
        Initialize the visualization service.
        
        Args:
            mcts_service: Optional MCTS service instance. If not provided,
                          a new instance will be created.
        """
        self.mcts_service = mcts_service or MCTSService()
        self.cache = {}
        
    def convert_search_result_to_vis_data(self, result: SearchResult) -> Dict[str, Any]:
        """
        Convert MCTS search result to visualization-friendly format.
        
        Args:
            result: The search result from the MCTS service
            
        Returns:
            Dictionary containing visualization data
        """
        # Convert the search result to a dictionary if it's not already
        if not isinstance(result, dict):
            result_dict = result.dict()
        else:
            result_dict = result
        
        # Extract search tree from result
        search_tree = result_dict.get("search_tree", {})
        
        # Extract node data
        node_visits = np.array(search_tree.get("node_visits", []))
        node_values = np.array(search_tree.get("node_values", []))
        children_index = search_tree.get("children_index", [])
        
        # Create parents mapping
        parents = {}
        for i, parent_actions in enumerate(search_tree.get("parents", [])):
            if i > 0:  # Skip root node
                parents[i] = parent_actions
        
        # Calculate node states
        states = []
        for i in range(len(node_visits)):
            if node_visits[i] == 0:
                states.append(NodeState.UNEXPLORED)
            elif i in result_dict.get("action", []):
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
            "visits": node_visits.tolist(),
            "values": node_values.tolist(),
            "parents": parents,
            "children": children,
            "states": states,
            "search_type": result_dict.get("search_type", "unknown")
        }
        
        return vis_data
    
    def cache_visualization_data(self, search_id: str, vis_data: Dict[str, Any]) -> None:
        """
        Cache visualization data for later retrieval.
        
        Args:
            search_id: Unique identifier for the search
            vis_data: Visualization data to cache
        """
        self.cache[search_id] = vis_data
        
    def get_cached_visualization(self, search_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached visualization data.
        
        Args:
            search_id: Unique identifier for the search
            
        Returns:
            Cached visualization data or None if not found
        """
        return self.cache.get(search_id)
    
    def create_mock_visualization_data(self, 
                                      num_nodes: int = 50, 
                                      branching_factor: int = 3,
                                      max_depth: int = 4) -> Dict[str, Any]:
        """
        Create mock visualization data for testing.
        
        Args:
            num_nodes: Number of nodes in the tree
            branching_factor: Average branching factor
            max_depth: Maximum tree depth
            
        Returns:
            Dictionary containing mock visualization data
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
        states.append(NodeState.EXPLORED)
        
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
                    states.append(NodeState.EXPLORED)
                    children[child] = []
                else:
                    # Internal nodes
                    visits[child] = np.random.randint(10, 30)
                    values[child] = np.random.uniform(-0.5, 0.8)
                    states.append(NodeState.EXPLORED)
                    children[child] = []
                    
                    # Recursively create subtree
                    create_subtree(child, depth + 1)
        
        # Create the full tree
        create_subtree(0, 1)
        
        # Mark some nodes as unexplored, optimal, or selected
        for i in range(1, num_nodes):
            if np.random.random() < 0.1:
                visits[i] = 0
                states[i] = NodeState.UNEXPLORED
            elif np.random.random() < 0.05:
                states[i] = NodeState.OPTIMAL
            elif np.random.random() < 0.05:
                states[i] = NodeState.SELECTED
        
        # Construct visualization data
        vis_data = {
            "node_count": num_nodes,
            "visits": visits.tolist(),
            "values": values.tolist(),
            "parents": parents,
            "children": children,
            "states": states,
            "search_type": "mock"
        }
        
        return vis_data