"""
MCTX API for Vercel deployment

Provides a lightweight API for MCTS services, optimized for serverless deployment.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

import numpy as np
import jax
import jax.numpy as jnp

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import mctx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mctx.api")

# Force JAX to use CPU for Vercel deployment
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# Create FastAPI app
app = FastAPI(
    title="MCTX API",
    description="API for Monte Carlo Tree Search in JAX",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Model definitions
class RootInput(BaseModel):
    """Input for the root node of a search tree."""
    prior_logits: List[List[float]]
    value: List[float]
    embedding: List[List[float]]
    batch_size: int
    num_actions: int


class SearchParams(BaseModel):
    """Parameters for MCTS search."""
    num_simulations: int
    max_depth: Optional[int] = None
    max_num_considered_actions: Optional[int] = None
    dirichlet_fraction: Optional[float] = None
    dirichlet_alpha: Optional[float] = None
    use_t4_optimizations: bool = False
    precision: str = "fp32"
    tensor_core_aligned: bool = False
    distributed: bool = False
    num_devices: int = 1
    partition_batch: bool = False


class MCTSRequest(BaseModel):
    """Request for MCTS search."""
    root_input: RootInput
    search_params: SearchParams
    search_type: str = "muzero"
    device_type: str = "cpu"


# API endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "MCTX API",
        "version": mctx.__version__,
        "description": "API for Monte Carlo Tree Search in JAX",
    }


@app.get("/info")
async def info():
    """Get information about the MCTX environment."""
    return {
        "mctx_version": mctx.__version__,
        "jax_version": jax.__version__,
        "devices": [str(d) for d in jax.devices()],
        "platform": jax.devices()[0].platform,
    }


@app.post("/search")
async def run_search(request: MCTSRequest):
    """
    Run MCTS search with the given parameters.
    
    This endpoint runs a lightweight MCTS search suitable for serverless environments.
    For larger searches, consider using the full MCTX library with GPU acceleration.
    """
    try:
        # Extract parameters
        root_input = request.root_input
        search_params = request.search_params
        search_type = request.search_type
        
        # Create random key
        key = jax.random.PRNGKey(42)
        
        # Create root input
        root = mctx.RootFnOutput(
            prior_logits=jnp.array(root_input.prior_logits),
            value=jnp.array(root_input.value),
            embedding=jnp.array(root_input.embedding),
        )
        
        # Create recurrent function
        def recurrent_fn(embeddings, actions):
            # Create simple recurrent function for demonstration
            batch_size = embeddings.shape[0]
            num_actions = root_input.num_actions
            
            # Generate random values between -1 and 1
            value = jnp.zeros((batch_size,))
            
            # Generate flat policy logits
            prior_logits = jnp.zeros((batch_size, num_actions))
            
            # Create new embedding (zeros)
            new_embedding = embeddings
            
            # Add reward of 0.1 for each step
            reward = jnp.ones((batch_size,)) * 0.1
            
            # Always valid actions
            is_terminal = jnp.zeros((batch_size,), dtype=jnp.bool_)
            
            return mctx.RecurrentFnOutput(
                prior_logits=prior_logits,
                value=value,
                embedding=new_embedding,
                reward=reward,
                is_terminal=is_terminal
            )
        
        # Set policy type based on search_type
        if search_type.lower() == "muzero":
            policy_type = mctx.PolicyType.MUZERO
        elif search_type.lower() == "gumbel_muzero":
            policy_type = mctx.PolicyType.GUMBEL_MUZERO
        elif search_type.lower() == "stochastic_muzero":
            policy_type = mctx.PolicyType.STOCHASTIC_MUZERO
        else:
            raise ValueError(f"Unknown search type: {search_type}")
        
        # Run search
        start_time = time.time()
        search_results = mctx.search(
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=min(search_params.num_simulations, 100),  # Limit for serverless
            max_depth=search_params.max_depth,
            max_num_considered_actions=search_params.max_num_considered_actions,
            policy_type=policy_type,
            dirichlet_fraction=search_params.dirichlet_fraction,
            dirichlet_alpha=search_params.dirichlet_alpha,
        )
        end_time = time.time()
        
        # Create search statistics
        search_statistics = {
            "duration_ms": (end_time - start_time) * 1000,
            "num_expanded_nodes": len(search_results.search_tree.node_values),
            "max_depth_reached": len(search_results.search_paths[0]) if search_results.search_paths else 0,
            "optimized": search_params.use_t4_optimizations,
        }
        
        # Convert tree to JSON-serializable format
        tree_data = {
            "node_values": search_results.search_tree.node_values.tolist(),
            "node_visits": search_results.search_tree.node_visits.tolist(),
            "children_index": [
                [int(child) for child in children]
                for children in search_results.search_tree.children_index
            ],
        }
        
        # Convert result to JSON-serializable format
        result = {
            "action": search_results.action.tolist(),
            "search_tree": tree_data,
            "search_statistics": search_statistics,
            "search_type": search_type,
        }
        
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Error running search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/visualize")
async def visualize_tree(tree_json: str = Query(None)):
    """
    Generate tree visualization data.
    
    Args:
        tree_json: JSON string with tree data
        
    Returns:
        Visualization data for the tree
    """
    try:
        from mctx.monitoring.visualization import TreeVisualizer
        
        # Parse tree data
        if tree_json:
            tree_data = json.loads(tree_json)
        else:
            # Create mock tree if no data provided
            tree_data = create_mock_tree(30, 2, 3)
        
        # Create visualizer
        visualizer = TreeVisualizer()
        
        # Convert tree to visualization format if needed
        if "node_values" in tree_data and "node_visits" in tree_data:
            # Convert from raw tree format
            node_values = tree_data.get("node_values", [])
            node_visits = tree_data.get("node_visits", [])
            children_index = tree_data.get("children_index", [])
            
            # Create parents mapping
            parents = {}
            for i, children in enumerate(children_index):
                for j, child in enumerate(children):
                    if child >= 0:  # Skip -1 (no child)
                        parents[child] = i
            
            # Determine node states
            states = []
            for i in range(len(node_visits)):
                if node_visits[i] == 0:
                    states.append("unexplored")
                elif i == 0:  # Root
                    states.append("selected")
                elif node_visits[i] > np.percentile(np.array(node_visits), 90):
                    states.append("optimal")
                else:
                    states.append("explored")
            
            # Create children mapping
            children = {}
            for parent, children_arr in enumerate(children_index):
                parent_children = []
                for child in children_arr:
                    if child >= 0:  # Skip -1 (no child)
                        parent_children.append(child)
                children[parent] = parent_children
            
            # Construct visualization data
            vis_data = {
                "node_count": len(node_values),
                "visits": node_values,
                "values": node_visits,
                "parents": parents,
                "children": children,
                "states": states
            }
        else:
            # Already in visualization format
            vis_data = tree_data
        
        return JSONResponse(content=vis_data)
    
    except Exception as e:
        logger.error(f"Error generating visualization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper function to create mock tree
def create_mock_tree(num_nodes=30, branching_factor=2, max_depth=3):
    """Create a mock tree for visualization."""
    visits = np.zeros(num_nodes)
    values = np.zeros(num_nodes)
    parents = {}
    children = {}
    states = []
    
    # Set root node values
    visits[0] = 20
    values[0] = 0.5
    children[0] = []
    states.append("explored")
    
    # Create tree structure
    node_counter = 1
    
    def create_subtree(parent, depth):
        nonlocal node_counter
        if depth >= max_depth or node_counter >= num_nodes:
            return
            
        # Determine number of children
        num_children = min(branching_factor, num_nodes - node_counter)
        
        # Create children
        for _ in range(num_children):
            if node_counter >= num_nodes:
                break
                
            # Create child
            child = node_counter
            node_counter += 1
            
            # Set parent-child relationship
            parents[child] = parent
            children[parent].append(child)
            
            # Initialize values
            if depth == max_depth - 1:
                # Leaf nodes
                visits[child] = np.random.randint(1, 5)
                values[child] = np.random.uniform(-0.5, 0.5)
                states.append("explored")
                children[child] = []
            else:
                # Internal nodes
                visits[child] = np.random.randint(5, 15)
                values[child] = np.random.uniform(-0.2, 0.8)
                states.append("explored")
                children[child] = []
                
                # Recursive subtree
                create_subtree(child, depth + 1)
    
    # Create full tree
    create_subtree(0, 1)
    
    # Mark some nodes as special states
    for i in range(1, num_nodes):
        if np.random.random() < 0.1:
            visits[i] = 0
            states[i] = "unexplored"
        elif np.random.random() < 0.05:
            states[i] = "optimal"
        elif np.random.random() < 0.05:
            states[i] = "selected"
    
    return {
        "node_count": num_nodes,
        "visits": visits.tolist(),
        "values": values.tolist(),
        "parents": parents,
        "children": children,
        "states": states
    }