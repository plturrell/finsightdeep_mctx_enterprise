import os
import time
import uuid
from typing import Dict, List, Optional, Any, Union
import json

import jax
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Header, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge

# Import MCTX
import mctx
from mctx.visualization import visualize_tree

# Optional: Import HANA connector if available
try:
    from mctx.integrations import hana_connector
    HANA_AVAILABLE = True
except ImportError:
    HANA_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(
    title="MCTX Enterprise API",
    description="Enterprise Decision Intelligence Platform powered by Monte Carlo Tree Search",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Prometheus metrics
SEARCH_REQUESTS = Counter('mctx_search_requests_total', 'Total number of search requests')
SEARCH_DURATION = Histogram('mctx_search_duration_seconds', 'Search request duration in seconds')
GPU_MEMORY_USAGE = Gauge('mctx_gpu_memory_usage_bytes', 'GPU memory usage in bytes')
ACTIVE_REQUESTS = Gauge('mctx_active_requests', 'Number of active requests')

# Load environment variables
NUM_WORKERS = int(os.getenv("MCTX_NUM_WORKERS", "4"))
MAX_BATCH_SIZE = int(os.getenv("MCTX_MAX_BATCH_SIZE", "32"))
DEFAULT_SIMULATIONS = int(os.getenv("MCTX_DEFAULT_SIMULATIONS", "128"))
TIMEOUT_SECONDS = int(os.getenv("MCTX_TIMEOUT_SECONDS", "60"))
GPU_MEMORY_FRACTION = float(os.getenv("MCTX_GPU_MEMORY_FRACTION", "0.9"))
USE_MIXED_PRECISION = os.getenv("MCTX_USE_MIXED_PRECISION", "true").lower() == "true"
ENABLE_DISTRIBUTED = os.getenv("MCTX_ENABLE_DISTRIBUTED", "false").lower() == "true"
NUM_DEVICES = int(os.getenv("MCTX_NUM_DEVICES", "1"))
ENABLE_REDIS_CACHE = os.getenv("MCTX_ENABLE_REDIS_CACHE", "true").lower() == "true"

# Initialize JAX
print(f"JAX devices: {jax.devices()}")
print(f"Number of devices: {jax.device_count()}")

# Initialize Redis cache if enabled
if ENABLE_REDIS_CACHE:
    import redis
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "redis"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        db=0
    )
    REDIS_CACHE_TTL = int(os.getenv("REDIS_CACHE_TTL", "3600"))
    
    # Test Redis connection
    try:
        redis_client.ping()
        print("Redis connection successful")
    except Exception as e:
        print(f"Redis connection failed: {e}")
        ENABLE_REDIS_CACHE = False

# Initialize HANA connector if available
if HANA_AVAILABLE:
    try:
        hana_host = os.getenv("HANA_HOST")
        hana_port = int(os.getenv("HANA_PORT", "443"))
        hana_user = os.getenv("HANA_USER")
        hana_password = os.getenv("HANA_PASSWORD")
        
        if all([hana_host, hana_user, hana_password]):
            hana_conn = hana_connector.HanaConnector(
                host=hana_host,
                port=hana_port,
                user=hana_user,
                password=hana_password,
                use_ssl=True
            )
            
            # Test connection
            if hana_conn.test_connection():
                print("HANA connection successful")
                # Initialize schema if it doesn't exist
                hana_conn.initialize_schema(drop_existing=False)
            else:
                print("HANA connection failed")
                hana_conn = None
        else:
            print("HANA credentials not provided")
            hana_conn = None
    except Exception as e:
        print(f"HANA initialization error: {e}")
        hana_conn = None
else:
    hana_conn = None

# Load models (simplified for this example)
# In a real implementation, you would load your specific models
class ModelRegistry:
    def __init__(self):
        self.models = {}
        self.load_default_models()
    
    def load_default_models(self):
        # This is a placeholder for model loading
        # In a real implementation, load your specific models
        pass
    
    def get_model(self, model_name):
        if model_name not in self.models:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        return self.models[model_name]

model_registry = ModelRegistry()

# API key validation (simplified)
def validate_api_key(api_key: str = Header(None)):
    if os.getenv("MCTX_API_KEY_REQUIRED", "false").lower() == "true":
        valid_keys = os.getenv("MCTX_API_KEYS", "").split(",")
        if not api_key or api_key not in valid_keys:
            raise HTTPException(status_code=401, detail="Invalid API key")
    return True

# Pydantic models for API
class State(BaseModel):
    observation: List[float] = Field(..., description="The observation vector")
    legal_actions: List[int] = Field(..., description="List of legal actions")
    
class SearchConfig(BaseModel):
    num_simulations: int = Field(DEFAULT_SIMULATIONS, description="Number of simulations to run")
    temperature: float = Field(1.0, description="Temperature for action selection")
    dirichlet_fraction: Optional[float] = Field(None, description="Fraction of Dirichlet noise to add")
    dirichlet_alpha: Optional[float] = Field(None, description="Alpha parameter for Dirichlet distribution")
    use_mixed_precision: bool = Field(USE_MIXED_PRECISION, description="Whether to use mixed precision")
    
class SearchRequest(BaseModel):
    state: State
    config: SearchConfig = Field(default_factory=SearchConfig)
    model: str = Field("muzero_default", description="Model to use for search")
    
class SearchResponse(BaseModel):
    action: int
    action_weights: List[float]
    root_value: float
    q_values: List[float]
    visit_counts: List[int]
    search_id: str
    computation_time_ms: float

# API routes
@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    gpu_info = {}
    try:
        # Check GPU availability
        devices = jax.devices()
        gpu_info = {
            f"device_{i}": {
                "platform": str(d.platform),
                "device_kind": d.device_kind,
            }
            for i, d in enumerate(devices)
        }
    except Exception as e:
        gpu_info = {"error": str(e)}
    
    # Check Redis if enabled
    redis_status = "disabled"
    if ENABLE_REDIS_CACHE:
        try:
            redis_client.ping()
            redis_status = "available"
        except:
            redis_status = "unavailable"
    
    # Check HANA if enabled
    hana_status = "disabled"
    if hana_conn:
        try:
            if hana_conn.test_connection():
                hana_status = "available"
            else:
                hana_status = "unavailable"
        except:
            hana_status = "unavailable"
    
    return {
        "status": "healthy",
        "gpu_status": "available" if gpu_info else "unavailable",
        "gpu_info": gpu_info,
        "redis_status": redis_status,
        "hana_status": hana_status,
        "version": "1.0.0",
        "uptime_seconds": time.time() - start_time
    }

@app.post("/api/v1/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    background_tasks: BackgroundTasks,
    api_key_valid: bool = Depends(validate_api_key)
):
    """Run a Monte Carlo Tree Search"""
    search_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Update metrics
    SEARCH_REQUESTS.inc()
    ACTIVE_REQUESTS.inc()
    
    try:
        # Check cache if enabled
        if ENABLE_REDIS_CACHE:
            cache_key = f"search:{hash(json.dumps(request.dict(), sort_keys=True))}"
            cached_result = redis_client.get(cache_key)
            if cached_result:
                result = json.loads(cached_result)
                result["search_id"] = search_id  # Generate new search ID
                computation_time = time.time() - start_time
                result["computation_time_ms"] = round(computation_time * 1000, 2)
                return result
        
        # Convert state to appropriate format
        # Note: This is a simplified example. In a real implementation,
        # you would process the state according to your model's requirements.
        observation = np.array(request.state.observation, dtype=np.float32)
        legal_actions = np.array(request.state.legal_actions, dtype=np.int32)
        
        # Create dummy root for this example
        # In a real implementation, you would create a proper root state
        root = mctx.RootFnOutput(
            prior_logits=np.ones(len(legal_actions), dtype=np.float32),
            value=np.array(0.0, dtype=np.float32),
            embedding=np.zeros(64, dtype=np.float32)  # Example embedding size
        )
        
        # Create dummy recurrent function for this example
        # In a real implementation, you would use your actual model
        def recurrent_fn(params, rng_key, action, embedding):
            next_embedding = embedding + np.array([0.1] * len(embedding), dtype=np.float32)
            return (
                mctx.RecurrentFnOutput(
                    reward=np.array(0.1, dtype=np.float32),
                    discount=np.array(0.99, dtype=np.float32),
                    prior_logits=np.ones(len(legal_actions), dtype=np.float32),
                    value=np.array(0.5, dtype=np.float32)
                ),
                next_embedding
            )
        
        # Run search
        if USE_MIXED_PRECISION:
            # Use T4-optimized search with mixed precision
            policy_output = mctx.t4_optimized_search(
                params=None,  # No actual params in this example
                rng_key=jax.random.PRNGKey(0),
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=request.config.num_simulations,
                dirichlet_fraction=request.config.dirichlet_fraction,
                dirichlet_alpha=request.config.dirichlet_alpha,
                temperature=request.config.temperature,
                use_mixed_precision=True
            )
        else:
            # Use standard search
            policy_output = mctx.muzero_policy(
                params=None,  # No actual params in this example
                rng_key=jax.random.PRNGKey(0),
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=request.config.num_simulations,
                dirichlet_fraction=request.config.dirichlet_fraction,
                dirichlet_alpha=request.config.dirichlet_alpha,
                temperature=request.config.temperature
            )
        
        # Format response
        result = {
            "action": int(policy_output.action),
            "action_weights": policy_output.action_weights.tolist(),
            "root_value": float(policy_output.root_value),
            "q_values": policy_output.q_values.tolist(),
            "visit_counts": policy_output.visit_counts.tolist(),
            "search_id": search_id,
            "computation_time_ms": round((time.time() - start_time) * 1000, 2)
        }
        
        # Store in cache if enabled
        if ENABLE_REDIS_CACHE:
            redis_client.setex(
                cache_key,
                REDIS_CACHE_TTL,
                json.dumps(result)
            )
        
        # Store in HANA if available
        if hana_conn:
            background_tasks.add_task(
                store_search_result_in_hana,
                hana_conn=hana_conn,
                search_id=search_id,
                policy_output=policy_output,
                metadata={
                    "request": request.dict(),
                    "computation_time_ms": result["computation_time_ms"]
                }
            )
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")
    
    finally:
        # Update metrics
        ACTIVE_REQUESTS.dec()
        SEARCH_DURATION.observe(time.time() - start_time)

@app.get("/api/v1/visualization/{search_id}", response_class=HTMLResponse)
async def get_visualization(
    search_id: str,
    api_key_valid: bool = Depends(validate_api_key)
):
    """Get visualization for a search"""
    # In a real implementation, you would retrieve the search tree from storage
    # This is a simplified example returning a placeholder visualization
    
    # Check if search_id exists in HANA
    if hana_conn:
        result = hana_conn.get_search_results(search_id)
        if result:
            # Generate visualization from the actual search tree
            html = visualize_tree(
                result.policy_output.search_tree,
                show_values=True,
                show_visit_counts=True,
                color_scheme="value",
                layout="radial",
                include_controls=True,
                include_metrics=True,
                title=f"Search ID: {search_id}"
            )
            return html
    
    # Return a placeholder if not found
    return """
    <html>
        <head>
            <title>MCTX Visualization</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; margin: 50px; }
            </style>
        </head>
        <body>
            <h1>Search Visualization</h1>
            <p>Search ID: {search_id}</p>
            <p>Visualization not available or search ID not found.</p>
        </body>
    </html>
    """.format(search_id=search_id)

@app.get("/api/v1/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return prometheus_client.generate_latest()

# Helper functions
async def store_search_result_in_hana(hana_conn, search_id, policy_output, metadata):
    """Store search result in HANA database"""
    try:
        hana_conn.store_search_results(
            search_id=search_id,
            policy_output=policy_output,
            metadata=metadata
        )
    except Exception as e:
        print(f"Error storing search result in HANA: {e}")

# Startup event
start_time = time.time()
@app.on_event("startup")
async def startup_event():
    print("MCTX Enterprise API is starting up...")
    print(f"Configuration:")
    print(f"  Workers: {NUM_WORKERS}")
    print(f"  Max Batch Size: {MAX_BATCH_SIZE}")
    print(f"  Default Simulations: {DEFAULT_SIMULATIONS}")
    print(f"  Timeout: {TIMEOUT_SECONDS}s")
    print(f"  GPU Memory Fraction: {GPU_MEMORY_FRACTION}")
    print(f"  Mixed Precision: {USE_MIXED_PRECISION}")
    print(f"  Distributed Mode: {ENABLE_DISTRIBUTED}")
    print(f"  Redis Cache: {ENABLE_REDIS_CACHE}")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    print("MCTX Enterprise API is shutting down...")

# Run the application if executed as script
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
EOF < /dev/null