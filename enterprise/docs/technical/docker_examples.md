# Docker Deployment Examples

This document provides examples of using the MCTX Docker containers for various enterprise scenarios.

## NVIDIA GPU Container Examples

### Basic MCTS with Visualization

```bash
# Start the NVIDIA container with visualization server
docker run -p 8050:8050 -p 8000:8000 --gpus all mctx-nvidia
```

Then access the visualization dashboard at `http://localhost:8050`

### Custom MCTS Script

Create a script `my_mcts.py`:

```python
import jax
import jax.numpy as jnp
from mctx import search, RootFnOutput

# Your custom MCTS code here
# ...

if __name__ == "__main__":
    # Run your search
    results = search(...)
    print(f"Selected actions: {results.action}")
```

Run it in the container:

```bash
docker run --gpus all -v $(pwd):/app/custom mctx-nvidia python /app/custom/my_mcts.py
```

### Interactive Visualization Demo

```bash
# Run the interactive visualization demo
docker run -p 8050:8050 --gpus all mctx-nvidia python examples/monitoring_demo.py
```

Then access the visualization dashboard at `http://localhost:8050`

### With SAP HANA Integration

```bash
# Run with HANA connection
docker run --gpus all \
  -e HANA_HOST=your-hana-host \
  -e HANA_PORT=30015 \
  -e HANA_USER=your-user \
  -e HANA_PASSWORD=your-password \
  mctx-nvidia python examples/hana_integration_demo.py
```

## Vercel API Container Examples

### Run API Server Locally

```bash
# Start the API server
docker run -p 3000:3000 mctx-api
```

Test with curl:

```bash
curl http://localhost:3000/info
```

### Deploy to Vercel

1. Copy the Vercel configuration:

```bash
cp docker/vercel.json .
```

2. Install Vercel CLI:

```bash
npm install -g vercel
```

3. Deploy to Vercel:

```bash
vercel
```

### Run With Custom Configuration

```bash
# Start with custom configuration
docker run -p 3000:3000 \
  -e MAX_BATCH_SIZE=16 \
  -e MAX_NUM_SIMULATIONS=200 \
  mctx-api
```

## Visualization Server Examples

### Run Standalone Visualization Server

```bash
# Start visualization server
docker run -p 8050:8050 mctx-vis
```

Then access at `http://localhost:8050`

### Visualize Custom Tree File

Create a JSON file `tree.json` with your tree data, then:

```bash
# Visualize custom tree
docker run -p 8050:8050 -v $(pwd):/app/data mctx-vis \
  python -m mctx.monitoring.cli server --tree-file /app/data/tree.json
```

### Generate Static Visualizations

```bash
# Generate visualizations
docker run -v $(pwd):/app/output mctx-vis \
  python -m mctx.monitoring.cli visualize \
  --tree-file /app/examples/test_data/muzero_tree.json \
  --output /app/output/visualizations
```

## Advanced Docker Compose Examples

### Complete Enterprise Stack

Create a `docker-compose.enterprise.yml` file:

```yaml
version: '3.8'

services:
  # T4-optimized MCTS service
  mctx-service:
    image: mctx-nvidia:latest
    ports:
      - "8000:8000"
    environment:
      - JAX_PLATFORM_NAME=gpu
      - MCTX_ENABLE_T4_OPTIMIZATIONS=1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
  
  # Visualization frontend
  mctx-dashboard:
    image: mctx-vis:latest
    ports:
      - "8050:8050"
    environment:
      - API_URL=http://mctx-service:8000
    depends_on:
      - mctx-service
  
  # Prometheus for metrics
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
  
  # Grafana for dashboards
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus
  
  # NGINX for SSL and load balancing
  nginx:
    image: nginx:latest
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - mctx-service
      - mctx-dashboard

volumes:
  grafana_data:
```

Run with:

```bash
docker-compose -f docker-compose.enterprise.yml up
```

## Performance Tuning

### T4 GPU Optimization

```bash
# Run with explicit T4 optimizations
docker run --gpus all \
  -e MCTX_ENABLE_T4_OPTIMIZATIONS=1 \
  -e MCTX_T4_TENSOR_CORE_ALIGNED=1 \
  -e MCTX_T4_MEMORY_OPTIMIZE=1 \
  -e MCTX_T4_MIXED_PRECISION=1 \
  mctx-nvidia python examples/t4_optimization_demo.py
```

### Multi-GPU Deployment

```bash
# Run with distributed configuration
docker run --gpus all \
  -e MCTX_DISTRIBUTED_DEVICES=2 \
  -e MCTX_DISTRIBUTED_BATCH_SPLIT=1 \
  mctx-nvidia python examples/enhanced_distributed_demo.py
```

## Production Deployment Tips

1. **Resource Allocation**: For T4 GPUs, allocate at least 8GB GPU memory
2. **High Availability**: Use Docker Swarm or Kubernetes for HA deployments
3. **Security**: Use environment variables for sensitive configuration
4. **Monitoring**: Connect Prometheus for performance monitoring
5. **Volume Mounts**: Mount persistent volumes for saving visualizations
6. **Networking**: Use Docker networks to isolate services
7. **Scaling**: Use horizontal scaling for API services
8. **Health Checks**: Implement health checks for all containers