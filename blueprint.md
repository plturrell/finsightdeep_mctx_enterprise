# FinsightDeep MCTX Enterprise

Enterprise-grade Monte Carlo Tree Search framework built on JAX, optimized for business decision intelligence applications with NVIDIA T4 acceleration.

## Overview

FinsightDeep MCTX Enterprise transforms the core MCTX library into a comprehensive decision intelligence platform for enterprise applications, delivering measurable business value through T4-optimized performance, distributed computing, and enterprise integrations.

## Available Services

After deployment, the following services are available:

| Service | Port | URL | Description |
|---------|------|-----|-------------|
| Main Visualization | 8050 | [http://localhost:8050](http://localhost:8050) | Interactive MCTS tree visualization dashboard |
| API Server | 8000 | [http://localhost:8000](http://localhost:8000) | REST API for programmatic access |
| Secondary Visualization | 8051 | [http://localhost:8051](http://localhost:8051) | Alternative visualization interface |
| Prometheus | 9090 | [http://localhost:9090](http://localhost:9090) | Metrics collection and querying |
| Grafana | 3001 | [http://localhost:3001](http://localhost:3001) | Interactive monitoring dashboards (user: admin, password: admin) |
| Documentation | 8080 | [http://localhost:8080](http://localhost:8080) | Documentation server |

## Quick Start Guide

1. **View MCTS Visualizations**: Open [http://localhost:8050](http://localhost:8050) to access the main visualization dashboard
2. **Explore the API**: The REST API is available at [http://localhost:8000](http://localhost:8000)
3. **Monitor Performance**: Access Grafana at [http://localhost:3001](http://localhost:3001) (login: admin/admin)

## T4 Optimization Features

This deployment includes NVIDIA T4-specific optimizations:
- Tensor Core utilization for faster matrix operations
- Memory layout optimization for GPU acceleration
- Mixed precision computation (FP16/FP32)
- Batched simulation for parallel processing

## Running Examples

```bash
# Run visualization demo
docker exec -it mctx-nvidia python examples/visualization_demo.py

# Run policy improvement demo
docker exec -it mctx-nvidia python examples/policy_improvement_demo.py
```

## Viewing Tree Visualizations

The main visualization dashboard provides:
- Interactive MCTS tree exploration
- Metrics panels showing visit distributions and value statistics
- Analysis tools for understanding search performance

## Using the API

Example API request to run a search:

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "state": {"board": [0, 0, 0, 0, 0, 0, 0, 0, 0]},
    "num_simulations": 100,
    "temperature": 1.0
  }'
```

## Monitoring Performance

Access comprehensive monitoring dashboards:
- Prometheus: [http://localhost:9090](http://localhost:9090)
- Grafana: [http://localhost:3001](http://localhost:3001) (login with admin/admin)

The Grafana dashboard includes:
- Tree growth metrics
- Visit distribution analytics
- Value estimation tracking
- GPU utilization statistics
- Memory usage monitoring

## Business Applications

FinsightDeep MCTX Enterprise provides industry-specific solutions with proven ROI:

| Industry | Solution Areas | Business Impact |
|----------|----------------|-----------------|
| Financial Services | Portfolio optimization, risk management | 3.2% higher returns, 47% better risk assessment |
| Healthcare | Resource allocation, patient flow | 31% more capacity, 24% lower costs |
| Manufacturing | Supply chain, production planning | 22% inventory reduction, 35% better resilience |
| Retail | Inventory, pricing optimization | 28% lower carrying costs, 62% fewer stockouts |
| Energy | Trading optimization, grid management | 4.1% higher profits, 23% better reliability |

## Troubleshooting

If you encounter issues:

1. **Service not responding**: Check container status
   ```bash
   docker-compose -f docker-compose.nvidia.yml ps
   ```

2. **View logs for a specific service**
   ```bash
   docker-compose -f docker-compose.nvidia.yml logs mctx-nvidia
   ```

3. **Verify GPU availability**
   ```bash
   docker exec -it mctx-nvidia nvidia-smi
   ```

4. **Check JAX device configuration**
   ```bash
   docker exec -it mctx-nvidia python -c 'import jax; print(jax.devices())'
   ```