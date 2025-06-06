# FinsightDeep MCTX Enterprise - NVIDIA LaunchPad Guide

Welcome to FinsightDeep MCTX Enterprise on NVIDIA LaunchPad! This guide will help you navigate the deployed services and get started with the MCTX platform.

## Available Services

After deployment with `docker-compose -f docker-compose.nvidia.yml up`, the following services are available:

| Service | Port | URL | Description |
|---------|------|-----|-------------|
| Main Visualization | 8050 | [http://localhost:8050](http://localhost:8050) | Interactive MCTS tree visualization dashboard |
| API Server | 8000 | [http://localhost:8000](http://localhost:8000) | REST API for programmatic access |
| Secondary Visualization | 8051 | [http://localhost:8051](http://localhost:8051) | Alternative visualization interface |
| Prometheus | 9090 | [http://localhost:9090](http://localhost:9090) | Metrics collection and querying |
| Grafana | 3001 | [http://localhost:3001](http://localhost:3001) | Interactive monitoring dashboards (user: admin, password: admin) |

## Quick Start Guide

### 1. View MCTS Visualizations

Open [http://localhost:8050](http://localhost:8050) in your browser to access the main visualization dashboard.

The visualization dashboard provides:
- Interactive MCTS tree exploration
- Metrics panels showing visit distributions and value statistics
- Analysis tools for understanding search performance

### 2. Use the API

The REST API is available at [http://localhost:8000](http://localhost:8000).

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

### 3. Monitor Performance

Access monitoring dashboards:
- Prometheus: [http://localhost:9090](http://localhost:9090)
- Grafana: [http://localhost:3001](http://localhost:3001) (login with admin/admin)

### 4. Run Example Workflows

Example scripts are available in the container:

```bash
# Run visualization demo
docker exec -it mctx-nvidia python examples/visualization_demo.py

# Run policy improvement demo
docker exec -it mctx-nvidia python examples/policy_improvement_demo.py
```

## T4 Optimization Features

This deployment includes NVIDIA T4-specific optimizations:

- **Tensor Core Utilization**: Leverages T4 Tensor Cores for matrix operations
- **Memory Layout Optimization**: Enhanced memory access patterns for GPU acceleration
- **Mixed Precision**: Uses FP16/FP32 mixed precision for faster computation
- **Batch Processing**: Optimized batching for T4 architecture

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

## Available Examples

The container includes several examples:

1. **Visualization Demo**: Interactive visualization of MCTS processes
2. **Policy Improvement Demo**: Demonstrates Gumbel MuZero policy improvement
3. **Monitoring Demo**: Shows real-time metrics collection
4. **T4 Optimization Demo**: Demonstrates T4-specific performance enhancements

To run an example:
```bash
docker exec -it mctx-nvidia python examples/[example_name].py
```

## Additional Resources

- [MCTX Documentation](https://github.com/plturrell/finsightdeep_mctx_enterprise/tree/enterprise-version/docs)
- [JAX Documentation](https://jax.readthedocs.io/)
- [NVIDIA T4 Optimization Guide](https://developer.nvidia.com/blog/optimizing-gpu-performance-tensor-cores/)