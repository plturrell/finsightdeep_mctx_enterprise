# MCTX Enterprise - NVIDIA Blueprint Edition

## Overview

MCTX Enterprise is a high-performance Monte Carlo Tree Search framework optimized for NVIDIA GPUs, particularly T4 GPUs. This version includes comprehensive T4-specific optimizations, monitoring capabilities, and production-ready deployment configurations.

## Features

- **T4-Optimized MCTS**: Advanced optimizations for NVIDIA T4 GPUs
- **Tensor Core Utilization**: Automatic alignment and operations for Tensor Cores
- **Mixed Precision**: FP16/FP32 mixed precision for optimal performance
- **Memory Layout Optimizations**: Cache-aware memory patterns for T4 architecture
- **Distributed Search**: Multi-GPU support with various parallelism strategies
- **Comprehensive Monitoring**: Integrated Prometheus/Grafana monitoring
- **Production-Ready**: Health checks, logging, and Kubernetes support

## Quick Start

### Prerequisites

- NVIDIA GPU with CUDA 11.0+ support (T4 recommended)
- NVIDIA Container Toolkit (nvidia-docker)
- Docker and Docker Compose

### Deployment

1. **Clone the repository**:
   ```bash
   git clone ${MCTX_REPO_URL:-$(git config --get remote.origin.url)}
   cd $(basename ${MCTX_REPO_URL:-$(git config --get remote.origin.url)} .git)
   ```

2. **Deploy using Docker Compose**:
   ```bash
   docker-compose -f docker-compose.nvidia.yml up -d
   ```

3. **Check deployment health**:
   ```bash
   ./scripts/check-health.sh
   ```

### Available Services

| Service | URL | Description |
|---------|-----|-------------|
| API Server | http://localhost:8000 | REST API endpoints |
| Main Visualization | http://localhost:8050 | Primary visualization interface |
| Secondary Visualization | http://localhost:8051 | Alternative visualization |
| Documentation | http://localhost:8080 | Documentation server |
| Prometheus | http://localhost:9090 | Metrics collection |
| Grafana | http://localhost:3001 | Dashboards (admin/admin) |

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `JAX_PLATFORM_NAME` | JAX platform (gpu/cpu) | gpu |
| `MCTX_ENABLE_T4_OPTIMIZATIONS` | Enable T4 optimizations | 1 |
| `MCTX_PRECISION` | Computation precision | fp16 |
| `MCTX_TENSOR_CORES` | Use Tensor Cores | 1 |
| `MCTX_CACHE_OPTIMIZATION_LEVEL` | Cache optimization level (0-3) | 2 |
| `LOG_LEVEL` | Logging level | INFO |

### Custom Configuration

For advanced configuration, check:
- `/config/distribution.yaml` - Distributed training settings
- `/config/prometheus/gpu.yml` - Prometheus configuration
- `/docker-compose.nvidia.yml` - Deployment configuration

## Architecture

The system consists of several components:

1. **MCTX Core**: T4-optimized MCTS implementation
2. **API Server**: FastAPI-based REST API
3. **Visualization Server**: Interactive tree visualization
4. **Monitoring Stack**: Prometheus, Grafana, exporters
5. **Documentation Server**: Self-hosted documentation

## Performance

Performance on T4 GPUs compared to CPU:

| Configuration | Typical Speedup | Memory Efficiency |
|---------------|-----------------|-------------------|
| T4 FP32 | 3-5x | Standard |
| T4 FP16 | 6-10x | 40% reduction |
| T4 with Tensor Cores | 8-15x | 40% reduction |

## Monitoring

The monitoring stack provides comprehensive visibility:

1. **System Metrics**: CPU, memory, disk usage
2. **GPU Metrics**: Utilization, memory, temperature, power
3. **Application Metrics**: Throughput, latency, tree statistics
4. **Health Checks**: Service health and readiness

## Health Checks

All services have integrated health checks accessible at:

- MCTX API: http://localhost:8000/health
- MCTX API Readiness: http://localhost:8000/health/ready
- MCTX API Liveness: http://localhost:8000/health/live

## Documentation

For more detailed information:

- [T4 Optimization Guide](./docs/T4_OPTIMIZATION_GUIDE.md)
- [NVIDIA Deployment Guide](./docs/NVIDIA_DEPLOYMENT_GUIDE.md)
- [Distributed Training Guide](./docs/distributed_mcts.md)
- [API Reference](./docs/api_reference.md)

## Kubernetes Deployment

For Kubernetes deployment with NVIDIA GPU Operator:

```bash
# Apply configuration
kubectl apply -f kubernetes/gpu-operator.yaml
```

## CI/CD Pipeline

The project includes a GitHub Actions workflow for automated testing on GPU instances. See `.github/workflows/gpu-ci.yml` for details.

## Benchmarking

Run performance benchmarks:

```bash
# Full benchmark suite
docker exec mctx-nvidia python examples/cpu_gpu_benchmark.py

# T4-specific optimizations benchmark
docker exec mctx-nvidia python examples/t4_optimization_demo.py
```

## Troubleshooting

For common issues:

1. **GPU not detected**: Check NVIDIA driver and container toolkit installation
2. **Performance issues**: Verify T4 optimizations are enabled
3. **Memory errors**: Adjust batch size or precision
4. **Service unavailable**: Check service logs and health endpoints

## Support

For issues related to NVIDIA GPU deployment:
- GitHub Issues: [${MCTX_ISSUES_URL:-$(git config --get remote.origin.url)/issues}](${MCTX_ISSUES_URL:-$(git config --get remote.origin.url)/issues})
- Email Support: ${MCTX_SUPPORT_EMAIL:-$(git config --get user.email)}