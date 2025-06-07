# NVIDIA GPU Deployment Guide for MCTX Enterprise

## Overview

This guide provides detailed instructions for deploying the MCTX Enterprise platform on NVIDIA GPU infrastructure, with special optimizations for T4 GPUs. The MCTX Enterprise platform provides high-performance Monte Carlo Tree Search capabilities for business decision intelligence applications.

## Requirements

### Hardware Requirements
- NVIDIA T4 GPU (recommended) or other NVIDIA GPU with Tensor Cores
- 16+ GB System RAM
- 10+ GB Storage

### Software Requirements
- Docker 19.03+
- NVIDIA Container Toolkit (nvidia-docker)
- NVIDIA Driver 450.80.02+ 
- CUDA 11.0+

## Architecture

The MCTX Enterprise platform consists of several containerized services:

- **MCTX Core Service**: Main computational engine with T4 optimizations
- **Visualization Server**: Interactive web UI for search visualization
- **API Server**: REST API for programmatic access
- **Prometheus**: Metrics collection
- **Grafana**: Performance dashboards
- **Documentation Server**: Self-hosted documentation

![MCTX Architecture](./images/mctx_architecture.png)

## Deployment Options

MCTX Enterprise supports three deployment options:

1. **Basic Deployment**: Core service only
2. **Standard Deployment**: Core service + visualization
3. **Full Stack Deployment**: All services including monitoring

## Quick Start Guide

### Prerequisites

Ensure the NVIDIA Container Toolkit is installed:
```bash
# Check if nvidia-docker is installed
nvidia-docker --version

# If not installed, follow the instructions at:
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### Deployment Steps

1. **Clone the repository**:
   ```bash
   git clone ${MCTX_REPO_URL:-$(git config --get remote.origin.url)}
   cd $(basename ${MCTX_REPO_URL:-$(git config --get remote.origin.url)} .git)
   ```

2. **Deploy using the automated script**:
   ```bash
   ./deploy-nvidia.sh
   ```
   
   The script will:
   - Verify NVIDIA driver and toolkit installation
   - Check for GPU availability
   - Prompt you to select deployment type
   - Build and start the containers

3. **Verify deployment**:
   ```bash
   docker-compose -f docker-compose.nvidia.yml ps
   ```

## Performance Tuning

### T4-Specific Optimizations

MCTX Enterprise includes several optimizations specifically for NVIDIA T4 GPUs:

1. **Tensor Core Utilization**: Automatically aligns matrix dimensions for optimal Tensor Core performance
2. **Mixed Precision**: Uses FP16/FP32 mixed precision for faster computation
3. **Memory Layout Optimization**: Enhanced memory access patterns for T4's memory hierarchy
4. **Cache Optimization**: Memory allocation optimized for T4's L1/L2 cache structure

### Environment Variables

Fine-tune performance with these environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `MCTX_ENABLE_T4_OPTIMIZATIONS` | Enable T4-specific optimizations | `1` |
| `MCTX_PRECISION` | Computation precision (fp16, fp32) | `fp16` |
| `MCTX_TENSOR_CORES` | Enable Tensor Core optimizations | `1` |
| `MCTX_CACHE_OPTIMIZATION_LEVEL` | Cache optimization level (0-3) | `2` |
| `MCTX_BATCH_SIZE` | Default batch size for operations | Auto |

Example:
```bash
docker-compose -f docker-compose.nvidia.yml up -d -e MCTX_PRECISION=fp32
```

## Monitoring

### GPU Performance Metrics

The integrated monitoring stack collects and displays GPU-specific metrics:

- GPU Utilization
- Memory Usage
- Tensor Core Utilization
- Throughput (simulations/second)
- Memory Bandwidth
- Temperature

### Accessing Dashboards

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3001 (login: admin/admin)
- **Pre-configured dashboards**:
  - MCTX Overview
  - GPU Performance
  - Search Metrics
  - Memory Analysis

## Benchmarking

MCTX Enterprise includes a comprehensive benchmarking suite for performance testing:

```bash
# Run the full benchmark suite
docker exec mctx-nvidia python examples/cpu_gpu_benchmark.py

# Run T4-specific optimizations benchmark
docker exec mctx-nvidia python examples/t4_optimization_demo.py
```

Benchmark results are saved to the `mctx_output` directory and include:
- Execution time comparison
- Speedup analysis
- Memory efficiency metrics
- JSON performance data for custom analysis

## GPU Operator Integration

For Kubernetes deployments, MCTX Enterprise supports the NVIDIA GPU Operator:

1. **Install GPU Operator**:
   ```bash
   helm repo add nvidia https://nvidia.github.io/gpu-operator
   helm repo update
   helm install --wait --generate-name \
     -n gpu-operator --create-namespace \
     nvidia/gpu-operator
   ```

2. **Deploy MCTX with GPU resources**:
   ```yaml
   # Example Kubernetes manifest excerpt
   resources:
     limits:
       nvidia.com/gpu: 1
   ```

3. **Device Plugin Configuration**:
   ```yaml
   # T4-specific configuration
   resources:
     limits:
       nvidia.com/gpu: 1
       nvidia.com/mig-config: all-balanced
   ```

## Distributed Training

MCTX Enterprise supports distributed training across multiple GPUs:

1. **Configure distribution strategy**:
   ```bash
   # Edit the distribution.yaml file
   nano config/distribution.yaml
   ```

2. **Launch distributed training**:
   ```bash
   docker exec mctx-nvidia python -m mctx.distributed.run \
     --config config/distribution.yaml
   ```

3. **Distribution modes**:
   - Data Parallel: Split batches across GPUs
   - Model Parallel: Split model across GPUs
   - Hybrid: Combination of both approaches

## Troubleshooting

### Common Issues

1. **GPU Not Detected**
   ```bash
   # Check NVIDIA driver
   nvidia-smi
   
   # Verify Docker can access GPU
   docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```

2. **Out of Memory Errors**
   ```bash
   # Reduce batch size
   docker-compose -f docker-compose.nvidia.yml up -d -e MCTX_BATCH_SIZE=32
   ```

3. **Performance Issues**
   ```bash
   # Check Tensor Core utilization
   docker exec mctx-nvidia python -c "import mctx; mctx.utils.check_tensor_cores()"
   ```

### Logs and Diagnostics

```bash
# View service logs
docker-compose -f docker-compose.nvidia.yml logs mctx-nvidia

# Run diagnostics tool
docker exec mctx-nvidia python -m mctx.monitoring.diagnostics
```

## Advanced Configuration

### Custom CUDA Kernels

MCTX Enterprise includes custom CUDA kernels for specific operations:

```bash
# Enable custom kernels
docker-compose -f docker-compose.nvidia.yml up -d -e MCTX_USE_CUSTOM_KERNELS=1
```

### Multi-GPU Setup

For multi-GPU setups, modify `docker-compose.nvidia.yml`:

```yaml
services:
  mctx-nvidia:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all  # Use all available GPUs
              capabilities: [gpu]
```

## References

- [NVIDIA T4 Product Documentation](https://www.nvidia.com/en-us/data-center/tesla-t4/)
- [NVIDIA Container Toolkit Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html)
- [JAX GPU Documentation](https://jax.readthedocs.io/en/latest/gpu_performance_tips.html)
- [NVIDIA Deep Learning Performance Guide](https://docs.nvidia.com/deeplearning/performance/index.html)

## Support

For issues related to NVIDIA GPU deployment:
- GitHub Issues: [${MCTX_ISSUES_URL:-$(git config --get remote.origin.url)/issues}](${MCTX_ISSUES_URL:-$(git config --get remote.origin.url)/issues})
- Email Support: ${MCTX_SUPPORT_EMAIL:-$(git config --get user.email)}