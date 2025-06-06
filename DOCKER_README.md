# MCTX Docker Deployment Guide

This guide provides a comprehensive overview of the Docker deployment options for MCTX (Monte Carlo Tree Search in JAX), with a focus on the NVIDIA T4 GPU optimizations.

## Quick Start

For the fastest deployment with NVIDIA T4 GPUs, use our deployment script:

```bash
./deploy-nvidia.sh
```

This script will:
1. Check your environment for required dependencies
2. Verify NVIDIA GPU availability and T4 compatibility
3. Guide you through deployment options
4. Start the selected services with the optimized configuration

## Deployment Options

MCTX offers three deployment configurations to meet different needs:

### 1. Development Setup (`docker-compose.yml`)
- CPU-only, lightweight configuration
- Ideal for development and testing
- Includes basic API service, frontend, and monitoring

### 2. Simplified NVIDIA GPU Setup (`docker-compose.nvidia.yml`)
- Optimized for NVIDIA T4 GPUs
- Includes visualization server and basic monitoring
- Perfect for quick GPU-accelerated demos and testing

### 3. Production NVIDIA GPU Setup (`deployment/docker-compose.yml`)
- Enterprise-grade configuration
- Includes Redis caching, comprehensive monitoring
- Optional SAP HANA integration and NGINX reverse proxy
- Designed for high-performance production environments

## Key Features

### T4 GPU Optimizations
The NVIDIA T4-optimized container includes:
- Memory layout optimizations for T4 GPU architecture
- Tensor core utilization for faster matrix operations
- Mixed precision computation (FP16/FP32)
- Optimized tree layout and memory management
- Distributed computation support

### Monitoring and Visualization
- Real-time performance metrics with Prometheus
- Interactive dashboards with Grafana
- Tree visualization and analysis tools

### SAP HANA Integration
- Enterprise database integration for large-scale deployments
- Connection pooling and efficient serialization
- Transaction management and query optimization

## System Requirements

- Docker Engine 19.03+
- Docker Compose 1.27+
- NVIDIA Container Toolkit (for GPU support)
- NVIDIA driver 450.80.02+ (for T4 GPUs)
- 8GB+ RAM (16GB+ recommended for full stack)
- 10GB+ disk space

## Configuration

Each Docker Compose file can be customized through environment variables:

- `MCTX_ENABLE_T4_OPTIMIZATIONS`: Enable T4-specific optimizations (default: 1 for NVIDIA setup)
- `JAX_PLATFORM_NAME`: Set to "gpu" for GPU acceleration, "cpu" for CPU-only mode
- `MCTX_MAX_BATCH_SIZE`: Maximum batch size for MCTS (default: 32)
- `LOG_LEVEL`: Logging level (default: INFO)

See `DOCKER_SETUP.md` for complete configuration details.

## Troubleshooting

If you encounter issues:

1. Check the container logs:
   ```bash
   docker-compose -f docker-compose.nvidia.yml logs -f
   ```

2. Ensure NVIDIA GPU is properly configured:
   ```bash
   nvidia-smi
   docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```

3. Check for port conflicts:
   ```bash
   netstat -tuln | grep -E '8000|8050|9090|3000'
   ```

4. Verify all required files exist:
   ```bash
   ls -la docker/Dockerfile.nvidia frontend/Dockerfile prometheus.yml
   ```

For more detailed troubleshooting, refer to the "Troubleshooting" section in `DOCKER_SETUP.md`.