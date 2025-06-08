# MCTX NVIDIA LaunchPad Blueprint

This document provides instructions for deploying MCTX on NVIDIA LaunchPad using the provided Docker Compose blueprint.

## Overview

The MCTX blueprint for NVIDIA LaunchPad provides a fully configured, production-ready deployment of the MCTX stack optimized for T4 GPUs. This deployment includes:

- FastAPI backend with T4-optimized Monte Carlo Tree Search algorithms
- Interactive visualization dashboards
- Performance monitoring tools
- Comprehensive documentation
- Metrics collection and visualization with Prometheus and Grafana

## Hardware Requirements

- NVIDIA T4 GPU (minimum 1)
- 16GB RAM (minimum)
- 8 CPU cores (recommended)
- 10GB storage (minimum)

## Services

The blueprint deploys the following services:

- **API Service (Port 8000)**: Main backend API with MCTX integration and T4 optimizations
- **Visualization Dashboard (Port 8050)**: Interactive MCTS tree visualization
- **Performance Monitoring (Port 8051)**: Advanced metrics and profiling dashboard
- **Documentation Server (Port 8080)**: Comprehensive documentation and tutorials
- **Prometheus (Port 9090)**: Metrics collection and storage
- **Grafana (Port 3001)**: Interactive dashboards for monitoring
- **Redis (Port 6379)**: For caching and rate limiting
- **SAP HANA Integration**: For enterprise data storage and processing

## Deployment

### Prerequisites

- Docker and Docker Compose installed
- NVIDIA Docker Runtime configured
- Access to a machine with an NVIDIA T4 GPU

### Steps

1. Clone the MCTX repository:
   ```bash
   git clone https://github.com/finsightdev/mctx.git
   cd mctx
   ```

2. Launch the blueprint:
   ```bash
   ./run-nvidia-blueprint.sh start
   ```

3. To check the status:
   ```bash
   ./run-nvidia-blueprint.sh status
   ```

4. To view logs:
   ```bash
   ./run-nvidia-blueprint.sh logs [service_name]
   ```

5. To check GPU status:
   ```bash
   ./run-nvidia-blueprint.sh gpu
   ```

6. To stop the services:
   ```bash
   ./run-nvidia-blueprint.sh stop
   ```

## Configuration

The blueprint is configured using environment variables defined in `.env.nvidia`. Key settings include:

- `MAX_BATCH_SIZE`: Maximum batch size for processing (default: 128)
- `MAX_NUM_SIMULATIONS`: Maximum simulations per request (default: 1000)
- `API_SECRET_KEY`: Secret key for API security
- `GRAFANA_PASSWORD`: Password for Grafana admin user

### SAP HANA Configuration

The blueprint includes integration with SAP HANA for enterprise data storage:

- `HANA_HOST`: SAP HANA hostname (default: d93a8739-44a8-4845-bef3-8ec724dea2ce.hana.prod-us10.hanacloud.ondemand.com)
- `HANA_PORT`: SAP HANA port (default: 443)
- `HANA_USER`: SAP HANA username (default: DBADMIN)
- `HANA_PASSWORD`: SAP HANA password
- `HANA_SCHEMA`: Schema name for MCTX tables (default: MCTX)
- `HANA_ENCRYPT`: Whether to use encryption for the connection (default: True)
- `HANA_POOL_SIZE`: Connection pool size (default: 10 for API, 5 for other services)
- `HANA_ENABLE_CACHING`: Whether to enable result caching (default: True)

The SAP HANA integration provides:
- Storage and retrieval of MCTS trees and search results
- Model caching for improved performance
- Search history tracking and analytics
- Enterprise-grade data persistence
- Automatic schema and table management

## T4 GPU Optimizations

The MCTX deployment automatically detects T4 GPUs and enables the following optimizations:

- Tensor Core-aware pruning strategies
- Mixed precision training for efficient GPU utilization
- Memory-optimized search algorithms
- Batch parallelism for improved throughput
- NCCL-based distributed computing capabilities

## Monitoring

Performance monitoring is available through:

1. Grafana dashboards at http://localhost:3001 (login: admin/admin)
2. Performance monitoring dashboard at http://localhost:8051
3. Prometheus metrics at http://localhost:9090

## Scaling Guidelines

For production workloads, the following scaling is recommended:

- Up to 100 concurrent users: 1x NVIDIA T4
- Up to 500 concurrent users: 2x NVIDIA T4
- Up to 1000 concurrent users: 4x NVIDIA T4 or 1x NVIDIA A100

For multi-node deployments, enable the `MCTX_ENABLE_MULTI_NODE` setting and configure the node count and rank for each instance.

## Troubleshooting

If you encounter issues:

1. Check GPU is properly detected: `./run-nvidia-blueprint.sh gpu`
2. Verify all services are healthy: `./run-nvidia-blueprint.sh status`
3. Check service logs: `./run-nvidia-blueprint.sh logs [service_name]`
4. Ensure proper GPU drivers are installed
5. Verify NVIDIA Docker runtime is properly configured