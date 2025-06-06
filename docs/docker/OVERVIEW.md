# MCTX Docker Deployment

This document provides a comprehensive overview of the Docker deployment options for MCTX.

## Docker Configuration Structure

All Docker configurations have been consolidated in the `config/docker/` directory:

```
config/
├── docker/
│   ├── development.yml      # Development setup (CPU-only)
│   ├── gpu-basic.yml        # Basic NVIDIA GPU setup
│   ├── gpu-production.yml   # Production NVIDIA GPU setup
│   ├── blue-green.yml       # Zero-downtime deployment
│   ├── cloud.yml            # Cloud-optimized deployment
│   └── README.md            # Docker configuration guide
├── prometheus/
│   ├── development.yml      # Prometheus config for development
│   ├── gpu.yml              # Prometheus config for GPU setups
│   ├── production.yml       # Prometheus config for production
│   ├── blue-green.yml       # Prometheus config for blue-green
│   └── cloud.yml            # Prometheus config for cloud
└── nginx/                   # NGINX configurations (for blue-green)
```

## Quick Start

For most users, the following commands will get you started quickly:

```bash
# For development (CPU-only)
docker-compose -f config/docker/development.yml up -d

# For NVIDIA GPU deployment
./scripts/deploy-nvidia.sh
```

## Deployment Options

MCTX supports five main deployment configurations:

1. **Development Setup**
   - CPU-only, lightweight configuration
   - Basic API and frontend services
   - Simple monitoring setup
   - Command: `docker-compose -f config/docker/development.yml up -d`

2. **Basic GPU Setup**
   - Optimized for NVIDIA T4 GPUs
   - Includes visualization server and monitoring
   - Command: `docker-compose -f config/docker/gpu-basic.yml up -d`

3. **Production GPU Setup**
   - Enterprise-grade configuration with Redis caching
   - Comprehensive monitoring stack
   - Command: `docker-compose -f config/docker/gpu-production.yml up -d`

4. **Blue-Green Deployment**
   - Zero-downtime deployment with dual environments
   - NGINX-based traffic routing
   - Command: `docker-compose -f config/docker/blue-green.yml up -d`

5. **Cloud Deployment**
   - Optimized for Vercel, AWS, and GCP
   - Lightweight, scalable configuration
   - Command: `docker-compose -f config/docker/cloud.yml up -d`

## Helper Scripts

For convenience, we provide helper scripts in the `scripts/` directory:

- `scripts/deploy-nvidia.sh`: Interactive script for NVIDIA GPU deployment
- `scripts/test-docker-deployment.sh`: Test script for validating Docker configurations

## Customizing Deployments

Each Docker Compose file supports customization through environment variables. You can create a `.env` file in the project root or pass variables directly to docker-compose:

```bash
# Example: Customizing ports and log level
LOG_LEVEL=DEBUG API_PORT=8002 GRAFANA_PORT=3002 docker-compose -f config/docker/development.yml up -d
```

## Troubleshooting

If you encounter issues with Docker deployment:

1. Check if Docker and Docker Compose are installed and running
2. For GPU deployments, verify NVIDIA Container Toolkit is installed
3. Check for port conflicts with existing services
4. Examine container logs for specific errors
5. Run the test script: `./scripts/test-docker-deployment.sh`

For detailed troubleshooting steps, see the "Troubleshooting" section in `config/docker/README.md`.