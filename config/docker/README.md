# MCTX Docker Configuration

This directory contains standardized Docker Compose configurations for different deployment scenarios.

## Available Configurations

1. **`development.yml`**: Lightweight CPU-only setup for development
2. **`gpu-basic.yml`**: Basic NVIDIA GPU setup for quick testing
3. **`gpu-production.yml`**: Full-featured NVIDIA GPU setup with Redis caching for production
4. **`blue-green.yml`**: Zero-downtime deployment configuration with blue/green environments
5. **`cloud.yml`**: Cloud-optimized configuration for Vercel/AWS/GCP deployment

## Usage

From the project root directory:

```bash
# Development setup
docker-compose -f config/docker/development.yml up -d

# Basic NVIDIA GPU setup
docker-compose -f config/docker/gpu-basic.yml up -d

# Production NVIDIA GPU setup
docker-compose -f config/docker/gpu-production.yml up -d

# Blue-Green deployment
docker-compose -f config/docker/blue-green.yml up -d
```

## Helper Scripts

We also provide helper scripts for common deployment scenarios:

```bash
# Quick NVIDIA GPU deployment
./scripts/deploy-nvidia.sh

# Test Docker deployment
./scripts/test-docker-deployment.sh
```

## Environment Variables

Each Docker Compose file uses environment variables with sensible defaults. You can override them by:

1. Creating a `.env` file in the project root
2. Setting environment variables before running docker-compose
3. Passing them directly to the docker-compose command

See the comments in each Docker Compose file for available variables.

## Configuration Files

Supporting configuration files (Prometheus, Grafana, etc.) are located in:
- `config/prometheus/` - Prometheus configuration files
- `config/grafana/` - Grafana dashboards and data sources
- `config/nginx/` - NGINX configurations for reverse proxy and load balancing