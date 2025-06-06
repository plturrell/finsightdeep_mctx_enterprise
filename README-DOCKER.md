# MCTX Docker Deployment Guide

This is a quick reference guide for deploying MCTX with Docker. For comprehensive documentation, see the [Docker Documentation](docs/docker/OVERVIEW.md).

## Quick Start

```bash
# Development setup (CPU-only)
docker-compose up -d

# NVIDIA GPU deployment
./bin/deploy-nvidia.sh
```

## Deployment Options

MCTX provides multiple deployment options:

1. **Development (CPU-only)**
   ```bash
   docker-compose up -d
   ```

2. **Basic NVIDIA GPU**
   ```bash
   docker-compose -f docker-compose.nvidia.yml up -d
   # or
   docker-compose -f config/docker/gpu-basic.yml up -d
   ```

3. **Production NVIDIA GPU**
   ```bash
   docker-compose -f config/docker/gpu-production.yml up -d
   ```

4. **Blue-Green Deployment**
   ```bash
   docker-compose -f config/docker/blue-green.yml up -d
   ```

5. **Cloud Deployment**
   ```bash
   docker-compose -f config/docker/cloud.yml up -d
   ```

## Testing

To validate your Docker deployment:

```bash
./bin/test-docker-deployment.sh
```

## Customization

Create a `.env` file based on `.env.docker` to customize the deployment:

```bash
cp .env.docker .env
# Edit .env with your preferred settings
```

## Documentation

For detailed documentation, see:
- [Docker Overview](docs/docker/OVERVIEW.md)
- [Docker Setup Guide](docs/docker/SETUP.md)
- [Docker Configuration Reference](config/docker/README.md)