# MCTX Docker Setup Guide

This project includes three Docker Compose configurations:

## 1. Development Setup (Project Root)

The Docker Compose file in the project root (`./docker-compose.yml`) is intended for **local development** with minimal setup. It runs on CPU only and includes:

- Basic API service
- Simple frontend
- Prometheus and Grafana for monitoring

To use the development setup:

```bash
# From the project root
docker-compose up -d
```

Access:
- API: http://localhost:8000
- Frontend: http://localhost:3000
- Grafana: http://localhost:3001 (admin/admin)
- Prometheus: http://localhost:9090

## 2. Simplified NVIDIA GPU Setup (Project Root)

The simplified NVIDIA GPU setup (`./docker-compose.nvidia.yml`) is designed for **quick deployment with T4 GPUs** and includes:

- NVIDIA GPU-optimized container with T4 performance enhancements
- Visualization server
- Basic monitoring with Prometheus and Grafana

To use the simplified NVIDIA GPU setup:

```bash
# From the project root
docker-compose -f docker-compose.nvidia.yml up -d

# Or use our convenient deployment script
./deploy-nvidia.sh
```

The deployment script performs environment checks and provides a guided setup experience.

Access:
- API: http://localhost:8000
- Visualization Dashboard: http://localhost:8050
- Grafana: http://localhost:3001 (admin/admin)
- Prometheus: http://localhost:9090

## 3. Production NVIDIA GPU Setup (Deployment Directory)

The production-ready Docker Compose file in the deployment directory (`./deployment/docker-compose.yml`) is optimized for **enterprise NVIDIA GPU deployment** and includes:

- FastAPI backend with NVIDIA GPU optimizations
- T4-specific performance enhancements
- Redis caching
- Comprehensive monitoring with Prometheus and Grafana
- Optional SAP HANA integration
- Optional NGINX reverse proxy with TLS

To use the production NVIDIA setup:

```bash
# Navigate to the deployment directory
cd deployment

# Copy and customize the environment file
cp .env.example .env

# Start the deployment
docker-compose up -d
```

Access:
- API: http://localhost:8000
- Grafana: http://localhost:3000 (login with credentials from .env)
- Prometheus: http://localhost:9090

## Which Docker Compose Should I Use?

- **For development and testing**: Use the setup in the project root (`./docker-compose.yml`)
- **For quick NVIDIA GPU deployment**: Use the simplified setup in the project root (`./docker-compose.nvidia.yml`) or the `./deploy-nvidia.sh` script
- **For production deployment with NVIDIA GPUs**: Use the setup in the deployment directory (`./deployment/docker-compose.yml`)

For more information about the production deployment, see the [deployment guide](./deployment/README.md).

## Troubleshooting Docker Deployment

### Common Issues

1. **Build context errors**: If you encounter build context errors, check that all referenced Dockerfiles exist and paths are correct.

2. **NVIDIA GPU not detected**: Ensure the NVIDIA Container Toolkit is installed and configured properly:
   ```bash
   # Check NVIDIA Docker runtime
   docker info | grep -i nvidia
   
   # Check GPU visibility
   nvidia-smi
   ```

3. **Port conflicts**: If services fail to start due to port conflicts, check if the ports are already in use and modify the port mappings in the docker-compose file.

4. **Missing configuration files**: Ensure all referenced configuration files (prometheus.yml, etc.) exist in the expected locations.

For additional help, refer to the [Docker documentation](https://docs.docker.com/compose/reference/) or open an issue in the project repository.