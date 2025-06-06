# MCTX Docker Setup Guide

This project includes two Docker Compose configurations:

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

## 2. Production NVIDIA GPU Setup (Deployment Directory)

The production-ready Docker Compose file in the deployment directory (`./deployment/docker-compose.yml`) is optimized for **NVIDIA GPU deployment** and includes:

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
- **For production deployment with NVIDIA GPUs**: Use the setup in the deployment directory (`./deployment/docker-compose.yml`)

For more information about the production deployment, see the [deployment guide](./deployment/README.md).