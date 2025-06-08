# MCTX Backend Docker Setup

This document provides instructions for running the MCTX backend services using Docker Compose for local development and testing.

## Services

The backend setup includes the following services:

- **API**: FastAPI application serving the MCTX API endpoints
- **Redis**: For caching and rate limiting
- **Prometheus**: For metrics collection
- **Grafana**: For metrics visualization
- **Documentation Server**: Nginx server for hosting documentation
- **Worker**: Background task processing

## Prerequisites

- Docker and Docker Compose installed
- Port 8000, 6379, 9090, 3001, and 8080 available

## Quick Start

To start all backend services:

```bash
./run-backend.sh start
```

To check the status of the services:

```bash
./run-backend.sh status
```

To stop all services:

```bash
./run-backend.sh stop
```

## Configuration

The backend services are configured using environment variables defined in `.env.backend`. The key settings include:

- `API_PORT`: Port for the FastAPI service (default: 8000)
- `REDIS_PORT`: Port for Redis (default: 6379)
- `PROMETHEUS_PORT`: Port for Prometheus (default: 9090)
- `GRAFANA_PORT`: Port for Grafana (default: 3001)
- `DOCS_PORT`: Port for the documentation server (default: 8080)

## Service Endpoints

- **API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/api/docs
- **Documentation Server**: http://localhost:8080
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3001 (admin/admin)

## Testing

To test the backend deployment:

```bash
./run-backend.sh test
```

This will:
1. Start all required services
2. Check the health of each service
3. Test the API endpoints
4. Verify the documentation server
5. Check Prometheus and Grafana

## Troubleshooting

If you encounter issues:

1. Check service status: `./run-backend.sh status`
2. View service logs: `./run-backend.sh logs` or `./run-backend.sh logs [service_name]`
3. Ensure ports are available: `netstat -tuln | grep <port>`
4. Check container health: `docker inspect --format='{{.State.Health.Status}}' mctx-backend-api`

## NVIDIA Deployment

This Docker Compose setup is designed as a prerequisite for NVIDIA Blueprint deployment. After ensuring everything works locally, you can proceed to deploy on NVIDIA LaunchPad using the blueprint configuration.