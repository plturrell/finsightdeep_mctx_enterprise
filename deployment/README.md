# MCTX Enterprise Deployment

This directory contains everything needed to deploy the MCTX Enterprise Decision Intelligence Platform in a production environment.

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit installed (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## Directory Structure

```
deployment/
├── docker-compose.yml     # Main Docker Compose configuration
├── .env                   # Environment variables (copy from .env.example)
├── fastapi/               # FastAPI backend code
│   ├── Dockerfile         # FastAPI Dockerfile
│   ├── main.py            # Main application code
│   └── requirements.txt   # Python dependencies
├── monitoring/            # Monitoring configuration
│   ├── prometheus.yml     # Prometheus configuration
│   └── grafana/           # Grafana dashboards and configuration
└── logs/                  # Log directory (created at runtime)
```

## Setup

1. Install Docker and NVIDIA Container Toolkit:

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

2. Create your environment file:

```bash
cp .env.example .env
# Edit .env with your specific configuration
```

3. Start the deployment:

```bash
docker-compose up -d
```

## Verification

1. Check that all services are running:

```bash
docker-compose ps
```

2. Verify the API is accessible:

```bash
curl http://localhost:8000/api/v1/health
```

3. Access the monitoring dashboard:

```
Grafana: http://localhost:3000 (default login: admin/admin)
Prometheus: http://localhost:9090
```

## Using the API

The API is available at `http://localhost:8000` and provides the following endpoints:

- `GET /api/v1/health` - Health check endpoint
- `POST /api/v1/search` - Run a search (requires API key if enabled)
- `GET /api/v1/visualization/{search_id}` - Get visualization for a search
- `GET /api/v1/metrics` - Prometheus metrics

Example search request:

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "state": {
      "observation": [0.1, 0.2, 0.3, 0.4, 0.5],
      "legal_actions": [0, 1, 2]
    },
    "config": {
      "num_simulations": 128,
      "temperature": 1.0
    },
    "model": "muzero_default"
  }'
```

## Configuration Options

The `.env` file supports the following configuration options:

- `MCTX_NUM_WORKERS`: Number of worker processes (default: 4)
- `MCTX_MAX_BATCH_SIZE`: Maximum batch size for inference (default: 32)
- `MCTX_DEFAULT_SIMULATIONS`: Default simulation count (default: 128)
- `MCTX_TIMEOUT_SECONDS`: Request timeout in seconds (default: 60)
- `MCTX_GPU_MEMORY_FRACTION`: Fraction of GPU memory to use (default: 0.9)
- `MCTX_USE_MIXED_PRECISION`: Use mixed precision (FP16) (default: true)
- `MCTX_ENABLE_DISTRIBUTED`: Enable distributed mode (default: false)
- `MCTX_NUM_DEVICES`: Number of devices to use (default: 1)
- `MCTX_ENABLE_REDIS_CACHE`: Enable Redis caching (default: true)
- `REDIS_HOST`, `REDIS_PORT`: Redis connection settings
- `MCTX_API_KEY_REQUIRED`: Require API key for access (default: true)
- `MCTX_API_KEYS`: Comma-separated list of valid API keys

## Scaling

To scale the API service:

```bash
docker-compose up -d --scale mctx-api=3
```

For production environments with multiple GPUs, you may need to modify the Docker Compose file to specify GPU allocation.

## Troubleshooting

### GPU Not Available

If the GPU is not available to the container:

1. Check if NVIDIA drivers are installed: `nvidia-smi`
2. Verify NVIDIA Container Toolkit: `docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`
3. Check Docker Compose configuration for GPU resource allocation

### API Performance Issues

1. Check GPU memory usage: `nvidia-smi`
2. Adjust batch size and worker count in `.env`
3. Check logs: `docker-compose logs mctx-api`
4. Monitor metrics in Grafana dashboard

## Security Considerations

1. Change default API keys in production
2. Use a reverse proxy with TLS for production deployments
3. Set up proper authentication for Grafana and Prometheus
4. Secure the Redis instance

## Contact

For enterprise support, contact enterprise@mctx-ai.com
EOF < /dev/null