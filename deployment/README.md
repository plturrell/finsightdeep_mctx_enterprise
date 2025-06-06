# MCTX Enterprise Deployment Guide

This guide provides comprehensive instructions for deploying the MCTX Enterprise Decision Intelligence Platform, including both the NVIDIA-optimized backend and the Vercel frontend.

## System Architecture

The deployment consists of two main components:

1. **Backend (FastAPI + NVIDIA GPU)**
   - Production-ready FastAPI service
   - NVIDIA GPU optimizations for T4
   - Distributed computing support
   - SAP HANA integration
   - Redis caching
   - Prometheus and Grafana monitoring

2. **Frontend (Next.js + Vercel)**
   - Interactive visualization interface
   - Real-time metrics dashboard
   - Responsive design
   - Dark/light mode support

## Directory Structure

```
deployment/
├── docker-compose.yml     # Main Docker Compose configuration
├── .env.example           # Environment variables template
├── README.md              # This file
├── fastapi/               # FastAPI backend code
│   ├── Dockerfile         # FastAPI Dockerfile
│   ├── main.py            # Main application code
│   └── requirements.txt   # Python dependencies
├── monitoring/            # Monitoring configuration
│   ├── prometheus.yml     # Prometheus configuration
│   └── grafana/           # Grafana dashboards and configuration
├── vercel/                # Frontend application
│   ├── package.json       # Frontend dependencies
│   ├── src/               # React components and pages
│   └── public/            # Static assets
└── logs/                  # Log directory (created at runtime)
```

## Backend Deployment

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit installed

### Setup

1. **Install NVIDIA Container Toolkit**

```bash
# For Ubuntu
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify installation
sudo docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

2. **Configure Environment**

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your specific configuration
nano .env
```

3. **Start Deployment**

```bash
docker-compose up -d
```

This will start the following services:
- `mctx-api`: The main FastAPI service
- `redis`: For caching search results
- `prometheus`: For metrics collection
- `grafana`: For monitoring dashboards
- `redis-exporter`: For Redis metrics

4. **Verify Deployment**

```bash
# Check container status
docker-compose ps

# Check API health
curl http://localhost:8000/api/v1/health

# Access monitoring
# Grafana: http://localhost:3000 (default login: admin/admin)
# Prometheus: http://localhost:9090
```

### API Usage

The API provides the following endpoints:

- `GET /api/v1/health`: Health check endpoint
- `POST /api/v1/search`: Run a search (requires API key if enabled)
- `GET /api/v1/visualization/{search_id}`: Get visualization for a search
- `GET /api/v1/metrics`: Prometheus metrics

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

### Production Configuration

For production deployments, consider the following:

1. **Use HTTPS with Nginx**

```bash
# Install Nginx
apt-get update
apt-get install -y nginx certbot python3-certbot-nginx

# Configure Nginx as reverse proxy
cat > /etc/nginx/sites-available/mctx-api << EOF
server {
    listen 80;
    server_name api.mctx.example.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
}
EOF

# Enable the site
ln -s /etc/nginx/sites-available/mctx-api /etc/nginx/sites-enabled/
nginx -t
systemctl reload nginx

# Set up SSL
certbot --nginx -d api.mctx.example.com
```

2. **Enable the NGINX service in docker-compose.yml**

Uncomment and configure the `nginx` service in the docker-compose.yml file.

3. **Change Default Credentials**

Update the following in your `.env` file:
- `MCTX_API_KEYS`: Generate strong API keys
- `GRAFANA_ADMIN_PASSWORD`: Change from default "admin"

4. **Configure SAP HANA Integration (if needed)**

Uncomment and configure the `hana-client` service in the docker-compose.yml file.

## Frontend Deployment

### Prerequisites

- Vercel account
- Node.js 16+ and npm

### Local Development

For local development of the frontend:

```bash
# Navigate to frontend directory
cd vercel

# Install dependencies
npm install

# Create environment file
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local

# Start development server
npm run dev
```

### Vercel Deployment

1. **Configure Environment Variables**

In the Vercel dashboard, add the following environment variable:

```
NEXT_PUBLIC_API_URL=https://api.mctx.example.com
```

2. **Deploy to Vercel**

```bash
# Install Vercel CLI
npm install -g vercel

# Login to Vercel
vercel login

# Deploy
cd vercel
vercel --prod
```

Alternatively, connect your GitHub repository to Vercel for automatic deployments.

3. **Verify Deployment**

Visit your deployed site at `https://mctx-visualization.vercel.app` (or your custom domain).

## Scaling

### Backend Scaling

1. **Horizontal Scaling**

```bash
# Scale the API service
docker-compose up -d --scale mctx-api=3
```

For multiple servers, set up a load balancer like HAProxy:

```bash
# Example HAProxy configuration
frontend http_front
   bind *:80
   stats uri /haproxy?stats
   default_backend http_back

backend http_back
   balance roundrobin
   server server1 10.0.0.1:8000 check
   server server2 10.0.0.2:8000 check
   server server3 10.0.0.3:8000 check
```

2. **Vertical Scaling**

Use more powerful NVIDIA GPUs like A100 or H100.

### GPU Configuration

For environments with multiple GPUs:

```yaml
# In docker-compose.yml
services:
  mctx-api:
    # ...
    environment:
      - NVIDIA_VISIBLE_DEVICES=0,1  # Specify GPU indices
      - MCTX_ENABLE_DISTRIBUTED=true
      - MCTX_NUM_DEVICES=2
```

## Monitoring and Maintenance

### Monitoring Dashboards

Access Grafana at http://localhost:3000 with the following dashboards:
- MCTX Performance: Overall system performance
- GPU Utilization: NVIDIA GPU metrics
- API Health: Endpoint health and response times
- Redis Cache: Cache hit rates and memory usage

### Setting Up Alerts

In Grafana:
1. Navigate to Alerting → Notification channels
2. Add channels for email, Slack, or PagerDuty
3. Create alert rules for high GPU utilization, API errors, etc.

### Backup Procedures

For data persistence:

```bash
# Backup volumes
docker run --rm -v mctx-data:/source -v /path/to/backup:/backup alpine tar -czf /backup/mctx-data-$(date +%Y%m%d).tar.gz -C /source .
docker run --rm -v mctx-redis-data:/source -v /path/to/backup:/backup alpine tar -czf /backup/redis-data-$(date +%Y%m%d).tar.gz -C /source .
```

## Troubleshooting

### GPU Issues

If the GPU is not available to the container:

1. Check if NVIDIA drivers are installed: `nvidia-smi`
2. Verify NVIDIA Container Toolkit: `docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`
3. Check docker-compose configuration

### API Errors

1. Check logs: `docker-compose logs mctx-api`
2. Verify environment variables: `docker-compose config`
3. Check container health: `docker inspect mctx-api`

### Redis Connection Issues

1. Check if Redis is running: `docker-compose ps redis`
2. Test Redis connection: `docker exec -it mctx-redis redis-cli ping`

### Visualization Problems

1. Check browser console for errors
2. Verify API URL configuration
3. Check CORS settings in the API

## Security Considerations

1. **API Security**
   - Change default API keys
   - Implement rate limiting
   - Use HTTPS for all communication

2. **Container Security**
   - Regularly update container images
   - Scan for vulnerabilities: `docker scan mctx-api:latest`
   - Restrict container capabilities

3. **Network Security**
   - Use internal networks for service communication
   - Restrict access to monitoring endpoints
   - Configure firewall rules

## Additional Resources

- [MCTX Documentation](../docs/)
- [T4 Optimization Guide](../docs/t4_optimizations.md)
- [Distributed MCTS](../docs/distributed_mcts.md)
- [SAP HANA Integration](../docs/hana_integration.md)
- [Visualization Guide](../docs/visualization.md)

## Contact

For enterprise support, contact enterprise@mctx-ai.com