# MCTX Deployment Guide

This guide provides comprehensive instructions for deploying MCTX in production environments, covering both backend and frontend deployments.

## Backend Deployment

### FastAPI Backend

MCTX provides a ready-to-use FastAPI backend optimized for NVIDIA GPUs.

#### Prerequisites

- Docker and Docker Compose
- NVIDIA Container Toolkit (for GPU support)
- NVIDIA GPU(s) with CUDA 11.2+ support
- 16GB+ system RAM
- 50GB+ disk space

#### Quick Start

```bash
# Clone the deployment repository
git clone https://github.com/your-org/mctx-deployment.git
cd mctx-deployment

# Configure environment variables
cp .env.example .env
# Edit .env with your configuration

# Start the deployment
docker-compose up -d
```

#### Configuration Options

Key environment variables in `.env`:

```bash
# Core Configuration
MCTX_NUM_WORKERS=4                 # Number of worker processes
MCTX_MAX_BATCH_SIZE=32             # Maximum batch size for inference
MCTX_DEFAULT_SIMULATIONS=128       # Default simulation count
MCTX_TIMEOUT_SECONDS=60            # Request timeout

# GPU Configuration
MCTX_GPU_MEMORY_FRACTION=0.9       # Fraction of GPU memory to use
MCTX_USE_MIXED_PRECISION=true      # Use mixed precision (FP16)

# Distributed Configuration
MCTX_ENABLE_DISTRIBUTED=false      # Enable distributed mode
MCTX_NUM_DEVICES=1                 # Number of devices to use

# Cache Configuration
MCTX_ENABLE_REDIS_CACHE=true       # Enable Redis caching
REDIS_HOST=redis                   # Redis host
REDIS_PORT=6379                    # Redis port
REDIS_CACHE_TTL=3600               # Cache TTL in seconds

# Monitoring Configuration
MCTX_ENABLE_PROMETHEUS=true        # Enable Prometheus metrics
PROMETHEUS_PORT=9090               # Prometheus port

# Security Configuration
MCTX_API_KEY_REQUIRED=true         # Require API key for access
MCTX_API_KEYS=key1,key2,key3       # Comma-separated list of valid API keys
```

#### Docker Compose Configuration

The `docker-compose.yml` includes:

```yaml
version: '3.8'

services:
  mctx-api:
    build:
      context: ./backend
      dockerfile: Dockerfile
    restart: always
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    env_file:
      - .env
    depends_on:
      - redis
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  redis:
    image: redis:7.0-alpine
    restart: always
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes

  prometheus:
    image: prom/prometheus:v2.40.0
    restart: always
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus

  grafana:
    image: grafana/grafana:9.3.1
    restart: always
    ports:
      - "3000:3000"
    volumes:
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
      - grafana-data:/var/lib/grafana
    depends_on:
      - prometheus

volumes:
  redis-data:
  prometheus-data:
  grafana-data:
```

#### API Endpoints

The FastAPI backend exposes the following endpoints:

##### MCTS Search

```
POST /api/v1/search
```

Request body:
```json
{
  "state": {
    "observation": [...],
    "legal_actions": [0, 1, 2, 3]
  },
  "config": {
    "num_simulations": 128,
    "temperature": 1.0,
    "dirichlet_fraction": 0.25,
    "dirichlet_alpha": 0.3
  },
  "model": "muzero_default"
}
```

Response:
```json
{
  "action": 2,
  "action_weights": [0.1, 0.2, 0.6, 0.1],
  "root_value": 0.75,
  "q_values": [0.5, 0.7, 0.8, 0.4],
  "visit_counts": [10, 20, 60, 10],
  "search_id": "3f7d53c2-7c9e-4a2a-8e7b-6f3a7d2c4b1a",
  "computation_time_ms": 235
}
```

##### Tree Visualization

```
GET /api/v1/visualization/{search_id}
```

Response: HTML visualization of the search tree.

##### Health Check

```
GET /api/v1/health
```

Response:
```json
{
  "status": "healthy",
  "gpu_status": "available",
  "version": "1.2.3",
  "uptime_seconds": 3600
}
```

#### Monitoring

The deployment includes Prometheus and Grafana for monitoring:

- **Prometheus**: Collects metrics from the API server
- **Grafana**: Provides visualization dashboards

Pre-configured dashboards include:
- GPU Usage Dashboard
- API Performance Dashboard
- Search Metrics Dashboard
- Redis Cache Dashboard

#### Scaling

For horizontal scaling:

```bash
# Scale to 3 API instances
docker-compose up -d --scale mctx-api=3
```

For load balancing with Nginx:

```nginx
upstream mctx_api {
    server mctx-api-1:8000;
    server mctx-api-2:8000;
    server mctx-api-3:8000;
}

server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://mctx_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Kubernetes Deployment

For more advanced deployments, MCTX can be deployed on Kubernetes.

#### Prerequisites

- Kubernetes cluster (1.19+)
- Helm (3.0+)
- NVIDIA GPU Operator installed

#### Deployment with Helm

```bash
# Add the MCTX Helm repository
helm repo add mctx https://helm.example.com/mctx
helm repo update

# Install the MCTX backend
helm install mctx-backend mctx/mctx-backend \
  --namespace mctx \
  --create-namespace \
  --set replicas=3 \
  --set gpu.enabled=true \
  --set gpu.count=1 \
  --set redis.enabled=true \
  --set monitoring.enabled=true \
  --values custom-values.yaml
```

#### Example `custom-values.yaml`

```yaml
# MCTX Backend Helm Values
replicas: 3

image:
  repository: ghcr.io/your-org/mctx-backend
  tag: v1.2.3
  pullPolicy: IfNotPresent

resources:
  requests:
    cpu: 2
    memory: 4Gi
  limits:
    cpu: 4
    memory: 8Gi

gpu:
  enabled: true
  count: 1
  memory: 16Gi

config:
  numWorkers: 4
  maxBatchSize: 32
  defaultSimulations: 128
  timeoutSeconds: 60
  useMixedPrecision: true
  enableDistributed: false

redis:
  enabled: true
  persistence:
    enabled: true
    size: 10Gi

monitoring:
  enabled: true
  prometheus:
    enabled: true
  grafana:
    enabled: true
    dashboards:
      - mctx-performance
      - mctx-gpu-usage

ingress:
  enabled: true
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: mctx-api.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: mctx-api-tls
      hosts:
        - mctx-api.example.com
```

## Frontend Deployment

### Vercel Deployment

MCTX includes a ready-to-use Next.js frontend for visualization and interaction.

#### Prerequisites

- Node.js 16+ and npm
- Vercel CLI (optional for local development)
- GitHub account (for Vercel integration)

#### Quick Start

```bash
# Clone the frontend repository
git clone https://github.com/your-org/mctx-frontend.git
cd mctx-frontend

# Install dependencies
npm install

# Configure environment
cp .env.example .env.local
# Edit .env.local with your configuration

# Run locally
npm run dev

# Deploy to Vercel
vercel --prod
```

#### Configuration Options

Key environment variables in `.env.local`:

```bash
# API Configuration
NEXT_PUBLIC_API_URL=https://mctx-api.example.com
NEXT_PUBLIC_API_VERSION=v1

# Authentication
NEXT_PUBLIC_AUTH_ENABLED=true
NEXT_PUBLIC_AUTH_PROVIDER=auth0
AUTH0_CLIENT_ID=your-auth0-client-id
AUTH0_CLIENT_SECRET=your-auth0-client-secret
AUTH0_DOMAIN=your-auth0-domain

# Features
NEXT_PUBLIC_ENABLE_VISUALIZATIONS=true
NEXT_PUBLIC_ENABLE_DASHBOARD=true
NEXT_PUBLIC_ENABLE_EXPERIMENTS=true
NEXT_PUBLIC_ENABLE_CUSTOM_MODELS=true

# UI Configuration
NEXT_PUBLIC_DEFAULT_THEME=light
NEXT_PUBLIC_PRIMARY_COLOR=#0066cc
```

#### Vercel Deployment

The easiest way to deploy the frontend is using Vercel:

1. Connect your GitHub repository to Vercel
2. Configure environment variables in the Vercel dashboard
3. Deploy the application

For custom domains, configure in the Vercel dashboard under "Domains".

#### Static Export

For environments where Vercel is not available:

```bash
# Build the application
npm run build

# Export as static HTML/JS/CSS
npm run export

# The result will be in the 'out' directory
```

This can be deployed to any static hosting service like Netlify, AWS S3, or GitHub Pages.

#### Frontend Features

The frontend includes:

- **Interactive Tree Visualization**: Explore search trees with zoom, pan, and detail views
- **Dashboard**: Monitor system performance and search metrics
- **Experiment Management**: Compare different MCTS configurations
- **Model Management**: Upload and manage custom models
- **API Explorer**: Interactive documentation for the API
- **Authentication**: Integration with Auth0 for secure access

### Docker Deployment

The frontend can also be deployed using Docker:

#### Dockerfile

```dockerfile
FROM node:16-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/out /usr/share/nginx/html
COPY nginx/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

#### Nginx Configuration

```nginx
server {
    listen 80;
    server_name frontend.example.com;
    root /usr/share/nginx/html;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass https://mctx-api.example.com/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

#### Docker Compose

```yaml
version: '3.8'

services:
  frontend:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "80:80"
    environment:
      - NEXT_PUBLIC_API_URL=https://mctx-api.example.com
      - NEXT_PUBLIC_API_VERSION=v1
      - NEXT_PUBLIC_AUTH_ENABLED=true
```

## Multi-Cloud Deployment

MCTX supports deployment across multiple cloud providers for maximum availability and performance.

### AWS Deployment

For AWS deployment:

1. Use AWS ECR for container registry
2. Deploy using ECS with EC2 instances for GPU support
3. Use EFS for shared model storage
4. Set up CloudWatch for monitoring
5. Use ALB for load balancing
6. Deploy frontend to Amplify or S3 + CloudFront

### GCP Deployment

For GCP deployment:

1. Use GCR for container registry
2. Deploy using GKE with GPU nodes
3. Use Cloud Storage for model storage
4. Set up Cloud Monitoring for observability
5. Use Cloud Load Balancing for traffic management
6. Deploy frontend to Cloud Run or Firebase Hosting

### Azure Deployment

For Azure deployment:

1. Use Azure Container Registry
2. Deploy using AKS with GPU nodes
3. Use Azure Files for model storage
4. Set up Azure Monitor for observability
5. Use Azure Application Gateway for load balancing
6. Deploy frontend to Azure Static Web Apps

## Performance Optimization

### GPU Optimization

For maximum GPU performance:

1. Enable mixed precision training (FP16)
2. Optimize batch sizes based on GPU memory
3. Configure memory limits to prevent OOM errors
4. Use tensor core-aligned operations
5. Monitor GPU memory usage and throughput

Example NVIDIA configuration for Docker:

```bash
# Run with specific NVIDIA runtime options
docker run --gpus all \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  mctx-backend:latest
```

### API Performance

For maximum API throughput:

1. Configure asynchronous workers:
   ```bash
   # In Docker Compose
   environment:
     - MCTX_NUM_WORKERS=4
     - MCTX_WORKER_CONNECTIONS=1000
   ```

2. Enable Redis caching:
   ```bash
   # In Docker Compose
   environment:
     - MCTX_ENABLE_REDIS_CACHE=true
     - REDIS_CACHE_TTL=3600
   ```

3. Configure request timeouts:
   ```bash
   # In Docker Compose
   environment:
     - MCTX_REQUEST_TIMEOUT=60
     - MCTX_KEEP_ALIVE=75
   ```

## Security Considerations

### API Security

Secure your API deployment with:

1. API Key Authentication:
   ```bash
   # In Docker Compose
   environment:
     - MCTX_API_KEY_REQUIRED=true
     - MCTX_API_KEYS=key1,key2,key3
   ```

2. TLS Encryption:
   ```nginx
   # In Nginx
   server {
       listen 443 ssl;
       server_name api.example.com;
       
       ssl_certificate /etc/nginx/certs/api.example.com.crt;
       ssl_certificate_key /etc/nginx/certs/api.example.com.key;
       ssl_protocols TLSv1.2 TLSv1.3;
       
       # ... other configuration
   }
   ```

3. Rate Limiting:
   ```nginx
   # In Nginx
   http {
       limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
       
       server {
           # ... other configuration
           
           location /api/ {
               limit_req zone=api burst=20 nodelay;
               proxy_pass http://backend;
           }
       }
   }
   ```

### Model Security

Protect your models with:

1. Signed URLs for model access
2. Encryption at rest for model storage
3. Versioning and access controls for model updates

### Frontend Security

Secure your frontend with:

1. Content Security Policy:
   ```html
   <!-- In index.html -->
   <meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'self'; connect-src 'self' https://api.example.com">
   ```

2. Authentication with Auth0 or similar provider
3. CSRF protection for API requests
4. XSS protection with proper input sanitization

## Disaster Recovery

### Backup Strategy

Implement a comprehensive backup strategy:

1. Regular model backups:
   ```bash
   # In cron job
   0 2 * * * aws s3 sync /path/to/models s3://mctx-backups/models/
   ```

2. Database backups:
   ```bash
   # In cron job
   0 3 * * * pg_dump -U postgres mctx | gzip > /backups/mctx-$(date +%Y%m%d).sql.gz
   ```

3. Configuration backups:
   ```bash
   # In cron job
   0 4 * * * tar -czf /backups/mctx-config-$(date +%Y%m%d).tar.gz /path/to/config/
   ```

### High Availability

For high availability:

1. Multi-region deployment
2. Automatic failover with health checks
3. Read replicas for databases
4. Redundant API instances
5. CDN for frontend assets

## Troubleshooting

### Common Issues

1. **GPU Memory Errors**:
   ```bash
   # Check GPU memory usage
   nvidia-smi
   
   # Reduce batch size in configuration
   MCTX_MAX_BATCH_SIZE=16
   ```

2. **API Timeouts**:
   ```bash
   # Increase timeout settings
   MCTX_TIMEOUT_SECONDS=120
   
   # Check logs for performance bottlenecks
   docker logs mctx-api
   ```

3. **Redis Connection Issues**:
   ```bash
   # Test Redis connection
   redis-cli -h redis ping
   
   # Check Redis logs
   docker logs redis
   ```

### Logging

Configure comprehensive logging:

```yaml
# In Docker Compose
environment:
  - MCTX_LOG_LEVEL=info
  - MCTX_LOG_FORMAT=json
  - MCTX_LOG_FILE=/app/logs/mctx.log

volumes:
  - ./logs:/app/logs
```

### Monitoring Alerts

Set up alerts for:

1. High GPU memory usage
2. High API latency
3. High error rates
4. Low cache hit rates
5. Disk space constraints

Example Prometheus alert:

```yaml
# In prometheus/alerts.yml
groups:
- name: mctx-alerts
  rules:
  - alert: HighGPUMemoryUsage
    expr: nvidia_gpu_memory_used_bytes / nvidia_gpu_memory_total_bytes > 0.9
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High GPU memory usage ({{ $value }})"
      description: "GPU memory usage is above 90% for 5 minutes."
```

## Conclusion

This deployment guide covers the essential aspects of deploying MCTX in production environments. For more specific configurations or custom deployments, please contact the MCTX team or refer to the [GitHub repository](https://github.com/your-org/mctx).