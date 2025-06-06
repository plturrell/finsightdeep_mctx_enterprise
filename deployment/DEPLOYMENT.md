# MCTX Deployment Guide

This guide covers how to deploy the MCTX visualization system with NVIDIA backend optimizations using FastAPI and Vercel.

## System Architecture

The system consists of two main components:

1. **Backend API (FastAPI + NVIDIA GPU)**
   - Production-ready FastAPI service
   - NVIDIA GPU optimizations for T4
   - Distributed computing support
   - SAP HANA integration
   - Prometheus monitoring

2. **Frontend (Next.js + Vercel)**
   - Elegant visualization interface
   - Interactive controls
   - Responsive design
   - Optimized for performance

## Backend Deployment

### Prerequisites

- NVIDIA GPU server with CUDA 11.8+
- Docker and Docker Compose
- SAP HANA database connection

### Deployment Steps

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/mctx.git
cd mctx
```

2. **Configure environment variables**

Create a `.env` file in the `deployment/fastapi` directory:

```
HANA_HOST=your-hana-host.example.com
HANA_PORT=443
HANA_USER=your-hana-user
HANA_PASSWORD=your-hana-password
```

3. **Build and start the containers**

```bash
cd deployment/fastapi
docker-compose up -d
```

This will start the following services:
- MCTX API (on port 8000)
- Redis (for caching)
- Prometheus (for monitoring)
- Grafana (for visualization dashboards)

4. **Verify deployment**

Check that the API is running:

```bash
curl http://localhost:8000
```

Access the monitoring dashboards:
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090

### Production Considerations

For production deployment, consider the following:

1. **Use NVIDIA GPU Cloud (NGC)**

```bash
docker pull nvcr.io/nvidia/tensorflow:22.12-tf2-py3
```

2. **Configure HTTPS with Nginx**

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

3. **Configure monitoring alerts**

Set up Prometheus alerting rules in `prometheus.yml`.

## Frontend Deployment

### Prerequisites

- Vercel account
- Node.js 16+

### Deployment Steps

1. **Configure environment variables**

In the Vercel dashboard, add the following environment variable:

```
NEXT_PUBLIC_API_URL=https://api.mctx.example.com
```

2. **Deploy to Vercel**

```bash
cd deployment/vercel
npm install
npm run build

# Deploy using Vercel CLI
vercel --prod
```

Alternatively, connect your GitHub repository to Vercel for automatic deployments.

3. **Verify deployment**

Visit your deployed site at `https://mctx-visualization.vercel.app` (or your custom domain).

### Production Considerations

1. **Custom domain**

Configure a custom domain in the Vercel dashboard.

2. **Performance optimization**

Enable Vercel Edge Functions and Analytics for improved performance.

3. **Access control**

Implement authentication using Auth0 or Next.js Auth.

## Monitoring and Maintenance

### Monitoring

- Use Grafana dashboards to monitor the backend performance
- Set up alerts for high GPU utilization or long search durations
- Monitor SAP HANA connection health

### Maintenance

- Regularly update dependencies
- Run backup jobs for the database
- Schedule system updates during low-traffic periods

## Scaling

### Backend Scaling

1. **Horizontal scaling**

Add more API servers behind a load balancer:

```bash
# Example using HAProxy
apt-get install -y haproxy

# Configure HAProxy
cat > /etc/haproxy/haproxy.cfg << EOF
frontend http_front
   bind *:80
   stats uri /haproxy?stats
   default_backend http_back

backend http_back
   balance roundrobin
   server server1 10.0.0.1:8000 check
   server server2 10.0.0.2:8000 check
   server server3 10.0.0.3:8000 check
EOF

systemctl restart haproxy
```

2. **Vertical scaling**

Use more powerful NVIDIA GPUs like A100 or H100.

### Frontend Scaling

Vercel automatically handles scaling for the frontend.

## Troubleshooting

### Backend Issues

1. **GPU not detected**

Check NVIDIA drivers and Docker configuration:

```bash
nvidia-smi
docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

2. **API errors**

Check logs:

```bash
docker-compose logs -f mctx-api
```

### Frontend Issues

1. **API connection errors**

Check CORS configuration and API URL.

2. **Visualization not rendering**

Check browser console for errors and ensure WebGL is enabled.

## Security Considerations

1. **API authentication**

Implement JWT authentication for the API.

2. **HANA security**

Use SSL for HANA connections and store credentials securely.

3. **Container security**

Regularly scan containers for vulnerabilities:

```bash
docker scan mctx-api:latest
```

## Conclusion

This deployment architecture provides a production-ready setup for MCTX with NVIDIA optimizations, elegant visualization, and comprehensive monitoring. Follow the security recommendations for a robust production deployment.