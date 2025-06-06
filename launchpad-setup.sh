#!/bin/bash
# This script sets up the MCTX environment in NVIDIA LaunchPad

set -e

# Create necessary directories
mkdir -p /home/ubuntu/workspace
mkdir -p /home/ubuntu/workspace/config/prometheus
mkdir -p /home/ubuntu/workspace/examples
mkdir -p /home/ubuntu/workspace/mctx_output

# Copy the docker-compose file to the standard location
cp docker-compose.launchpad.yml /home/ubuntu/workspace/docker-compose.yaml

# Create a minimal Prometheus config
cat > /home/ubuntu/workspace/config/prometheus/gpu.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'mctx'
    static_configs:
      - targets: ['mctx-nvidia:8000']
EOF

# Copy README files to accessible locations
cp README.md /home/ubuntu/workspace/
cp NVIDIA_LAUNCHPAD.md /home/ubuntu/workspace/
cp blueprint.md /home/ubuntu/workspace/

echo "LaunchPad setup complete. Starting Docker Compose..."

# Start Docker Compose
cd /home/ubuntu/workspace
docker-compose up -d

echo "Docker Compose started. Services should be available shortly."
echo "If services don't appear, check logs with: docker-compose logs"