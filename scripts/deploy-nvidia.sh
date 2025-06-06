#!/bin/bash
set -e

# MCTX NVIDIA GPU Deployment Script
# This script simplifies the deployment of MCTX with NVIDIA GPU support

# Color definitions
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== MCTX NVIDIA GPU Deployment ===${NC}"
echo "Starting deployment at $(date)"

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed.${NC}"
    echo "Please install Docker before continuing."
    exit 1
fi

# Check for Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed.${NC}"
    echo "Please install Docker Compose before continuing."
    exit 1
fi

# Check for NVIDIA Docker
if ! docker info | grep -q "Runtimes:.*nvidia"; then
    echo -e "${YELLOW}Warning: NVIDIA Docker runtime not detected.${NC}"
    echo "For GPU support, please install the NVIDIA Container Toolkit:"
    echo "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    
    read -p "Do you want to continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Deployment aborted."
        exit 1
    fi
fi

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}NVIDIA GPUs detected:${NC}"
    nvidia-smi -L
    
    # Check for T4 GPU
    if nvidia-smi -L | grep -q "T4"; then
        echo -e "${GREEN}T4 GPU detected - T4 optimizations will be enabled by default.${NC}"
    else
        echo -e "${YELLOW}No T4 GPU detected - using standard configurations.${NC}"
        echo "Performance optimizations may not be optimal for your GPU model."
    fi
else
    echo -e "${YELLOW}Warning: No NVIDIA GPU detected.${NC}"
    echo "This deployment is optimized for NVIDIA GPUs, especially T4."
    
    read -p "Do you want to continue anyway? (CPU mode will be used) (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Deployment aborted."
        exit 1
    fi
fi

# Ensure output directory exists
mkdir -p mctx_output
echo -e "${GREEN}Created output directory: mctx_output${NC}"

# Check if config files exist
if [ ! -d "config/prometheus" ]; then
    echo -e "${RED}Error: Missing configuration files.${NC}"
    echo "Please ensure the repository is up to date."
    exit 1
fi

# Build and start the containers
echo -e "${GREEN}Building and starting MCTX containers...${NC}"
echo "This may take several minutes for the first build."

# Check which deployment option to use
echo "Which deployment option would you like to use?"
echo "1. Basic GPU setup (Simplified T4-optimized environment)"
echo "2. Production GPU setup (Enterprise-grade with Redis caching)"
echo "3. Blue-Green deployment (Zero-downtime deployment with dual environments)"

read -p "Enter option (1-3): " -n 1 -r DEPLOY_OPTION
echo

case $DEPLOY_OPTION in
    1)
        echo -e "${GREEN}Deploying Basic GPU setup...${NC}"
        docker-compose -f config/docker/gpu-basic.yml up -d
        ;;
    2)
        echo -e "${GREEN}Deploying Production GPU setup...${NC}"
        docker-compose -f config/docker/gpu-production.yml up -d
        ;;
    3)
        echo -e "${GREEN}Deploying Blue-Green environment...${NC}"
        docker-compose -f config/docker/blue-green.yml up -d
        ;;
    *)
        echo -e "${RED}Invalid option. Deploying Basic GPU setup...${NC}"
        docker-compose -f config/docker/gpu-basic.yml up -d
        ;;
esac

# Check if deployment was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}MCTX deployment completed successfully!${NC}"
    echo
    
    if [ "$DEPLOY_OPTION" == "1" ]; then
        echo "Services available at:"
        echo "- MCTX API: http://localhost:8000"
        echo "- Visualization Dashboard: http://localhost:8050"
        echo "- Grafana: http://localhost:3001 (admin/admin)"
        echo "- Prometheus: http://localhost:9090"
    elif [ "$DEPLOY_OPTION" == "2" ]; then
        echo "Services available at:"
        echo "- MCTX API: http://localhost:8000"
        echo "- Visualization Dashboard: http://localhost:8050"
        echo "- Grafana: http://localhost:3000 (admin/admin)"
        echo "- Prometheus: http://localhost:9090"
        echo "- Redis: localhost:6379"
    else
        echo "Services available at:"
        echo "- MCTX API (Blue): http://localhost:80 (routed through NGINX)"
        echo "- MCTX API (Green): http://localhost:80 (inactive until switched)"
        echo "- Visualization (Blue): http://localhost:8051"
        echo "- Visualization (Green): http://localhost:8052"
        echo "- Grafana: http://localhost:3001 (admin/admin)"
        echo "- Prometheus: http://localhost:9090"
        
        echo
        echo "To switch between Blue and Green deployments:"
        echo "  ./config/nginx/scripts/switch-deployment.sh"
    fi
    
    echo
    echo "To view logs:"
    echo "  docker-compose -f config/docker/<deployment-file>.yml logs -f"
    echo
    echo "To stop the deployment:"
    echo "  docker-compose -f config/docker/<deployment-file>.yml down"
else
    echo -e "${RED}Deployment failed.${NC}"
    echo "Please check the error messages above and try again."
    exit 1
fi