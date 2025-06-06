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

# Build and start the containers
echo -e "${GREEN}Building and starting MCTX containers...${NC}"
echo "This may take several minutes for the first build."

# Check which services to deploy
echo "Which services would you like to deploy?"
echo "1. MCTX NVIDIA only (main service with T4 optimizations)"
echo "2. MCTX NVIDIA + Visualization server"
echo "3. Full stack (MCTX NVIDIA + Visualization + Monitoring)"

read -p "Enter option (1-3): " -n 1 -r DEPLOY_OPTION
echo

case $DEPLOY_OPTION in
    1)
        echo -e "${GREEN}Deploying MCTX NVIDIA service only...${NC}"
        docker-compose -f docker-compose.nvidia.yml up -d mctx-nvidia
        ;;
    2)
        echo -e "${GREEN}Deploying MCTX NVIDIA + Visualization services...${NC}"
        docker-compose -f docker-compose.nvidia.yml up -d mctx-nvidia mctx-vis
        ;;
    3)
        echo -e "${GREEN}Deploying full stack with monitoring...${NC}"
        docker-compose -f docker-compose.nvidia.yml up -d
        ;;
    *)
        echo -e "${RED}Invalid option. Deploying MCTX NVIDIA service only...${NC}"
        docker-compose -f docker-compose.nvidia.yml up -d mctx-nvidia
        ;;
esac

# Check if deployment was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}MCTX deployment completed successfully!${NC}"
    echo
    echo "Services available at:"
    echo "- MCTX API: http://localhost:8000"
    echo "- Visualization Dashboard: http://localhost:8050"
    
    if [ "$DEPLOY_OPTION" == "3" ]; then
        echo "- Prometheus: http://localhost:9090"
        echo "- Grafana: http://localhost:3001 (admin/admin)"
    fi
    
    echo
    echo "To view logs:"
    echo "  docker-compose -f docker-compose.nvidia.yml logs -f"
    echo
    echo "To stop the deployment:"
    echo "  docker-compose -f docker-compose.nvidia.yml down"
else
    echo -e "${RED}Deployment failed.${NC}"
    echo "Please check the error messages above and try again."
    exit 1
fi