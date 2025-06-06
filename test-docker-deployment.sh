#!/bin/bash
set -e

# MCTX Docker Deployment Test Script
# Tests the Docker deployment without using actual GPUs

# Color definitions
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== MCTX Docker Deployment Test ===${NC}"
echo "Starting test at $(date)"

# Function to check if a service is running
check_service() {
  local service=$1
  local container_name=$2
  
  echo -e "${YELLOW}Testing $service service...${NC}"
  
  if docker ps | grep -q "$container_name"; then
    echo -e "${GREEN}✓ $service container is running${NC}"
    return 0
  else
    echo -e "${RED}✗ $service container is not running${NC}"
    return 1
  fi
}

# Function to check if a port is accessible
check_port() {
  local service=$1
  local port=$2
  local wait_time=${3:-5}
  
  echo -e "${YELLOW}Waiting $wait_time seconds for $service to start...${NC}"
  sleep $wait_time
  
  echo -e "${YELLOW}Testing $service on port $port...${NC}"
  
  if nc -z localhost $port; then
    echo -e "${GREEN}✓ $service is accessible on port $port${NC}"
    return 0
  else
    echo -e "${RED}✗ $service is not accessible on port $port${NC}"
    return 1
  fi
}

# Test CPU-only deployment (safer for testing)
echo -e "${GREEN}Testing basic Docker deployment...${NC}"
echo "Starting containers in CPU mode..."

# Start the containers in detached mode using the simplified config
docker-compose up -d api prometheus

# Check if containers are running
check_service "API" "mctx_api_1"
API_RUNNING=$?

check_service "Prometheus" "mctx_prometheus_1"
PROMETHEUS_RUNNING=$?

# Check if services are accessible
if [ $API_RUNNING -eq 0 ]; then
  check_port "API" 8001 10
  API_ACCESSIBLE=$?
else
  API_ACCESSIBLE=1
fi

if [ $PROMETHEUS_RUNNING -eq 0 ]; then
  check_port "Prometheus" 9090 5
  PROMETHEUS_ACCESSIBLE=$?
else
  PROMETHEUS_ACCESSIBLE=1
fi

# Stop the containers
echo -e "${YELLOW}Stopping containers...${NC}"
docker-compose down

# Print test summary
echo -e "${GREEN}=== Test Summary ===${NC}"
echo "API Service: $([ $API_RUNNING -eq 0 ] && [ $API_ACCESSIBLE -eq 0 ] && echo -e "${GREEN}PASS${NC}" || echo -e "${RED}FAIL${NC}")"
echo "Prometheus: $([ $PROMETHEUS_RUNNING -eq 0 ] && [ $PROMETHEUS_ACCESSIBLE -eq 0 ] && echo -e "${GREEN}PASS${NC}" || echo -e "${RED}FAIL${NC}")"

if [ $API_RUNNING -eq 0 ] && [ $API_ACCESSIBLE -eq 0 ] && [ $PROMETHEUS_RUNNING -eq 0 ] && [ $PROMETHEUS_ACCESSIBLE -eq 0 ]; then
  echo -e "${GREEN}All tests passed! Docker deployment is working correctly.${NC}"
  echo
  echo "You can now use the full deployment with:"
  echo "  ./deploy-nvidia.sh"
  exit 0
else
  echo -e "${RED}Some tests failed. Please check the Docker configuration.${NC}"
  exit 1
fi