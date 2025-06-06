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

# Check which deployment to test
echo "Which deployment would you like to test?"
echo "1. Development setup (CPU-only)"
echo "2. Basic GPU setup (T4-optimized)"
echo "3. Production GPU setup (Enterprise)"
echo "4. Blue-Green deployment"
echo "5. Cloud deployment"

read -p "Enter option (1-5): " -n 1 -r DEPLOY_OPTION
echo

case $DEPLOY_OPTION in
    1)
        echo -e "${GREEN}Testing Development setup...${NC}"
        COMPOSE_FILE="config/docker/development.yml"
        docker-compose -f $COMPOSE_FILE up -d api prometheus
        
        # Check services
        check_service "API" "mctx_api_1"
        API_RUNNING=$?
        
        check_service "Prometheus" "mctx_prometheus_1"
        PROMETHEUS_RUNNING=$?
        
        # Check ports
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
        ;;
        
    2)
        echo -e "${GREEN}Testing Basic GPU setup...${NC}"
        COMPOSE_FILE="config/docker/gpu-basic.yml"
        docker-compose -f $COMPOSE_FILE up -d mctx-nvidia prometheus
        
        # Check services
        check_service "MCTX NVIDIA" "mctx-nvidia"
        MCTX_RUNNING=$?
        
        check_service "Prometheus" "mctx_prometheus_1"
        PROMETHEUS_RUNNING=$?
        
        # Check ports
        if [ $MCTX_RUNNING -eq 0 ]; then
          check_port "MCTX API" 8000 15
          API_ACCESSIBLE=$?
          
          check_port "Visualization" 8050 5
          VIS_ACCESSIBLE=$?
        else
          API_ACCESSIBLE=1
          VIS_ACCESSIBLE=1
        fi
        
        if [ $PROMETHEUS_RUNNING -eq 0 ]; then
          check_port "Prometheus" 9090 5
          PROMETHEUS_ACCESSIBLE=$?
        else
          PROMETHEUS_ACCESSIBLE=1
        fi
        ;;
        
    3)
        echo -e "${GREEN}Testing Production GPU setup...${NC}"
        COMPOSE_FILE="config/docker/gpu-production.yml"
        docker-compose -f $COMPOSE_FILE up -d mctx-api redis prometheus
        
        # Check services
        check_service "MCTX API" "mctx-api"
        API_RUNNING=$?
        
        check_service "Redis" "mctx-redis"
        REDIS_RUNNING=$?
        
        check_service "Prometheus" "mctx-prometheus"
        PROMETHEUS_RUNNING=$?
        
        # Check ports
        if [ $API_RUNNING -eq 0 ]; then
          check_port "MCTX API" 8000 15
          API_ACCESSIBLE=$?
        else
          API_ACCESSIBLE=1
        fi
        
        if [ $REDIS_RUNNING -eq 0 ]; then
          check_port "Redis" 6379 5
          REDIS_ACCESSIBLE=$?
        else
          REDIS_ACCESSIBLE=1
        fi
        
        if [ $PROMETHEUS_RUNNING -eq 0 ]; then
          check_port "Prometheus" 9090 5
          PROMETHEUS_ACCESSIBLE=$?
        else
          PROMETHEUS_ACCESSIBLE=1
        fi
        ;;
        
    4)
        echo -e "${GREEN}Testing Blue-Green deployment...${NC}"
        COMPOSE_FILE="config/docker/blue-green.yml"
        docker-compose -f $COMPOSE_FILE up -d nginx-router mctx-blue prometheus
        
        # Check services
        check_service "NGINX Router" "nginx-router"
        NGINX_RUNNING=$?
        
        check_service "MCTX Blue" "mctx-blue"
        BLUE_RUNNING=$?
        
        check_service "Prometheus" "mctx_prometheus_1"
        PROMETHEUS_RUNNING=$?
        
        # Check ports
        if [ $NGINX_RUNNING -eq 0 ]; then
          check_port "NGINX" 80 5
          NGINX_ACCESSIBLE=$?
        else
          NGINX_ACCESSIBLE=1
        fi
        
        if [ $BLUE_RUNNING -eq 0 ]; then
          # Blue is behind nginx, so we don't check direct access
          BLUE_ACCESSIBLE=0
        else
          BLUE_ACCESSIBLE=1
        fi
        
        if [ $PROMETHEUS_RUNNING -eq 0 ]; then
          check_port "Prometheus" 9090 5
          PROMETHEUS_ACCESSIBLE=$?
        else
          PROMETHEUS_ACCESSIBLE=1
        fi
        ;;
        
    5)
        echo -e "${GREEN}Testing Cloud deployment...${NC}"
        COMPOSE_FILE="config/docker/cloud.yml"
        docker-compose -f $COMPOSE_FILE up -d api redis
        
        # Check services
        check_service "API" "mctx-api"
        API_RUNNING=$?
        
        check_service "Redis" "mctx-cache"
        REDIS_RUNNING=$?
        
        # Check ports
        if [ $API_RUNNING -eq 0 ]; then
          check_port "API" 3000 10
          API_ACCESSIBLE=$?
        else
          API_ACCESSIBLE=1
        fi
        
        if [ $REDIS_RUNNING -eq 0 ]; then
          check_port "Redis" 6379 5
          REDIS_ACCESSIBLE=$?
        else
          REDIS_ACCESSIBLE=1
        fi
        ;;
        
    *)
        echo -e "${RED}Invalid option. Testing Development setup...${NC}"
        COMPOSE_FILE="config/docker/development.yml"
        docker-compose -f $COMPOSE_FILE up -d api prometheus
        
        # Check services
        check_service "API" "mctx_api_1"
        API_RUNNING=$?
        
        check_service "Prometheus" "mctx_prometheus_1"
        PROMETHEUS_RUNNING=$?
        
        # Check ports
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
        ;;
esac

# Stop the containers
echo -e "${YELLOW}Stopping containers...${NC}"
docker-compose -f $COMPOSE_FILE down

# Print test summary
echo -e "${GREEN}=== Test Summary ===${NC}"

case $DEPLOY_OPTION in
    1)
        echo "API Service: $([ $API_RUNNING -eq 0 ] && [ $API_ACCESSIBLE -eq 0 ] && echo -e "${GREEN}PASS${NC}" || echo -e "${RED}FAIL${NC}")"
        echo "Prometheus: $([ $PROMETHEUS_RUNNING -eq 0 ] && [ $PROMETHEUS_ACCESSIBLE -eq 0 ] && echo -e "${GREEN}PASS${NC}" || echo -e "${RED}FAIL${NC}")"
        
        if [ $API_RUNNING -eq 0 ] && [ $API_ACCESSIBLE -eq 0 ] && [ $PROMETHEUS_RUNNING -eq 0 ] && [ $PROMETHEUS_ACCESSIBLE -eq 0 ]; then
          TEST_PASSED=0
        else
          TEST_PASSED=1
        fi
        ;;
        
    2)
        echo "MCTX NVIDIA: $([ $MCTX_RUNNING -eq 0 ] && [ $API_ACCESSIBLE -eq 0 ] && [ $VIS_ACCESSIBLE -eq 0 ] && echo -e "${GREEN}PASS${NC}" || echo -e "${RED}FAIL${NC}")"
        echo "Prometheus: $([ $PROMETHEUS_RUNNING -eq 0 ] && [ $PROMETHEUS_ACCESSIBLE -eq 0 ] && echo -e "${GREEN}PASS${NC}" || echo -e "${RED}FAIL${NC}")"
        
        if [ $MCTX_RUNNING -eq 0 ] && [ $API_ACCESSIBLE -eq 0 ] && [ $VIS_ACCESSIBLE -eq 0 ] && [ $PROMETHEUS_RUNNING -eq 0 ] && [ $PROMETHEUS_ACCESSIBLE -eq 0 ]; then
          TEST_PASSED=0
        else
          TEST_PASSED=1
        fi
        ;;
        
    3)
        echo "MCTX API: $([ $API_RUNNING -eq 0 ] && [ $API_ACCESSIBLE -eq 0 ] && echo -e "${GREEN}PASS${NC}" || echo -e "${RED}FAIL${NC}")"
        echo "Redis: $([ $REDIS_RUNNING -eq 0 ] && [ $REDIS_ACCESSIBLE -eq 0 ] && echo -e "${GREEN}PASS${NC}" || echo -e "${RED}FAIL${NC}")"
        echo "Prometheus: $([ $PROMETHEUS_RUNNING -eq 0 ] && [ $PROMETHEUS_ACCESSIBLE -eq 0 ] && echo -e "${GREEN}PASS${NC}" || echo -e "${RED}FAIL${NC}")"
        
        if [ $API_RUNNING -eq 0 ] && [ $API_ACCESSIBLE -eq 0 ] && [ $REDIS_RUNNING -eq 0 ] && [ $REDIS_ACCESSIBLE -eq 0 ] && [ $PROMETHEUS_RUNNING -eq 0 ] && [ $PROMETHEUS_ACCESSIBLE -eq 0 ]; then
          TEST_PASSED=0
        else
          TEST_PASSED=1
        fi
        ;;
        
    4)
        echo "NGINX Router: $([ $NGINX_RUNNING -eq 0 ] && [ $NGINX_ACCESSIBLE -eq 0 ] && echo -e "${GREEN}PASS${NC}" || echo -e "${RED}FAIL${NC}")"
        echo "MCTX Blue: $([ $BLUE_RUNNING -eq 0 ] && [ $BLUE_ACCESSIBLE -eq 0 ] && echo -e "${GREEN}PASS${NC}" || echo -e "${RED}FAIL${NC}")"
        echo "Prometheus: $([ $PROMETHEUS_RUNNING -eq 0 ] && [ $PROMETHEUS_ACCESSIBLE -eq 0 ] && echo -e "${GREEN}PASS${NC}" || echo -e "${RED}FAIL${NC}")"
        
        if [ $NGINX_RUNNING -eq 0 ] && [ $NGINX_ACCESSIBLE -eq 0 ] && [ $BLUE_RUNNING -eq 0 ] && [ $BLUE_ACCESSIBLE -eq 0 ] && [ $PROMETHEUS_RUNNING -eq 0 ] && [ $PROMETHEUS_ACCESSIBLE -eq 0 ]; then
          TEST_PASSED=0
        else
          TEST_PASSED=1
        fi
        ;;
        
    5)
        echo "API: $([ $API_RUNNING -eq 0 ] && [ $API_ACCESSIBLE -eq 0 ] && echo -e "${GREEN}PASS${NC}" || echo -e "${RED}FAIL${NC}")"
        echo "Redis: $([ $REDIS_RUNNING -eq 0 ] && [ $REDIS_ACCESSIBLE -eq 0 ] && echo -e "${GREEN}PASS${NC}" || echo -e "${RED}FAIL${NC}")"
        
        if [ $API_RUNNING -eq 0 ] && [ $API_ACCESSIBLE -eq 0 ] && [ $REDIS_RUNNING -eq 0 ] && [ $REDIS_ACCESSIBLE -eq 0 ]; then
          TEST_PASSED=0
        else
          TEST_PASSED=1
        fi
        ;;
        
    *)
        echo "API Service: $([ $API_RUNNING -eq 0 ] && [ $API_ACCESSIBLE -eq 0 ] && echo -e "${GREEN}PASS${NC}" || echo -e "${RED}FAIL${NC}")"
        echo "Prometheus: $([ $PROMETHEUS_RUNNING -eq 0 ] && [ $PROMETHEUS_ACCESSIBLE -eq 0 ] && echo -e "${GREEN}PASS${NC}" || echo -e "${RED}FAIL${NC}")"
        
        if [ $API_RUNNING -eq 0 ] && [ $API_ACCESSIBLE -eq 0 ] && [ $PROMETHEUS_RUNNING -eq 0 ] && [ $PROMETHEUS_ACCESSIBLE -eq 0 ]; then
          TEST_PASSED=0
        else
          TEST_PASSED=1
        fi
        ;;
esac

if [ $TEST_PASSED -eq 0 ]; then
  echo -e "${GREEN}All tests passed! Docker deployment is working correctly.${NC}"
  echo
  echo "You can now use the full deployment with:"
  echo "  ./scripts/deploy-nvidia.sh"
  exit 0
else
  echo -e "${RED}Some tests failed. Please check the Docker configuration.${NC}"
  exit 1
fi