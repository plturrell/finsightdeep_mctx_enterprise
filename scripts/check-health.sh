#!/bin/bash
# Health check script for MCTX services
# This script checks the health of all deployed MCTX services

set -e

# ANSI color codes for better formatting
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color
BOLD='\033[1m'

echo -e "${BOLD}MCTX T4 GPU Services Health Check${NC}"
echo "Running health checks at $(date)"
echo "----------------------------------------"

# Function to check service health
check_service() {
    local service=$1
    local url=$2
    local expected_status=$3
    
    echo -n "Checking $service... "
    
    # Use curl to check the service
    status_code=$(curl -s -o /dev/null -w "%{http_code}" $url)
    
    if [ "$status_code" == "$expected_status" ]; then
        echo -e "${GREEN}OK${NC} (HTTP $status_code)"
        return 0
    else
        echo -e "${RED}FAILED${NC} (HTTP $status_code, expected $expected_status)"
        return 1
    fi
}

# Check if we're running in Docker Compose environment
if [ -f ".env" ]; then
    source .env
fi

# Set default host if not provided
HOST=${HOST:-localhost}

# Count success and failures
success=0
failure=0

# Check main MCTX service
if check_service "MCTX API" "http://${HOST}:8000/health" "200"; then
    ((success++))
else
    ((failure++))
    echo -e "${YELLOW}  → Try: docker-compose -f docker-compose.nvidia.yml logs mctx-nvidia${NC}"
fi

# Check visualization service
if check_service "MCTX Visualization" "http://${HOST}:8050" "200"; then
    ((success++))
else
    ((failure++))
    echo -e "${YELLOW}  → Try: docker-compose -f docker-compose.nvidia.yml logs mctx-vis${NC}"
fi

# Check secondary visualization service
if check_service "MCTX Secondary Visualization" "http://${HOST}:8051" "200"; then
    ((success++))
else
    ((failure++))
    echo -e "${YELLOW}  → Try: docker-compose -f docker-compose.nvidia.yml logs mctx-vis${NC}"
fi

# Check Prometheus
if check_service "Prometheus" "http://${HOST}:9090/-/healthy" "200"; then
    ((success++))
else
    ((failure++))
    echo -e "${YELLOW}  → Try: docker-compose -f docker-compose.nvidia.yml logs prometheus${NC}"
fi

# Check Grafana
if check_service "Grafana" "http://${HOST}:3001/api/health" "200"; then
    ((success++))
else
    ((failure++))
    echo -e "${YELLOW}  → Try: docker-compose -f docker-compose.nvidia.yml logs grafana${NC}"
fi

# Check documentation server
if check_service "Documentation Server" "http://${HOST}:8080" "200"; then
    ((success++))
else
    ((failure++))
    echo -e "${YELLOW}  → Try: docker-compose -f docker-compose.nvidia.yml logs docs-server${NC}"
fi

# Print summary
echo "----------------------------------------"
echo -e "Health Check Summary: ${BOLD}$success services healthy${NC}, ${BOLD}$failure services unhealthy${NC}"

if [ $failure -gt 0 ]; then
    echo -e "\n${YELLOW}Troubleshooting Tips:${NC}"
    echo "1. Check service logs: docker-compose -f docker-compose.nvidia.yml logs [service-name]"
    echo "2. Verify GPU is accessible: docker exec mctx-nvidia nvidia-smi"
    echo "3. Restart specific service: docker-compose -f docker-compose.nvidia.yml restart [service-name]"
    echo "4. Check NVIDIA driver: nvidia-smi"
    echo "5. For more details: docker exec mctx-nvidia cat /app/logs/mctx.log"
    exit 1
else
    echo -e "\n${GREEN}All services are healthy!${NC}"
    exit 0
fi