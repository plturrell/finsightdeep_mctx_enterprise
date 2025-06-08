#!/bin/bash

# Script to deploy and test the MCTX backend services

set -e  # Exit on any error

echo "========================================"
echo "MCTX Backend Deployment and Test Script"
echo "========================================"
echo

# Check if docker is running
if ! docker info > /dev/null 2>&1; then
  echo "Error: Docker is not running. Please start Docker and try again."
  exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
  echo "Error: docker-compose is not installed. Please install it and try again."
  exit 1
fi

# Ensure data directory exists
mkdir -p data

# Function to check if a service is healthy
check_service_health() {
  local service=$1
  local max_attempts=$2
  local attempt=0
  local required=$3  # 1 for required, 0 for optional
  
  echo -n "Checking health of $service service: "
  
  while [ $attempt -lt $max_attempts ]; do
    if docker-compose -f docker-compose.backend.yml ps $service | grep "(healthy)" > /dev/null; then
      echo "HEALTHY"
      return 0
    fi
    
    attempt=$((attempt+1))
    echo -n "."
    sleep 5
  done
  
  echo "NOT HEALTHY"
  echo "Service $service is not healthy after $max_attempts attempts."
  docker-compose -f docker-compose.backend.yml logs $service --tail 20
  
  # If service is required, exit with error
  if [ "$required" -eq 1 ]; then
    return 1
  else
    echo "Continuing anyway as this service is optional..."
    return 0
  fi
}

# Start the services
echo "Starting backend services..."
docker-compose -f docker-compose.backend.yml --env-file .env.backend up -d

# Allow some time for services to start
sleep 10

# Check health of each service
echo
echo "Checking service health status..."
check_service_health api 6 1        # 30 seconds (required)
check_service_health redis 3 1      # 15 seconds (required)
# Prometheus and Grafana can take longer
check_service_health prometheus 6 0 # 30 seconds (optional)
check_service_health grafana 6 0    # 30 seconds (optional)

echo
echo "Testing API endpoints..."

# Test health endpoint
echo -n "Testing /health/ endpoint: "
if curl -s http://localhost:8000/health/ | grep -q "healthy"; then
  echo "OK"
else
  echo "FAILED"
  echo "Health endpoint response:"
  curl -v http://localhost:8000/health/
  exit 1
fi

# Test root endpoint
echo -n "Testing root (/) endpoint: "
if curl -s http://localhost:8000/ | grep -q "version"; then
  echo "OK"
else
  echo "FAILED"
  echo "Root endpoint response:"
  curl -v http://localhost:8000/
  exit 1
fi

# Test metrics endpoint
echo -n "Testing /metrics/ endpoint: "
if curl -s http://localhost:8000/metrics/ | grep -q "metrics"; then
  echo "OK"
else
  echo "FAILED"
  echo "Metrics endpoint response:"
  curl -v http://localhost:8000/metrics/
  exit 1
fi

# Test test endpoint
echo -n "Testing /test/ endpoint: "
if curl -s http://localhost:8000/test/ | grep -q "Test endpoint working"; then
  echo "OK"
else
  echo "FAILED"
  echo "Test endpoint response:"
  curl -v http://localhost:8000/test/
  exit 1
fi

# Test HANA connection (optional)
echo -n "Testing HANA connection: "
HANA_RESPONSE=$(curl -s http://localhost:8000/hana/test/)
if echo "$HANA_RESPONSE" | grep -q "connected"; then
  echo "OK - HANA connection successful"
else
  echo "WARNING - HANA connection failed"
  echo "HANA test endpoint response:"
  echo "$HANA_RESPONSE"
  echo "Continuing anyway as HANA might not be available in all environments..."
fi

# Check if docs are available (optional)
echo -n "Testing documentation server: "
if curl -s http://localhost:8080/ | grep -q "html"; then
  echo "OK"
else
  echo "WARNING - Documentation server not responding properly"
  echo "Documentation server response:"
  curl -v http://localhost:8080/
  echo "Continuing anyway as this service is optional..."
fi

# Check Prometheus (optional)
echo -n "Testing Prometheus: "
if curl -s http://localhost:9090/-/healthy | grep -q "Prometheus"; then
  echo "OK"
else
  echo "WARNING - Prometheus not responding properly"
  echo "Prometheus health check response:"
  curl -v http://localhost:9090/-/healthy
  echo "Continuing anyway as this service is optional..."
fi

# Check Grafana (optional)
echo -n "Testing Grafana: "
if curl -s http://localhost:3001/api/health | grep -q "ok"; then
  echo "OK"
else
  echo "WARNING - Grafana not responding properly"
  echo "Grafana health check response:"
  curl -v http://localhost:3001/api/health
  echo "Continuing anyway as this service is optional..."
fi

echo
echo "All backend services have been deployed and verified successfully!"
echo
echo "Service endpoints:"
echo "- API: http://localhost:8000"
echo "- API Documentation: http://localhost:8000/api/docs"
echo "- Documentation Server: http://localhost:8080"
echo "- Prometheus: http://localhost:9090"
echo "- Grafana: http://localhost:3001 (admin/admin)"
echo
echo "To stop the services, run:"
echo "docker-compose -f docker-compose.backend.yml down"

# Make the script executable
chmod +x test-backend-deployment.sh