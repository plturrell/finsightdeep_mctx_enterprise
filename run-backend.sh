#!/bin/bash

# Script to start and manage the MCTX backend services

# Default action is to start
ACTION=${1:-start}

COMPOSE_FILE="docker-compose.backend.yml"
ENV_FILE=".env.backend"

case $ACTION in
  start)
    echo "Starting MCTX backend services..."
    docker-compose -f $COMPOSE_FILE --env-file $ENV_FILE up -d
    echo "Services started. Use './run-backend.sh status' to check status."
    echo
    echo "Service endpoints:"
    echo "- API: http://localhost:8000"
    echo "- API Documentation: http://localhost:8000/api/docs"
    echo "- Documentation Server: http://localhost:8080"
    echo "- Prometheus: http://localhost:9090"
    echo "- Grafana: http://localhost:3001 (admin/admin)"
    ;;
  
  stop)
    echo "Stopping MCTX backend services..."
    docker-compose -f $COMPOSE_FILE down
    ;;
  
  restart)
    echo "Restarting MCTX backend services..."
    docker-compose -f $COMPOSE_FILE --env-file $ENV_FILE down
    docker-compose -f $COMPOSE_FILE --env-file $ENV_FILE up -d
    ;;
  
  status)
    echo "MCTX backend services status:"
    docker-compose -f $COMPOSE_FILE ps
    ;;
  
  logs)
    # Get the service name if provided, otherwise show all logs
    SERVICE=${2:-""}
    echo "Showing logs for MCTX backend services ${SERVICE}..."
    docker-compose -f $COMPOSE_FILE logs --tail=100 -f $SERVICE
    ;;
  
  test)
    echo "Running backend deployment tests..."
    bash test-backend-deployment.sh
    ;;
  
  *)
    echo "Usage: $0 {start|stop|restart|status|logs [service]|test}"
    exit 1
    ;;
esac

exit 0