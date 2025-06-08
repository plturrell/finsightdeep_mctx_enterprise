#!/bin/bash

# Script to start and manage the MCTX services on NVIDIA LaunchPad

# Default action is to start
ACTION=${1:-start}

COMPOSE_FILE="docker-compose.nvidia-blueprint.yml"
ENV_FILE=".env.nvidia"

case $ACTION in
  start)
    echo "Starting MCTX NVIDIA LaunchPad services..."
    docker-compose -f $COMPOSE_FILE --env-file $ENV_FILE up -d
    echo "Services started. Use './run-nvidia-blueprint.sh status' to check status."
    echo
    echo "Service endpoints:"
    echo "- API: http://localhost:8000"
    echo "- API Documentation: http://localhost:8000/api/docs"
    echo "- Visualization Dashboard: http://localhost:8050"
    echo "- Performance Monitoring: http://localhost:8051"
    echo "- Documentation Server: http://localhost:8080"
    echo "- Prometheus: http://localhost:9090"
    echo "- Grafana: http://localhost:3001 (admin/admin)"
    ;;
  
  stop)
    echo "Stopping MCTX NVIDIA LaunchPad services..."
    docker-compose -f $COMPOSE_FILE down
    ;;
  
  restart)
    echo "Restarting MCTX NVIDIA LaunchPad services..."
    docker-compose -f $COMPOSE_FILE --env-file $ENV_FILE down
    docker-compose -f $COMPOSE_FILE --env-file $ENV_FILE up -d
    ;;
  
  status)
    echo "MCTX NVIDIA LaunchPad services status:"
    docker-compose -f $COMPOSE_FILE ps
    ;;
  
  logs)
    # Get the service name if provided, otherwise show all logs
    SERVICE=${2:-""}
    echo "Showing logs for MCTX NVIDIA LaunchPad services ${SERVICE}..."
    docker-compose -f $COMPOSE_FILE logs --tail=100 -f $SERVICE
    ;;
  
  gpu)
    echo "Checking GPU status..."
    # Run nvidia-smi inside the container to check GPU status
    docker exec -it mctx-nvidia-api nvidia-smi
    ;;
  
  *)
    echo "Usage: $0 {start|stop|restart|status|logs [service]|gpu}"
    exit 1
    ;;
esac

exit 0