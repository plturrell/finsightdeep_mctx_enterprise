#!/bin/bash
# Build and run MCTX Docker containers

# Set error handling
set -e

# Terminal colors
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print colored message
print_msg() {
  echo -e "${CYAN}[MCTX Docker]${NC} $1"
}

# Print error message
print_error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

# Print help message
print_help() {
  echo -e "${CYAN}MCTX Docker Build Script${NC}"
  echo ""
  echo "Usage: $0 [COMMAND]"
  echo ""
  echo "Commands:"
  echo "  nvidia    Build and run the NVIDIA GPU optimized container"
  echo "  api       Build and run the API container (Vercel-compatible)"
  echo "  vis       Build and run the visualization server"
  echo "  all       Build all containers"
  echo "  help      Show this help message"
  echo ""
  echo "Examples:"
  echo "  $0 nvidia     # Build and run NVIDIA container"
  echo "  $0 api        # Build and run API container"
  echo "  $0 all        # Build all containers"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
  print_error "Docker is not installed. Please install Docker first."
  exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
  print_error "Docker Compose is not installed. Please install Docker Compose first."
  exit 1
fi

# Build NVIDIA container
build_nvidia() {
  print_msg "Building MCTX NVIDIA container..."
  docker-compose build mctx-nvidia
  
  # Check if build was successful
  if [ $? -eq 0 ]; then
    print_msg "${GREEN}MCTX NVIDIA container built successfully!${NC}"
    
    # Check if NVIDIA drivers are installed
    if command -v nvidia-smi &> /dev/null; then
      print_msg "NVIDIA drivers detected."
    else
      print_msg "${YELLOW}NVIDIA drivers not detected. Make sure they are installed for GPU support.${NC}"
    fi
    
    # Check if user wants to run the container
    read -p "Do you want to run the NVIDIA container now? (y/n): " run_container
    if [[ "$run_container" =~ ^[Yy]$ ]]; then
      docker-compose up mctx-nvidia
    else
      print_msg "To run the container later, use: docker-compose up mctx-nvidia"
    fi
  else
    print_error "Failed to build MCTX NVIDIA container."
  fi
}

# Build API container
build_api() {
  print_msg "Building MCTX API container (Vercel-compatible)..."
  docker-compose build mctx-api
  
  # Check if build was successful
  if [ $? -eq 0 ]; then
    print_msg "${GREEN}MCTX API container built successfully!${NC}"
    
    # Check if user wants to run the container
    read -p "Do you want to run the API container now? (y/n): " run_container
    if [[ "$run_container" =~ ^[Yy]$ ]]; then
      docker-compose up mctx-api
    else
      print_msg "To run the container later, use: docker-compose up mctx-api"
    fi
  else
    print_error "Failed to build MCTX API container."
  fi
}

# Build visualization server
build_vis() {
  print_msg "Building MCTX visualization server..."
  docker-compose build mctx-vis
  
  # Check if build was successful
  if [ $? -eq 0 ]; then
    print_msg "${GREEN}MCTX visualization server built successfully!${NC}"
    
    # Check if user wants to run the container
    read -p "Do you want to run the visualization server now? (y/n): " run_container
    if [[ "$run_container" =~ ^[Yy]$ ]]; then
      docker-compose up mctx-vis
    else
      print_msg "To run the container later, use: docker-compose up mctx-vis"
    fi
  else
    print_error "Failed to build MCTX visualization server."
  fi
}

# Build all containers
build_all() {
  print_msg "Building all MCTX containers..."
  docker-compose build
  
  # Check if build was successful
  if [ $? -eq 0 ]; then
    print_msg "${GREEN}All MCTX containers built successfully!${NC}"
    
    # Check if user wants to run all containers
    read -p "Do you want to run all containers now? (y/n): " run_containers
    if [[ "$run_containers" =~ ^[Yy]$ ]]; then
      docker-compose up
    else
      print_msg "To run the containers later, use: docker-compose up"
    fi
  else
    print_error "Failed to build all MCTX containers."
  fi
}

# Parse command
case "$1" in
  nvidia)
    build_nvidia
    ;;
  api)
    build_api
    ;;
  vis)
    build_vis
    ;;
  all)
    build_all
    ;;
  help|--help|-h)
    print_help
    ;;
  *)
    # Default to help if no command specified
    if [ -z "$1" ]; then
      print_help
    else
      print_error "Unknown command: $1"
      print_help
    fi
    ;;
esac