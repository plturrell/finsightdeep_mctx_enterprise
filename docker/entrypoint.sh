#!/bin/bash
set -e

# Log information about the environment
echo "=== MCTX Container Information ==="
echo "Container started at: $(date)"
echo "Python version: $(python --version)"
echo "JAX version: $(python -c 'import jax; print(jax.__version__)')"

# Set up documentation
if [ -f "/app/docker/setup-docs.sh" ]; then
    echo "Setting up documentation..."
    bash /app/docker/setup-docs.sh
    
    # Start a simple HTTP server to serve documentation on port 8080
    if command -v python3 &> /dev/null; then
        echo "Starting documentation server on port 8080..."
        python3 -m http.server 8080 --directory /docs &
    elif command -v python &> /dev/null; then
        echo "Starting documentation server on port 8080..."
        python -m http.server 8080 --directory /docs &
    else
        echo "Python not found, documentation server not started"
    fi
    
    echo "Documentation available at http://localhost:8080"
else
    echo "Documentation setup script not found"
fi

# Run the fix-readme script to make README accessible in many locations
if [ -f "/app/docker/fix-readme.sh" ]; then
    echo "Running README fix script..."
    bash /app/docker/fix-readme.sh
else
    echo "README fix script not found"
fi

# Check for NVIDIA GPU
echo "Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo "CUDA version: $(nvcc --version | grep release | cut -d',' -f2 | cut -d' ' -f3)"
    
    # Check for T4 GPU
    if nvidia-smi -L | grep -q "T4"; then
        echo "T4 GPU detected - T4 optimizations will be enabled by default"
        export MCTX_ENABLE_T4_OPTIMIZATIONS=1
    else
        echo "No T4 GPU detected - using standard configurations"
    fi
else
    echo "No NVIDIA GPU detected - running in CPU mode"
    export JAX_PLATFORM_NAME="cpu"
fi

# Log JAX devices
echo "JAX devices:"
python -c 'import jax; print(jax.devices())'

# Check if we're starting the API server
if [[ "$1" == "api" ]]; then
    echo "Starting API server..."
    exec uvicorn api.app.main:app --host 0.0.0.0 --port 8000 "${@:2}"
fi

# Check if we're starting the visualization server
if [[ "$1" == "visualize" ]]; then
    echo "Starting visualization server..."
    exec python -m mctx.monitoring.cli server --port 8050 "${@:2}"
fi

# Check if we're running a custom command
if [[ "$1" ]]; then
    echo "Running command: $@"
    exec "$@"
fi

# If no command specified, run the default command
echo "Running default monitoring demo..."
exec python examples/monitoring_demo.py --save-visualizations