#!/bin/bash
set -e

# Log information about the environment
echo "=== MCTX Container Information ==="
echo "Container started at: $(date)"
echo "Python version: $(python --version)"
echo "JAX version: $(python -c 'import jax; print(jax.__version__)')"

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