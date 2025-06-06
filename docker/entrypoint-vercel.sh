#!/bin/bash
set -e

# Log information about the environment
echo "=== MCTX Vercel Container Information ==="
echo "Container started at: $(date)"
echo "Python version: $(python --version)"
echo "JAX version: $(python -c 'import jax; print(jax.__version__)')"

# Set JAX to use CPU for Vercel deployment
export JAX_PLATFORM_NAME="cpu"

# Log JAX devices
echo "JAX devices:"
python -c 'import jax; print(jax.devices())'

# Check if we're starting the API server with a custom port
if [[ "$1" == "api" ]]; then
    PORT="${2:-3000}"
    echo "Starting API server on port $PORT..."
    exec uvicorn api.main:app --host 0.0.0.0 --port $PORT "${@:3}"
fi

# Check if we're running a custom command
if [[ "$1" ]]; then
    echo "Running command: $@"
    exec "$@"
fi

# If no command specified, run the default command (API server)
PORT="${PORT:-3000}"
echo "Running API server on port $PORT..."
exec uvicorn api.main:app --host 0.0.0.0 --port $PORT