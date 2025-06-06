# MCTX Docker Deployment

This directory contains Docker configurations for deploying MCTX in various environments, with a focus on NVIDIA GPU acceleration and cloud deployment options.

## Deployment Options

MCTX provides multiple deployment options to suit different environments:

1. **NVIDIA GPU Optimized**: For high-performance environments with T4 GPUs
2. **Vercel-compatible API**: For serverless API deployment
3. **Visualization Server**: For interactive exploration of MCTS trees
4. **Blue-Green Deployment**: For zero-downtime production environments

## Quick Start

Use the provided build script to easily build and run the containers:

```bash
# Build and run the NVIDIA GPU optimized container
./docker-build.sh nvidia

# Build and run the API container
./docker-build.sh api

# Build and run the visualization server
./docker-build.sh vis

# Build all containers
./docker-build.sh all
```

Alternatively, use Docker Compose directly:

```bash
# Build the containers
docker-compose build

# Run the NVIDIA container
docker-compose up mctx-nvidia

# Run the API container
docker-compose up mctx-api

# Run the visualization server
docker-compose up mctx-vis
```

## Container Details

### NVIDIA GPU Optimized (`Dockerfile.nvidia`)

Designed for high-performance MCTS with T4 GPU acceleration:

- Based on NVIDIA's TensorFlow container with CUDA support
- Includes T4 optimizations for tensor cores and memory layout
- Provides visualization and monitoring capabilities
- Exposes ports for both API (8000) and visualization (8050)

```bash
# Run with GPU support
docker run --gpus all -p 8000:8000 -p 8050:8050 mctx-nvidia
```

### Vercel-compatible API (`Dockerfile.vercel`)

Designed for serverless API deployment:

- Lightweight container optimized for CPU inference
- Compatible with Vercel and similar serverless platforms
- Provides minimal MCTS API with JSON I/O
- Exposes port 3000 by default

```bash
# Run the API server
docker run -p 3000:3000 mctx-api
```

### Visualization Server

Dedicated server for visualizing MCTS trees:

- Based on the NVIDIA container but can run on CPU
- Provides interactive visualization of trees
- Includes metrics panels and analysis tools
- Exposes visualization port (8050)

```bash
# Run the visualization server
docker run -p 8050:8050 mctx-nvidia visualize
```

## Environment Variables

Configure the containers with these environment variables:

- `JAX_PLATFORM_NAME`: Set to `gpu` or `cpu` to control JAX's hardware target
- `MCTX_ENABLE_T4_OPTIMIZATIONS`: Set to `1` to enable T4-specific optimizations
- `XLA_PYTHON_CLIENT_ALLOCATOR`: Memory allocator for JAX (use `platform` for GPU)
- `PORT`: Port for the API server (default: 3000 for Vercel, 8000 for others)
- `LOG_LEVEL`: Logging verbosity (default: INFO)
- `MAX_BATCH_SIZE`: Maximum batch size for search requests
- `MAX_NUM_SIMULATIONS`: Maximum number of simulations for search requests

## Vercel Deployment

To deploy to Vercel:

1. Copy the Vercel configuration:
   ```bash
   cp docker/vercel.json .
   ```

2. Create a `vercel.json` file in your project root with Vercel-specific configurations

3. Deploy to Vercel:
   ```bash
   vercel
   ```

## SAP HANA Integration

The NVIDIA container includes SAP HANA integration capabilities. To use them:

1. Configure HANA connection with environment variables:
   ```bash
   docker run -e HANA_HOST=your-hana-host -e HANA_PORT=39015 -e HANA_USER=your-user -e HANA_PASSWORD=your-password mctx-nvidia
   ```

2. Use the HANA integration module:
   ```python
   from mctx.enterprise.hana_integration import HanaConfig, HanaConnection
   
   config = HanaConfig(
       host=os.environ["HANA_HOST"],
       port=int(os.environ["HANA_PORT"]),
       user=os.environ["HANA_USER"],
       password=os.environ["HANA_PASSWORD"]
   )
   
   connection = HanaConnection(config)
   ```

## Blue-Green Deployment

For production environments, MCTX provides a blue-green deployment configuration for zero-downtime updates:

```bash
# Start blue-green deployment
docker-compose -f docker/blue-green-deployment.yml up -d
```

This configuration includes:

- **Dual Environments**: Blue and green environments running in parallel
- **NGINX Router**: Intelligent routing between blue and green deployments
- **Zero-Downtime Updates**: Deploy updates without service interruption
- **Gradual Rollouts**: Test new versions before full deployment
- **Instant Rollbacks**: Switch back to previous version if issues arise

### Switching Between Environments

To switch the active deployment:

```bash
# Switch to green deployment
curl -X POST -u admin:password https://mctx.example.com/deployment/switch?color=green

# Switch to blue deployment
curl -X POST -u admin:password https://mctx.example.com/deployment/switch?color=blue
```

### Update Process

1. Deploy new version to inactive environment (e.g., green if blue is active)
2. Run tests against the new deployment
3. Switch router to new environment
4. Verify new deployment is working correctly
5. Update the previously active environment when convenient

For more examples, see [docker_examples.md](../enterprise/docs/technical/docker_examples.md).

## Resource Requirements

- **NVIDIA GPU container**:
  - NVIDIA GPU with CUDA 11.x support (T4 recommended)
  - 8GB+ GPU memory
  - 16GB+ system RAM
  - 10GB+ disk space

- **API container**:
  - 4GB+ system RAM
  - 5GB+ disk space

- **Visualization server**:
  - 4GB+ system RAM
  - 5GB+ disk space