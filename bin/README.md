# MCTX Executable Scripts

This directory contains executable scripts and symlinks for common operations:

## Available Scripts

- `deploy-nvidia.sh` - Interactive script for NVIDIA GPU deployment
- `test-docker-deployment.sh` - Test script for validating Docker deployments
- `remove_nonproduction_files.sh` - Utility for removing non-production files before release

## Usage

All scripts can be run from the project root:

```bash
# Deploy with NVIDIA GPU support
./bin/deploy-nvidia.sh

# Test Docker deployment
./bin/test-docker-deployment.sh
```

For more detailed documentation, see:
- Docker deployment: [Docker Documentation](../docs/docker/OVERVIEW.md)
- Development guide: [Contributing Guide](../CONTRIBUTING.md)