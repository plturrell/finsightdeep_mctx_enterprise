# MCTX Directory Structure

This document explains the overall directory structure of the MCTX project, highlighting the separation between the core library and enterprise features.

## Overview

The project is organized into two main components:

1. **Core MCTX Library**: The original JAX-native Monte Carlo Tree Search implementation
2. **Enterprise Extensions**: Production-ready features, optimizations, and deployment configurations

## Core Library Structure

```
mctx/                     # Project root
├── mctx/                 # Core library package
│   ├── __init__.py       # Package initialization
│   ├── py.typed          # Type hints marker
│   └── _src/             # Source code
│       ├── action_selection.py
│       ├── base.py
│       ├── policies.py
│       ├── qtransforms.py
│       ├── search.py
│       ├── seq_halving.py
│       └── tree.py
│
├── examples/             # Core library examples
│   ├── policy_improvement_demo.py
│   └── visualization_demo.py
│
├── docs/                 # Core library documentation
│   └── api/              # API documentation
│
├── requirements/         # Dependency management
│   ├── requirements.txt
│   ├── requirements-test.txt
│   └── requirements_examples.txt
│
├── setup.py              # Package installation
├── MANIFEST.in           # Package manifest
├── LICENSE               # License file
└── README.md             # Main README
```

## Enterprise Extensions Structure

```
enterprise/               # Enterprise extensions root
├── backend/              # Backend components
│   ├── distributed/      # Distributed MCTS implementation
│   ├── fastapi/          # FastAPI service implementation
│   ├── hana/             # SAP HANA integration
│   └── t4/               # NVIDIA T4 GPU optimizations
│
├── deployment/           # Deployment configurations
│   ├── docker/           # Docker Compose deployment
│   ├── kubernetes/       # Kubernetes deployment
│   └── monitoring/       # Prometheus and Grafana monitoring
│
├── frontend/             # Frontend components
│   ├── react/            # React components
│   └── vercel/           # Vercel deployment
│
└── docs/                 # Documentation
    ├── business/         # Business-focused documentation
    └── technical/        # Technical documentation
```

## File Relationships

- **Core to Enterprise**: The enterprise extensions build upon the core MCTX library, importing and extending its functionality.
- **Backend to Frontend**: The backend services provide APIs consumed by the frontend components.
- **Deployment to Components**: The deployment configurations package both backend and frontend components.

## Working with the Repository

### Core Library Development

For core algorithm improvements and bug fixes:

```bash
# Navigate to the core library
cd mctx

# Make changes to the source code
vim mctx/_src/search.py

# Run tests
pytest
```

### Enterprise Feature Development

For enterprise feature development:

```bash
# Navigate to the enterprise directory
cd enterprise

# Work on T4 optimizations
vim backend/t4/t4_optimizations.py

# Work on FastAPI service
vim backend/fastapi/main.py

# Work on frontend components
vim frontend/vercel/src/components/TreeVisualization.tsx
```

### Deployment

For deployment operations:

```bash
# Deploy with Docker Compose
cd enterprise/deployment/docker
docker-compose up -d

# Deploy frontend to Vercel
cd enterprise/frontend/vercel
vercel --prod
```

## Documentation

- Core library documentation is in `docs/`
- Enterprise documentation is in `enterprise/docs/`
  - Business-focused docs in `enterprise/docs/business/`
  - Technical docs in `enterprise/docs/technical/`

## Best Practices

1. **Keep concerns separated**: Core library changes should be independent of enterprise features
2. **Maintain backward compatibility**: Enterprise features should not break core functionality
3. **Document relationships**: When adding new features, document how they relate to existing components
4. **Follow consistent naming**: Use consistent naming conventions across all components
5. **Test extensively**: Ensure all components have appropriate tests