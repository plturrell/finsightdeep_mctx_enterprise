# Contributing to MCTX

Thank you for your interest in contributing to MCTX! This document provides guidelines and instructions for contributing to the project.

## Code Structure

MCTX uses the following directory structure:

- `mctx/`: Core Monte Carlo Tree Search implementation
  - `_src/`: Internal implementation details
  - `enterprise/`: Enterprise features and integrations
  - `monitoring/`: Monitoring and visualization tools

- `src/`: Extended modules and utilities
  - `aiq/`: AI-powered query capabilities
    - `owl/`: Optimized Workflow Library

- `api/`: API implementation
  - `app/`: FastAPI application

- `examples/`: Example usage of MCTX

- `config/`: Configuration files
  - `docker/`: Docker configurations
  - `prometheus/`: Prometheus configurations
  - `grafana/`: Grafana configurations
  - `nginx/`: NGINX configurations

- `docs/`: Documentation

- `tests/`: Unit and integration tests

## Getting Started

1. **Set up your development environment:**

   ```bash
   # Clone the repository
   git clone https://github.com/google-deepmind/mctx.git
   cd mctx

   # Create a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install -r requirements/requirements.txt
   pip install -r requirements/requirements-test.txt
   
   # Install pre-commit hooks
   pip install pre-commit
   pre-commit install
   ```

2. **Run tests to verify your setup:**

   ```bash
   pytest tests/
   ```

## Development Workflow

1. **Create a new branch for your feature or bugfix:**

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes following the coding standards**

3. **Run tests to ensure your changes don't break existing functionality:**

   ```bash
   pytest tests/
   ```

4. **Commit your changes with a descriptive message:**

   ```bash
   git commit -m "Add feature: description of your changes"
   ```

5. **Push your branch and create a pull request:**

   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style Guidelines

MCTX follows these style guidelines:

- **Python**: [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- **Line length**: 100 characters
- **Indentation**: 2 spaces
- **Documentation**: [Google docstring format](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)

We use the following tools to enforce code style:

- **Black**: Code formatting
- **isort**: Import sorting
- **Pylint**: Linting
- **mypy**: Type checking
- **Ruff**: Fast linting

These tools are configured in `.pre-commit-config.yaml` and `pyproject.toml`.

## Package Structure

New code should follow these guidelines:

1. **Core MCTS Algorithms**: Add to `mctx/_src/`
2. **Enterprise Features**: Add to `mctx/enterprise/`
3. **API Components**: Add to `src/aiq/owl/`
4. **Visualization Tools**: Add to `mctx/monitoring/`

## Import Order

Imports should be organized in the following order:

1. Standard library imports
2. Third-party imports
3. MCTX imports

Example:

```python
import os
import time
from typing import Dict, List, Optional

import numpy as np
import jax
import jax.numpy as jnp

from mctx import base
from mctx._src import tree
from src.aiq.owl import config
```

## Commit Message Guidelines

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters
- Reference issues and pull requests after the first line

## Pull Request Process

1. Ensure your code passes all tests and style checks
2. Update documentation if necessary
3. Include tests for new functionality
4. Ensure the PR description clearly describes the problem and solution

## Testing Guidelines

- Write unit tests for all new functionality
- Maintain or improve test coverage
- Place tests in the `tests/` directory with a matching structure to the code being tested
- Name test files with a `test_` prefix

## Documentation

- Update README.md if necessary
- Add docstrings to all public functions, classes, and methods
- Keep documentation up to date with code changes

## Code Review

All submissions require review. We use GitHub pull requests for this purpose.

## License

By contributing to MCTX, you agree that your contributions will be licensed under the project's [Apache License](LICENSE).