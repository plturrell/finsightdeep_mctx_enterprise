# MCTX Service API

This directory contains the service API for MCTX, providing a clean interface for running Monte Carlo Tree Search algorithms on SAP HANA infrastructure.

## Features

- **Multiple Search Algorithms**: Support for MuZero, Gumbel MuZero, and Stochastic MuZero.
- **T4 GPU Optimizations**: Special optimizations for NVIDIA T4 GPUs including mixed precision and tensor core alignment.
- **Distributed Search**: Ability to distribute search across multiple GPUs for larger models and datasets.
- **SAP HANA Integration**: Complete integration with SAP HANA for storing search history and aggregated statistics.
- **Performance Monitoring**: Detailed metrics for search performance across different configurations.

## Directory Structure

- `models/`: Pydantic models for request/response handling
- `services/`: Business logic including the MCTS service
- `db/`: Database connections and utilities
- `core/`: Core utilities, configuration, and error handling
- `docs/`: Documentation on features and usage

## Getting Started

To use the service in your application:

```python
from api.app.models.mcts_models import MCTSRequest, SearchParams, RootInput
from api.app.services.mcts_service import MCTSService

# Create a service instance
service = MCTSService()

# Create a request
request = MCTSRequest(
    root_input=RootInput(...),
    search_params=SearchParams(
        num_simulations=50,
        use_t4_optimizations=True,
        precision="fp16"
    ),
    search_type="gumbel_muzero"
)

# Run the search
result = service.run_search(request)
```

See `examples/mcts_service_demo.py` for a complete working example.

## T4 GPU Optimizations

The service includes special optimizations for NVIDIA T4 GPUs:

- Mixed precision (FP16) support
- Tensor core alignment
- Memory monitoring
- Optimized kernel fusion

See `docs/optimized_mcts.md` for more details.

## Distributed MCTS

The service supports distributing MCTS across multiple GPUs:

- Partition search across devices
- Partition batch across devices
- Auto-synchronization of results

See `docs/optimized_mcts.md` for usage instructions.

## SAP HANA Integration

All search results are stored in SAP HANA with detailed metrics. The service automatically updates daily statistics for performance monitoring.

Tables:
- `MCTX_SEARCH_HISTORY`: Detailed history of each search run
- `MCTX_SEARCH_STATISTICS`: Aggregated daily statistics

## Examples

See the `examples/` directory for usage examples:
- `mcts_service_demo.py`: Demonstrates different search configurations
- `t4_optimization_demo.py`: Shows T4-specific optimizations
- `distributed_mcts_demo.py`: Shows distributed search capabilities