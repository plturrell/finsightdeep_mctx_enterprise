# SAP HANA Integration for MCTX

This document provides detailed information on integrating the MCTX decision intelligence platform with SAP HANA databases for enterprise deployment.

## Overview

The MCTX SAP HANA integration provides:

- Secure storage of search trees, models, and simulation results
- High-performance querying and retrieval of decision data
- Enterprise-grade security and access control
- Batch operations for efficient data management
- Comprehensive metadata management
- Connection pooling for optimal performance

## Prerequisites

- SAP HANA database (version 2.0 or later)
- Python 3.8 or later
- `hdbcli` Python package (SAP HANA client library)
- MCTX with enterprise extensions

## Installation

Install MCTX with enterprise extensions:

```bash
pip install mctx[enterprise]
```

Or install the SAP HANA client separately:

```bash
pip install hdbcli>=2.16.26
```

## Configuration

The SAP HANA integration is configured using the `HanaConfig` class:

```python
from mctx.enterprise import HanaConfig

hana_config = HanaConfig(
    host="your_hana_host",       # Required: Hostname or IP
    port=30015,                  # Required: Port number (default: 30015)
    user="your_username",        # Required: Username
    password="your_password",    # Required: Password
    schema="MCTX",               # Optional: Schema name (default: "MCTX")
    encryption=True,             # Optional: Use encryption (default: True)
    autocommit=True,             # Optional: Auto-commit transactions (default: True)
    timeout=30,                  # Optional: Connection timeout in seconds (default: 30)
    pool_size=10,                # Optional: Connection pool size (default: 10)
    use_compression=True,        # Optional: Compress data (default: True)
    compression_level=6,         # Optional: Compression level 1-9 (default: 6)
    enable_caching=True,         # Optional: Enable result caching (default: True)
    cache_ttl=3600               # Optional: Cache TTL in seconds (default: 3600)
)
```

## Connection Management

### Establishing a Connection

```python
from mctx.enterprise import connect_to_hana

connection = connect_to_hana(hana_config)
```

The connection manager handles:
- Connection pooling
- Automatic reconnection
- Transaction management
- Connection health monitoring

### Closing Connections

```python
# Close all connections in the pool
connection.close_all()
```

## Data Schema

The integration creates the following tables in the specified schema:

1. **MCTS_TREES**: Stores search trees
   - `tree_id`: UUID primary key
   - `name`: Human-readable name
   - `batch_size`: Batch size of the tree
   - `num_actions`: Number of actions
   - `num_simulations`: Number of simulations
   - `tree_data`: Serialized tree data (BLOB)
   - `metadata`: Additional metadata (JSON)
   - `created_at`: Creation timestamp
   - `updated_at`: Last update timestamp

2. **MODEL_CACHE**: Stores model parameters
   - `model_id`: UUID primary key
   - `name`: Model name
   - `model_type`: Type of model
   - `model_data`: Serialized model data (BLOB)
   - `metadata`: Additional metadata (JSON)
   - `created_at`: Creation timestamp
   - `updated_at`: Last update timestamp

3. **SIMULATION_RESULTS**: Stores simulation results
   - `result_id`: UUID primary key
   - `tree_id`: Foreign key to MCTS_TREES
   - `model_id`: Foreign key to MODEL_CACHE
   - `batch_idx`: Batch index
   - `visit_counts`: Visit counts data (BLOB)
   - `visit_probs`: Visit probabilities data (BLOB)
   - `value`: Value data (BLOB)
   - `qvalues`: Q-values data (BLOB)
   - `metadata`: Additional metadata (JSON)
   - `created_at`: Creation timestamp

## Core API Functions

### Managing Trees

```python
from mctx.enterprise import save_tree_to_hana, load_tree_from_hana

# Save a tree
tree_id = save_tree_to_hana(
    connection,
    tree,
    tree_id=None,             # Optional: UUID (generated if None)
    name="DecisionTree",      # Optional: Human-readable name
    metadata={"key": "value"} # Optional: Additional metadata
)

# Load a tree
tree, metadata = load_tree_from_hana(connection, tree_id)
```

### Managing Models

```python
from mctx.enterprise import save_model_to_hana, load_model_from_hana

# Save a model
model_id = save_model_to_hana(
    connection,
    model,
    model_id=None,           # Optional: UUID (generated if None)
    name="MyModel",          # Optional: Human-readable name
    model_type="muzero",     # Optional: Model type
    metadata={"key": "value"} # Optional: Additional metadata
)

# Load a model
model = load_model_from_hana(connection, model_id)
```

### Managing Simulation Results

```python
from mctx.enterprise import save_simulation_results, load_simulation_results

# Save simulation results
result_id = save_simulation_results(
    connection,
    tree_id,
    model_id,
    batch_idx=0,
    summary=tree.summary(),
    metadata={"key": "value"}
)

# Load simulation results
results = load_simulation_results(
    connection,
    result_id=None,          # Optional: Specific result ID
    tree_id="tree_uuid",     # Optional: Filter by tree ID
    batch_idx=None           # Optional: Filter by batch index
)
```

### Batch Operations

For improved performance with multiple operations:

```python
from mctx.enterprise import batch_tree_operations

results = batch_tree_operations(
    connection,
    [
        {
            "operation": "save_tree",
            "tree": tree1,
            "name": "Tree1"
        },
        {
            "operation": "save_model",
            "model": model1,
            "name": "Model1"
        },
        {
            "operation": "load_tree",
            "tree_id": "existing_tree_id"
        }
    ]
)
```

## Advanced Usage

### Metadata Management

Metadata is stored as JSON and can contain arbitrary information:

```python
metadata = {
    "project_id": "PRJ-123",
    "decision_type": "supply_chain_optimization",
    "business_unit": "operations",
    "decision_owner": "Jane Smith",
    "confidence_level": 0.92,
    "creation_time": "2023-06-15T12:30:00Z",
    "tags": ["production", "high_priority", "q2_2023"],
    "model_version": "1.2.3",
    "parameters": {
        "discount_factor": 0.99,
        "exploration_constant": 1.25,
        "num_simulations": 1000
    }
}

tree_id = save_tree_to_hana(connection, tree, metadata=metadata)
```

### Transaction Management

For custom transaction handling:

```python
conn = connection.get_connection()
try:
    # Perform operations
    cursor = conn.cursor()
    cursor.execute("BEGIN")
    
    # Your operations here
    
    cursor.execute("COMMIT")
finally:
    connection.release_connection(conn)
```

### Connection Pooling

The integration includes automatic connection pooling:

```python
# Configure larger pool for high-throughput applications
hana_config = HanaConfig(
    host="your_hana_host",
    port=30015,
    user="your_username",
    password="your_password",
    pool_size=50  # Larger pool for high concurrency
)
```

## Performance Considerations

1. **Data Compression**
   - Enable compression for large trees or models
   - Adjust compression level based on CPU vs. storage tradeoff

2. **Connection Pooling**
   - Use appropriate pool size for your workload
   - Default of 10 connections works well for most applications

3. **Batch Operations**
   - Use batch operations for multiple related operations
   - Significantly reduces network roundtrips

4. **Query Optimization**
   - Use specific filters when loading results
   - Avoid retrieving unnecessary data

## Security Considerations

1. **Authentication**
   - Use strong passwords or SSO where available
   - Consider using environment variables for credentials

2. **Encryption**
   - Keep encryption enabled for all connections
   - SAP HANA supports TLS for secure connections

3. **Access Control**
   - Use schema-level permissions in SAP HANA
   - Restrict user access to only necessary schemas

4. **Audit Logging**
   - Enable SAP HANA audit logging for compliance
   - Track all data modifications

## Troubleshooting

### Common Issues

1. **Connection Failures**
   - Verify hostname and port
   - Check network connectivity and firewall settings
   - Ensure user has appropriate permissions

2. **Performance Issues**
   - Check connection pool utilization
   - Monitor SAP HANA system load
   - Consider increasing cache TTL for read-heavy workloads

3. **Memory Usage**
   - Large trees can consume significant memory
   - Use batching for large data operations

### Logging

Enable detailed logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Example Workflows

### Decision Storage and Retrieval

```python
# Store a decision tree after evaluation
tree_id = save_tree_to_hana(
    connection,
    tree,
    name="SupplyChainDecision",
    metadata={
        "decision_id": "D-12345",
        "decision_date": "2023-06-15",
        "decision_maker": "Operations Team"
    }
)

# Retrieve the decision later for audit
tree, metadata = load_tree_from_hana(connection, tree_id)
```

### Model Versioning

```python
# Save model versions with metadata
model_id_v1 = save_model_to_hana(
    connection,
    model_v1,
    name="InventoryModel",
    model_type="muzero",
    metadata={"version": "1.0.0", "accuracy": 0.91}
)

model_id_v2 = save_model_to_hana(
    connection,
    model_v2,
    name="InventoryModel",
    model_type="muzero",
    metadata={"version": "2.0.0", "accuracy": 0.94}
)
```

### Results Comparison

```python
# Save multiple simulation results
for i, batch_result in enumerate(batch_results):
    save_simulation_results(
        connection,
        tree_id,
        model_id,
        batch_idx=i,
        summary=batch_result.summary,
        metadata={"scenario": f"scenario_{i}"}
    )

# Retrieve all results for comparison
all_results = load_simulation_results(connection, tree_id=tree_id)
```

## API Reference

For complete API documentation, see the inline docstrings in the source code:

- `HanaConfig`: Configuration for SAP HANA connection
- `HanaConnection`: Connection manager for SAP HANA
- `HanaTreeSerializer`: Serializes and deserializes MCTS trees
- `HanaModelCache`: Caches models in SAP HANA
- `connect_to_hana()`: Establishes a connection to SAP HANA
- `save_tree_to_hana()`: Saves a tree to SAP HANA
- `load_tree_from_hana()`: Loads a tree from SAP HANA
- `save_model_to_hana()`: Saves a model to SAP HANA
- `load_model_from_hana()`: Loads a model from SAP HANA
- `save_simulation_results()`: Saves simulation results to SAP HANA
- `load_simulation_results()`: Loads simulation results from SAP HANA
- `batch_tree_operations()`: Performs batch operations on trees

## License

The SAP HANA integration is licensed under the Apache License, Version 2.0.