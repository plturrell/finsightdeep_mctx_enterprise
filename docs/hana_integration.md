# SAP HANA Integration

This guide provides detailed information on integrating MCTX with SAP HANA for enterprise-grade storage, retrieval, and analysis of Monte Carlo Tree Search (MCTS) results.

## Overview

The SAP HANA integration allows MCTX users to:
- Store search trees and policy outputs in a performant, enterprise-grade database
- Query and analyze search results across multiple experiments
- Maintain a persistent history of search operations
- Share results across distributed systems
- Leverage HANA's analytics capabilities for MCTS result analysis

## Setup

### Prerequisites

- SAP HANA database (on-premises or cloud instance)
- SAP HANA Python client (`hdbcli` package)
- Valid HANA user credentials with appropriate permissions

### Installation

```bash
# Install MCTX with HANA integration
pip install mctx[hana]

# Or manually install dependencies
pip install mctx
pip install hdbcli pyhdb
```

## Basic Usage

### Connecting to SAP HANA

```python
from mctx.integrations import hana_connector

# Connect to SAP HANA
connector = hana_connector.HanaConnector(
    host="your_hana_host.example.com",
    port=443,  # Default is 30015 for non-SSL, 443 for SSL
    user="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    use_ssl=True  # Recommended for production
)

# Test connection
if connector.test_connection():
    print("Successfully connected to SAP HANA")
else:
    print("Connection failed")
```

### Schema Initialization

The first time you use the connector, you need to initialize the MCTX schema:

```python
# Initialize MCTX schema in HANA
connector.initialize_schema(
    schema_name="MCTX_DATA",  # Optional, defaults to "MCTX_DATA"
    drop_existing=False  # Set to True to recreate schema if it exists
)
```

This creates the following tables:
- `SEARCH_RESULTS`: Main table for search results metadata
- `SEARCH_TREES`: Stores serialized search trees
- `POLICY_OUTPUTS`: Stores policy outputs
- `SEARCH_METRICS`: Stores performance metrics
- `SEARCH_PARAMETERS`: Stores search configuration parameters

### Storing Search Results

After running an MCTS search, store the results:

```python
import mctx
import jax

# Run a search
params = ...  # Your model parameters
root = ...    # Root state
recurrent_fn = ...  # Your environment model

policy_output = mctx.muzero_policy(
    params, 
    jax.random.PRNGKey(0), 
    root, 
    recurrent_fn,
    num_simulations=128
)

# Store the search results
search_id = connector.store_search_results(
    search_id=None,  # Auto-generate an ID if None
    policy_output=policy_output,
    metadata={
        "experiment": "chess_training",
        "iteration": 42,
        "environment": "chess",
        "player": "white"
    }
)

print(f"Stored search results with ID: {search_id}")
```

### Retrieving Search Results

Retrieve results by ID:

```python
# Get search results by ID
result = connector.get_search_results("12345-abcde-67890")

if result:
    policy_output = result.policy_output
    metadata = result.metadata
    timestamp = result.timestamp
    
    # Use the retrieved policy output
    action = policy_output.action
    search_tree = policy_output.search_tree
    
    print(f"Retrieved search from {timestamp}")
    print(f"Metadata: {metadata}")
    print(f"Selected action: {action}")
else:
    print("Search results not found")
```

### Querying Search Results

Find search results matching specific criteria:

```python
# Query search results
results = connector.query_search_results(
    experiment="chess_training",
    min_timestamp="2023-01-01",
    max_timestamp="2023-12-31",
    metadata_filters={
        "player": "white",
        "environment": "chess"
    },
    limit=10,
    order_by="timestamp",
    order_direction="DESC"
)

print(f"Found {len(results)} matching search results")

for result in results:
    print(f"ID: {result.search_id}, Timestamp: {result.timestamp}")
    print(f"Action: {result.policy_output.action}")
    print(f"Value: {result.policy_output.root_value}")
    print("---")
```

## Advanced Features

### Batch Operations

For improved performance when storing many search results:

```python
# Start a batch operation
with connector.batch_operation(batch_size=100) as batch:
    for i in range(1000):
        # Run search
        policy_output = mctx.muzero_policy(...)
        
        # Add to batch
        batch.add_search_result(
            policy_output=policy_output,
            metadata={"iteration": i}
        )
        
        # Batch will automatically commit every 100 items
        # and on context exit
```

### Tree Analysis

Analyze search trees directly in HANA:

```python
# Get statistical metrics for a specific experiment
metrics = connector.analyze_search_trees(
    experiment="chess_training",
    metrics=["avg_depth", "avg_branching_factor", "max_value", "min_value"]
)

print(f"Average tree depth: {metrics['avg_depth']}")
print(f"Average branching factor: {metrics['avg_branching_factor']}")
print(f"Value range: [{metrics['min_value']}, {metrics['max_value']}]")

# Compare metrics across experiments
comparison = connector.compare_experiments(
    experiment_ids=["exp1", "exp2", "exp3"],
    metrics=["avg_depth", "avg_value"]
)

for exp_id, exp_metrics in comparison.items():
    print(f"Experiment {exp_id}:")
    print(f"  Average depth: {exp_metrics['avg_depth']}")
    print(f"  Average value: {exp_metrics['avg_value']}")
```

### Performance Monitoring

Store and analyze performance metrics:

```python
# Store performance metrics
connector.store_performance_metrics(
    search_id="12345-abcde-67890",
    metrics={
        "total_time_seconds": 15.23,
        "simulations_per_second": 8.4,
        "max_memory_usage_mb": 1250,
        "avg_batch_time_ms": 18.5,
        "tree_size_nodes": 2048
    }
)

# Query performance over time
performance_trend = connector.query_performance_trend(
    experiment="chess_training",
    metric="simulations_per_second",
    group_by="day",
    start_date="2023-01-01",
    end_date="2023-12-31"
)

for date, value in performance_trend:
    print(f"{date}: {value} simulations per second")
```

## Security Best Practices

### Credential Management

For secure credential management:

```python
import os
from mctx.integrations import hana_connector

# Using environment variables (recommended)
connector = hana_connector.HanaConnector(
    host=os.environ.get("HANA_HOST"),
    port=int(os.environ.get("HANA_PORT", "443")),
    user=os.environ.get("HANA_USER"),
    password=os.environ.get("HANA_PASSWORD"),
    use_ssl=True
)

# Using credential file
connector = hana_connector.HanaConnector.from_credential_file(
    "/path/to/secure/credentials.json",
    use_ssl=True
)

# Using credential manager
connector = hana_connector.HanaConnector.from_credential_manager(
    "sap_hana_credentials",
    use_ssl=True
)
```

### SSL Configuration

For enhanced security:

```python
# Connect with custom SSL configuration
connector = hana_connector.HanaConnector(
    host="your_hana_host.example.com",
    port=443,
    user="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    use_ssl=True,
    ssl_validate_cert=True,
    ssl_ca_file="/path/to/ca_cert.pem"
)
```

## Schema Details

The MCTX HANA integration uses the following schema structure:

### SEARCH_RESULTS Table

```sql
CREATE TABLE MCTX_DATA.SEARCH_RESULTS (
    SEARCH_ID NVARCHAR(100) PRIMARY KEY,
    TIMESTAMP TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    EXPERIMENT_ID NVARCHAR(100),
    METADATA NCLOB,  -- JSON metadata
    POLICY_OUTPUT_ID NVARCHAR(100),
    SEARCH_TREE_ID NVARCHAR(100),
    PARAMETERS_ID NVARCHAR(100)
);
```

### SEARCH_TREES Table

```sql
CREATE TABLE MCTX_DATA.SEARCH_TREES (
    TREE_ID NVARCHAR(100) PRIMARY KEY,
    SEARCH_ID NVARCHAR(100),
    TREE_DATA NCLOB,  -- Serialized tree data (JSON)
    NODE_COUNT INTEGER,
    MAX_DEPTH INTEGER,
    ROOT_VALUE DOUBLE
);
```

### POLICY_OUTPUTS Table

```sql
CREATE TABLE MCTX_DATA.POLICY_OUTPUTS (
    OUTPUT_ID NVARCHAR(100) PRIMARY KEY,
    SEARCH_ID NVARCHAR(100),
    ACTION NVARCHAR(100),
    ACTION_WEIGHTS NCLOB,  -- JSON array
    Q_VALUES NCLOB,  -- JSON array
    VISIT_COUNTS NCLOB,  -- JSON array
    ROOT_VALUE DOUBLE
);
```

## Example: Complete Integration Workflow

Here's a complete example of integrating MCTX with SAP HANA in a training loop:

```python
import jax
import mctx
from mctx.integrations import hana_connector
import time
import uuid

# Connect to SAP HANA
connector = hana_connector.HanaConnector(
    host="your_hana_host.example.com",
    port=443,
    user="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    use_ssl=True
)

# Initialize schema if needed
connector.initialize_schema(drop_existing=False)

# Setup your model and environment
params = ...  # Your model parameters
env = ...     # Your environment
recurrent_fn = ...  # Your environment model

# Create experiment ID
experiment_id = f"experiment_{uuid.uuid4().hex[:8]}"
print(f"Starting experiment: {experiment_id}")

# Training loop
for iteration in range(100):
    print(f"Iteration {iteration}/100")
    
    # Reset environment
    observation = env.reset()
    done = False
    total_reward = 0
    
    # Game loop
    game_trajectory = []
    while not done:
        # Convert observation to root state
        root = create_root_from_observation(observation, params)
        
        # Run MCTS
        start_time = time.time()
        policy_output = mctx.muzero_policy(
            params, 
            jax.random.PRNGKey(iteration), 
            root, 
            recurrent_fn,
            num_simulations=128
        )
        search_time = time.time() - start_time
        
        # Store search results
        search_id = connector.store_search_results(
            search_id=None,
            policy_output=policy_output,
            metadata={
                "experiment": experiment_id,
                "iteration": iteration,
                "game_step": len(game_trajectory),
                "observation": observation.tolist()
            }
        )
        
        # Store performance metrics
        connector.store_performance_metrics(
            search_id=search_id,
            metrics={
                "search_time_seconds": search_time,
                "simulations_per_second": 128 / search_time,
                "tree_size": policy_output.search_tree.node_count
            }
        )
        
        # Select action from policy
        action = policy_output.action
        
        # Execute action in environment
        next_observation, reward, done, info = env.step(action)
        
        # Record step
        game_trajectory.append({
            "observation": observation,
            "action": action,
            "reward": reward,
            "search_id": search_id
        })
        
        # Update for next step
        observation = next_observation
        total_reward += reward
    
    # Store game result
    connector.store_game_result(
        experiment_id=experiment_id,
        iteration=iteration,
        trajectory=game_trajectory,
        total_reward=total_reward,
        metadata={"game_length": len(game_trajectory)}
    )
    
    # Update model parameters (simplified)
    params = update_model(params, game_trajectory)

# Query results at the end
results = connector.query_search_results(
    experiment=experiment_id,
    limit=10,
    order_by="timestamp",
    order_direction="DESC"
)

print(f"Latest search results from experiment {experiment_id}:")
for result in results:
    print(f"ID: {result.search_id}, Action: {result.policy_output.action}")

# Get performance trend
performance = connector.query_performance_trend(
    experiment=experiment_id,
    metric="simulations_per_second",
    group_by="iteration"
)

print("Performance trend:")
for iteration, sims_per_second in performance:
    print(f"Iteration {iteration}: {sims_per_second:.2f} sims/second")
```

## Troubleshooting

### Common Issues

1. **Connection Failures**

   ```python
   # Test connection details
   connector.test_connection_details()
   ```

2. **Slow Performance**

   ```python
   # Enable performance logging
   connector = hana_connector.HanaConnector(
       # connection details...
       enable_performance_logging=True
   )
   
   # Check logs
   performance_logs = connector.get_performance_logs()
   print(f"Average query time: {performance_logs['avg_query_time_ms']}ms")
   ```

3. **Memory Issues**

   ```python
   # Configure batch size for large trees
   connector = hana_connector.HanaConnector(
       # connection details...
       max_tree_size_mb=100,  # Split trees larger than 100MB
       compression_level=5    # 0-9, higher means more compression
   )
   ```

For more information, see the [SAP HANA Client Interface for Python](https://help.sap.com/docs/SAP_HANA_CLIENT/f1b440ded6144a54ada97ff95dac7adf/f3b8fabf34324302b123297cdbe710f0.html) documentation.