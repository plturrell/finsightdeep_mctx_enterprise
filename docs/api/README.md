# MCTX API Documentation

## Overview

The MCTX API provides a RESTful interface to the MCTX (Monte Carlo Tree Search in JAX) library. It allows clients to run various MCTS algorithms without directly implementing the algorithms themselves.

This API is designed for enterprise use cases where:

1. Multiple client applications need to use MCTS algorithms
2. Computation should be offloaded to dedicated servers
3. A standardized interface is required for integration with other systems
4. Logging, monitoring, and error handling are essential

## Key Features

- **Multiple MCTS Algorithms**: Support for MuZero, Gumbel MuZero, and Stochastic MuZero
- **Batched Processing**: Efficient processing of multiple search requests simultaneously
- **Comprehensive Logging**: Structured JSON logging for monitoring and debugging
- **Robust Error Handling**: Detailed error responses with appropriate HTTP status codes
- **Health Monitoring**: Health check endpoint for system status
- **Scalable Architecture**: Designed to scale horizontally for high-demand scenarios

## API Endpoints

### POST /api/v1/mcts/search

Run Monte Carlo Tree Search with specified parameters.

#### Request Body

```json
{
  "root_input": {
    "prior_logits": [[0.1, 0.2, ...], ...],
    "value": [0.5, ...],
    "embedding": [0, ...],
    "batch_size": 32,
    "num_actions": 4
  },
  "search_params": {
    "num_simulations": 32,
    "max_depth": 50,
    "max_num_considered_actions": 16,
    "dirichlet_fraction": 0.25,
    "dirichlet_alpha": 0.3
  },
  "search_type": "gumbel_muzero"
}
```

#### Response

```json
{
  "action": [2, 1, ...],
  "action_weights": [[0.1, 0.7, 0.2, ...], ...],
  "search_statistics": {
    "duration_ms": 125.45,
    "num_expanded_nodes": 128,
    "max_depth_reached": 12
  }
}
```

### GET /api/v1/mcts/health

Check API health status.

#### Response

```json
{
  "status": "ok",
  "version": "1.0.0",
  "timestamp": 1622825892.4563272
}
```

## Error Handling

The API uses standard HTTP status codes and provides detailed error information:

- **400 Bad Request**: Invalid input parameters
- **422 Unprocessable Entity**: Validation errors in the request
- **429 Too Many Requests**: Resource limits exceeded
- **500 Internal Server Error**: Server-side errors

Error response format:

```json
{
  "status_code": 400,
  "message": "Invalid input data",
  "details": {
    "error": "Batch size exceeds maximum allowed"
  }
}
```

## Authentication

The API supports API key authentication for secure access. To enable:

1. Set `API_KEY_REQUIRED=True` in configuration
2. Include the API key in the `X-API-Key` header with each request

## Performance Considerations

- **Batch Processing**: Use batched requests whenever possible for better performance
- **Simulation Count**: Higher simulation counts provide better results but take longer
- **Maximum Depth**: Limit search depth for better performance in complex environments

## Deployment

The API is designed to be deployed in containers and can be scaled horizontally. For production deployments:

1. Configure environment variables for optimal settings
2. Enable API key authentication
3. Set up proper logging infrastructure
4. Configure appropriate resource limits

## Integration with Frontend

The API is designed to integrate with the MCTX frontend application, which provides:

1. A visual interface for configuring MCTS parameters
2. Visualization of search trees
3. Result analysis and comparison tools
4. Batch job management