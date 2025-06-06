from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field


class RootInput(BaseModel):
    """
    Input model for the root node of a Monte Carlo Tree Search.
    """
    prior_logits: List[List[float]] = Field(
        ..., description="Prior logits for each batch item and action"
    )
    value: List[float] = Field(
        ..., description="Value estimate for each batch item"
    )
    embedding: List[Any] = Field(
        ..., description="State embedding for each batch item"
    )
    batch_size: int = Field(
        ..., description="Number of batch items"
    )
    num_actions: int = Field(
        ..., description="Number of possible actions"
    )


class SearchParams(BaseModel):
    """
    Parameters for MCTS search configuration.
    """
    num_simulations: int = Field(
        32, description="Number of simulations to run"
    )
    max_depth: Optional[int] = Field(
        None, description="Maximum search depth"
    )
    max_num_considered_actions: Optional[int] = Field(
        16, description="Maximum number of actions to consider at the root"
    )
    dirichlet_fraction: Optional[float] = Field(
        None, description="Fraction of exploration noise to add to prior logits"
    )
    dirichlet_alpha: Optional[float] = Field(
        None, description="Alpha parameter for Dirichlet distribution"
    )
    use_t4_optimizations: bool = Field(
        False, description="Whether to use T4 GPU optimizations"
    )
    precision: str = Field(
        "fp32", description="Precision to use (fp16 or fp32)"
    )
    tensor_core_aligned: bool = Field(
        True, description="Whether to align dimensions for tensor cores"
    )
    distributed: bool = Field(
        False, description="Whether to use distributed MCTS across multiple GPUs"
    )
    num_devices: int = Field(
        1, description="Number of devices to distribute across"
    )
    partition_batch: bool = Field(
        True, description="Whether to partition the batch across devices"
    )


class MCTSRequest(BaseModel):
    """
    Request model for running MCTS search.
    """
    root_input: RootInput
    search_params: SearchParams
    search_type: str = Field(
        "gumbel_muzero", description="Type of search algorithm to use"
    )
    device_type: str = Field(
        "gpu", description="Type of devices to use (gpu, tpu, cpu)"
    )


class SearchResult(BaseModel):
    """
    Result model containing MCTS search results.
    """
    action: List[int] = Field(
        ..., description="Selected action for each batch item"
    )
    action_weights: List[List[float]] = Field(
        ..., description="Action probabilities for each batch item"
    )
    search_statistics: Dict[str, Any] = Field(
        ..., description="Statistics from the search process"
    )
    distributed_stats: Optional[Dict[str, Any]] = Field(
        None, description="Statistics from distributed execution"
    )


class ErrorResponse(BaseModel):
    """
    Standardized error response model.
    """
    status_code: int
    message: str
    details: Optional[Dict[str, Any]] = None