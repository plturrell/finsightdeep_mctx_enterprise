import time
import datetime
import jax
import jax.numpy as jnp
import mctx
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from functools import partial

from ..core.exceptions import InvalidInputError, ModelError, ResourceExceededError
from ..core.logging import log_search_metrics, with_logging
from ..core.config import get_settings
from ..models.mcts_models import MCTSRequest, RootInput, SearchParams, SearchResult
from mctx._src.t4_search import t4_search
from mctx._src.distributed import distributed_search, DistributedConfig
from ..db.hana_connector import hana_manager
from ..db.hana_sdi import hana_sdi

logger = logging.getLogger("mctx")
settings = get_settings()


class MCTSService:
    """
    Service class to handle Monte Carlo Tree Search operations.
    
    This class provides methods to run different MCTS algorithms
    and process their results.
    """
    
    def __init__(self):
        """Initialize the MCTS service."""
        # Initialize HANA connection
        hana_manager.initialize()
        
        # Try to initialize SDI, but don't fail if it's not available
        try:
            hana_sdi.initialize()
            self.sdi_available = True
        except Exception as e:
            logger.warning(f"HANA Smart Data Integration not available: {str(e)}")
            self.sdi_available = False
    
    @staticmethod
    def _convert_root_input(root_input: RootInput) -> mctx.RootFnOutput:
        """
        Convert API root input to MCTX RootFnOutput.
        
        Args:
            root_input: Root input from API request
            
        Returns:
            MCTX RootFnOutput object
        """
        try:
            prior_logits = jnp.array(root_input.prior_logits)
            value = jnp.array(root_input.value)
            embedding = jnp.array(root_input.embedding)
            
            return mctx.RootFnOutput(
                prior_logits=prior_logits,
                value=value,
                embedding=embedding,
            )
        except Exception as e:
            logger.error(f"Error converting root input: {str(e)}")
            raise InvalidInputError(
                message="Failed to convert root input to MCTX format",
                details={"error": str(e)}
            )
    
    @staticmethod
    def _validate_input(request: MCTSRequest) -> None:
        """
        Validate input request parameters.
        
        Args:
            request: MCTS request to validate
            
        Raises:
            InvalidInputError: If validation fails
        """
        # Validate batch size
        if request.root_input.batch_size > settings.MAX_BATCH_SIZE:
            raise InvalidInputError(
                message=f"Batch size exceeds maximum allowed ({settings.MAX_BATCH_SIZE})",
                details={"batch_size": request.root_input.batch_size}
            )
        
        # Validate simulation count
        if request.search_params.num_simulations > settings.MAX_NUM_SIMULATIONS:
            raise InvalidInputError(
                message=f"Number of simulations exceeds maximum allowed ({settings.MAX_NUM_SIMULATIONS})",
                details={"num_simulations": request.search_params.num_simulations}
            )
        
        # Validate search type
        valid_search_types = ["muzero", "gumbel_muzero", "stochastic_muzero"]
        if request.search_type not in valid_search_types:
            raise InvalidInputError(
                message=f"Invalid search type. Must be one of: {', '.join(valid_search_types)}",
                details={"search_type": request.search_type}
            )
            
        # Validate precision if T4 optimizations enabled
        if request.search_params.use_t4_optimizations:
            valid_precisions = ["fp16", "fp32"]
            if request.search_params.precision not in valid_precisions:
                raise InvalidInputError(
                    message=f"Invalid precision. Must be one of: {', '.join(valid_precisions)}",
                    details={"precision": request.search_params.precision}
                )
        
        # Validate distributed settings
        if request.search_params.distributed:
            if request.search_params.num_devices <= 1:
                raise InvalidInputError(
                    message="Distributed mode requires num_devices > 1",
                    details={"num_devices": request.search_params.num_devices}
                )
            
            valid_device_types = ["gpu", "tpu", "cpu"]
            if request.device_type not in valid_device_types:
                raise InvalidInputError(
                    message=f"Invalid device type. Must be one of: {', '.join(valid_device_types)}",
                    details={"device_type": request.device_type}
                )
    
    @staticmethod
    def _get_mock_recurrent_fn(qvalues: jnp.ndarray) -> Callable:
        """
        Create a mock recurrent function for demonstration purposes.
        
        In a production setting, this would be replaced with the actual
        recurrent function from the user's model.
        
        Args:
            qvalues: Q-values for each state-action pair
            
        Returns:
            A recurrent function compatible with MCTX
        """
        def recurrent_fn(params, rng_key, action, embedding):
            batch_size = action.shape[0]
            
            # Simple mock reward based on action and embedding
            reward = jnp.ones_like(action, dtype=jnp.float32)
            
            # Always use discount=1.0 in this mock example
            discount = jnp.ones_like(reward)
            
            # Create dummy prior logits
            prior_logits = jnp.zeros((batch_size, qvalues.shape[1]), dtype=jnp.float32)
            
            # Create dummy value
            value = jnp.zeros_like(reward)
            
            # Increment embedding to track depth
            next_embedding = embedding + 1
            
            recurrent_fn_output = mctx.RecurrentFnOutput(
                reward=reward,
                discount=discount,
                prior_logits=prior_logits,
                value=value
            )
            
            return recurrent_fn_output, next_embedding
        
        return recurrent_fn
    
    def _run_t4_optimized_search(self, params, rng_key, root, recurrent_fn, 
                           search_params, search_type):
        """
        Run T4-optimized MCTS search.
        
        Args:
            params: Parameters for recurrent_fn
            rng_key: Random key
            root: Root node
            recurrent_fn: Recurrent function
            search_params: Search parameters
            search_type: Type of search algorithm
            
        Returns:
            Policy output
        """
        # Extract T4-specific parameters
        precision = search_params.precision
        tensor_core_aligned = search_params.tensor_core_aligned
        
        # Create action selection functions based on search type
        if search_type == "muzero":
            root_fn = partial(
                mctx.muzero_policy,
                params=params,
                rng_key=rng_key,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=search_params.num_simulations,
                max_depth=search_params.max_depth,
                dirichlet_fraction=search_params.dirichlet_fraction,
                dirichlet_alpha=search_params.dirichlet_alpha,
            )
            root_action_selection_fn = mctx.puct_action_selection
            interior_action_selection_fn = mctx.puct_action_selection
        elif search_type == "gumbel_muzero":
            root_fn = partial(
                mctx.gumbel_muzero_policy,
                params=params,
                rng_key=rng_key,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=search_params.num_simulations,
                max_depth=search_params.max_depth,
                max_num_considered_actions=search_params.max_num_considered_actions,
            )
            root_action_selection_fn = mctx.gumbel_muzero_root_action_selection
            interior_action_selection_fn = mctx.muzero_interior_action_selection
        elif search_type == "stochastic_muzero":
            root_fn = partial(
                mctx.stochastic_muzero_policy,
                params=params,
                rng_key=rng_key,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=search_params.num_simulations,
                max_depth=search_params.max_depth,
            )
            root_action_selection_fn = mctx.stochastic_muzero_root_action_selection
            interior_action_selection_fn = mctx.stochastic_muzero_interior_action_selection
        else:
            # This should never happen due to validation
            raise InvalidInputError(f"Invalid search type: {search_type}")
        
        # Run T4-optimized search
        tree = t4_search(
            params=params,
            rng_key=rng_key,
            root=root,
            recurrent_fn=recurrent_fn,
            root_action_selection_fn=root_action_selection_fn,
            interior_action_selection_fn=interior_action_selection_fn,
            num_simulations=search_params.num_simulations,
            max_depth=search_params.max_depth,
            precision=precision,
            tensor_core_aligned=tensor_core_aligned,
            monitor_memory=True
        )
        
        # Convert tree to policy output
        if search_type == "muzero":
            action = mctx.muzero_action_selection(rng_key, tree)
        elif search_type == "gumbel_muzero":
            action = mctx.gumbel_muzero_action_selection(rng_key, tree)
        elif search_type == "stochastic_muzero":
            action = mctx.stochastic_muzero_action_selection(rng_key, tree)
        
        # Calculate action weights (visit counts normalized)
        visit_counts = tree.children_visits[0]  # Root node visits
        action_weights = visit_counts / jnp.sum(visit_counts, axis=-1, keepdims=True)
        
        # Create policy output NamedTuple
        PolicyOutput = Tuple[jnp.ndarray, jnp.ndarray, mctx._src.tree.Tree]
        return PolicyOutput(action=action, action_weights=action_weights, search_tree=tree)

    def _run_distributed_search(self, params, rng_key, root, recurrent_fn, 
                              search_params, search_type):
        """
        Run distributed MCTS search across multiple GPUs.
        
        Args:
            params: Parameters for recurrent_fn
            rng_key: Random key
            root: Root node
            recurrent_fn: Recurrent function
            search_params: Search parameters
            search_type: Type of search algorithm
            
        Returns:
            Policy output and distributed statistics
        """
        # Create distributed configuration
        distributed_config = DistributedConfig(
            num_devices=search_params.num_devices,
            partition_search=True,
            partition_batch=search_params.partition_batch,
            device_type=search_params.device_type,
            precision=search_params.precision,
            tensor_core_aligned=search_params.tensor_core_aligned,
        )
        
        # Create action selection functions based on search type
        if search_type == "muzero":
            root_action_selection_fn = mctx.puct_action_selection
            interior_action_selection_fn = mctx.puct_action_selection
        elif search_type == "gumbel_muzero":
            root_action_selection_fn = mctx.gumbel_muzero_root_action_selection
            interior_action_selection_fn = mctx.muzero_interior_action_selection
        elif search_type == "stochastic_muzero":
            root_action_selection_fn = mctx.stochastic_muzero_root_action_selection
            interior_action_selection_fn = mctx.stochastic_muzero_interior_action_selection
        else:
            # This should never happen due to validation
            raise InvalidInputError(f"Invalid search type: {search_type}")
        
        # Track distributed performance
        distributed_start_time = time.time()
        
        # Run distributed search
        tree = distributed_search(
            params=params,
            rng_key=rng_key,
            root=root,
            recurrent_fn=recurrent_fn,
            root_action_selection_fn=root_action_selection_fn,
            interior_action_selection_fn=interior_action_selection_fn,
            num_simulations=search_params.num_simulations,
            max_depth=search_params.max_depth,
            config=distributed_config
        )
        
        # Calculate distributed execution time
        distributed_duration_ms = (time.time() - distributed_start_time) * 1000
        
        # Convert tree to policy output
        if search_type == "muzero":
            action = mctx.muzero_action_selection(rng_key, tree)
        elif search_type == "gumbel_muzero":
            action = mctx.gumbel_muzero_action_selection(rng_key, tree)
        elif search_type == "stochastic_muzero":
            action = mctx.stochastic_muzero_action_selection(rng_key, tree)
        
        # Calculate action weights (visit counts normalized)
        visit_counts = tree.children_visits[0]  # Root node visits
        action_weights = visit_counts / jnp.sum(visit_counts, axis=-1, keepdims=True)
        
        # Create policy output NamedTuple
        PolicyOutput = Tuple[jnp.ndarray, jnp.ndarray, mctx._src.tree.Tree]
        
        # Create distributed statistics
        distributed_stats = {
            "duration_ms": distributed_duration_ms,
            "num_devices": search_params.num_devices,
            "partition_batch": search_params.partition_batch,
            "device_type": search_params.device_type,
        }
        
        return PolicyOutput(action=action, action_weights=action_weights, search_tree=tree), distributed_stats
        
    @with_logging
    def run_search(self, request: MCTSRequest, user_id: Optional[str] = None) -> SearchResult:
        """
        Run Monte Carlo Tree Search with the specified parameters.
        
        Args:
            request: MCTS request containing root input and search parameters
            user_id: Optional user identifier for history tracking
            
        Returns:
            SearchResult containing actions and search statistics
            
        Raises:
            InvalidInputError: If input validation fails
            ModelError: If search execution fails
            ResourceExceededError: If execution exceeds resource limits
        """
        try:
            # Validate input
            self._validate_input(request)
            
            # Start timing
            start_time = time.time()
            
            # Convert root input to MCTX format
            root = self._convert_root_input(request.root_input)
            
            # Get a random key
            rng_key = jax.random.PRNGKey(int(time.time() * 1000) % (2**32))
            
            # Create mock qvalues for demonstration
            # In production, this would be replaced with real model outputs
            qvalues = jnp.ones((request.root_input.batch_size, request.root_input.num_actions))
            
            # Get a recurrent function
            # In production, this would be provided by the user's model
            recurrent_fn = self._get_mock_recurrent_fn(qvalues)
            
            # Extract search parameters
            search_params = request.search_params
            search_type = request.search_type
            
            # Initialize distributed stats
            distributed_stats = None
            
            # Determine which search method to use
            if search_params.distributed and search_params.num_devices > 1:
                # Run distributed search
                logger.info(f"Running distributed MCTS search with {search_params.num_devices} devices")
                policy_output, distributed_stats = self._run_distributed_search(
                    params=(),
                    rng_key=rng_key,
                    root=root,
                    recurrent_fn=recurrent_fn,
                    search_params=search_params,
                    search_type=search_type
                )
            elif search_params.use_t4_optimizations:
                # Run T4-optimized search
                logger.info("Running T4-optimized MCTS search")
                policy_output = self._run_t4_optimized_search(
                    params=(),
                    rng_key=rng_key,
                    root=root,
                    recurrent_fn=recurrent_fn,
                    search_params=search_params,
                    search_type=search_type
                )
            else:
                # Run standard search algorithm
                logger.info(f"Running standard {search_type} search")
                
                # Run the appropriate search algorithm
                if search_type == "muzero":
                    policy_output = mctx.muzero_policy(
                        params=(),
                        rng_key=rng_key,
                        root=root,
                        recurrent_fn=recurrent_fn,
                        num_simulations=search_params.num_simulations,
                        max_depth=search_params.max_depth,
                        dirichlet_fraction=search_params.dirichlet_fraction,
                        dirichlet_alpha=search_params.dirichlet_alpha,
                    )
                elif search_type == "gumbel_muzero":
                    policy_output = mctx.gumbel_muzero_policy(
                        params=(),
                        rng_key=rng_key,
                        root=root,
                        recurrent_fn=recurrent_fn,
                        num_simulations=search_params.num_simulations,
                        max_depth=search_params.max_depth,
                        max_num_considered_actions=search_params.max_num_considered_actions,
                    )
                elif search_type == "stochastic_muzero":
                    policy_output = mctx.stochastic_muzero_policy(
                        params=(),
                        rng_key=rng_key,
                        root=root,
                        recurrent_fn=recurrent_fn,
                        num_simulations=search_params.num_simulations,
                        max_depth=search_params.max_depth,
                    )
                else:
                    # This should never happen due to validation
                    raise InvalidInputError(f"Invalid search type: {search_type}")
            
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Extract search statistics
            search_tree = policy_output.search_tree
            action = policy_output.action.tolist()
            action_weights = policy_output.action_weights.tolist()
            
            # Calculate metrics for logging
            num_expanded_nodes = jnp.sum(search_tree.node_visits > 0).item()
            max_depth_reached = 0  # In a real implementation, track this during search
            
            # Log metrics
            log_search_metrics(
                search_type=request.search_type,
                batch_size=request.root_input.batch_size,
                num_simulations=search_params.num_simulations,
                duration=duration_ms,
                num_expanded_nodes=num_expanded_nodes,
                max_depth_reached=max_depth_reached
            )
            
            # Return formatted search results
            search_statistics = {
                "duration_ms": duration_ms,
                "num_expanded_nodes": num_expanded_nodes,
                "max_depth_reached": max_depth_reached,
                "search_type": search_type,
                "optimized": search_params.use_t4_optimizations or search_params.distributed,
                "precision": search_params.precision
            }
            
            result = SearchResult(
                action=action,
                action_weights=action_weights,
                search_statistics=search_statistics,
                distributed_stats=distributed_stats
            )
            
            # Save search history to database
            try:
                today = datetime.datetime.now().strftime("%Y-%m-%d")
                
                # Save detailed search history
                hana_manager.save_search_history(
                    search_type=request.search_type,
                    batch_size=request.root_input.batch_size,
                    num_simulations=search_params.num_simulations,
                    max_depth=search_params.max_depth,
                    config={
                        "search_params": dict(search_params),
                        "root_input_shape": {
                            "batch_size": request.root_input.batch_size,
                            "num_actions": request.root_input.num_actions
                        },
                        "optimizations": {
                            "use_t4": search_params.use_t4_optimizations,
                            "distributed": search_params.distributed,
                            "num_devices": search_params.num_devices if search_params.distributed else 1,
                            "precision": search_params.precision,
                            "tensor_core_aligned": search_params.tensor_core_aligned,
                            "device_type": request.device_type
                        }
                    },
                    duration_ms=duration_ms,
                    num_expanded_nodes=num_expanded_nodes,
                    max_depth_reached=max_depth_reached,
                    result=result.dict(),
                    user_id=user_id
                )
                
                # Update daily statistics
                hana_manager.update_daily_statistics(
                    date_key=today,
                    search_type=request.search_type
                )
                
                # Store large tree data if SDI is available and tree is large
                if self.sdi_available and policy_output.search_tree.node_visits.size > 10000:
                    try:
                        # Generate unique tree ID
                        import uuid
                        tree_id = f"{request.search_type}_{uuid.uuid4().hex}"
                        
                        # Create tree data for storage
                        tree_data = {
                            "search_type": request.search_type,
                            "batch_size": request.root_input.batch_size,
                            "num_simulations": search_params.num_simulations,
                            "node_count": int(policy_output.search_tree.node_visits.size),
                            "max_depth": search_params.max_depth,
                            "optimization": {
                                "use_t4": search_params.use_t4_optimizations,
                                "distributed": search_params.distributed,
                                "num_devices": search_params.num_devices
                            },
                            "tree_summary": {
                                "visit_count": policy_output.search_tree.node_visits.tolist(),
                                "values": policy_output.search_tree.node_values.tolist()
                            }
                        }
                        
                        # Store using SDI
                        hana_sdi.save_large_tree(tree_id, tree_data, user_id)
                    except Exception as e:
                        # Log error but don't fail if SDI storage fails
                        logger.error(f"Failed to store large tree data: {str(e)}")
                        
                # Add workload class information for HANA resource management
                conn = hana_manager.get_connection().__enter__()
                try:
                    with conn.cursor() as cursor:
                        cursor.execute("ALTER SYSTEM ALTER CONFIGURATION ('indexserver.ini', 'SYSTEM') SET ('workload_classes', 'MCTS_SERVICE') = 'default_workload_class';")
                    conn.commit()
                except Exception as e:
                    logger.warning(f"Could not set workload class: {str(e)}")
                finally:
                    try:
                        hana_manager.get_connection().__exit__(None, None, None)
                    except:
                        pass
            except Exception as e:
                # Log error but don't fail the request if history saving fails
                logger.error(f"Failed to save search history: {str(e)}")
            
            return result
            
        except Exception as e:
            if isinstance(e, (InvalidInputError, ModelError, ResourceExceededError)):
                raise
            
            logger.exception(f"Unexpected error in MCTS search: {str(e)}")
            raise ModelError(
                message="Failed to execute MCTS search",
                details={"error": str(e)}
            )