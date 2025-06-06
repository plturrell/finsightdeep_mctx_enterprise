import logging
import jax
import jax.numpy as jnp
import mctx
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..core.config import get_settings

logger = logging.getLogger("mctx")
settings = get_settings()


class RecurrentFnService:
    """
    Service for managing MCTS recurrent functions.
    
    This service provides implementations of recurrent functions
    for various domains and models.
    """
    
    @staticmethod
    def create_environment_model(
        model_params: Dict[str, Any],
        model_type: str = "muzero",
    ) -> Callable:
        """
        Create an environment model recurrent function.
        
        Args:
            model_params: Parameters for the model
            model_type: Type of model to create
            
        Returns:
            RecurrentFn function compatible with MCTX
        """
        if model_type == "muzero":
            return RecurrentFnService.create_muzero_recurrent_fn(model_params)
        elif model_type == "gumbel_muzero":
            return RecurrentFnService.create_gumbel_muzero_recurrent_fn(model_params)
        elif model_type == "bandit":
            return RecurrentFnService.create_bandit_recurrent_fn(model_params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    @staticmethod
    def create_muzero_recurrent_fn(model_params: Dict[str, Any]) -> Callable:
        """
        Create a MuZero-style recurrent function using JAX.
        
        Args:
            model_params: Neural network parameters
            
        Returns:
            RecurrentFn function compatible with MCTX
        """
        # Import here to avoid circular imports
        import haiku as hk
        import numpy as np
        
        # Extract model architecture parameters
        embedding_size = model_params.get("embedding_size", 64)
        hidden_sizes = model_params.get("hidden_sizes", [64, 64])
        num_actions = model_params.get("num_actions", 4)
        
        # Define dynamics network
        def dynamics_net(action, embedding):
            """MuZero dynamics network predicting next state and reward."""
            # One-hot encode action
            action_one_hot = jax.nn.one_hot(action, num_actions)
            
            # Concatenate action and embedding
            x = jnp.concatenate([action_one_hot, embedding], axis=-1)
            
            # Hidden layers
            for size in hidden_sizes:
                x = hk.Linear(size)(x)
                x = jax.nn.relu(x)
            
            # Predict reward
            reward = hk.Linear(1)(x)
            reward = jnp.squeeze(reward, axis=-1)
            
            # Predict next state embedding
            next_embedding = hk.Linear(embedding_size)(x)
            
            return reward, next_embedding
        
        # Define prediction network
        def prediction_net(embedding):
            """MuZero prediction network for policy and value."""
            x = embedding
            
            # Hidden layers
            for size in hidden_sizes:
                x = hk.Linear(size)(x)
                x = jax.nn.relu(x)
            
            # Predict policy logits and value
            policy_logits = hk.Linear(num_actions)(x)
            value = hk.Linear(1)(x)
            value = jnp.squeeze(value, axis=-1)
            
            return policy_logits, value
        
        # Initialize networks
        dynamics_transformed = hk.transform(dynamics_net)
        prediction_transformed = hk.transform(prediction_net)
        
        # Recurrent function
        def recurrent_fn(params, rng_key, action, embedding):
            """
            MuZero recurrent function.
            
            Args:
                params: Neural network parameters
                rng_key: JAX random key
                action: Selected actions
                embedding: State embeddings
                
            Returns:
                Tuple of (RecurrentFnOutput, next_embedding)
            """
            # Predict reward and next embedding
            reward, next_embedding = dynamics_transformed.apply(
                params["dynamics"], rng_key, action, embedding
            )
            
            # Predict policy and value for next state
            policy_logits, value = prediction_transformed.apply(
                params["prediction"], rng_key, next_embedding
            )
            
            # Use discount of 1.0 (no termination in this example)
            discount = jnp.ones_like(reward)
            
            # Create RecurrentFnOutput
            recurrent_fn_output = mctx.RecurrentFnOutput(
                reward=reward,
                discount=discount,
                prior_logits=policy_logits,
                value=value
            )
            
            return recurrent_fn_output, next_embedding
        
        return recurrent_fn
    
    @staticmethod
    def create_gumbel_muzero_recurrent_fn(model_params: Dict[str, Any]) -> Callable:
        """
        Create a Gumbel MuZero-style recurrent function.
        
        Args:
            model_params: Neural network parameters
            
        Returns:
            RecurrentFn function compatible with MCTX
        """
        # For Gumbel MuZero, we can use the same architecture as MuZero
        # The difference is in the action selection, not the model
        return RecurrentFnService.create_muzero_recurrent_fn(model_params)
    
    @staticmethod
    def create_bandit_recurrent_fn(model_params: Dict[str, Any]) -> Callable:
        """
        Create a simple bandit model recurrent function.
        
        This is a deterministic bandit with known reward values.
        
        Args:
            model_params: Parameters including reward values
            
        Returns:
            RecurrentFn function compatible with MCTX
        """
        # Extract parameters
        qvalues = model_params.get("qvalues")
        if qvalues is None:
            raise ValueError("qvalues must be provided for bandit model")
        
        qvalues = jnp.array(qvalues)
        
        def recurrent_fn(params, rng_key, action, embedding):
            """
            Bandit recurrent function.
            
            For a bandit, the state doesn't change, but we still need to
            return a valid RecurrentFnOutput.
            
            Args:
                params: Not used for bandit
                rng_key: Not used for deterministic bandit
                action: Selected actions
                embedding: Used to track depth (0 = root)
                
            Returns:
                Tuple of (RecurrentFnOutput, next_embedding)
            """
            batch_size = action.shape[0]
            num_actions = qvalues.shape[-1]
            
            # For bandit, reward is the Q-value of the selected action,
            # but only at the root (embedding == 0)
            reward = jnp.where(
                embedding == 0,
                qvalues[jnp.arange(batch_size), action],
                jnp.zeros_like(embedding, dtype=jnp.float32)
            )
            
            # Always use discount=1.0
            discount = jnp.ones_like(reward)
            
            # Zero prior_logits and value for simplicity
            prior_logits = jnp.zeros((batch_size, num_actions), dtype=jnp.float32)
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
    
    @staticmethod
    def create_chess_recurrent_fn(model_params: Dict[str, Any]) -> Callable:
        """
        Create a recurrent function for a chess-like board game.
        
        Args:
            model_params: Neural network parameters and rules
            
        Returns:
            RecurrentFn function compatible with MCTX
        """
        # Extract model parameters
        board_size = model_params.get("board_size", 8)
        embedding_size = model_params.get("embedding_size", 128)
        use_legal_actions = model_params.get("use_legal_actions", True)
        
        # This would be a neural network in a real implementation
        # Here we create a simple stub with the right API
        def get_board_embedding(board_state):
            """Convert board state to embedding."""
            # In a real implementation, this would use a neural network
            # Return dummy embedding of the right shape
            batch_size = board_state.shape[0]
            return jnp.zeros((batch_size, embedding_size))
        
        def get_legal_moves(board_state):
            """Get legal moves for each board state."""
            # In a real implementation, this would use game rules
            # Return dummy legal moves mask
            batch_size = board_state.shape[0]
            num_actions = board_size * board_size
            return jnp.ones((batch_size, num_actions), dtype=jnp.bool_)
        
        def get_policy_and_value(embedding):
            """Get policy and value predictions."""
            # In a real implementation, this would use a neural network
            batch_size = embedding.shape[0]
            num_actions = board_size * board_size
            # Return uniform policy and zero value
            return (
                jnp.zeros((batch_size, num_actions)),
                jnp.zeros((batch_size,))
            )
        
        def apply_action(board_state, action):
            """Apply action to board state."""
            # In a real implementation, this would use game rules
            # Return dummy next state
            return board_state
        
        def compute_reward(board_state, action, next_board_state):
            """Compute reward for transition."""
            # In a real implementation, this would use game rules
            # Return zero reward
            batch_size = board_state.shape[0]
            return jnp.zeros((batch_size,))
        
        def is_terminal(board_state):
            """Check if board state is terminal."""
            # In a real implementation, this would use game rules
            # Return all False (not terminal)
            batch_size = board_state.shape[0]
            return jnp.zeros((batch_size,), dtype=jnp.bool_)
        
        def recurrent_fn(params, rng_key, action, embedding):
            """
            Chess-like game recurrent function.
            
            Args:
                params: Neural network parameters
                rng_key: JAX random key
                action: Selected actions
                embedding: State embeddings or board states
                
            Returns:
                Tuple of (RecurrentFnOutput, next_embedding)
            """
            # Apply action to get next state
            next_board_state = apply_action(embedding, action)
            
            # Compute reward
            reward = compute_reward(embedding, action, next_board_state)
            
            # Check for terminal states
            terminal = is_terminal(next_board_state)
            discount = 1.0 - terminal
            
            # Get policy and value for next state
            next_embedding = get_board_embedding(next_board_state)
            prior_logits, value = get_policy_and_value(next_embedding)
            
            # If using legal actions, mask out illegal moves
            if use_legal_actions:
                legal_mask = get_legal_moves(next_board_state)
                prior_logits = jnp.where(
                    legal_mask, prior_logits, -1e9 * jnp.ones_like(prior_logits)
                )
            
            # Create RecurrentFnOutput
            recurrent_fn_output = mctx.RecurrentFnOutput(
                reward=reward,
                discount=discount,
                prior_logits=prior_logits,
                value=value
            )
            
            return recurrent_fn_output, next_board_state
        
        return recurrent_fn