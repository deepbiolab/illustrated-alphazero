import torch
import torch.optim as optim
from copy import copy
from typing import Dict, Tuple, Any

from .search import MonteCarloTree
from .network import Policy
from .environment import Environment

class MCTSAgent:
    """
    An agent that uses Monte Carlo Tree Search with a neural network policy
    for playing and learning games.
    
    The agent combines:
    - Neural network policy for move evaluation
    - MCTS for look-ahead search
    - Self-play for training
    """
    def __init__(self, env_settings: Dict[str, Any], learning_rate: float = 0.01, 
                 weight_decay: float = 1.0e-4) -> None:
        """
        Initialize the MCTS agent.
        
        Example env_settings:
        {
            "size": (3, 3),  # Board size
            "N": 3,          # Number in a row to win
        }
        
        Args:
            env_settings: Dictionary containing environment settings
            learning_rate: Learning rate for the optimizer (default: 0.01)
            weight_decay: L2 regularization parameter (default: 1.0e-4)
        """
        self.env_settings = env_settings
        # Initialize neural network policy
        self.policy = Policy(env=Environment(**env_settings))
        # Setup Adam optimizer with L2 regularization
        self.optimizer = optim.Adam(
            self.policy.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )

    def select_action(self, env: Environment, num_simulations: int = 50, 
                     temperature: float = 0.1) -> Tuple[int, int]:
        """
        Select an action using MCTS with neural network policy guidance.
        
        Example:
        If playing TicTacToe, might return:
        - (3, 3) representing move at coordinates (3,3) on the board
        - (pass_move, pass_move) for a pass move
        
        Args:
            env: The game environment
            num_simulations: Number of MCTS simulations to run (default: 50)
            temperature: Controls exploration vs exploitation:
                        - Lower (e.g., 0.1): More exploitation (picks best moves)
                        - Higher (e.g., 1.0): More exploration (more random)
        
        Returns:
            tuple: The selected move as (x, y) coordinates
        """
        # Create fresh search tree from current game state
        root_node = MonteCarloTree(copy(env))
        
        # Perform MCTS simulations to build search tree
        for _ in range(num_simulations):
            root_node.explore(self.policy)
        
        # Select next move based on visit counts and temperature
        next_node, _ = root_node.next(temperature=temperature)
        return next_node.game.last_move

    def learn(self, num_simulations: int = 100) -> Tuple[float, float]:
        """
        Perform one episode of learning through self-play.
        
        The learning process:
        1. Uses MCTS to play a complete game
        2. At each move:
           - Collects policy targets (MCTS visit counts)
           - Collects value targets (final game outcome)
        3. Updates neural network to predict both policy and value
        
        Example outcomes:
        - 1.0: First player wins
        - -1.0: Second player wins
        - 0.0: Draw
        
        Example loss computation:
        value_loss = (predicted_value - actual_outcome)^2
        policy_loss = -Σ(target_prob * log(predicted_prob))
        total_loss = value_loss + policy_loss
        
        Args:
            num_simulations: Number of MCTS simulations per move (default: 100)
            
        Returns:
            tuple: (episode_outcome, episode_loss)
                  e.g., (1.0, 2.34) for win with loss of 2.34
        """
        # Initialize new game and tracking variables
        search_tree = MonteCarloTree(Environment(**self.env_settings))
        value_terms: list[torch.Tensor] = []    # List to store value predictions: [0.7, 0.3, ...]
        policy_terms: list[torch.Tensor] = []   # List to store policy losses for each move

        # game loop - continue until game ends
        while search_tree.outcome is None:
            # Build search tree through MCTS simulations
            for _ in range(num_simulations):
                search_tree.explore(self.policy)

            # Store whose perspective we're learning from (1 or -1)
            current_player = search_tree.game.player
            
            # Make move and get training targets
            search_tree, (value, pred_value, mcts_probs, prior_probs) = search_tree.next()
            search_tree.detach_parent()  # Memory optimization

            # Compute losses for this move
            # Policy loss calculation
            log_probs = torch.log(prior_probs) * mcts_probs
            
            # Importance sampling correction term
            constant = torch.where(
                mcts_probs > 0,                         # Where MCTS visited
                mcts_probs * torch.log(mcts_probs),     # KL divergence term
                torch.tensor(0.0)                       # Zero for unvisited moves
            )
            
            # Store policy loss for this move
            policy_terms.append(-torch.sum(log_probs - constant))
            
            # Store value prediction from current player's perspective
            value_terms.append(pred_value * current_player)

        # Game is finished - compute total loss
        outcome = search_tree.outcome  # Final result: +1, -1, or 0
        
        # - Value loss: (predicted - actual)^2 for each move
        # - Policy loss: Sum of cross entropy losses
        loss = torch.sum(
            (torch.stack(value_terms) - outcome) ** 2 +  # Value loss term
            torch.stack(policy_terms)                    # Policy loss term
        )
        
        # Update network weights
        self.optimizer.zero_grad()
        loss.backward()       
        self.optimizer.step()

        return outcome, float(loss)

    def save_model(self, path: str) -> None:
        """
        Save the policy network to a file.
        
        Example path: 'checkpoints/best_model.pth'
        """
        torch.save(self.policy.state_dict(), path)

    def load_model(self, path: str) -> None:
        """
        Load the policy network from a file.
        
        Example path: 'checkpoints/best_model.pth'
        """
        self.policy.load_state_dict(torch.load(path, weights_only=True))