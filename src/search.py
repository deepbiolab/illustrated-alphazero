import torch
import random
from copy import copy
from math import sqrt
from typing import Optional, Tuple, List
from torch import Tensor

from .transform import BoardTransform
from .environment import Environment
from .network import Policy

device = "cpu"

class MonteCarloTree:
    """
    Monte Carlo Tree Search implementation for AlphaZero algorithm.

    This class implements the MCTS algorithm used in AlphaZero, combining
    neural network evaluation with tree search.

    Attributes:
        game (Environment): Current game state
        child (Dict[Tuple[int, ...], 'MonteCarloTree']): Child nodes
        U (float): Upper Confidence Bound score
        N (int): Visit count
        V (float): Expected value [-1, 1]
        pred_V (Tensor): Neural network value prediction
        prior_prob (Tensor): Prior probability from policy
        outcome (Optional[int]): Game result if terminal
        parent (Optional[MonteCarloTree]): Parent node

    Examples:
        >>> game = Environment()
        >>> mcts = MonteCarloTree(game)
        >>> for _ in range(100):
        >>>     mcts.explore(policy)
        >>> next_state, info = mcts.next(temperature=0.1)

    Notes:
        - UCB formula: V + C * P * sqrt(N_parent) / (1 + N)
        - Values are always from parent player's perspective

    References:
        - Silver, D. et al. (2017). Mastering Chess and Shogi by Self-Play
          with a General Reinforcement Learning Algorithm.
    """

    def __init__(
        self,
        game: Environment,
        parent: Optional["MonteCarloTree"] = None,
        prior_prob: Tensor = torch.tensor(0.0, dtype=torch.float),
    ) -> None:
        """
        Initialize a Monte Carlo Tree Search node.

        Args:
            game (Environment): The game state this node represents
            parent (Optional[MonteCarloTree]): Parent node in the MCTS tree (None for root)
            prior_prob (Tensor): Prior probability from policy network for reaching this node
        """
        # Current game state
        self.game = game

        # Dictionary of child nodes, mapping actions to nodes
        self.child = {}

        # Upper Confidence Bound (UCB) score for this node
        # Used in selection phase to balance exploration vs exploitation
        self.U = 0

        # Number of times this node has been visited during search
        # Used to calculate UCB and determine move selection
        self.N = 0

        # Expected value from MCTS simulations
        # Average of all simulation outcomes through this node
        self.V = 0

        # Probability from policy network for reaching this node
        self.prior_prob = prior_prob

        # Predicted value from policy network for this node
        self.pred_V = torch.tensor(0.0, dtype=torch.float)

        # Terminal state indicator
        # None: game not finished
        # 0: draw
        # 1: win for player who just moved
        # -1: loss for player who just moved
        self.outcome = self.game.score

        # If game has ended, set terminal values
        if self.game.score is not None:
            # Set value based on game outcome and current player
            self.V = self.game.score * self.game.player

            # Set UCB:
            # - For draws (score=0): U = 0
            # - For wins/losses: U = ±inf to ensure selection/avoidance
            self.U = 0 if self.game.score == 0 else self.V * float("inf")

        # Reference to parent node for backpropagation
        # None for root node of the tree
        self.parent = parent

        # Parameter for balance between exploration and exploitation in UCB
        self.C = 1.0

    def selection(self):
        """
        Selection phase: traverse the tree from the current node to a leaf node
        by selecting nodes with the highest UCB value.

        Returns:
            MonteCarloTree: selected leaf node
            bool: True if special termination condition was met
        """
        current = self
        terminate = False

        while current.child and current.outcome is None:
            child = current.child
            max_U = max(c.U for c in child.values())
            actions = [a for a, c in child.items() if c.U == max_U]

            if len(actions) == 0:
                print("error zero length ", max_U)
                print(current.game.state)
                return current, True

            action = random.choice(actions)

            if max_U == -float("inf"):
                current.U = float("inf")
                current.V = 1.0
                terminate = True
                break

            elif max_U == float("inf"):
                current.U = -float("inf")
                current.V = -1.0
                terminate = True
                break

            current = child[action]

        return current, terminate

    def expansion(self, policy: Policy):
        """
        Expansion phase: create children for the current node if it's a leaf
        and the game isn't over.

        Args:
            policy: Neural network policy model

        Returns:
            tuple: (next_actions, prior_probs, pred_value)
            if expansion happened, None otherwise
        """
        if not self.child and self.outcome is None:
            # Get actions, probabilities and value from policy network
            next_actions, prior_probs, pred_value = BoardTransform.evaluate_policy(
                policy, self.game, device=device
            )

            # Create children nodes
            self.create_child(next_actions, prior_probs)

            return next_actions, prior_probs, pred_value

        return None

    def evaluation(self, pred_value: Tensor) -> None:
        """
        Evaluation phase: set the value of this node based on the policy evaluation.

        Args:
            pred_value (Tensor): Value prediction from policy network,
                typically in range [-1, 1]
                where 1 indicates win, 0 draw, and -1 loss

        Returns:
            None
        """
        self.pred_V = -pred_value
        self.V = -float(pred_value)

    def backpropagation(self):
        """
        Backpropagation phase: update statistics for all nodes from this node to root.
        """
        current = self
        current.N += 1

        while current.parent:
            parent = current.parent
            parent.N += 1

            # Update parent's value estimate
            parent.V += (-current.V - parent.V) / parent.N

            # Update UCB values for all siblings
            for sibling in parent.child.values():
                if sibling.U is not float("inf") and sibling.U is not -float("inf"):
                    sibling.U = sibling.V + self.C * float(sibling.prior_prob) * sqrt(
                        parent.N
                    ) / (1 + sibling.N)

            current = parent

    def create_child(self, 
                    actions: List[List[int]], 
                    prior_probs: Tensor) -> None:
        """
        Create child nodes for all possible actions from current state.
        
        This method expands the current node by:
        1. Creating copies of current game state for each action
        2. Applying each action to its respective game copy
        3. Creating child MCTS nodes with the new game states
        4. Storing children in a dictionary mapped by action tuples
        
        Args:
            actions (List[List[int]]): List of valid moves, where each move is 
                                    represented as [row, col] coordinates
            prior_probs (Tensor): Prior probabilities for each action from policy network.
                                Shape: [num_actions], sum to 1.0
        
        Returns:
            None: Updates self.child dictionary in place
        
        Example:
            actions = [[0, 0], [0, 1]]  # Two possible moves
            prior_probs = tensor([0.6, 0.4])  # Their respective probabilities
            create_child(actions, prior_probs)  # Creates two child nodes
        """
        # Create copies of current game state for each possible action
        # Using _ since we don't need the action in the list comprehension
        games = [copy(self.game) for _ in actions]

        # Apply each action to its respective game copy
        for action, game in zip(actions, games):
            game.move(action)  # Updates game state with the move

        # Create dictionary mapping actions to new MCTS nodes
        child = {
            tuple(action): MonteCarloTree(game, self, prior_prob)
            for action, game, prior_prob in zip(actions, games, prior_probs)
        }
        
        # Update this node's children dictionary
        self.child = child

    def explore(self, policy: Policy) -> None:
        """
        Perform one iteration of Monte Carlo Tree Search using the policy network.
        
        This method implements the four key phases of MCTS:
        1. Selection: Traverse tree from root to leaf following UCB values
        2. Expansion: Create child nodes for all possible moves from leaf
        3. Evaluation: Get value prediction from policy network
        4. Backpropagation: Update statistics back up the tree
        
        Args:
            policy (Policy): Neural network policy model that provides:
                - Prior probabilities for moves
                - Value estimation for positions
        
        Raises:
            ValueError: If called on a terminal game state
        
        Example flow:
            1. Start at root
            2. Select child nodes using UCB until leaf is reached
            3. If leaf is non-terminal, expand and evaluate with policy
            4. Backpropagate results to update node statistics
        """
        if self.game.score is not None:
            raise ValueError("game has ended with score {0:d}".format(self.game.score))

        # Selection phase
        leaf_node, terminated = self.selection()

        # If selection didn't terminate early due to special conditions
        if not terminated:
            # Expansion phase
            expansion_result = leaf_node.expansion(policy)

            # Evaluation phase (only if expansion happened)
            if expansion_result is not None:
                _, _, pred_value = expansion_result
                leaf_node.evaluation(pred_value)

        # Backpropagation phase
        leaf_node.backpropagation()

    def next(
        self, temperature: float = 1.0
    ) -> Tuple["MonteCarloTree", Tuple[float, Tensor, Tensor, Tensor]]:
        """
        Select the next move based on MCTS statistics and return relevant information.

        This method implements the final move selection after completing all MCTS simulations.
        It uses visit counts and temperature to compute selection probabilities.

        Args:
            temperature (float): Temperature parameter for visit count scaling.
                - T → 0: Deterministic selection of best move (exploitation)
                - T → ∞: Random selection (exploration)

        Returns:
            Tuple containing:
            - MonteCarloTree: Selected child node representing next game state
            - Tuple containing:
                - float: Negative MCTS value (-V) from current player's perspective
                - Tensor: Negative policy network prediction (-pred_V) from current player's perspective
                - Tensor: Move probabilities from MCTS visit counts (after temperature scaling)
                - Tensor: Prior probabilities from policy network

        Raises:
            ValueError: If game has ended or no valid moves are available
        """
        # Check if game has ended
        if self.game.score is not None:
            raise ValueError("game has ended with score {0:d}".format(self.game.score))

        # Ensure there are valid moves available
        if not self.child:
            print(self.game.state)
            raise ValueError("no children found and game hasn't ended")

        child = self.child

        # Get maximum UCB score among children
        max_U = max(c.U for c in child.values())

        # If winning moves exist, select only from those
        if max_U == float("inf"):
            mcts_probs = torch.tensor(
                [1.0 if c.U == float("inf") else 0 for c in child.values()],
                device=device,
            )
        else:
            # Convert visit counts to probabilities using temperature
            # Higher temperature → more uniform distribution
            # Lower temperature → more focused on most visited nodes
            max_N = (
                max(node.N for node in child.values()) + 1
            )  # +1 for numerical stability
            mcts_probs = torch.tensor(
                [(node.N / max_N) ** (1 / temperature) for node in child.values()],
                device=device,
            )

        # Normalize probabilities to sum to 1
        if torch.sum(mcts_probs) > 0:
            mcts_probs /= torch.sum(mcts_probs)
        else:
            # If all probabilities are zero, use uniform distribution
            mcts_probs = torch.tensor(1.0 / len(child), device=device).repeat(
                len(child)
            )

        # Get prior probabilities from policy network for all children
        prior_probs = torch.stack([node.prior_prob for node in child.values()]).to(device)

        # Randomly select next state based on MCTS probabilities
        next_state = random.choices(list(child.values()), weights=mcts_probs)[0]

        # Return selected state and information tuple
        # Note: Values are negated to convert from parent's perspective 
        # to current player's perspective
        return next_state, (-self.V, -self.pred_V, mcts_probs, prior_probs)

    def detach_parent(self) -> None:
        """
        Detach this node from its parent by removing the parent reference.
        This is typically called on the root node after selecting a move,
        to free memory by allowing the old tree to be garbage collected.

        The detached node becomes a new root node that can be used for
        future searches from the current game state.

        Returns:
            None
        """
        del self.parent  # Remove parent reference to break the tree connection
        self.parent = None  # Explicitly set to None to indicate this is now a root node
