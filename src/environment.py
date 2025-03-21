import numpy as np

from .utils import (
	has_consecutive_values, find_consecutive_sequences, get_winning_lines
)


class Environment:
    """Game environment for N-in-a-row game (like Tic-tac-toe or Gomoku).
    
    Game Rules:
        1. Players take turns placing pieces on empty cells
        2. Player 1 uses piece value 1, Player 2 uses -1
        3. First player to get N pieces in a row (horizontal, vertical, or diagonal) wins
        4. Game ends in draw if board is full with no winner
    
    Args:
        size (tuple): Board dimensions as (width, height)
        N (int): Number of pieces needed in a row to win
    
    Example:
        ```python
        # Create a standard tic-tac-toe game (3x3, need 3 in a row)
        env = Environment(size=(3,3), N=3)
        
        # Make moves
        env.move((1,1))  # Player 1 takes center
        env.move((0,0))  # Player 2 takes top-left
        ```
    """

    def __init__(self, size, N):
        """
        Initialize game environment for N-in-a-row game.
        
        Attributes:
            w (int): Board width
            h (int): Board height
            state (np.ndarray): Game board state matrix:
                - 0: Empty cell
                - 1: Player 1's piece
                - -1: Player 2's piece
            player (int): Current player (1 or -1)
            n_moves (int): Number of moves made in the game
            last_move (tuple): Coordinates of the last move made (i,j)
            score (None|int): Game status:
                - None: Game ongoing
                - 0: Draw
                - 1: Player 1 wins
                - -1: Player 2 wins
        """
        self.size = size
        self.N = N
        self.w, self.h = size

        self._validate_game_parameters()

        # Initialize game board as empty (all zeros)
        self.state = np.zeros(size, dtype=float)
        
        # Player 1 starts the game
        self.player = 1
        
        # Game state tracking
        self.n_moves = 0        # Number of moves made
        self.last_move = None   # Coordinates of last move
        self.score = None       # Game outcome

    def _validate_game_parameters(self):
        """
        Validate game initialization parameters.
        
        Raises:
            ValueError: If game parameters are invalid
        """
        if self.w < 0 or self.h < 0:
            raise ValueError(f"Board dimensions must be positive, got {self.w}x{self.h}")
        
        if self.N < 2:
            raise ValueError(f"Winning condition must be at least 2, got {self.N}")
        
        if self.N > max(self.w, self.h):
            raise ValueError(
                f"Winning condition {self.N} cannot be larger than both board dimensions {self.w}x{self.h}"
            )

    @staticmethod
    def _calculate_min_moves_for_win(N):
        """
        Calculate minimum moves needed for a win to be possible.
        
        For N pieces in a row:
        - Player 1 needs N moves
        - Player 2 needs (N-1) moves
        Total minimum moves = 2N-1
        
        Args:
            N (int): Number of pieces needed in a row to win
            
        Returns:
            int: Minimum number of moves needed
        """
        return 2 * N - 1

    def _check_minimum_moves_for_win(self):
        """
        Check if enough moves have been made for a win to be possible.
        
        Example:
            For 3-in-a-row (N=3), need at least 5 moves total:
            Player 1: 3 moves, Player 2: 2 moves
        
        Returns:
            bool: True if enough moves have been made, False otherwise
        """
        min_moves = self._calculate_min_moves_for_win(self.N)
        return self.n_moves >= min_moves

    @property
    def is_game_over(self):
        """Check if game has ended (win or draw)."""
        return self.score is not None

    def _is_valid_move(self, i, j):
        """Check if move is valid (in bounds and cell is empty)."""
        return (
            0 <= i < self.w and 
            0 <= j < self.h and 
            self.state[i, j] == 0
        )

    def get_score(self):
        """
        Check if game is over and return score.
        
        The game can end in three ways:
        1. Current player wins by connecting N pieces
        2. Board is full with no winner (draw)
        3. Game is still ongoing
        
        Process:
        1. First check if minimum moves for win is reached
        2. Check all possible winning lines through last move
        3. Check for draw if no winner found
        
        Returns:
            int or None: Game outcome
                1: Player 1 wins
                -1: Player 2 wins
                0: Draw (board full)
                None: Game ongoing
        
        Example:
            >>> env = Environment(size=(3,3), N=3)
            >>> env.move((1,1))  # Center
            >>> env.move((0,0))  # Top-left
            >>> env.get_score()
            None  # Game still ongoing
        """
        # Not enough moves made for any player to win
        if not self._check_minimum_moves_for_win():
            return None

        # Check all possible winning lines through the last move:
        # - Horizontal line (row)
        # - Vertical line (column)
        # - Diagonal (top-left to bottom-right)
        # - Diagonal (top-right to bottom-left)
        for line in get_winning_lines(self.state, self.last_move):
            # For each line, check if current player has N consecutive pieces
            if has_consecutive_values(line, self.N, self.player):
                return self.player  # Current player wins

        # If no winning line found, check for draw
        # Draw occurs when board is full (no empty cells)
        if np.all(self.state != 0):
            return 0  # Draw

        # Game is still ongoing
        return None

    def move(self, loc):
        """
        Make a move at the specified location.
        
        Args:
            loc (tuple): (x,y) coordinates where to place the piece
            
        Returns:
            bool: True if move was valid and made, False otherwise
            
        Example:
            ```python
            # Place piece at center of 3x3 board
            env.move((1,1))
            ```
        """
        i, j = loc
        
        # Validate move
        if not self._is_valid_move(i, j):
            return False
            
        # Make move
        self.state[i, j] = self.player
        self.n_moves += 1
        self.last_move = (i, j)
        
        # Update game state
        self.score = self.get_score()
        
        # Switch players if game continues
        if not self.is_game_over:
            self.player *= -1
            
        return True

    def _get_coordinate_matrix(self):
        """
        Generate matrix of (x,y) coordinates for each board position.
        
        Returns:
            np.ndarray: Matrix of shape (width, height, 2) containing coordinates
        """
        i, j = np.meshgrid(
            np.arange(self.state.shape[0]), 
            np.arange(self.state.shape[1]), 
            indexing='ij'
        )
        return np.dstack([i, j])

    def available_moves(self):
        """
        Get list of available moves.
        
        Returns:
            np.ndarray: Array of (x,y) coordinates where moves are possible
        """
        coord_matrix = self._get_coordinate_matrix()
        return coord_matrix[self.available_mask() == 1]

    def available_mask(self):
        """
        Get binary mask of available moves.
        
        Returns:
            Array of same shape as board with:
            1: Move possible
            0: Move not possible
        """
        return (np.abs(self.state) != 1).astype(np.uint8)

    def get_winning_loc(self):
        """
        Get coordinates of winning line.
        
        Returns:
            List of (x,y) coordinates forming winning line
            Empty list if no winner
        """
        if not self._check_minimum_moves_for_win():
            return []

        # i = [[0, 0, 0],
        #      [1, 1, 1],
        #      [2, 2, 2]]
        # j = [[0, 1, 2],
        #      [0, 1, 2],
        #      [0, 1, 2]]
        i, j = np.meshgrid(np.arange(self.state.shape[0]), 
                           np.arange(self.state.shape[1]), 
                           indexing='ij')
        # [[[0 0] [0 1] [0 2]]
        #  [[1 0] [1 1] [1 2]]
        #  [[2 0] [2 1] [2 2]]]
        index_matrix = np.dstack([i, j])

        pieces = get_winning_lines(self.state, self.last_move)
        indices = get_winning_lines(index_matrix, self.last_move)

        for line, index in zip(pieces, indices):
            starts, ends, runs = find_consecutive_sequences(line, self.player)

            winning = runs >= self.N
            if not np.any(winning):
                continue

            starts_ind = starts[winning][0]
            ends_ind = ends[winning][0]
            indices = index[starts_ind:ends_ind]
            return indices

        return []

    def __copy__(self):
        """Create a deep copy of the game state."""
        cls = self.__class__
        new_game = cls.__new__(cls)
        new_game.__dict__.update(self.__dict__)

        new_game.N = self.N
        new_game.state = self.state.copy()
        new_game.n_moves = self.n_moves
        new_game.last_move = self.last_move
        new_game.player = self.player
        new_game.score = self.score
        return new_game

