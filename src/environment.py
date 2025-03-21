import numpy as np

from .utils import (
	has_consecutive_values, find_consecutive_sequences, get_winning_lines
)


class Environment:
    """
    Game environment for N-in-a-row game (like Tic-tac-toe or Gomoku).
    
    State representation:
    - 0: Empty cell
    - 1: Player 1's piece
    - -1: Player 2's piece
    - 0.5: First move under pie rule
    """

    def __init__(self, size, N, pie_rule=False):
        """
        Initialize game environment.
        
        Example:
        env = Environment(size=(3,3), N=3)  # 3x3 board, need 3 in a row to win
        """
        self.size = size
        self.w, self.h = size
        self.N = N  # Number needed in a row to win

        # Validate game parameters
        if (
            self.w < 0
            or self.h < 0
            or self.N < 2
            or (self.N > self.w and self.N > self.h)
        ):
            raise ValueError(
                "Game cannot initialize with a {0:d}x{1:d} grid, and winning condition {2:d} in a row".format(
                    self.w, self.h, self.N
                )
            )

        self.score = None  # None: ongoing, 0: draw, 1/-1: winner
        self.state = np.zeros(size, dtype=float)  # Game board
        self.player = 1  # Current player (1 or -1)
        self.last_move = None
        self.n_moves = 0
        self.pie_rule = pie_rule  # Whether pie rule is enabled
        self.switched_side = False

    def __copy__(self):
        """Create a deep copy of the game state."""
        cls = self.__class__
        new_game = cls.__new__(cls)
        new_game.__dict__.update(self.__dict__)

        new_game.N = self.N
        new_game.pie_rule = self.pie_rule
        new_game.state = self.state.copy()
        new_game.switched_side = self.switched_side
        new_game.n_moves = self.n_moves
        new_game.last_move = self.last_move
        new_game.player = self.player
        new_game.score = self.score
        return new_game

    def get_score(self):
        """
        Check if game is over and return score.
        
        Returns:
            None: Game ongoing
            0: Draw
            1: Player 1 wins
            -1: Player 2 wins
        """
        # Too few moves for anyone to win
        if self.n_moves < 2 * self.N - 1:
            return None

        i, j = self.last_move
        hor, ver, diag_right, diag_left = get_winning_lines(self.state, (i, j))

        # Check all lines through last move for winner
        for line in [ver, hor, diag_right, diag_left]:
            if has_consecutive_values(line, self.N, self.player):
                return self.player

        # Check for draw (board full)
        if np.all(self.state != 0):
            return 0

        return None

    def get_winning_loc(self):
        """
        Get coordinates of winning line.
        
        Returns:
            List of (x,y) coordinates forming winning line
            Empty list if no winner
        """
        if self.n_moves < 2 * self.N - 1:
            return []

        loc = self.last_move
        hor, ver, diag_right, diag_left = get_winning_lines(self.state, loc)
        ind = np.indices(self.state.shape)
        ind = np.moveaxis(ind, 0, -1)
        hor_ind, ver_ind, diag_right_ind, diag_left_ind = get_winning_lines(ind, loc)

        pieces = [hor, ver, diag_right, diag_left]
        indices = [hor_ind, ver_ind, diag_right_ind, diag_left_ind]

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

    def move(self, loc):
        """
        Make a move at the specified location.
        
        Example:
        env.move((1,1))  # Place piece at center of 3x3 board
        
        Returns:
            bool: True if move was valid and made, False otherwise
        """
        i, j = loc
        success = False
        if self.w > i >= 0 and self.h > j >= 0:
            if self.state[i, j] == 0:
                # Make normal move
                self.state[i, j] = self.player

                # Handle pie rule
                if self.pie_rule:
                    if self.n_moves == 1:
                        self.state[tuple(self.last_move)] = -self.player
                        self.switched_side = False
                    elif self.n_moves == 0:
                        # First move under pie rule is marked as 0.5
                        self.state[i, j] = self.player / 2.0
                        self.switched_side = False

                success = True

            # Handle pie rule switch
            elif self.pie_rule and self.state[i, j] == -self.player / 2.0:
                self.state[i, j] = self.player
                self.switched_side = True
                success = True

        if success:
            self.n_moves += 1
            self.last_move = tuple((i, j))
            self.score = self.get_score()

            # Switch players if game continues
            if self.score is None:
                self.player *= -1

            return True

        return False

    def available_moves(self):
        """
        Get list of available moves.
        
        Returns:
            Array of (x,y) coordinates where moves are possible
        """
        indices = np.moveaxis(np.indices(self.state.shape), 0, -1)
        return indices[np.abs(self.state) != 1]

    def available_mask(self):
        """
        Get binary mask of available moves.
        
        Returns:
            Array of same shape as board with:
            1: Move possible
            0: Move not possible
        """
        return (np.abs(self.state) != 1).astype(np.uint8)