import numpy as np

def get_runs(v, i):
    """
    Find continuous runs of a value in a vector.
    
    Example:
    v = [0,0,1,1,1,0,0], i = 1
    Returns: 
        starts = [2]    # where runs start
        ends = [5]      # where runs end
        lengths = [3]   # length of runs
    """
    bounded = np.hstack(([0], (v == i).astype(int), [0]))
    difs = np.diff(bounded)
    (starts,) = np.where(difs > 0)
    (ends,) = np.where(difs < 0)
    return starts, ends, ends - starts


def in_a_row(v, N, i):
    """
    Check if vector contains N consecutive occurrences of i.
    
    Example:
    v = [0,1,1,1,0], N = 3, i = 1
    Returns: True (has 3 ones in a row)
    """
    if len(v) < N:
        return False
    else:
        _, _, total = get_runs(v, i)
        return np.any(total >= N)


def get_lines(matrix, loc):
    """
    Get all lines (horizontal, vertical, diagonal) passing through loc.
    
    Example for 3x3 matrix with loc=(1,1):
    Returns 4 lines:
    - horizontal: matrix[1,:]
    - vertical: matrix[:,1]
    - diagonal right: top-left to bottom-right diagonal
    - diagonal left: top-right to bottom-left diagonal
    """
    i, j = loc
    flat = matrix.reshape(-1, *matrix.shape[2:])

    w = matrix.shape[0]
    h = matrix.shape[1]

    def flat_pos(pos):
        return pos[0] * h + pos[1]

    pos = flat_pos((i, j))

    # Calculate complementary positions
    ic = w - 1 - i
    jc = h - 1 - j

    # Calculate diagonal endpoints
    tl = (i - j, 0) if i > j else (0, j - i)
    tl = flat_pos(tl)

    bl = (w - 1 - (ic - j), 0) if ic > j else (w - 1, j - ic)
    bl = flat_pos(bl)

    tr = (i - jc, h - 1) if i > jc else (0, h - 1 - (jc - i))
    tr = flat_pos(tr)

    br = (w - 1 - (ic - jc), h - 1) if ic > jc else (w - 1, h - 1 - (jc - ic))
    br = flat_pos(br)

    # Extract lines
    hor = matrix[:, j]  # Horizontal line
    ver = matrix[i, :]  # Vertical line
    diag_right = np.concatenate([flat[tl : pos : h + 1], flat[pos : br + 1 : h + 1]])
    diag_left = np.concatenate([flat[tr : pos : h - 1], flat[pos : bl + 1 : h - 1]])

    return hor, ver, diag_right, diag_left


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
        hor, ver, diag_right, diag_left = get_lines(self.state, (i, j))

        # Check all lines through last move for winner
        for line in [ver, hor, diag_right, diag_left]:
            if in_a_row(line, self.N, self.player):
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
        hor, ver, diag_right, diag_left = get_lines(self.state, loc)
        ind = np.indices(self.state.shape)
        ind = np.moveaxis(ind, 0, -1)
        hor_ind, ver_ind, diag_right_ind, diag_left_ind = get_lines(ind, loc)

        pieces = [hor, ver, diag_right, diag_left]
        indices = [hor_ind, ver_ind, diag_right_ind, diag_left_ind]

        for line, index in zip(pieces, indices):
            starts, ends, runs = get_runs(line, self.player)

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