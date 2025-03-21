import matplotlib.pyplot as plt
from collections import deque
import numpy as np
from typing import Tuple, Union
from numpy.typing import NDArray


def find_consecutive_sequences(
    v: NDArray, i: Union[int, float]
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Find continuous runs of a value in a vector.

    Example for `v = [0,0,1,1,1,0,0], i = 1`:
        ```
        Input v:     [0,0,1,1,1,0,0]
        binary:      [0,0,1,1,1,0,0]  # Convert to boolean/binary array
                     F,F,T,T,T,F,F    # where v == i
        padded:      [0,0,0,1,1,1,0,0,0]  # Add 0s at both ends
                     F,F,F,T,T,T,F,F,F
        diffs:       [0,0,1,0,0,-1,0,0]   # Difference between adjacent elements
                          ↑     ↑
                         start end
                          (2)  (5)
        starts:      [2]    # Where difs > 0  (0→1 transitions)
        ends:        [5]    # Where difs < 0  (1→0 transitions)
        lengths:     [3]    # ends - starts = 5 - 2 = 3
        ```

    Another example `v = [0,1,1,0,1,1,1], v=1` :
        ```
        Input v:     [0,1,1,0,1,1,1]
        binary:      [0,1,1,0,1,1,1]
        padded:      [0,0,1,1,0,1,1,1,0]
        diff:        [0,1,0,-1,1,0,0,-1]
                        ↑   ↑  ↑     ↑
                        s1  e1 s2    e2
        starts:      [1,4]    # Two sequences start
        ends:        [3,7]    # Two sequences end
        lengths:     [2,3]   # Sequence lengths
        ```
    """
    bounded = np.hstack(([0], (v==i).astype(int), [0]))
    diffs = np.diff(bounded)
    (starts,) = np.where(diffs > 0)
    (ends,) = np.where(diffs < 0)
    return starts, ends, ends - starts


def has_consecutive_values(v: NDArray, N: int, i: Union[int, float]) -> bool:
    """
    Check if vector contains N consecutive occurrences of i.

    Args:
        v: Input vector
        N: Number of consecutive values needed
        i: Value to check for

    Example:
        ```
        v = [0,1,1,1,0], N = 3, i = 1
        ```
    Returns: True (has 3 ones in a row)
    """
    # Convert to binary array where target value is 1, others 0
    binary = (v == i).astype(int)

    # Create kernel for convolution
    kernel = np.ones(N)

    # Convolve and check if any position has N consecutive values
    return np.any(np.convolve(binary, kernel, mode="valid") == N)


def get_winning_lines(
    matrix: NDArray, loc: Tuple[int, int]
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Get all continuous lines (horizontal, vertical, diagonal) passing through loc.

    Args:
        matrix: Game board matrix
        loc: (row, col) position to check

    Returns:
        tuple: (horizontal, vertical, diagonal_right, diagonal_left) lines

    Example for 3x3 matrix with `loc=(1,1)`:
        ```
        [
            [0, 1, 1],
            [1,-1, 0],
            [-1,1, 0]
        ]
        ```
    Returns the continuous lines through position `(1,1)`
    """
    row, col = loc
    height, width = matrix.shape[:2]

    # Horizontal and vertical lines
    horizontal = matrix[row, :]
    vertical   = matrix[:, col]

    # Create diagonal indices (for top-left to bottom-right diagonal)
    max_left   = min(row, col)  # For (1,1): min(1,1) = 1 steps up-left
    max_right  = min((height - 1) - row, (width - 1) - col)  # For (1,1) in 3x3: min(1,1) = 1 steps down-right
    rows_right = np.arange(row - max_left, row + max_right + 1)  # For (1,1): [0,1,2] row indices
    cols_right = np.arange(col - max_left, col + max_right + 1)  # For (1,1): [0,1,2] col indices
    diag_right = matrix[rows_right, cols_right]  # For (1,1): [matrix[0,0], matrix[1,1], matrix[2,2]]

    # Anti-diagonal indices (for top-right to bottom-left diagonal)
    max_up     = min(row, width - col - 1)  # For (1,1): min(1,1) = 1 steps up-right
    max_down   = min(height - row - 1, col)  # For (1,1): min(1,1) = 1 steps down-left
    rows_left  = np.arange(row - max_up, row + max_down + 1)  # For (1,1): [0,1,2] row indices
    cols_left  = np.arange(col + max_up, col - max_down - 1, -1)  # For (1,1): [2,1,0] col indices
    diag_left  = matrix[rows_left, cols_left]  # For (1,1): [matrix[0,2], matrix[1,1], matrix[2,0]]

    return horizontal, vertical, diag_right, diag_left


def plot_losses(losses, window=50):
    """
    Plot training losses with moving average.

    Args:
        losses: List of training loss values
        window: Size of moving average window (default: 50)
    """
    # Calculate moving average
    moving_averages = []
    window_values = deque(maxlen=window)

    for loss in losses:
        window_values.append(loss)
        moving_averages.append(np.mean(list(window_values)))

    # Create figure
    fig = plt.figure(figsize=(12, 6))

    # Plot raw loss values
    plt.plot(losses, alpha=0.3, color="blue", label="Raw Losses")

    # Plot moving average
    plt.plot(
        moving_averages,
        color="red",
        linewidth=2,
        label=f"Moving Average (window={window})",
    )

    # Set plot labels and properties
    plt.title("Training Losses")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.show()
