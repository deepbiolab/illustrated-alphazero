# Illustrated AlphaZero

A Python implementation of the AlphaZero algorithm for board games, with detailed illustrations and explanations.

<div style="display:flex; justify-content:center; align-items:center; flex-wrap:wrap;">
  <div style="display:flex; flex-direction:column; margin-right:20px;">
  	<div style="text-align:center; margin-bottom:20px;">
  		<img src="./assets/3x3-game-ai-ai.gif" style="width:200px; height:200px; object-fit:contain;" />
   	</div>
  	<div style="text-align:center;">
      <img src="./assets/3x3-game-human-ai.gif" style="width:200px; height:200px; object-fit:contain;" />
	</div>
    <p style="text-align:center; margin-top:10px;"><strong>TicTacToe (3×3)</strong></p>
  </div>
  <div style="text-align:center;">
    <img src="./assets/7x4-game.gif" style="width:400px; height:430px; object-fit:contain;" />
    <p><strong>Connect Four (7×6)</strong></p>
  </div>
</div>

A detailed blog-style explanation of the AlphaZero algorithm can be found [here](blog/dive-into-alphazero.pdf).

<img src="./assets/blog-cover.png" style="zoom: 50%;" >


## Overview

![arch](./assets/arch.svg)

AlphaZero is a groundbreaking algorithm that combines Monte Carlo Tree Search (MCTS) with deep neural networks to achieve superhuman performance in board games. This project provides a detailed implementation of AlphaZero, including the core components and a user-friendly interface for interactive play.

This project implements the core concepts of the AlphaZero algorithm, featuring:
- Monte Carlo Tree Search (MCTS)
- Deep Neural Network Policy
- Self-play Training
- Board Game Environment (e.g., TicTacToe)

## Project Structure

```
illustrated-alphazero/
├── src/
│   ├── agent.py        # MCTS Agent implementation
│   ├── config.py       # Configuration settings
│   ├── environment.py  # Game environment
│   ├── interact.py     # GUI for human interaction
│   ├── network.py      # Neural network architecture
│   ├── search.py       # Monte Carlo Tree Search
│   ├── transform.py    # Board state transformations
│   └── utils.py        # Utility functions
├── checkpoints/        # Model checkpoints
├── main.py             # Training entry point
├── playground.py       # Interactive game environment
└── README.md
```

## Key Components

### MCTS Search
- Selection: Choose promising nodes using UCB scores
- Expansion: Create child nodes for unexplored states
- Evaluation: Use neural network to evaluate positions
- Backpropagation: Update node statistics

### Neural Network
- Policy Head: Predicts move probabilities
- Value Head: Estimates position value
- Convolutional Features: Extracts board patterns

### Self-play Training
- Game simulation through MCTS
- Data collection from self-play games
- Neural network training with collected examples


## Installation

```bash
git clone https://github.com/deepbiolab/illustrated-alphazero.git
cd illustrated-alphazero
pip install -r requirements.txt
```

## Usage


### Training AlphaZero

```bash
python main.py
```

The `main.py` script trains the AlphaZero model using the specified configuration. It simulates self-play games, collects data, and trains the neural network.

### Playing Against Trained Model

```bash
python playground.py
```

The `playground.py` script provides a graphical interface for playing against the trained model. You can interact with the game using the provided GUI in different game modes:

1. **Human vs Human**: Two players take turns making moves.
2. **Human vs Random**: Play against a random AI opponent.
3. **Human vs AI**: Play against the trained AlphaZero model.
4. **AI vs AI**: Watch the trained AlphaZero model play against itself.

```bash
Available Game Modes:
1: Red: Human vs Blue: Human
2: Red: Human vs Blue: Random
3: Red: Human vs Blue: AI
4: Red: AI vs Blue: AI

Select game mode (1-4): 
```

### Configuration

Key parameters in `config.py`, here is an example for `3 x 3`, `7 x 6` board.

#### Settings for `3 x 3` board and connect `3` to win

- Environment settings
```python
ENV_SETTINGS = {
	"size": (3, 3),  # Board size
	"N": 3,          # Number in a row to win
}

# Agent settings
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1.0e-4

# Training settings
NUM_EPISODES = 400
SIMULATIONS_PER_MOVE = 100
EVAL_FREQUENCY = 50  # Episodes between evaluations

# MCTS settings
MCTS_SIMULATIONS = 50  # Number of MCTS simulations for action selection
TEMPERATURE = 0.1     # Temperature for move selection
...
```
- Evaluating model performance during training
![loss](./assets/3x3-training-loss.png)

#### Settings for `7 x 6` board and connect `4` to win

- Environment settings
```python
ENV_SETTINGS = {
	"size": (7, 6),  # Board size
	"N": 4,          # Number in a row to win
}

# Agent settings
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1.0e-4

# Training settings
NUM_EPISODES = 1000
SIMULATIONS_PER_MOVE = 150
EVAL_FREQUENCY = 50  # Episodes between evaluations

# MCTS settings
MCTS_SIMULATIONS = 150  # Number of MCTS simulations for action selection
TEMPERATURE = 0.2     # Temperature for move selection
...
```

- Evaluating model performance during training
![loss](./assets/7x4-training-loss.png)


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by DeepMind's AlphaZero papers
- Built with PyTorch deep learning framework

## References

1. Silver, D., et al. (2017). Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
2. Silver, D., et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play