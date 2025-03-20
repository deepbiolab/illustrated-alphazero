"""
Configuration settings for the MCTS agent and training process.
"""
from pathlib import Path

class Config:
    # Environment settings
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

    # Model paths
    CHECKPOINT_DIR = Path("checkpoints")
    BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"
    FINAL_MODEL_PATH = CHECKPOINT_DIR / "final_model.pth"