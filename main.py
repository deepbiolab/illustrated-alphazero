import numpy as np
from tqdm import tqdm
from collections import deque

from src.config import Config
from src.agent import MCTSAgent
from src.plots import plot_losses



def train_agent(agent: MCTSAgent):
    """
    Train the MCTS agent and save best model.
    
    Args:
        agent: The MCTSAgent instance
    """
    # Create checkpoints directory if it doesn't exist
    Config.CHECKPOINT_DIR.mkdir(exist_ok=True)
    
    outcomes = deque(maxlen=Config.EVAL_FREQUENCY)
    losses_window = deque(maxlen=Config.EVAL_FREQUENCY)
    losses = []
    best_mean_loss = float('inf')

    for episode in tqdm(range(Config.NUM_EPISODES), desc="Training", unit="episode"):
        # Run one episode of learning
        outcome, loss = agent.learn(num_simulations=Config.SIMULATIONS_PER_MOVE)
        
        # Store results
        outcomes.append(outcome)
        losses_window.append(loss)
        losses.append(loss)

        # Calculate and save progress periodically
        if (episode + 1) % Config.EVAL_FREQUENCY == 0:
            mean_loss = np.mean(list(losses_window))
            print(
                f"Episode: {episode + 1}, "
                f"Mean Loss: {mean_loss:.2f}, "
                f"Recent Outcomes: {list(outcomes)}"
            )
            
            # Save model if we have a new best mean loss
            if mean_loss < best_mean_loss:
                best_mean_loss = mean_loss
                agent.save_model(Config.BEST_MODEL_PATH)
                print(f"New best model saved with mean loss: {mean_loss:.4f}")

    # Save final model
    agent.save_model(Config.FINAL_MODEL_PATH)
    print("Final model saved")
    
    return losses


if __name__ == "__main__":
    
    # Create agent with settings from config
    agent = MCTSAgent(
        env_settings=Config.ENV_SETTINGS,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    losses = train_agent(agent)
    plot_losses(losses)

        