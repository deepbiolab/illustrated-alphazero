import os
import sys
import random
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import Config
from src.agent import MCTSAgent
from src.environment import Environment
from src.interact import Play

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

def random_player(env):
    """Simple random player with game end handling."""
    if env.score is not None:  # Game has ended
        return None
    moves = list(env.available_moves())
    if not moves:  # No moves available
        return None
    return moves[random.randint(0, len(moves)-1)]


def create_environment(board_size=None, N=None):
    """
    Create game environment with either custom settings or config defaults.
    
    Args:
        board_size: Optional board size (defaults to Config setting)
        N: Optional number in a row to win (defaults to Config setting)
    """
    if board_size is None or N is None:
        return Environment(**Config.ENV_SETTINGS)
    
    env_setting = {"size": (board_size, board_size), "N": N}
    return Environment(**env_setting)


def setup_gameplay(settings=None, agent=None):
    """
    Set up gameplay with either custom settings or defaults.
    
    Args:
        settings: Optional dictionary containing game settings
        agent: Optional pre-configured MCTSAgent instance
    """
    # Create environment
    if settings:
        env = create_environment(settings['board_size'], settings['N'])
    else:
        env = create_environment()
    
    # If no settings provided, default to human vs AI with config settings
    if settings is None:
        if agent is None:
            agent = MCTSAgent(
                env_settings=Config.ENV_SETTINGS,
                learning_rate=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY
            )
        
        return Play(
            env,
            player1=None,  # Human player
            player2=lambda env: agent.select_action(
                env,
                num_simulations=Config.MCTS_SIMULATIONS,
                temperature=Config.TEMPERATURE
            )
        )
    
    # Mode mapping for custom settings
    MODE_MAPPING = {
        1: ('Human vs Human', None, None),
        2: ('Human vs Random', None, random_player),
        3: ('Human vs AI', None, create_ai_player(settings['model_path'])),
        4: ('Random vs AI', random_player, create_ai_player(settings['model_path']))
    }
    
    mode_name, player1, player2 = MODE_MAPPING[settings['mode']]
    print(f"\nStarting game in {mode_name} mode...")
    
    return Play(env, player1=player1, player2=player2)

def create_ai_player(model_path):
    """Create an AI player using the trained model."""
    # Create agent with settings from config
    agent = MCTSAgent(
        env_settings=Config.ENV_SETTINGS,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    if Path(model_path).exists():
        agent.load_model(model_path)
        print(f"Loaded AI model from {model_path}")
        
        def ai_player(env):
            """AI player with game end handling."""
            if env.score is not None:  # Game has ended
                return None
            return agent.select_action(
                env,
                num_simulations=Config.MCTS_SIMULATIONS,
                temperature=Config.TEMPERATURE
            )
        
        return ai_player
    else:
        print(f"No model found at {model_path}, falling back to random player")
        return random_player

def get_user_input():
    """Get game settings from user input."""
    print("\nAvailable Game Modes:")
    print("1: Human (Red) vs Human (Blue)")
    print("2: Human (Red) vs Random (Blue)")
    print("3: Human (Red) vs AI (Blue)")
    print("4: Random (Red) vs AI (Blue)")
    
    while True:
        try:
            mode = int(input("\nSelect game mode (1-4): "))
            if mode not in [1, 2, 3, 4]:
                print("Invalid mode! Please select 1-4.")
                continue
            
            # Get default values from config
            default_size = Config.ENV_SETTINGS['size'][0]
            default_n = Config.ENV_SETTINGS['N']
            
            # Handle board size input
            size_input = input(f"Enter board size (default {default_size}): ")
            board_size = int(size_input) if size_input.strip() else default_size
            if board_size < 3:
                print("Board size must be at least 3!")
                continue

            # Handle connect N input
            n_input = input(f"Enter number in a row to win (default {default_n}): ")
            connect_n = int(n_input) if n_input.strip() else default_n
            if connect_n < 3:
                print("Connect N must be at least 3!")
                continue
            
            if connect_n > board_size:
                print("Number in a row cannot be larger than board size!")
                continue
                
            return {
                'mode': mode,
                'board_size': board_size,
                'N': connect_n,
                'model_path': str(Config.BEST_MODEL_PATH)
            }
            
        except ValueError:
            print("Invalid input! Please enter numbers only.")


def run_interactive_game(game):
    """Run the interactive game."""
    try:
        plt.show()
    except KeyboardInterrupt:
        print("\nGame interrupted by user")
    except ValueError as e:
        if "game has ended" in str(e):
            print("\nGame has ended!")
        else:
            print(f"\nError during gameplay: {e}")
    except Exception as e:
        print(f"\nError during gameplay: {e}")

if __name__ == "__main__":
    try:
        # Get game settings from user
        settings = get_user_input()
        
        # Setup and run game
        game = setup_gameplay(settings=settings)
        print("Game started! Close the window to exit.")
        run_interactive_game(game)
        
    except KeyboardInterrupt:
        print("\nGame interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)