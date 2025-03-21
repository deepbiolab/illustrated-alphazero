import os
import sys
import random
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import Config
from src.agent import MCTSAgent
from src.environment import Environment
from src.interact import Play

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"


def random_player(env):
    """Simple random player with game end handling."""
    if env.score is not None:  # Game has ended
        return None
    moves = list(env.available_moves())
    if not moves:  # No moves available
        return None
    return moves[random.randint(0, len(moves) - 1)]


def create_ai_player(model_path):
    """Create an AI player using the trained model."""
    agent = MCTSAgent(
        env_settings=Config.ENV_SETTINGS,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
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
                temperature=Config.TEMPERATURE,
            )

        return ai_player
    else:
        print(f"No model found at {model_path}, falling back to random player")
        return random_player


def setup_gameplay(mode=None):
    """
    Set up gameplay with config settings.

    Args:
        mode: Game mode (1-4)
    """
    env = Environment(**Config.ENV_SETTINGS)

    # Mode mapping with lazy initialization
    def get_mode_config(mode):
        if mode == 1:
            return "Red: Human vs Blue: Human", None, None
        elif mode == 2:
            return "Red: Human vs Blue: Random", None, random_player
        elif mode == 3:
            return "Red: Human vs Blue: AI", None, create_ai_player(str(Config.BEST_MODEL_PATH))
        elif mode == 4:
            return "Red: AI vs Blue: AI", create_ai_player(str(Config.BEST_MODEL_PATH)), create_ai_player(str(Config.BEST_MODEL_PATH))

    mode_name, player1, player2 = get_mode_config(mode)
    print(f"\nStarting game in {mode_name} mode...")

    return Play(env, player1=player1, player2=player2, mode_name=mode_name)


def get_user_input():
    """Get game mode from user input."""
    print("\nAvailable Game Modes:")
    print("1: Red: Human vs Blue: Human")
    print("2: Red: Human vs Blue: Random")
    print("3: Red: Human vs Blue: AI")
    print("4: Red: AI vs Blue: AI")

    while True:
        try:
            mode = int(input("\nSelect game mode (1-4): "))
            if mode not in [1, 2, 3, 4]:
                print("Invalid mode! Please select 1-4.")
                continue
            return mode
        except ValueError:
            print("Invalid input! Please enter a number between 1 and 4.")


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
        # Get game mode from user
        mode = get_user_input()

        # Setup and run game
        game = setup_gameplay(mode=mode)
        print("Game started! Close the window to exit.")
        run_interactive_game(game)

    except KeyboardInterrupt:
        print("\nGame interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)