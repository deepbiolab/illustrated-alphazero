import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvBlock(nn.Module):
    """Convolutional block module for feature extraction"""
    def __init__(self, in_channels=1, out_channels=32):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
    
    def forward(self, x):
        return F.relu(self.conv(x))

class PolicyHead(nn.Module):
    """Policy head module that outputs action probabilities"""
    def __init__(self, input_size=64, hidden_size=32, action_size=9):
        super(PolicyHead, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        self.action_size = action_size
    
    def forward(self, x, available_actions):
        """
        Args:
            x: Feature tensor
            available_actions: Binary mask of legal moves
                Example for a 3x3 board:
                If the board state is:
                [[ 1,  0, -1],
                 [ 0,  1,  0],
                 [-1,  0,  0]]
                Then available_actions would be:
                [0, 1, 0, 1, 0, 1, 0, 1, 1]
                where 1 indicates empty positions (legal moves)
        """
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        
        # Calculate probability distribution using numerically stable masked softmax
        # 1. Find maximum logit value for numerical stability
        max_logit = torch.max(logits)
        
        # 2. Subtract max_logit before exponential to prevent overflow
        # Why we subtract max_logit:
        # - When logits contain large values, exp(logits) can explode to infinity
        # - Since exp(x - C) / sum(exp(x - C)) = exp(x) / sum(exp(x)) for any C,
        #   subtracting max_logit doesn't change the final probability distribution
        # - This trick keeps the exponentials in a reasonable numerical range
        # 
        # Example:
        # If logits = [100, 101, 102]
        # Without stabilization:
        #   exp(logits) = [e^100, e^101, e^102] -> potential overflow
        # With stabilization (max_logit = 102):
        #   exp(logits - 102) = [e^-2, e^-1, e^0] = [0.135, 0.368, 1.0] -> numerically stable
        exp = available_actions * torch.exp(logits - max_logit)
        
        # 3. Normalize to get probabilities
        probabilities = exp / torch.sum(exp)
        
        return probabilities

class ValueHead(nn.Module):
    """Value head module that estimates the state value"""
    def __init__(self, input_size=64, hidden_size=16):
        super(ValueHead, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        """Returns a value between -1 and 1 indicating the estimated game outcome"""
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
    
        # Apply tanh to bound value prediction to [-1, 1]
        # Why tanh:
        # 1. Zero-sum Game Properties:
        #    - Symmetric around 0 (win = +1, draw = 0, loss = -1)
        #    - Perfect play tends toward 0 in zero-sum games
        #
        # 2. Training Efficiency:
        #    - Stronger gradients near 0 compared to sigmoid
        #    - Optimal for positions that should be evaluated as neutral
        return self.tanh(x)

class Policy(nn.Module):
    """AlphaZero policy network for TicTacToe"""
    def __init__(self, env):
        """
        Args:
            env: Game environment with size attribute (e.g., (3,3) for standard TicTacToe)
        Example:
            For standard 3x3 TicTacToe:
            env.size = (3,3)
            action_size = 9
        """
        super(Policy, self).__init__()
        self.env = env
        self.action_size = np.prod(env.size).item()
        
        # Feature extraction layers
        self.conv_out_channels = 32
        self.conv_block = ConvBlock(in_channels=1, out_channels=self.conv_out_channels)
        
        # Calculate feature size based on board dimensions
        # feature_size = height * width * conv_output_channels
        self.feature_size = (
            self.env.size[0] *  # height
            self.env.size[1] *  # width
            self.conv_out_channels
        )
        self.feature_layer = nn.Linear(self.feature_size, 64)
        
        # Policy and value heads
        self.policy_head = PolicyHead(
            input_size=64,
            hidden_size=32,
            action_size=self.action_size
        )
        self.value_head = ValueHead(
            input_size=64,
            hidden_size=16
        )

    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input state tensor of shape (batch_size, 1, height, width)
               Example for a single 3x3 board state:
               x = [[[[ 1,  0, -1],
                      [ 0,  1,  0],
                      [-1,  0,  0]]]]
               where: 1 = X, -1 = O, 0 = empty
        
        Returns:
            probabilities: Action probability distribution shaped as env.size
                         Example: [[0.1, 0.2, 0.0],
                                   [0.3, 0.0, 0.2],
                                   [0.0, 0.1, 0.1]]
            value: Estimated state value between -1 and 1
                   Example: 0.8 (indicating advantage for current player)
        """
        # Feature extraction
        features = self.conv_block(x)
        features = features.view(-1, self.feature_size)
        features = F.relu(self.feature_layer(features))
        
        # Calculate available actions mask
        # abs(x) != 1 gives True for empty positions (0)
        # and False for occupied positions (1 or -1)
        available_actions = (torch.abs(x.squeeze()) != 1).type(torch.FloatTensor)
        available_actions = available_actions.reshape(-1, self.action_size)
        
        # Get policy and value predictions
        probabilities = self.policy_head(features, available_actions)
        value = self.value_head(features)
        
        # Reshape probabilities to match board dimensions
        probabilities = probabilities.view(self.env.size)
        
        return probabilities, value


def main():
    """
    Demonstrates the AlphaZero policy network usage with example data
    """
    import torch
    from dataclasses import dataclass
    
    # Create a simple environment class for demonstration
    @dataclass
    class TicTacToeEnv:
        size: tuple = (3, 3)

    # Initialize environment and network
    env = TicTacToeEnv()
    policy = Policy(env)
    
    # Create example board state
    # 1=X, -1=O, 0=empty
    # Current board state:
    # X _ O
    # _ X _
    # O _ _
    board = torch.tensor([[
        [ 1,  0, -1],
        [ 0,  1,  0],
        [-1,  0,  0]
    ]], dtype=torch.float32).unsqueeze(0)  # Shape: (1, 1, 3, 3)
    
    # Forward pass through the network
    probabilities, value = policy(board)
    
    # Print results with visualization
    print("Example AlphaZero Network Inference:")
    print("\nInput Board State:")
    symbols = {1: 'X', -1: 'O', 0: '_'}
    for row in board.squeeze():
        print(' '.join([symbols[int(x.item())] for x in row]))
    
    print("\nNetwork Outputs:")
    print("1. Action Probabilities:")
    prob_array = probabilities.detach().numpy()
    
	# Format probabilities with 2 decimal places
    for row in prob_array:
        print(' '.join([f'{x:5.2f}' for x in row]))
    
    print("\n2. State Value:", f"{value.item():5.2f}")
    print("   (1.0 = winning for X, -1.0 = winning for O)")
    
    # Print feature dimensions
    print("\nNetwork Architecture Details:")
    print(f"Board Size: {env.size}")
    print(f"Conv Output Channels: {policy.conv_out_channels}")
    print(f"Flattened Feature Size: {policy.feature_size}")
    print(f"Total Actions: {policy.action_size}")

if __name__ == "__main__":
    main()