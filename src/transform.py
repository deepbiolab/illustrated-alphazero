import torch
import random
from enum import Enum
from typing import Callable, List, Tuple, Any

from .network import Policy
from .interact import Play


class Transform(Enum):
    """Enumeration of board transformations"""
    IDENTITY = 0    # Original state
    HORIZONTAL = 1  # Horizontal flip
    VERTICAL = 2    # Vertical flip
    BOTH = 3        # Both horizontal and vertical flip
    TRANSPOSE = 4   # Transpose
    TRANS_HOR = 5   # Transpose + horizontal flip
    TRANS_VER = 6   # Transpose + vertical flip
    TRANS_BOTH = 7  # Transpose + both flips


class BoardTransform:
    """Handle board transformations and their inverses"""
    
    @staticmethod
    def flip(x: torch.Tensor, dim: int) -> torch.Tensor:
        """Flip tensor along specified dimension"""
        return torch.flip(x, [dim])

    @classmethod
    def transform(cls, x: torch.Tensor, trans_type: Transform) -> torch.Tensor:
        """Apply specified transformation to board state"""
        if trans_type == Transform.IDENTITY:
            return x.clone()
        elif trans_type == Transform.HORIZONTAL:
            return torch.flip(x, [1])
        elif trans_type == Transform.VERTICAL:
            return torch.flip(x, [0])
        elif trans_type == Transform.BOTH:
            return torch.flip(x, [0, 1])
        elif trans_type == Transform.TRANSPOSE:
            return x.T.clone()
        elif trans_type == Transform.TRANS_HOR:
            return torch.flip(x, [1]).T.clone()
        elif trans_type == Transform.TRANS_VER:
            return torch.flip(x, [0]).T.clone()
        elif trans_type == Transform.TRANS_BOTH:
            return torch.flip(x, [0, 1]).T.clone()

    @classmethod
    def inverse_transform(cls, x: torch.Tensor, trans_type: Transform) -> torch.Tensor:
        """Apply inverse transformation to board state"""
        if trans_type == Transform.IDENTITY:
            return x.clone()
        elif trans_type == Transform.HORIZONTAL:
            return cls.flip(x, 1)
        elif trans_type == Transform.VERTICAL:
            return cls.flip(x, 0)
        elif trans_type == Transform.BOTH:
            return cls.flip(cls.flip(x, 0), 1)
        elif trans_type == Transform.TRANSPOSE:
            return x.T.clone()
        elif trans_type == Transform.TRANS_HOR:
            return cls.flip(x, 0).T.clone()
        elif trans_type == Transform.TRANS_VER:
            return cls.flip(x, 1).T.clone()
        elif trans_type == Transform.TRANS_BOTH:
            return cls.flip(cls.flip(x, 0), 1).T.clone()

    @classmethod
    def get_transformation_pairs(cls, include_rotations: bool = True) -> List[Tuple[Callable, Callable]]:
        """Get list of (transform, inverse_transform) pairs"""
        transforms = list(Transform) if include_rotations else list(Transform)[:4]
        return [(lambda x, t=t: cls.transform(x, t),
                lambda x, t=t: cls.inverse_transform(x, t)) 
                for t in transforms]

    @classmethod
    def evaluate_policy(cls, policy: Policy, game: Play, device: torch.device) -> Tuple[List[Any], torch.Tensor, torch.Tensor]:
        """
        Evaluate policy network on game state with random symmetry transformation.
        
        Args:
            policy: Policy network that takes board state and returns probabilities and value
            game: Game instance containing current state and player information
            device: PyTorch device to use for tensor operations
        
        Returns:
            Tuple containing:
            - List of available moves
            - Transformed probabilities for available moves
            - Value prediction for the current state
        """
        # For square boards (equal height and width), include all transformations
        # For rectangular boards, only include reflections since rotations would change dimensions
        is_square_board = game.size[0] == game.size[1]
        trans_list = cls.get_transformation_pairs(include_rotations=is_square_board)
        transform_func, inverse_func = random.choice(trans_list)
        
        # Convert numpy array to torch tensor and transform game state
        current_state = torch.tensor(game.state * game.player, dtype=torch.float, device=device)
        transformed_state = transform_func(current_state)
        
        # Prepare network input
        input_tensor = transformed_state.unsqueeze(0).unsqueeze(0)
        
        # Get predictions from policy network
        transformed_probs, value = policy(input_tensor)
        
        # Get available moves mask and apply inverse transformation
        available_mask = torch.tensor(game.available_mask(), dtype=torch.bool)
        probs = inverse_func(transformed_probs)[available_mask].view(-1)
        
        return game.available_moves(), probs, value.squeeze().squeeze()


# Test code
if __name__ == "__main__":
    # Create transformation list
    trans_list = BoardTransform.get_transformation_pairs()

    # Create example board state
    board_state = torch.tensor([[ 1,   0,   1],
                                [-1,  -1,   0],
                                [ 0,   1,   1]])
    
    print("Original board:")
    print(board_state)

    # Test all transformations
    for trans_type in Transform:
        transform_func, inverse_func = trans_list[trans_type.value]
        
        transformed = transform_func(board_state)
        restored = inverse_func(transformed)
        
        print(f"\n{trans_type.name} transformation:")
        print(transformed)
        
        # Verify transformation correctness
        assert torch.allclose(board_state, restored), f"Failed for {trans_type.name}"
        print("Restoration verified ✓")