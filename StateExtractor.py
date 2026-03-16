import numpy as np

class StateExtractor:

    @staticmethod
    def get_state(game, current_action: int) -> np.ndarray:
        """
        Extracts the raw 104-dimensional environment state.
        - 100 dimensions: The 10x10 board, normalized to [0.0, 1.0]
        - 4 dimensions: One-hot encoded current heading
        """
        # 1. O(1) Memory View and Normalization
        # game.board max value is 3 (food). Dividing by 3.0 normalizes to [0.0, 1.0].
        # We cast to float32 to match PyTorch's default tensor type.
        board_flat = (game.board.ravel() / 3.0).astype(np.float32)

        # 2. One-Hot Encode the Current Direction
        # current_action mapping: 0=Up, 1=Right, 2=Down, 3=Left
        direction = np.zeros(4, dtype=np.float32)
        direction[current_action] = 1.0

        # 3. Concatenate into the final 104-D vector
        state = np.concatenate((board_flat, direction))

        return state