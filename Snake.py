import numpy as np
from collections import deque
from typing import Tuple


class SnakeEnv:
    def __init__(self, width: int = 10, height: int = 10) -> None:
        """Initializes the game variables but does not start the game."""
        self.width: int = width
        self.height: int = height

        # The 2D grid representing our AI's state
        self.board: np.ndarray = np.zeros((self.height, self.width), dtype=np.int8)

        # The snake's body, where index 0 is always the head
        self.snake: deque = deque()
        self.food_pos: Tuple[int, int] = (0, 0)
        self.done: bool = False

    def reset(self) -> np.ndarray:
        """
        Clears the board, spawns the snake and food, and returns the starting state.
        Must be called before starting a new game episode.
        """
        self.board.fill(0)
        self.snake.clear()
        self.done = False

        # TODO: Add logic to place the snake and spawn the first food

        return self.board.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Executes one frame of the game based on the AI's action.

        Args:
            action (int): 0 (Up), 1 (Right), 2 (Down), 3 (Left)

        Returns:
            Tuple containing (new_state, reward, is_done)
        """
        if self.done:
            raise RuntimeError("You must call reset() before stepping a finished game.")

        reward: float = 0.0

        # TODO: Add movement logic, collision checking, and reward calculation

        return self.board.copy(), reward, self.done