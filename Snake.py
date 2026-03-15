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

        # 1. Map the action to a direction (Y change, X change)
        directions = {
            0: (-1, 0),  # Up: Y decreases
            1: (0, 1),  # Right: X increases
            2: (1, 0),  # Down: Y increases
            3: (0, -1)  # Left: X decreases
        }

        # Calculate the new head coordinates
        head_y, head_x = self.snake[0]
        move_y, move_x = directions[action]
        new_head = (head_y + move_y, head_x + move_x)

        # 2. Check Wall Collisions
        if (new_head[0] < 0 or new_head[0] >= self.height or
                new_head[1] < 0 or new_head[1] >= self.width):
            self.done = True
            return self.board.copy(), -10.0, self.done

        # 3. Process Food and the Tail
        reward = 0.0
        if new_head == self.food_pos:
            reward = 10.0
            # We grew! Do not remove the tail.
            self._spawn_food()  # We will need to create this helper function
        else:
            reward = -0.1  # Small penalty for wasting a step
            # We did not grow. Remove the old tail to move forward.
            tail_y, tail_x = self.snake.pop()
            self.board[tail_y, tail_x] = 0  # Clear the tail from the board array

        # 4. Check Body Collisions (Self-Collision)
        if new_head in self.snake:
            self.done = True
            return self.board.copy(), -10.0, self.done

        # 5. Update the Data Structures
        self.snake.appendleft(new_head)  # Add new head to the front of the queue

        # Update the board array to show the new snake body and head
        if len(self.snake) > 1:
            old_head = self.snake[1]
            self.board[old_head[0], old_head[1]] = 1  # Mark old head as body (1)

        self.board[new_head[0], new_head[1]] = 2  # Mark new head as head (2)

        return self.board.copy(), reward, self.done

    def _spawn_food(self) -> None:
        """
        Finds all empty spaces and randomly places food.
        Handles the rare edge case where the board is completely full.
        """
        # 1. Find all Y and X coordinates where the board is exactly 0 (empty)
        empty_y, empty_x = np.where(self.board == 0)

        # 2. Check for a perfect win
        # If there are no empty spaces, the snake fills the entire board.
        if len(empty_y) == 0:
            self.done = True
            return

        # 3. Choose one random index from the list of empty spaces
        random_index = np.random.randint(0, len(empty_y))

        # Extract the exact Y and X coordinate
        food_y = empty_y[random_index]
        food_x = empty_x[random_index]

        # 4. Save the position and update the board array
        self.food_pos = (int(food_y), int(food_x))
        self.board[food_y, food_x] = 3