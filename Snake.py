import numpy as np
from collections import deque
from typing import Tuple


class SnakeEnv:
    def __init__(self, width: int = 10, height: int = 10) -> None:
        self.width: int = width
        self.height: int = height
        self.board: np.ndarray = np.zeros((self.height, self.width), dtype=np.int8)
        self.snake: deque = deque()
        self.food_pos: Tuple[int, int] = (0, 0)
        self.done: bool = False

    def reset(self) -> np.ndarray:
        self.board.fill(0)
        self.snake.clear()
        self.done = False

        mid_x = self.width // 2
        mid_y = self.height // 2
        self.snake.append((mid_y, mid_x))
        self.snake.append((mid_y, mid_x - 1))

        self.board[mid_y][mid_x] = 1
        self.board[mid_y][mid_x - 1] = 2

        self._spawn_food()
        return self.board.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        if self.done:
            raise RuntimeError("You must call reset() before stepping a finished game.")

        directions = {
            0: (-1, 0),
            1: (0, 1),
            2: (1, 0),
            3: (0, -1)
        }

        head_y, head_x = self.snake[0]
        move_y, move_x = directions[action]
        new_head = (head_y + move_y, head_x + move_x)

        if (new_head[0] < 0 or new_head[0] >= self.height or
                new_head[1] < 0 or new_head[1] >= self.width):
            self.done = True
            return self.board.copy(), -10.0, self.done

        reward = 0.0
        if new_head == self.food_pos:
            reward = 10.0
            self._spawn_food()
        else:
            reward = -0.1
            tail_y, tail_x = self.snake.pop()
            self.board[tail_y, tail_x] = 0

        if new_head in self.snake:
            self.done = True
            return self.board.copy(), -10.0, self.done

        self.snake.appendleft(new_head)

        if len(self.snake) > 1:
            old_head = self.snake[1]
            self.board[old_head[0], old_head[1]] = 1

        self.board[new_head[0], new_head[1]] = 2

        return self.board.copy(), reward, self.done

    def _spawn_food(self) -> None:
        empty_y, empty_x = np.where(self.board == 0)

        if len(empty_y) == 0:
            self.done = True
            return

        random_index = np.random.randint(0, len(empty_y))
        food_y = empty_y[random_index]
        food_x = empty_x[random_index]

        self.food_pos = (int(food_y), int(food_x))
        self.board[food_y, food_x] = 3

    def play(self) -> None:
        import os

        action_map = {'w': 0, 'd': 1, 's': 2, 'a': 3}
        symbols = {0: ' . ', 1: ' H ', 2: ' O ', 3: ' F '}

        self.reset()
        score = 0

        while not self.done:
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"Score: {score}")
            print(' ---' * self.width)
            for row in self.board:
                print('|' + ''.join(symbols[cell] for cell in row) + '|')
            print(' ---' * self.width)
            print("Controls: W=Up  S=Down  A=Left  D=Right  Q=Quit")

            key = input("Move: ").strip().lower()
            if key == 'q':
                break
            if key not in action_map:
                continue

            _, reward, self.done = self.step(action_map[key])
            if reward == 10.0:
                score += 1

        print(f"\nGame Over! Final Score: {score}")


if __name__ == "__main__":
    game = SnakeEnv()
    game.play()