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

        self.frame_iteration = 0

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
        self.frame_iteration += 1
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
            self.frame_iteration = 0
        elif self.frame_iteration > 100 * len(self.snake):
            self.frame_iteration = 0
            reward = -15.0
            self.done = True
            return self.board.copy(), reward, self.done
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
        import pygame

        CELL_SIZE = 40
        COLORS = {
            0: (30, 30, 30),
            1: (0, 255, 100),  # head
            2: (0, 180, 60),  # body
            3: (255, 60, 60),  # food
        }
        BG_COLOR = (20, 20, 20)
        TEXT_COLOR = (255, 255, 255)

        pygame.init()
        screen = pygame.display.set_mode((self.width * CELL_SIZE, self.height * CELL_SIZE + 40))
        pygame.display.set_caption("Snake")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("Arial", 24)

        self.reset()
        score = 0
        current_action = 1  # start moving right

        running = True
        while running:
            clock.tick(4)  # speed — increase for faster snake

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_w, pygame.K_UP) and current_action != 2:
                        current_action = 0
                    elif event.key in (pygame.K_d, pygame.K_RIGHT) and current_action != 3:
                        current_action = 1
                    elif event.key in (pygame.K_s, pygame.K_DOWN) and current_action != 0:
                        current_action = 2
                    elif event.key in (pygame.K_a, pygame.K_LEFT) and current_action != 1:
                        current_action = 3
                    elif event.key == pygame.K_q:
                        running = False

            # Move the snake every frame
            _, reward, self.done = self.step(current_action)
            if reward == 10.0:
                score += 1

            if self.done:
                running = False

            # Draw
            screen.fill(BG_COLOR)
            score_text = font.render(f"Score: {score}", True, TEXT_COLOR)
            screen.blit(score_text, (10, 5))

            for y in range(self.height):
                for x in range(self.width):
                    cell = self.board[y][x]
                    rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE + 40, CELL_SIZE - 2, CELL_SIZE - 2)
                    pygame.draw.rect(screen, COLORS[cell], rect, border_radius=6)

            pygame.display.flip()

        # Game over screen
        screen.fill(BG_COLOR)
        over_text = font.render(f"Game Over! Score: {score}", True, TEXT_COLOR)
        screen.blit(over_text, (self.width * CELL_SIZE // 2 - 120, self.height * CELL_SIZE // 2))
        pygame.display.flip()
        pygame.time.wait(2000)
        pygame.quit()


if __name__ == "__main__":
    game = SnakeEnv()
    game.play()