import torch
import random
import numpy as np
from collections import deque
from Snake import SnakeEnv
from Brain import CNN_QNet, QTrainer
import json
import os

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


def get_state(game: SnakeEnv) -> np.ndarray:
    h, w = game.height, game.width
    head_ch = np.zeros((h, w), dtype=np.float32)
    body_ch = np.zeros((h, w), dtype=np.float32)
    food_ch = np.zeros((h, w), dtype=np.float32)

    snake_list = list(game.snake)
    if snake_list:
        head_ch[snake_list[0]] = 1.0
        for seg in snake_list[1:]:
            body_ch[seg] = 1.0

    food_ch[game.food_pos] = 1.0

    return np.stack([head_ch, body_ch, food_ch])  # shape: (3, 10, 10)


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 200
        self.gamma   = 0.99

        self.memory  = deque(maxlen=MAX_MEMORY)

        self.model   = CNN_QNet(output_size=3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = list(self.memory)

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(
            np.array(states),
            actions,
            rewards,
            np.array(next_states),
            dones,
        )

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state: np.ndarray) -> int:
        if random.randint(0, 200) < self.epsilon:
            return random.randint(0, 2)

        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        prediction   = self.model(state_tensor)
        return torch.argmax(prediction).item()


def train():
    record = 0
    score  = 0
    agent  = Agent()
    game   = SnakeEnv()
    game.reset()

    current_action = 1
    os.makedirs("replays", exist_ok=True)
    step_history = []

    print("Beginning Training Loop...")

    while True:
        is_recording = (agent.n_games % 100 == 0)

        state_old     = get_state(game)
        relative_move = agent.get_action(state_old)

        clock_wise = [0, 1, 2, 3]
        idx = clock_wise.index(current_action)

        if relative_move == 0:
            absolute_move = clock_wise[idx]
        elif relative_move == 1:
            absolute_move = clock_wise[(idx + 1) % 4]
        else:
            absolute_move = clock_wise[(idx - 1) % 4]

        current_action = absolute_move

        board, reward, done = game.step(absolute_move)

        if reward == 10.0:
            score += 1

        if is_recording:
            step_history.append({
                "snake": [[int(y), int(x)] for y, x in game.snake],
                "food":  [int(game.food_pos[0]), int(game.food_pos[1])],
                "score": score,
            })

        state_new = get_state(game)
        agent.train_short_memory(state_old, relative_move, reward, state_new, done)
        agent.remember(state_old, relative_move, reward, state_new, done)

        if done:
            if is_recording and step_history:
                filename = f"replays/run_{agent.n_games}.json"
                with open(filename, "w") as f:
                    json.dump(step_history, f)
                print(f"[*] Saved replay: {filename}")

            step_history = []
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            agent.epsilon = max(5, agent.epsilon - 0.1)

            if score > record:
                record = score

            print(f"Game: {agent.n_games} | Score: {score} | Record: {record} | Epsilon: {agent.epsilon:.1f}")
            score = 0


if __name__ == "__main__":
    train()