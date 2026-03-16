import torch
import random
import numpy as np
from collections import deque
from Snake import SnakeEnv
from StateExtractor import StateExtractor
from Brain import Linear_QNet, QTrainer
import json
import os

# 1. Define hardware accelerator (CUDA for NVIDIA, MPS for Apple Silicon, CPU fallback)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

# Hyperparameters
MAX_MEMORY = 100_000
BATCH_SIZE = 1000  # Increased from 500 to handle the larger 104-D state space
LR = 0.001


class Agent:
    def __init__(self):
        self.n_games = 0

        # CRITICAL FIX: Epsilon must start at 1.0 (100% exploration)
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.gamma = 0.99  # Discount rate

        self.memory = deque(maxlen=MAX_MEMORY)

        # 2. Map the network to the hardware accelerator
        self.model = Linear_QNet(104, 256, 128, 3).to(DEVICE)

        # 3. Pass the device context to the trainer
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma, device=DEVICE)

    def get_state(self, game, current_action: int) -> np.ndarray:
        return StateExtractor.get_state(game, current_action)

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self) -> None:
        """Experience Replay: Trains the model on a randomized batch of past experiences."""
        if len(self.memory) == 0:
            return

        # Cast to list prevents TypeError in Python 3.11+ when sampling deques
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(list(self.memory), BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def get_action(self, state: np.ndarray) -> int:
        """Implements the epsilon-greedy policy for action selection."""
        # Standard uniform probability check between [0.0, 1.0)
        if random.random() < self.epsilon:
            final_move = random.randint(0, 2)
        else:
            # 4. Zero-copy memory mapping, sent directly to DEVICE
            state_tensor = torch.from_numpy(state).float().to(DEVICE)
            prediction = self.model(state_tensor)
            final_move = torch.argmax(prediction).item()

        return final_move


def train():
    record = 0
    score = 0
    agent = Agent()
    game = SnakeEnv()

    game.reset()
    current_action = 1

    os.makedirs("replays", exist_ok=True)
    step_history = []

    print(f"Beginning Training Loop. Hardware Accelerator: {DEVICE}")

    while True:
        is_recording = (agent.n_games % 100 == 0)

        state_old = agent.get_state(game, current_action)
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
            frame_data = {
                "snake": [[int(y), int(x)] for y, x in game.snake],
                "food": [int(game.food_pos[0]), int(game.food_pos[1])],
                "score": score
            }
            step_history.append(frame_data)

        state_new = agent.get_state(game, current_action)

        # Store transition. Note: We DO NOT train on a single step anymore.
        agent.remember(state_old, relative_move, reward, state_new, done)

        if done:
            # Decay epsilon at the end of the episode
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

            if is_recording and step_history:
                filename = f"replays/run_{agent.n_games}.json"
                with open(filename, "w") as f:
                    json.dump(step_history, f)

            step_history = []
            game.reset()
            agent.n_games += 1

            # Execute batch training only between games
            agent.train_long_memory()

            if score > record:
                record = score

            print(f"Game: {agent.n_games} | Score: {score} | Record: {record} | Epsilon: {agent.epsilon:.3f}")
            score = 0


if __name__ == "__main__":
    train()