import torch
import random
import numpy as np
from collections import deque
from Snake import SnakeEnv
from Brain import Linear_QNet
from StateExtractor import StateExtractor

# Hyperparameters
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # Exploration parameter
        self.gamma = 0.9  # Discount rate for future rewards

        # O(1) time complexity for appending/popping from ends
        self.memory = deque(maxlen=MAX_MEMORY)

        # Instantiate the neural network (11 inputs, 256 hidden, 4 outputs)
        self.model = Linear_QNet(11, 256, 4)

        # TODO: A Trainer class is required to execute the backpropagation
        # self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game, current_action: int) -> np.ndarray:
        """Delegates state extraction to the dedicated module."""
        return StateExtractor.get_state(game, current_action)

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """Stores the transition tuple into the replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self) -> None:
        """Experience Replay: Trains the model on a randomized batch of past experiences."""
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        # Unzip the tuples into distinct arrays
        states, actions, rewards, next_states, dones = zip(*mini_sample)

        # TODO: Pass unzipped batches to the trainer
        # self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray,
                           done: bool) -> None:
        """Trains the model immediately on the most recent single step."""
        # TODO: Pass the single step to the trainer
        # self.trainer.train_step(state, action, reward, next_state, done)
        pass

    def get_action(self, state: np.ndarray) -> int:
        """
        Implements the epsilon-greedy policy for action selection.
        """
        # Epsilon decay: The randomness decreases linearly as n_games increases.
        self.epsilon = 80 - self.n_games

        if random.randint(0, 200) < self.epsilon:
            # Exploration: Choose a random action
            final_move = random.randint(0, 3)
        else:
            # Exploitation: Forward pass through the network
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_tensor)

            # Select the action index with the highest Q-value
            final_move = torch.argmax(prediction).item()

        return final_move