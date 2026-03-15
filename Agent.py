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