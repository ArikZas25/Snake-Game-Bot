import torch
import random
import numpy as np
from collections import deque
from Snake import SnakeEnv
from StateExtractor import StateExtractor
from Brain import Linear_QNet, QTrainer

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

        # Initialize the trainer with the learning rate and discount factor
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

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

        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray,
                           done: bool) -> None:
        """Trains the model immediately on the most recent single step."""
        self.trainer.train_step(state, action, reward, next_state, done)
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


def train():
    """
    The main execution loop for the DQN Agent.
    Orchestrates the environment, state extraction, action selection, and backpropagation.
    """
    record = 0
    score = 0
    agent = Agent()
    game = SnakeEnv()

    game.reset()

    # Initialize heading: 1 corresponds to moving Right in your environment logic
    current_action = 1

    print("Beginning Training Loop...")

    while True:
        # 1. Extract the current state
        state_old = agent.get_state(game, current_action)

        # 2. Agent predicts the optimal move (or explores randomly)
        final_move = agent.get_action(state_old)

        # Update current action for the next state extraction
        current_action = final_move

        # 3. Environment processes the move
        board, reward, done = game.step(final_move)

        # Track score (Food yields a reward of 10.0)
        if reward == 10.0:
            score += 1

        # 4. Extract the newly resulting state
        state_new = agent.get_state(game, current_action)

        # 5. Train the short-term memory (single step)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # 6. Store the transition in the replay memory buffer
        agent.remember(state_old, final_move, reward, state_new, done)

        # 7. Evaluate terminal state
        if done:
            # The game ended. Reset the environment and train on the batch.
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                # Note: Model saving logic belongs here

            print(f"Game: {agent.n_games} | Score: {score} | Record: {record}")

            # Reset score for the next game iteration
            score = 0


if __name__ == "__main__":
    train()