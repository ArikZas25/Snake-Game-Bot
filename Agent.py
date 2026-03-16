import torch
import random
import numpy as np
from collections import deque
from Snake import SnakeEnv
from StateExtractor import StateExtractor
from Brain import Linear_QNet, QTrainer
import json
import os

# Hyperparameters
MAX_MEMORY = 100_000
BATCH_SIZE = 500
LR = 0.001


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # Exploration parameter
        self.gamma = 0.99  # Discount rate for future rewards

        # O(1) time complexity for appending/popping from ends
        self.memory = deque(maxlen=MAX_MEMORY)

        # Instantiate the neural network (14 inputs, 256 hidden,128 hidden, 3 outputs)
        self.model = Linear_QNet(14, 256, 128, 3)

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
        self.epsilon = max(0.05, self.epsilon * 0.995)

        if random.randint(0, 200) < self.epsilon:
            # Exploration: Choose a random action
            final_move = random.randint(0, 2)
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

    os.makedirs("replays", exist_ok=True)
    step_history = []  # Will hold the sequence of frames

    print("Beginning Training Loop...")

    while True:

        is_recording = (agent.n_games % 100 == 0)

        # 1. Extract the current state
        state_old = agent.get_state(game, current_action)

        # 2. Agent predicts the optimal RELATIVE move: 0=Straight, 1=Right Turn, 2=Left Turn
        relative_move = agent.get_action(state_old)

        # 3. Translate Relative Move to Absolute Heading
        # Clockwise order mapping: 0=Up, 1=Right, 2=Down, 3=Left
        clock_wise = [0, 1, 2, 3]
        idx = clock_wise.index(current_action)

        if relative_move == 0:
            absolute_move = clock_wise[idx]  # No change, go straight
        elif relative_move == 1:
            absolute_move = clock_wise[(idx + 1) % 4]  # Turn right (clockwise)
        else:  # relative_move == 2
            absolute_move = clock_wise[(idx - 1) % 4]  # Turn left (counter-clockwise)

        # Update current action for state extraction BEFORE the next frame
        current_action = absolute_move

        # 4. Environment processes the ABSOLUTE move
        board, reward, done = game.step(absolute_move)

        if reward == 10.0:
            score += 1

        if is_recording:
            # Cast NumPy types to native Python ints for JSON serialization
            frame_data = {
                "snake": [[int(y), int(x)] for y, x in game.snake],
                "food": [int(game.food_pos[0]), int(game.food_pos[1])],
                "score": score
            }
            step_history.append(frame_data)


        # 5. Extract the newly resulting state
        state_new = agent.get_state(game, current_action)

        # 6. Train short-term memory using the RELATIVE move
        agent.train_short_memory(state_old, relative_move, reward, state_new, done)

        # 7. Store the transition in memory using the RELATIVE move
        agent.remember(state_old, relative_move, reward, state_new, done)

        # 8. Evaluate terminal state
        if done:
            #Save to Disk if we were recording
            if is_recording and step_history:
                filename = f"replays/run_{agent.n_games}.json"
                with open(filename, "w") as f:
                    json.dump(step_history, f)
                print(f"[*] Saved replay telemetry: {filename}")

                # Reset variables for the next game
            step_history = []
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score

            print(f"Game: {agent.n_games} | Score: {score} | Record: {record}")
            score = 0


if __name__ == "__main__":
    train()