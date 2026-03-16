import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# numpy does math, torch does math and learns from mistakes

# ── WHAT IS A LINEAR LAYER? ───────────────────────────────────────────────────
# nn.Linear(in, out) does this math: output = input × weight + bias
# It takes an array of `in` numbers and transforms it into `out` numbers
# The weights are what the network LEARNS over time

# ── WHAT IS RELU? ─────────────────────────────────────────────────────────────
# ReLU(x) = max(0, x)
# It simply kills any negative numbers → turns them to 0
# This lets the network learn non-linear patterns (not just straight lines)
# Without it, stacking layers would be pointless mathematically


# ── THE NETWORK ───────────────────────────────────────────────────────────────
class Linear_QNet(nn.Module):
    def __init__(self, input_size: int, hidden_size_1: int, hidden_size_2: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)   # Layer 1
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2) # Layer 2
        self.fc3 = nn.Linear(hidden_size_2, output_size)  # Layer 3 (Output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))  # Pass through layer 1, apply ReLU
        x = F.relu(self.fc2(x))  # Pass through layer 2, apply ReLU
        x = self.fc3(x)          # Pass through layer 3 (raw Q-values, no ReLU)
        return x

# ── TEST IT ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Create the network: 18 inputs → 256 hidden → 4 outputs
    model = Linear_QNet(input_size=18, hidden_size=256, output_size=4)

    # Create a dummy state — 18 random float numbers simulating game state
    dummy_state = torch.rand(18)
    print(f"Input:  {dummy_state}")
    print(f"Shape:  {dummy_state.shape}")

    # example
    #  S  r  l   u  R  d  l   A  b  r  l
    # [1, 0, 0,  0, 1, 0, 0,  1, 0, 0, 0] that is the 11 numbers
    #  danger    direction     food       == Snake moving right, food is above, wall straight ahead

    # Pass it through the network
    output = model(dummy_state)
    print(f"\nOutput: {output}")
    print(f"Shape:  {output.shape}")

    # This should print exactly 4 values — one Q-value per action
    # Action 0 = Up, 1 = Right, 2 = Down, 3 = Left
# ── TEST IT END ───────────────────────────────────────────────────────────────────
# Right now the weights are random so the output is meaningless.
# Training is the process of slowly adjusting those weights until the outputs make sense.

class QTrainer:
    def __init__(self, model: nn.Module, lr: float, gamma: float):
        self.lr = lr
        self.gamma = gamma
        self.model = model

        # Rule 2 Implementation: The Optimizer and Loss Function
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # 1. Cast all inputs to PyTorch tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # done can remain a tuple of booleans

        # Handle short memory (single step) by adding a batch dimension (1, x)
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 2. Forward pass: Get predicted Q values with current state
        pred = self.model(state)

        # 3. Rule 1 Implementation: The Bellman Equation
        # Clone predictions so we only calculate loss on the action actually taken
        target = pred.clone().detach()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # R + gamma * max(Q(S', a'))
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx])).item()

            # Map the calculated target Q-value strictly to the action taken
            # Assuming action is a 1D array/tensor containing the action index (0, 1, 2, or 3)
            action_idx = action[idx].item() if action.dim() == 1 else torch.argmax(action[idx]).item()
            target[idx][action_idx] = Q_new

        # 4. Rule 2 Implementation: Backpropagation
        self.optimizer.zero_grad()  # Clear old gradients
        loss = self.criterion(target, pred)  # Calculate MSE Loss
        loss.backward()  # Compute gradients (Backpropagation)
        self.optimizer.step()  # Update neural network weights