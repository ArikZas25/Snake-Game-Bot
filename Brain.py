import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

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
    # Test the new 104-D architecture
    model = Linear_QNet(input_size=104, hidden_size_1=256, hidden_size_2=128, output_size=3)

    # Create a dummy state — 104 random float numbers simulating game state
    dummy_state = torch.rand(104)
    print(f"Input:  {dummy_state}")
    print(f"Shape:  {dummy_state.shape}")

    # Pass it through the network
    output = model(dummy_state)
    print(f"\nOutput: {output}")
    print(f"Shape:  {output.shape}")

# ── TRAINER ───────────────────────────────────────────────────────────────────
class QTrainer:
    def __init__(self, model: nn.Module, lr: float, gamma: float, device: torch.device):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.device = device  # Hardware Accelerator Reference

        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # 1. Condense tuples to contiguous NumPy arrays first, then cast to PyTorch device tensors
        state = torch.tensor(np.array(state), dtype=torch.float, device=self.device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float, device=self.device)
        action = torch.tensor(np.array(action), dtype=torch.long, device=self.device)
        reward = torch.tensor(np.array(reward), dtype=torch.float, device=self.device)
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

        # 3. The Bellman Equation
        target = pred.clone().detach()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # R + gamma * max(Q(S', a'))
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx])).item()

            action_idx = action[idx].item() if action.dim() == 1 else torch.argmax(action[idx]).item()
            target[idx][action_idx] = Q_new

        # 4. Backpropagation
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()