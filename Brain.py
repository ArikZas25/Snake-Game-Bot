import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ── WHAT IS A LINEAR LAYER? ───────────────────────────────────────────────────
# nn.Linear(in, out) does this math: output = input × weight + bias
# It takes an array of `in` numbers and transforms it into `out` numbers
# The weights are what the network LEARNS over time

# ── WHAT IS RELU? ─────────────────────────────────────────────────────────────
# ReLU(x) = max(0, x)
# It simply kills any negative numbers → turns them to 0
# This lets the network learn non-linear patterns (not just straight lines)
# Without it, stacking layers would be pointless mathematically

# ── WHY A DEEPER NETWORK? ─────────────────────────────────────────────────────
# Layer 1: learns low-level patterns (wall proximity, space density)
# Layer 2: learns high-level strategies (am I about to trap myself?)


class Linear_QNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)        # layer 1: raw features
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)  # layer 2: composed features
        self.fc3 = nn.Linear(hidden_size // 2, output_size)  # layer 3: Q-values output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 21 → 256 → 128 → 3


if __name__ == "__main__":
    model = Linear_QNet(input_size=21, hidden_size=256, output_size=3)
    dummy = torch.rand(21)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output: {model(dummy)}")
    print(f"Output shape: {model(dummy).shape}")


class QTrainer:
    def __init__(self, model: nn.Module, lr: float, gamma: float):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state      = torch.tensor(state,      dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action     = torch.tensor(action,     dtype=torch.long)
        reward     = torch.tensor(reward,     dtype=torch.float)

        if len(state.shape) == 1:
            state      = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action     = torch.unsqueeze(action, 0)
            reward     = torch.unsqueeze(reward, 0)
            done       = (done,)

        pred   = self.model(state)
        target = pred.clone().detach()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx])).item()

            action_idx = action[idx].item() if action.dim() == 1 else torch.argmax(action[idx]).item()
            target[idx][action_idx] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()