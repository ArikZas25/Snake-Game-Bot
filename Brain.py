import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CNN_QNet(nn.Module):
    def __init__(self, output_size: int = 3):
        super().__init__()
        # Input: (batch, 3, 10, 10)
        # Channel 0: snake head
        # Channel 1: snake body
        # Channel 2: food
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   # → (32, 10, 10)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # → (64, 10, 10)
        self.fc1   = nn.Linear(64 * 10 * 10, 256)
        self.fc2   = nn.Linear(256, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class QTrainer:
    def __init__(self, model: nn.Module, lr: float, gamma: float):
        self.lr        = lr
        self.gamma     = gamma
        self.model     = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state      = torch.tensor(state,      dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action     = torch.tensor(action,     dtype=torch.long)
        reward     = torch.tensor(reward,     dtype=torch.float)

        # Make sure we always have a batch dimension
        if state.dim() == 3:          # single sample: (3, 10, 10)
            state      = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action     = action.unsqueeze(0)
            reward     = reward.unsqueeze(0)
            done       = (done,)

        pred   = self.model(state)
        target = pred.clone().detach()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx].unsqueeze(0))).item()

            action_idx = action[idx].item() if action.dim() == 1 else torch.argmax(action[idx]).item()
            target[idx][action_idx] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    model = CNN_QNet(output_size=3)
    dummy = torch.rand(1, 3, 10, 10)
    out   = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output: {out}")