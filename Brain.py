import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)   # layer 1
        self.fc2 = nn.Linear(hidden_size, output_size)  # layer 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))  # pass through layer 1, then ReLU
        x = self.fc2(x)          # pass through layer 2 (no ReLU at output)
        return x


# ── TEST IT ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Create the network: 11 inputs → 256 hidden → 4 outputs
    model = Linear_QNet(input_size=11, hidden_size=256, output_size=4)

    # Create a dummy state — 11 random float numbers simulating game state
    dummy_state = torch.rand(11)
    print(f"Input:  {dummy_state}")
    print(f"Shape:  {dummy_state.shape}")

    # Pass it through the network
    output = model(dummy_state)
    print(f"\nOutput: {output}")
    print(f"Shape:  {output.shape}")

    # This should print exactly 4 values — one Q-value per action
    # Action 0 = Up, 1 = Right, 2 = Down, 3 = Left
```

---

Run it and you should see something like:
```
Input:  tensor([0.23, 0.87, 0.45, ...])   ← 11 numbers
Output: tensor([0.12, -0.34, 0.56, 0.01]) ← 4 numbers