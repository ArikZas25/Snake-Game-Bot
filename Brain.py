import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)   # layer 1
        self.fc2 = nn.Linear(hidden_size, output_size)  # layer 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))  # pass through layer 1, then ReLU
        x = self.fc2(x)          # pass through layer 2 (no ReLU at output)
        return x

# 11 inputs → 256 hidden → 4 outputs

# ── TEST IT ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Create the network: 11 inputs → 256 hidden → 4 outputs
    model = Linear_QNet(input_size=11, hidden_size=256, output_size=4)

    # Create a dummy state — 11 random float numbers simulating game state
    dummy_state = torch.rand(11)
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