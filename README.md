# 🐍 Snake-Game-Bot

> A Snake game with a Deep Q-Network (DQN) AI that learns to play by itself — built from scratch using Python, PyTorch, and Pygame.

---

## 📁 Project Structure

| File | Role | Description |
|------|------|-------------|
| `Snake.py` | The Game World | Game logic, board, movement, food, collisions |
| `Brain.py` | The Neural Network | PyTorch model that takes 11 inputs → outputs 4 Q-values |
| `Agent.py` | The Decision Maker | Connects the brain to the game, handles learning |
| `Main.py` | The Entry Point | Runs everything together |

---

## 🎮 How the Game Works

The board is a 10×10 grid where every cell is a number:

```
0 = empty
1 = snake head
2 = snake body
3 = food
```

Every step, `Snake.py` moves the snake and returns:
- **New board state** — what the grid looks like now
- **Reward** — was that a good move?
- **Done** — is the game over?

### Rewards
| Event | Reward |
|-------|--------|
| Ate food 🍎 | +10 |
| Hit wall or itself 💀 | -10 |
| Just moved | -0.1 |

---

## 🧠 How the AI Works

Instead of a human pressing WASD, a neural network decides the next move.

### The 11 Input Numbers (State)

Rather than feeding the whole 10×10 board (100 numbers) to the AI, we give it 11 key facts:

```
[ danger_straight, danger_right, danger_left,        ← 3 numbers
  moving_up, moving_right, moving_down, moving_left, ← 4 numbers
  food_up, food_right, food_down, food_left ]         ← 4 numbers
```

**Example:** Snake moving right, food above, wall straight ahead:
```
[1, 0, 0,  0, 1, 0, 0,  1, 0, 0, 0]
```

### The Neural Network (Brain.py)

```
11 numbers → [Linear Layer: 11→256] → [ReLU] → [Linear Layer: 256→4] → 4 Q-values
```

- **`nn.Linear`** — multiplies inputs by learned weights. This is where knowledge is stored.
- **`ReLU`** — turns all negative numbers to 0. Lets the network learn complex patterns, not just straight lines.
- **4 Q-values** — one per direction (Up, Right, Down, Left). The highest = chosen action.

### The Learning Loop

```
1. Look at board → extract 11 numbers (state)
2. Feed into Brain → get 4 Q-values → pick highest (action)
3. Make that move → get reward from Snake.py
4. Store: (state, action, reward, next_state) in memory
5. Sample random memories → train the brain
6. Repeat thousands of times
```

The AI starts completely random and gradually learns which moves lead to food and which lead to death.

---

## ⚙️ Tech Stack

- **Python 3** — core language
- **NumPy** — board representation and math
- **PyTorch** — neural network and training
- **Pygame** — visual game window

---

## 🚀 Setup

```bash
# Install dependencies
pip install numpy torch pygame

# Play the game manually
python Main.py

# Train the AI (coming soon)
python Agent.py
```

---

## 👥 Team

| Role | Responsibility |
|------|---------------|
| **Arik & Yuval** — Game Engineers | `Snake.py` — game logic, movement, reward system, Pygame window |
| **Role A** — Agent Architect | `Agent.py` — DQN agent, replay memory, training loop |
| **Role B** — DL Architect | `Brain.py` — neural network design, PyTorch model |

---

## 📈 Roadmap

- [x] Snake game environment
- [x] Pygame visual window
- [x] Neural network (Brain.py)
- [ ] State extraction (11 numbers from board)
- [ ] Replay memory
- [ ] DQN training loop
- [ ] Trained model that beats the game

---