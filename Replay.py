import json
import pygame
import sys
import os
from typing import List, Dict


class ReplayViewer:
    def __init__(self, width: int = 10, height: int = 10):
        pygame.init()
        self.CELL_SIZE = 40
        self.width = width
        self.height = height

        # Screen dimensions based on your Snake.py logic
        self.screen = pygame.display.set_mode((self.width * self.CELL_SIZE, self.height * self.CELL_SIZE + 40))
        pygame.display.set_caption("Snake AI - Telemetry Viewer")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 24)
        self.small_font = pygame.font.SysFont("Arial", 16)

        self.COLORS = {
            "bg": (20, 20, 20),
            "text": (255, 255, 255),
            "text_dim": (150, 150, 150),
            "highlight": (0, 150, 255),
            "head": (0, 255, 100),
            "body": (0, 180, 60),
            "food": (255, 60, 60)
        }

        # Application State
        self.state = "MENU"  # Can be "MENU" or "PLAYBACK"
        self.fps = 10  # Default playback speed
        self.replay_files: List[str] = []
        self.selected_index = 0
        self.menu_scroll = 0  # For pagination

    def _load_replay_files(self) -> None:
        """Dynamically reads the replays directory and sorts them numerically."""
        if not os.path.exists("replays"):
            self.replay_files = []
            return

        files = [f for f in os.listdir("replays") if f.endswith(".json")]

        # Sort files by extracting the integer from 'run_XXX.json'
        try:
            files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        except ValueError:
            files.sort()  # Fallback to alphabetical if naming convention fails

        self.replay_files = files

    def run(self) -> None:
        """The main application loop acting as the State Machine."""
        while True:
            if self.state == "MENU":
                self._menu_loop()
            elif self.state == "PLAYBACK":
                self._playback_loop()

    def _menu_loop(self) -> None:
        """Handles rendering and logic for the file selection menu."""
        self._load_replay_files()

        # Guard against empty directory
        if not self.replay_files:
            print("[!] No replays found in the 'replays/' directory.")
            pygame.quit()
            sys.exit()

        menu_running = True
        while menu_running:
            self.screen.fill(self.COLORS["bg"])

            # 1. Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.selected_index = max(0, self.selected_index - 1)
                    elif event.key == pygame.K_DOWN:
                        self.selected_index = min(len(self.replay_files) - 1, self.selected_index + 1)
                    elif event.key == pygame.K_RETURN:
                        self.state = "PLAYBACK"
                        menu_running = False  # Exit menu loop to transition state
                    elif event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()

            # Adjust scroll window if cursor goes out of bounds
            if self.selected_index < self.menu_scroll:
                self.menu_scroll = self.selected_index
            elif self.selected_index >= self.menu_scroll + 10:
                self.menu_scroll = self.selected_index - 9

            # 2. Render UI
            title = self.font.render("Select Replay (Arrows to Navigate, Enter to Play)", True, self.COLORS["text"])
            self.screen.blit(title, (10, 10))

            for i in range(10):  # Show 10 items at a time
                file_idx = self.menu_scroll + i
                if file_idx >= len(self.replay_files):
                    break

                filename = self.replay_files[file_idx]
                is_selected = (file_idx == self.selected_index)

                color = self.COLORS["text"] if is_selected else self.COLORS["text_dim"]
                prefix = "> " if is_selected else "  "

                item_text = self.font.render(f"{prefix}{filename}", True, color)
                self.screen.blit(item_text, (20, 50 + (i * 30)))

            pygame.display.flip()
            self.clock.tick(30)  # Menu doesn't need high FPS

    def _playback_loop(self) -> None:
        """Handles rendering a specific JSON payload frame by frame."""
        filename = self.replay_files[self.selected_index]
        filepath = os.path.join("replays", filename)

        try:
            with open(filepath, 'r') as f:
                history = json.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            self.state = "MENU"
            return

        pygame.display.set_caption(f"Playing: {filename} - FPS: {self.fps}")

        frame_idx = 0
        total_frames = len(history)
        playback_running = True

        while playback_running and frame_idx < total_frames:
            # 1. Input Handling for Playback Controls
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        # Transition back to menu
                        self.state = "MENU"
                        playback_running = False
                    elif event.key == pygame.K_RIGHT:
                        self.fps = min(60, self.fps + 5)  # Speed up
                        pygame.display.set_caption(f"Playing: {filename} - FPS: {self.fps}")
                    elif event.key == pygame.K_LEFT:
                        self.fps = max(5, self.fps - 5)  # Slow down
                        pygame.display.set_caption(f"Playing: {filename} - FPS: {self.fps}")

            if not playback_running:
                break  # Exit the frame iteration early if user pressed ESC

            # 2. Extract Data
            frame = history[frame_idx]
            self.screen.fill(self.COLORS["bg"])

            # 3. Draw UI Overlay
            ui_text = self.font.render(f"Score: {frame['score']}", True, self.COLORS["text"])
            self.screen.blit(ui_text, (10, 5))

            controls_text = self.small_font.render(f"[ESC] Back to Menu | [Left/Right] Speed: {self.fps}", True,
                                                   self.COLORS["highlight"])
            self.screen.blit(controls_text, (10, self.height * self.CELL_SIZE + 15))

            # 4. Draw Food
            fy, fx = frame["food"]
            food_rect = pygame.Rect(fx * self.CELL_SIZE, fy * self.CELL_SIZE + 40, self.CELL_SIZE - 2,
                                    self.CELL_SIZE - 2)
            pygame.draw.rect(self.screen, self.COLORS["food"], food_rect, border_radius=6)

            # 5. Draw Snake
            for idx, (sy, sx) in enumerate(frame["snake"]):
                color = self.COLORS["head"] if idx == 0 else self.COLORS["body"]
                snake_rect = pygame.Rect(sx * self.CELL_SIZE, sy * self.CELL_SIZE + 40, self.CELL_SIZE - 2,
                                         self.CELL_SIZE - 2)
                pygame.draw.rect(self.screen, color, snake_rect, border_radius=6)

            pygame.display.flip()

            # 6. Time Step
            self.clock.tick(self.fps)
            frame_idx += 1

        # End of recording behavior: Go back to menu automatically
        if self.state != "MENU":
            pygame.time.wait(1000)  # Pause to see the final frame
            self.state = "MENU"


if __name__ == "__main__":
    viewer = ReplayViewer()
    viewer.run()