import json
import pygame
import numpy as np
import imageio
import os
import sys


class GifExporter:
    def __init__(self, width: int = 10, height: int = 10, cell_size: int = 40):
        self.width = width
        self.height = height
        self.cell_size = cell_size

        # We initialize the Pygame display in a hidden/dummy mode if possible,
        # but standard initialization works fine for a brief export process.
        pygame.init()
        self.screen = pygame.Surface((self.width * self.cell_size, self.height * self.cell_size + 40))

        self.font = pygame.font.SysFont("Arial", 24)

        self.COLORS = {
            "bg": (20, 20, 20),
            "text": (255, 255, 255),
            "head": (0, 255, 100),
            "body": (0, 180, 60),
            "food": (255, 60, 60)
        }

    def export_run(self, run_number: int, fps: int = 15, output_format: str = 'gif') -> None:
        """
        Parses a JSON telemetry file, renders it to a Pygame surface,
        extracts the pixel matrices, and encodes them to a media file.
        """
        input_filepath = os.path.join("replays", f"run_{run_number}.json")
        output_filepath = f"run_{run_number}_showcase.{output_format}"

        try:
            with open(input_filepath, 'r') as f:
                history = json.load(f)
        except FileNotFoundError:
            print(f"[Error] Telemetry file {input_filepath} not found.")
            return

        print(f"[*] Rendering {len(history)} frames into memory...")
        frames = []

        for frame_data in history:
            self.screen.fill(self.COLORS["bg"])

            # 1. Render Score
            score_text = self.font.render(f"Score: {frame_data['score']}", True, self.COLORS["text"])
            self.screen.blit(score_text, (10, 5))

            # 2. Render Food
            fy, fx = frame_data["food"]
            food_rect = pygame.Rect(fx * self.cell_size, fy * self.cell_size + 40, self.cell_size - 2,
                                    self.cell_size - 2)
            pygame.draw.rect(self.screen, self.COLORS["food"], food_rect, border_radius=6)

            # 3. Render Snake
            for idx, (sy, sx) in enumerate(frame_data["snake"]):
                color = self.COLORS["head"] if idx == 0 else self.COLORS["body"]
                snake_rect = pygame.Rect(sx * self.cell_size, sy * self.cell_size + 40, self.cell_size - 2,
                                         self.cell_size - 2)
                pygame.draw.rect(self.screen, color, snake_rect, border_radius=6)

            # 4. Matrix Extraction and Transformation
            # Extract 3D numpy array from the Pygame surface (W x H x 3)
            pixel_matrix = pygame.surfarray.array3d(self.screen)

            # Transpose axes 0 and 1 to convert from (W, H, 3) to (H, W, 3)
            corrected_matrix = np.transpose(pixel_matrix, (1, 0, 2))

            frames.append(corrected_matrix)

        print(f"[*] Encoding to {output_format.upper()} format at {fps} FPS. This may take a moment...")

        # 5. Media Encoding
        if output_format.lower() == 'gif':
            # duration is time per frame in milliseconds
            imageio.mimsave(output_filepath, frames, format='GIF', duration=1000 / fps, loop=0)
        elif output_format.lower() == 'mp4':
            imageio.mimsave(output_filepath, frames, format='FFMPEG', fps=fps)

        print(f"[+] Successfully exported artifact: {output_filepath}")


if __name__ == "__main__":
    exporter = GifExporter()
    # Replace 100 with whichever run you wish to export
    exporter.export_run(run_number=100, fps=15, output_format='gif')