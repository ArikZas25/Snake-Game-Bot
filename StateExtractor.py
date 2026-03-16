import numpy as np


class StateExtractor:

    @staticmethod
    def cast_ray(game, start_y: int, start_x: int, dy: int, dx: int) -> float:
        """
        Projects a ray in direction (dy, dx) and returns the inverse grid distance
        to the nearest obstacle (wall or snake body).
        """
        distance = 0
        curr_y = start_y + dy
        curr_x = start_x + dx

        while True:
            distance += 1

            # 1. Boundary Collision Check
            if curr_y < 0 or curr_y >= game.height or curr_x < 0 or curr_x >= game.width:
                break

            # 2. Body Collision Check
            if (curr_y, curr_x) in game.snake:
                break

            # Step forward
            curr_y += dy
            curr_x += dx

        # Return inverse distance (1.0 = immediate danger, approaching 0.0 = safe)
        return 1.0 / distance

    @staticmethod
    def get_state(game, current_action: int) -> np.ndarray:
        """
        Translates the game board into a 14-element continuous state vector.
        """
        head_y, head_x = game.snake[0]

        # Step 1: Raycast in 8 absolute directions
        # Order: Up, Up-Right, Right, Down-Right, Down, Down-Left, Left, Up-Left
        directions = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)
        ]

        ray_distances = [
            StateExtractor.cast_ray(game, head_y, head_x, dy, dx)
            for dy, dx in directions
        ]

        # Step 2: Extract Current Heading (One-hot encoded)
        dir_u = 1.0 if current_action == 0 else 0.0
        dir_r = 1.0 if current_action == 1 else 0.0
        dir_d = 1.0 if current_action == 2 else 0.0
        dir_l = 1.0 if current_action == 3 else 0.0

        # Step 3: Calculate Relative Food Location (Normalized by board dimensions)
        food_y, food_x = game.food_pos
        rel_food_y = (food_y - head_y) / game.height
        rel_food_x = (food_x - head_x) / game.width

        # Step 4: Construct and Cast the Vector
        state = ray_distances + [dir_u, dir_r, dir_d, dir_l, rel_food_y, rel_food_x]

        # Cast to float32 - critical for PyTorch execution speed
        return np.array(state, dtype=np.float32)