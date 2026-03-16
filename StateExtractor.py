import numpy as np
from collections import deque

class StateExtractor:

    @staticmethod
    def flood_fill(game, start_y: int, start_x: int) -> int:
        """
        Executes a Breadth-First Search to count contiguous safe squares.
        """
        # 1. Base Case: If the starting square is immediately fatal, return 0 space.
        if start_y < 0 or start_y >= game.height or start_x < 0 or start_x >= game.width:
            return 0
        if (start_y, start_x) in game.snake:
            return 0

        # 2. Initialization
        visited = set()
        queue = deque([(start_y, start_x)])
        visited.add((start_y, start_x))
        valid_space_count = 0

        # Convert snake deque to a hash set for O(1) lookup time.
        # This is computationally critical as we will check it hundreds of times per frame.
        obstacle_set = set(game.snake)

        # 3. BFS Loop
        while queue:
            curr_y, curr_x = queue.popleft()
            valid_space_count += 1

            # Check all 4 adjacent cells (Up, Down, Left, Right)
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = curr_y + dy, curr_x + dx

                # If within boundaries and not visited and not a snake body part
                if 0 <= ny < game.height and 0 <= nx < game.width:
                    if (ny, nx) not in visited and (ny, nx) not in obstacle_set:
                        visited.add((ny, nx))
                        queue.append((ny, nx))

        return valid_space_count

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
        head_y, head_x = game.snake[0]

        # 1. Define Ego-centric basis vectors based on current heading
        # Clockwise: 0=Up, 1=Right, 2=Down, 3=Left
        clock_wise = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        forward = clock_wise[current_action]
        right = clock_wise[(current_action + 1) % 4]
        backward = clock_wise[(current_action + 2) % 4]
        left = clock_wise[(current_action + 3) % 4]

        # 2. Project Food Position (Dot Product)
        food_y, food_x = game.food_pos
        food_dy = food_y - head_y
        food_dx = food_x - head_x

        # Normalize by board dimensions
        max_dim = max(game.height, game.width)
        food_forward = (food_dy * forward[0] + food_dx * forward[1]) / max_dim
        food_right = (food_dy * right[0] + food_dx * right[1]) / max_dim

        # 3. Rotated Flood Fill Volumes
        relative_deltas = [forward, right, backward, left]
        flood_fill_areas = []
        total_board_area = game.width * game.height

        for dy, dx in relative_deltas:
            simulated_y = head_y + dy
            simulated_x = head_x + dx
            raw_area = StateExtractor.flood_fill(game, simulated_y, simulated_x)
            flood_fill_areas.append(raw_area / total_board_area)

        # 4. Rotated 8-Directional Raycasts
        clock_wise_8 = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)
        ]

        ray_distances = []
        start_idx = current_action * 2
        for i in range(8):
            idx = (start_idx + i) % 8
            dy, dx = clock_wise_8[idx]
            ray_distances.append(StateExtractor.cast_ray(game, head_y, head_x, dy, dx))

        # 5. Construct Final 14-Dimensional Vector
        # 8 (rays) + 4 (flood fills) + 2 (food projections) = 14 inputs total
        state = ray_distances + flood_fill_areas + [food_forward, food_right]

        return np.array(state, dtype=np.float32)