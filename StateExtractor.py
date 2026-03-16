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

        # Step 1: Raycast in 8 absolute directions
        directions = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)
        ]
        ray_distances = [
            StateExtractor.cast_ray(game, head_y, head_x, dy, dx)
            for dy, dx in directions
        ]

        # Step 2: Extract Current Heading
        dir_u = 1.0 if current_action == 0 else 0.0
        dir_r = 1.0 if current_action == 1 else 0.0
        dir_d = 1.0 if current_action == 2 else 0.0
        dir_l = 1.0 if current_action == 3 else 0.0

        # Step 3: Calculate Relative Food Location
        food_y, food_x = game.food_pos
        rel_food_y = (food_y - head_y) / game.height
        rel_food_x = (food_x - head_x) / game.width

        # Step 4: Simulate adjacent moves and calculate Flood Fill volumes
        # Order must match action indices: Up (0), Right (1), Down (2), Left (3)
        action_deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        total_board_area = game.width * game.height

        flood_fill_areas = []
        for dy, dx in action_deltas:
            simulated_y = head_y + dy
            simulated_x = head_x + dx
            raw_area = StateExtractor.flood_fill(game, simulated_y, simulated_x)
            # Normalize between 0.0 and 1.0
            flood_fill_areas.append(raw_area / total_board_area)

        # Step 5: Construct the Final Vector
        # 8 (rays) + 4 (directions) + 2 (food) + 4 (flood fills) = 18 total inputs
        state = ray_distances + flood_fill_areas + [dir_u, dir_r, dir_d, dir_l, rel_food_y, rel_food_x]

        return np.array(state, dtype=np.float32)