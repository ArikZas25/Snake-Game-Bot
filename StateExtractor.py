import numpy as np
from collections import deque


class StateExtractor:

    @staticmethod
    def flood_fill(game, start_y: int, start_x: int, limit: int = None) -> int:
        """
        BFS to count reachable safe squares.
        limit: stops early once count reaches limit (used for trap detection).
        """
        if start_y < 0 or start_y >= game.height or start_x < 0 or start_x >= game.width:
            return 0
        if (start_y, start_x) in game.snake:
            return 0

        visited = set()
        queue = deque([(start_y, start_x)])
        visited.add((start_y, start_x))
        count = 0
        obstacle_set = set(game.snake)

        while queue:
            cy, cx = queue.popleft()
            count += 1
            if limit and count >= limit:
                return count
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < game.height and 0 <= nx < game.width:
                    if (ny, nx) not in visited and (ny, nx) not in obstacle_set:
                        visited.add((ny, nx))
                        queue.append((ny, nx))

        return count

    @staticmethod
    def cast_ray(game, start_y: int, start_x: int, dy: int, dx: int) -> float:
        distance = 0
        cy, cx = start_y + dy, start_x + dx
        obstacle_set = set(game.snake)

        while True:
            distance += 1
            if cy < 0 or cy >= game.height or cx < 0 or cx >= game.width:
                break
            if (cy, cx) in obstacle_set:
                break
            cy += dy
            cx += dx

        return 1.0 / distance

    @staticmethod
    def is_safe(game, y: int, x: int) -> bool:
        if y < 0 or y >= game.height or x < 0 or x >= game.width:
            return False
        if (y, x) in game.snake:
            return False
        return True

    @staticmethod
    def get_state(game, current_action: int) -> np.ndarray:
        head_y, head_x = game.snake[0]
        snake_length = len(game.snake)
        total_board_area = game.width * game.height

        # ── Step 1: Raycasts in 8 directions ─────────────────────────────────
        ray_directions = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)
        ]
        ray_distances = [
            StateExtractor.cast_ray(game, head_y, head_x, dy, dx)
            for dy, dx in ray_directions
        ]

        # ── Step 2: Current Heading (one-hot) ─────────────────────────────────
        dir_u = 1.0 if current_action == 0 else 0.0
        dir_r = 1.0 if current_action == 1 else 0.0
        dir_d = 1.0 if current_action == 2 else 0.0
        dir_l = 1.0 if current_action == 3 else 0.0

        # ── Step 3: Relative Food Location ────────────────────────────────────
        food_y, food_x = game.food_pos
        rel_food_y = (food_y - head_y) / game.height
        rel_food_x = (food_x - head_x) / game.width

        # ── Step 4: Flood fills — skip BFS for dead cells ─────────────────────
        action_deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # Up, Right, Down, Left

        raw_flood = {}
        for i, (dy, dx) in enumerate(action_deltas):
            ny, nx = head_y + dy, head_x + dx
            if not StateExtractor.is_safe(game, ny, nx):
                raw_flood[i] = 0  # wall or body — skip BFS entirely
            else:
                raw_flood[i] = StateExtractor.flood_fill(game, ny, nx)

        flood_fill_areas = [raw_flood[i] / total_board_area for i in range(4)]

        # ── Step 5: Trap scores for relative moves ────────────────────────────
        clock_wise = [0, 1, 2, 3]
        idx = clock_wise.index(current_action)
        absolute_straight = clock_wise[idx]
        absolute_right    = clock_wise[(idx + 1) % 4]
        absolute_left     = clock_wise[(idx - 1) % 4]

        def trap_score(action_abs):
            area = raw_flood[action_abs]
            if area == 0:
                return 0.0
            return min(area / max(snake_length, 1), 1.0)

        trap_straight = trap_score(absolute_straight)
        trap_right    = trap_score(absolute_right)
        trap_left     = trap_score(absolute_left)

        # ── Step 6: Final state vector (21 features) ──────────────────────────
        # 8 rays + 4 flood fills + 4 heading + 2 food + 3 trap = 21
        state = (
            ray_distances
            + flood_fill_areas
            + [dir_u, dir_r, dir_d, dir_l]
            + [rel_food_y, rel_food_x]
            + [trap_straight, trap_right, trap_left]
        )

        return np.array(state, dtype=np.float32)