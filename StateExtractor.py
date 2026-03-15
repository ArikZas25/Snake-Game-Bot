import numpy as np


class StateExtractor:

    @staticmethod
    def is_collision(point: tuple, game) -> bool:
        """
        Evaluates if a given (y, x) coordinate results in a terminal state.
        """
        y, x = point

        # 1. Boundary Collision Check
        if y < 0 or y >= game.height or x < 0 or x >= game.width:
            return True

        # 2. Body Collision Check
        if point in game.snake:
            return True

        return False

    @staticmethod
    def get_state(game, current_action: int) -> np.ndarray:
        """
        Translates the absolute game board into an 11-element relative state vector.
        """
        head = game.snake[0]
        head_y, head_x = head

        # Step 1: Define Absolute Adjacent Coordinates
        point_u = (head_y - 1, head_x)
        point_d = (head_y + 1, head_x)
        point_l = (head_y, head_x - 1)
        point_r = (head_y, head_x + 1)

        # Step 2: Extract Current Heading
        # Directions: 0: Up, 1: Right, 2: Down, 3: Left
        dir_u = current_action == 0
        dir_r = current_action == 1
        dir_d = current_action == 2
        dir_l = current_action == 3

        # Step 3: Calculate Relative Danger (The most complex logical step)
        danger_straight = (
                (dir_u and StateExtractor.is_collision(point_u, game)) or
                (dir_r and StateExtractor.is_collision(point_r, game)) or
                (dir_d and StateExtractor.is_collision(point_d, game)) or
                (dir_l and StateExtractor.is_collision(point_l, game))
        )

        danger_right = (
                (dir_u and StateExtractor.is_collision(point_r, game)) or
                (dir_r and StateExtractor.is_collision(point_d, game)) or
                (dir_d and StateExtractor.is_collision(point_l, game)) or
                (dir_l and StateExtractor.is_collision(point_u, game))
        )

        danger_left = (
                (dir_u and StateExtractor.is_collision(point_l, game)) or
                (dir_r and StateExtractor.is_collision(point_u, game)) or
                (dir_d and StateExtractor.is_collision(point_r, game)) or
                (dir_l and StateExtractor.is_collision(point_d, game))
        )

        # Step 4: Calculate Relative Food Location
        food_y, food_x = game.food_pos
        food_u = food_y < head_y
        food_d = food_y > head_y
        food_l = food_x < head_x
        food_r = food_x > head_x

        # Step 5: Construct and Cast the Vector
        state = [
            danger_straight, danger_right, danger_left,
            dir_u, dir_r, dir_d, dir_l,
            food_u, food_r, food_d, food_l
        ]

        # Convert the boolean array to an integer array (1s and 0s)
        return np.array(state, dtype=int)