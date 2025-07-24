
import random
import numpy as np
import collections
from .snake import Snake

class Board:
    DIRECTIONS = {
        0: (-1, 0),  # Up
        1: (1, 0),   # Down
        2: (0, -1),  # Left
        3: (0, 1),   # Right
    }
    def __init__(self, width, height, num_snakes=2):
        self.width = width
        self.height = height
        self.num_snakes = num_snakes
        self.snakes = []
        self.food = None
        self.is_game_over = False
        self.winner = None

    def initialize(self):
        self.snakes = []
        # For single snake, ensure it spawns at least 3 units away from walls
        # Assuming board dimensions are large enough (e.g., > 6)
        min_coord_x = 3
        max_coord_x = self.width - 4
        min_coord_y = 3
        max_coord_y = self.height - 4

        # Ensure valid range for random.randint
        if min_coord_x > max_coord_x:
            min_coord_x = 0
            max_coord_x = self.width - 1
        if min_coord_y > max_coord_y:
            min_coord_y = 0
            max_coord_y = self.height - 1

        initial_pos = (random.randint(min_coord_x, max_coord_x), random.randint(min_coord_y, max_coord_y))
        initial_direction = random.choice(list(self.DIRECTIONS.values()))
        self.snakes.append(Snake(initial_pos, initial_direction, 3))

        self._place_food()
        self.is_game_over = False
        self.winner = None

        self._place_food()
        self.is_game_over = False
        self.winner = None

    def _place_food(self):
        while True:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if (x, y) not in [body for s in self.snakes for body in s.get_body()]:
                self.food = (x, y)
                return

    def update(self, snake_id, action):
        snake = self.snakes[snake_id]
        if not snake.is_alive:
            return

        direction = self.DIRECTIONS.get(action)

        if direction and self._is_valid_direction(snake, direction):
            snake.move(direction)
        else: # Continue in the same direction if action is invalid
            snake.move(snake.direction)

        self._handle_collisions()
        self._check_for_food()

    def _is_valid_direction(self, snake, new_direction):
        if not snake or not snake.direction:
            return True
        # Prevent the snake from reversing
        return (new_direction[0] * -1, new_direction[1] * -1) != snake.direction

    def get_valid_moves(self, snake_id):
        valid_moves = []
        snake = self.snakes[snake_id]
        if not snake.is_alive:
            return [] # No valid moves for a dead snake

        current_head = snake.get_head()
        snake_body_parts = list(snake.get_body())

        for action_idx, direction_tuple in self.DIRECTIONS.items():
            # Check if the direction is valid (not reversing)
            if not self._is_valid_direction(snake, direction_tuple):
                continue

            # Calculate the next head position
            next_head_x = current_head[0] + direction_tuple[0]
            next_head_y = current_head[1] + direction_tuple[1]
            next_head = (next_head_x, next_head_y)

            # Check for wall collision
            if not (0 <= next_head_x < self.width and 0 <= next_head_y < self.height):
                continue # This move leads to wall collision

            # Check for self-collision (excluding the last segment, which will move)
            # If the snake is about to eat food, its tail won't pop, so we need to consider the full body
            # For simplicity, we'll check against all body parts except the current head
            # A more robust check would involve simulating the move and then checking collision
            # For now, let's assume the tail moves out of the way unless it's growing.
            # The simplest safe check is against all body parts except the very last one (tail)
            # if next_head in snake_body_parts[:-1]: # This is problematic if snake is growing
            
            # A more accurate check: if the next_head is any part of the body *except* the current tail position
            # (because the tail will move out of the way unless the snake just ate food and is growing)
            # For now, let's use a simpler check that might be slightly conservative but safe:
            # If the next head is any part of the body except the current tail, it's a collision.
            # This is a common simplification for valid move checks.
            
            # Let's create a temporary body to simulate the move for collision check
            temp_body = collections.deque(snake_body_parts)
            temp_body.appendleft(next_head)
            if len(temp_body) > snake.length: # If not growing, pop tail
                temp_body.pop()
            
            # Check if the new head collides with any part of the *new* body (excluding itself)
            is_self_collision = False
            for i, part in enumerate(list(temp_body)):
                if i == 0: # Skip the new head itself
                    continue
                if part == next_head:
                    is_self_collision = True
                    break
            
            if is_self_collision:
                continue # This move leads to self-collision

            valid_moves.append(action_idx)
        
        # If no valid moves are found, it means the snake is trapped.
        # In such a case, the game should end, or the snake should be considered dead.
        # For MCTS, returning an empty list means no moves are possible, which will lead to game over.
        return valid_moves

    def _handle_collisions(self):
        for i, snake in enumerate(self.snakes):
            if not snake.is_alive:
                continue

            head = snake.get_head()

            # Wall collision
            if not (0 <= head[0] < self.width and 0 <= head[1] < self.height):
                snake.is_alive = False
                continue

            # Self collision
            if head in list(snake.get_body())[1:]:
                snake.is_alive = False
                continue

            # Other snake collision
            for j, other_snake in enumerate(self.snakes):
                if i == j:
                    continue
                if head in other_snake.get_body():
                    snake.is_alive = False
                    break
        
        # For single snake game, game is over when the snake is not alive
        if not self.snakes[0].is_alive:
            self.is_game_over = True
            self.winner = None # No winner if the only snake dies

    def _check_for_food(self):
        for snake in self.snakes:
            if not snake.is_alive:
                continue
            if snake.get_head() == self.food:
                snake.grow()
                self._place_food()
                break # Only one snake can eat the food per turn

    

    def get_state(self, snake_id):
        # Create a state representation from the perspective of a specific snake
        state = np.zeros((3, self.width, self.height), dtype=np.float32)

        # Channel 0: Player's snake head
        player_snake = self.snakes[snake_id]
        if player_snake.is_alive:
            head_x, head_y = player_snake.get_head()
            if 0 <= head_x < self.width and 0 <= head_y < self.height:
                state[0, head_x, head_y] = 1

        # Channel 1: Player's snake body
        if player_snake.is_alive:
            for part in list(player_snake.get_body())[1:]:
                body_x, body_y = part
                if 0 <= body_x < self.width and 0 <= body_y < self.height:
                    state[1, body_x, body_y] = 1

        # Channel 2: Other snakes' bodies (as obstacles)
        for i, snake in enumerate(self.snakes):
            if i == snake_id or not snake.is_alive:
                continue
            for part in snake.get_body():
                body_x, body_y = part
                if 0 <= body_x < self.width and 0 <= body_y < self.height:
                    state[1, body_x, body_y] = 1 # Treat other snakes' bodies as obstacles

        # Channel 3: Food
        if self.food:
            food_x, food_y = self.food
            if 0 <= food_x < self.width and 0 <= food_y < self.height:
                state[2, food_x, food_y] = 1
        
        return state

    def copy(self):
        new_board = Board(self.width, self.height, self.num_snakes)
        new_board.snakes = [s.copy() for s in self.snakes]
        new_board.food = self.food
        new_board.is_game_over = self.is_game_over
        new_board.winner = self.winner
        return new_board
