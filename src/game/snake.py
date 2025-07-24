
import collections

class Snake:
    def __init__(self, initial_pos, initial_direction, initial_length):
        self.body = collections.deque([initial_pos])
        self.direction = initial_direction
        self.length = initial_length
        self.is_alive = True

    def move(self, new_direction):
        self.direction = new_direction
        head = self.body[0]
        
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        self.body.appendleft(new_head)

        if len(self.body) > self.length:
            self.body.pop()

    def grow(self):
        self.length += 1

    def get_head(self):
        return self.body[0]

    def get_body(self):
        return list(self.body)

    def copy(self):
        new_snake = Snake((0,0), (0,1), 1) # Dummy values
        new_snake.body = self.body.copy()
        new_snake.direction = self.direction
        new_snake.length = self.length
        new_snake.is_alive = self.is_alive
        return new_snake
