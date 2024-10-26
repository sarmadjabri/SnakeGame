import pygame
import random
import time

# Constants
WIDTH = 20
HEIGHT = 20
CELL_SIZE = 20
SCREEN_WIDTH = WIDTH * CELL_SIZE
SCREEN_HEIGHT = HEIGHT * CELL_SIZE

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# Directions in order of edge following
DIRECTIONS = [RIGHT, DOWN, LEFT, UP]

class Snake:
    def __init__(self):
        self.body = [(0, 0)]
        self.direction_index = 0
        self.grow = False

    def head(self):
        return self.body[0]

    def move(self):
        dx, dy = DIRECTIONS[self.direction_index]
        new_head = (self.head()[0] + dx, self.head()[1] + dy)

        # Check for self-collision or wall collision
        if (new_head in self.body or
                new_head[0] < 0 or new_head[0] >= WIDTH or
                new_head[1] < 0 or new_head[1] >= HEIGHT):
            return False

        # Move the snake
        self.body.insert(0, new_head)

        if not self.grow:
            self.body.pop()  # Remove tail unless growing
        else:
            self.grow = False

        return True

    def change_direction(self):
        self.direction_index = (self.direction_index + 1) % len(DIRECTIONS)

    def grow_snake(self):
        self.grow = True

class Game:
    def __init__(self):
        self.snake = Snake()
        self.food = self.place_food()
        self.running = True

    def place_food(self):
        while True:
            food_position = (random.randint(0, WIDTH - 1), random.randint(0, HEIGHT - 1))
            if food_position not in self.snake.body:
                return food_position

    def update(self):
        if not self.snake.move():
            self.running = False  # Game over

        if self.snake.head() == self.food:
            self.snake.grow_snake()
            self.food = self.place_food()  # Place new food

    def render(self, screen):
        screen.fill((0, 0, 0))  # Clear the screen
        for x, y in self.snake.body:
            pygame.draw.rect(screen, (0, 255, 0), (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        fx, fy = self.food
        pygame.draw.rect(screen, (255, 0, 0), (fx * CELL_SIZE, fy * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Snake Game")
        clock = pygame.time.Clock()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            self.update()
            self.render(screen)
            pygame.display.flip()
            clock.tick(10)  # Game speed

            self.snake.change_direction()  # Change direction for edge following logic

        pygame.quit()

# Start the game
if __name__ == "__main__":
    game = Game()
    game.run()
