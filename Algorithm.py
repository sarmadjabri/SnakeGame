import pygame
import random
import heapq

# Constants
WIDTH = 30
HEIGHT = 20
CELL_SIZE = 40  # Adjusted for better visibility
SCREEN_WIDTH = WIDTH * CELL_SIZE
SCREEN_HEIGHT = HEIGHT * CELL_SIZE

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
DIRECTIONS = [UP, DOWN, LEFT, RIGHT]

class Snake:
    def __init__(self):
        self.body = [(0, HEIGHT - 1)]  # Start at the bottom
        self.grow = False
        self.last_direction = RIGHT  # Start moving to the right

    def head(self):
        return self.body[0]

    def move(self, direction):
        # Prevent moving in the opposite direction
        if direction == (self.last_direction[0] * -1, self.last_direction[1] * -1):
            return False

        dx, dy = direction
        new_head = (self.head()[0] + dx, self.head()[1] + dy)

        # Check for self-collision or wall collision
        if (new_head in self.body or
                new_head[1] < 0 or new_head[0] < 0 or
                new_head[0] >= WIDTH or new_head[1] >= HEIGHT):  # Check for walls
            return False

        # Move the snake
        self.body.insert(0, new_head)
        self.last_direction = direction  # Update the direction

        if not self.grow:
            self.body.pop()  # Remove tail unless growing
        else:
            self.grow = False

        return True

    def grow_snake(self):
        self.grow = True

class Game:
    def __init__(self):
        self.snake = Snake()
        self.food = self.place_food()
        self.running = True
        self.score = 0  # Initialize score

    def place_food(self):
        while True:
            food_position = (random.randint(0, WIDTH - 1), random.randint(0, HEIGHT - 1))
            if food_position not in self.snake.body:
                return food_position

    def a_star_search(self):
        start = self.snake.head()
        goal = self.food
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal:
                return self.reconstruct_path(came_from, current)

            for direction in DIRECTIONS:
                neighbor = (current[0] + direction[0], current[1] + direction[1])

                # Valid move check (no body or wall)
                if (0 <= neighbor[0] < WIDTH and
                        0 <= neighbor[1] < HEIGHT and
                        neighbor not in self.snake.body):  # Valid move
                    tentative_g_score = g_score[current] + 1
                    if tentative_g_score < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)

                        if neighbor not in [i[1] for i in open_set]:
                            heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []  # No path found

    def heuristic(self, a, b):
        # Manhattan distance
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        total_path.reverse()
        return total_path

    def update(self):
        path = self.a_star_search()
        if path and len(path) > 1:
            # Calculate the direction for the snake
            direction = (path[1][0] - path[0][0], path[1][1] - path[0][1])

            # Move the snake in the calculated direction
            if not self.snake.move(direction):
                self.running = False  # Game over if snake hits itself or wall
        else:
            self.running = False  # No path found, game over

        if self.snake.head() == self.food:
            self.snake.grow_snake()
            self.food = self.place_food()  # Place new food
            self.score += 1  # Increment score when snake eats food

    def render(self, screen):
        screen.fill((0, 0, 0))  # Clear the screen

        # Draw the snake
        for x, y in self.snake.body:
            pygame.draw.rect(screen, (0, 255, 0), (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        # Draw the food
        fx, fy = self.food
        pygame.draw.rect(screen, (255, 0, 0), (fx * CELL_SIZE, fy * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        # Render the score
        font = pygame.font.SysFont('Arial', 24)
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        screen.blit(score_text, (10, 10))

    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Row-Clearing Snake Game")
        clock = pygame.time.Clock()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            self.update()
            self.render(screen)
            pygame.display.flip()
            clock.tick(25)  # Game speed

        pygame.quit()
        print(f"Game Over! Final Score: {self.score}")  # Print the final score

# Start the game
if __name__ == "__main__":
    game = Game()
    game.run()
