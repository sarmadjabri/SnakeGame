import numpy as np
import matplotlib.pyplot as plt
import json
import time

# Load scores from scores.json
def load_scores(file_path='scores.json'):
    with open(file_path, 'r') as f:
        return json.load(f)

class SnakeGame:
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        self.snake_position = [(0, 0)]
        self.direction = (0, 1)
        self.food_position = self._generate_food()
        self.score = 0

    def _generate_food(self):
        while True:
            food_x = np.random.randint(0, self.width)
            food_y = np.random.randint(0, self.height)
            if (food_x, food_y) not in self.snake_position:
                return (food_x, food_y)

    def step(self, action):
        self._update_direction(action)
        head_x, head_y = self.snake_position[0]
        new_head_x = (head_x + self.direction[0]) % self.width
        new_head_y = (head_y + self.direction[1]) % self.height
        self.snake_position.insert(0, (new_head_x, new_head_y))

        # Check for collisions with self
        if len(self.snake_position) > 1 and (new_head_x, new_head_y) in self.snake_position[1:]:
            return True  # kill game

        # Check for food
        if (new_head_x, new_head_y) == self.food_position:
            self.food_position = self._generate_food()
            self.score += 1
        else:
            self.snake_position.pop()  # Remove tail to keep za length

        return False  # Continue games

    def _update_direction(self, action):
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        self.direction = directions[action]

    def render(self):
        # Create a grid for rendering
        grid = np.zeros((self.height, self.width))
        for (x, y) in self.snake_position:
            grid[x, y] = 1  # Snake
        grid[self.food_position] = 2  # Food

        plt.imshow(grid, cmap='hot', interpolation='nearest')
        plt.axis('off')  # Hid
        plt.draw()
        plt.pause(0.1)

def main():
    scores = load_scores()
    plt.ion()
    game = SnakeGame()

    for score in scores:
        game.reset()
        action = 0  #
        while True:
            game_over = game.step(action)
            game.render()
            if game_over:
                break
            time.sleep(0.1)  # Slow down the renderin

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
